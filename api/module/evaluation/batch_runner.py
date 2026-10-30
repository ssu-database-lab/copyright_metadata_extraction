"""배치 실행기 — 계약서+저작물 세트를 동시 처리하고 채점한다.

CLI 하네스와 웹 배치 API가 **이 클래스 하나**를 공유한다. 실행 경로가 달라도
같은 코드가 돌아야 "UI로 돌린 결과"와 "CLI로 돌린 결과"가 갈리지 않는다.

세트 1건 처리 순서(v3 채점 기준 plan/tta):
    1) 계약서 → 파이프라인(OCR → LLM ∥ NER → 통합검증)  … 제목·저자·라이선스·권리
    2) 저작물 → 파이프라인(VLM 또는 OCR 경로)            … 설명·키워드·해상도·파일속성
    3) 상속 병합(LLM 호출 없음)                          … 계약서 권리정보를 저작물에 병합
    4) 정답 대조 채점

계약서가 있어야 제목·저자·라이선스가 **문서에서 추출된 값**이 된다.
저작물 파일만 넣으면 이 3속성은 비거나(과소평가) 입력 메타데이터를 그대로
되돌려주게 되어(과대평가) 어느 쪽도 실제 성능이 아니다.

**v4 — 채점 기준 "eval23" (docs/v4/M2_runner_spec.md §1.B B2·B3)** 에서는 세트가 **다리 하나**다:
    T-001 계약서 단독 → 파이프라인(텍스트층/OCR → LLM ∥ NER → 통합검증) → response["test_prediction"](23키)
                        를 그대로 기록, 17~23 키는 null 로 고정
    T-002 저작물      → 계약서는 규칙 파싱(제목·종별, 모델 호출 없음)으로 **문맥만** 주고 저작물 leg 만 부른다
                        (consolidate_work=False · frame_schedule="v4"); 1~16 은 문맥의 제목·종별만 채우고 나머지 null
  여기서는 채점하지 않는다 — <out>/test_predictions/<set_id>.json (+ test_predictions.jsonl) 에 기록만 하고,
  지표는 M1 평가기(api.module.evaluation.v4.evaluate) 가 gold23.jsonl 과 대조해 test_metrics.json · report.md 를
  낸다(--gold 를 주면 finalize_eval23 이 호출). 다리가 하나뿐이므로 세트 = 스레드풀의 작업 단위 하나다.
  <out>/test.log 에 세트별 시작·종료·소요시간·오류를 한국어로 남긴다.

설계 요점:
  - 세트 단위 체크포인트(results.jsonl append) → 중단·재개 안전. 4,000건은 반드시 끊긴다.
  - 세트 단위 예외 격리 → 파일 1건이 깨져도 나머지 3,999건이 살아남는다.
  - 누적 비용 상한 → 폭주 루프가 곧 요금이다.
  - 취소 이벤트 → UI에서 중단 가능.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import log_context as LC
from .manifest import ManifestEntry, effective_test
from .scoring import attributes_for, score_set
from .v4.keys23 import KEY_NAMES, T001_KEYS, T002_KEYS, empty_prediction

logger = logging.getLogger(__name__)

# 계약서 문맥 파서(M2 A4, 구현자 A) — 규칙만으로 제목·종별을 읽는다. 아직 없으면 T-002 저작물 leg 는
# 문맥 없이 돌아가며, 이 모듈의 import 자체는 실패하지 않아야 한다. **첫 사용 때** 지연 import 한다 —
# 모듈 로드 때 api 패키지(torch·transformers)를 끌어오면 배치 CLI 의 --help 가 50초 걸린다.
#
# 스레드 안전성(M2b 수정 1): --workers > 1 이면 T-002 세트 여러 개가 동시에 첫 사용에 들어온다. 예전에는
# import 가 **끝나기 전에** 적재 플래그를 세워서, 두 번째 스레드부터는 "이미 적재됨 + 함수 None" 을 보고
# 문맥 None 으로 조용히 지나갔다(fitz 등을 올리는 수 초 동안 들어온 세트 전부). 잠금 + 성공한 뒤에만 플래그.
parse_contract_context = None          # 로드되면 함수, 없으면 None (테스트에서 바꿔 끼울 수 있다)
_CONTEXT_PARSER_LOADED = False         # import 가 **성공한 뒤에만** True
_CONTEXT_PARSER_LOCK = threading.Lock()
_CONTEXT_PARSER_FAILURE: Optional[BaseException] = None   # 첫 실패의 예외 — 한 번만 로그하고 기억한다


def _import_context_parser():
    """실제 import — api.module.… 우선, api/ 가 루트인 CLI·웹 경로는 module.… 로. 테스트는 느린 가짜로 바꿔 끼운다."""
    try:
        from api.module.clip_extraction.contract_context import parse as impl
    except ImportError:
        from module.clip_extraction.contract_context import parse as impl  # type: ignore
    return impl


def _load_context_parser():
    """계약서 문맥 파서를 **한 번만**, 스레드 안전하게 지연 import 한다.

    - 성공: parse_contract_context 에 함수를 두고 **그 뒤에** 플래그를 세운다 → 잠금을 기다린 스레드도 같은 함수를 받는다.
    - ImportError(모듈 없음): 경고를 **한 번만** 남기고 이후로는 조용히 None (A4 미반영 환경 — 문맥 없이 실행).
    - 그 외 예외(파서 모듈 자체가 깨짐): 사유를 한 번 기록하고 호출자에게 다시 던진다 — 4,000세트가 문맥 없이
      조용히 도는 것보다 세트별 FAIL 로 드러나는 쪽이 맞다. 이후 호출도 (다시 import 하지 않고) 같은 사유로 던진다.
    """
    global parse_contract_context, _CONTEXT_PARSER_LOADED, _CONTEXT_PARSER_FAILURE
    if _CONTEXT_PARSER_LOADED or parse_contract_context is not None:
        return parse_contract_context
    with _CONTEXT_PARSER_LOCK:
        if _CONTEXT_PARSER_LOADED or parse_contract_context is not None:   # 잠금을 기다리는 동안 다른 스레드가 끝냈다
            return parse_contract_context
        if _CONTEXT_PARSER_FAILURE is not None:                            # 이미 실패 — 다시 시도·다시 로그하지 않는다
            if isinstance(_CONTEXT_PARSER_FAILURE, ImportError):
                return None
            raise RuntimeError(f"계약서 문맥 파서 import 실패(앞서 기록됨): "
                               f"{type(_CONTEXT_PARSER_FAILURE).__name__}: {_CONTEXT_PARSER_FAILURE}") from _CONTEXT_PARSER_FAILURE
        try:
            impl = _import_context_parser()
        except ImportError as e:
            _CONTEXT_PARSER_FAILURE = e
            logger.warning("계약서 문맥 파서(api.module.clip_extraction.contract_context)를 올릴 수 없습니다 (%s) — "
                           "T-002 저작물 leg 는 계약서 문맥(제목·종별) 없이 실행됩니다 (A4 반영 시 자동 사용)", e)
            return None
        except Exception as e:
            _CONTEXT_PARSER_FAILURE = e
            logger.error("계약서 문맥 파서 import 중 예외 — %s: %s", type(e).__name__, e, exc_info=True)
            raise
        parse_contract_context = impl
        _CONTEXT_PARSER_LOADED = True
        return impl

EVAL23 = "eval23"            # v4 23키 기록 모드 — 채점은 M1 평가기가 한다
V3_SCORING_SETS = ("plan", "tta")

# 검증된 단가 (docs/API_비용산정서_20260730.md · 환율 $1=₩1,400)
COST_OCR_PER_PAGE = 2.7
COST_LLM_EXTRACT = 10.2
COST_CONSOLIDATION_DOC = 34.6
# 생각 끔(기본, 2026-10-20) 실측 — 계약서 200건 평균 토큰: 추출 입력 7,239 / 출력 1,505 · 중재 4,866 / 2,672
COST_LLM_EXTRACT_NO_THINK = 10.8
COST_CONSOLIDATION_DOC_NO_THINK = 14.7
COST_CONSOLIDATION_IMG = 35.4
COST_VLM_IMAGE = 1.34     # qwen3.8-omni-flash 실측 (파일럿 이미지 100건, 2026-10-11)
COST_VLM_VIDEO = 1.77     # qwen3.8-omni-flash 실측 (파일럿 영상, 2026-10-10/11)


@dataclass
class RunConfig:
    workers: int = 4
    limit: Optional[int] = None
    resume: bool = True
    max_cost_krw: Optional[float] = None
    model_name: str = "alibaba-qwen3.5-122b-a10b"
    ocr_provider: str = "alibaba"
    ocr_model: Optional[str] = None
    ner_model: str = "klue-roberta-large"
    consolidate: bool = True
    consolidation_model: str = "alibaba-qwen3.5-122b-a10b"
    vlm_prefer: str = "auto"
    contract_document_type: str = "저작재산권 이용허락 계약서"
    work_document_type: str = "기타문서"
    # tier D(파일 생성 날짜)는 기본 제외 — 정답은 '저작물 창작 시점'이고 추출값은
    # '파일 생성 시각'이라 서로 다른 양이다. 같은 값으로 채점하면 구조적 오답이 된다.
    # 시험기관과 정의가 합의되면 빈 튜플로 두어 채점에 포함시킨다.
    skip_tiers: tuple = ("D",)
    # 채점 기준: "plan" = 연구개발계획서 11속성(기본), "tta" = TTA 표준 14속성,
    # "eval23" = v4 23키 기록(채점은 M1 평가기). TTA 쪽은 계약서에 실제로 인쇄되는 항목만 담고 있어 정답 출처가 다르다.
    scoring_set: str = "plan"
    # --- v4(eval23) 전용 ---
    test: Optional[str] = None            # 작업 수준 시험 구분 T-001 | T-002 (엔트리의 test 가 우선)
    consolidate_work: bool = False        # 저작물 leg 중재자 — v4 기본 생략(M2 A3; 저작물 leg 비용의 97%), --consolidate-work 로 켠다
    frame_schedule: str = "v4"            # 영상 키프레임 수 스케줄 (M2 A6; 웹앱은 v3 기본값 그대로)
    gold: Optional[str] = None            # gold23.jsonl — 주면 끝에 M1 평가기를 돌려 test_metrics.json·report.md 를 쓴다
    tau: float = 0.65                     # 설명 코사인 임계값(M1 §5) — 평가기로 그대로 넘긴다
    content_check: bool = True            # 설명 판정의 내용 토큰 검사(M1 §5) — 평가기로 그대로 넘긴다
    embed_cache: Optional[str] = None     # 임베딩 캐시 jsonl(M1 §5) — 재평가 때 모델을 다시 올리지 않는다
    embed_revision: Optional[str] = None  # KURE-v1 의 HF revision 고정(M1 §5) — 산출물 settings.embed 에 기록된다
    n_boot: Optional[int] = None          # 부트스트랩 반복 수(M1 §7) — None 이면 평가기 기본값(1,000)
    title_fallback: bool = False          # T-002 계약서가 없는 세트에 매니페스트 제목(extra.title)을 문맥으로 준다 — 모든 저작물에 계약서가 있는 본 시험을 흉내 내는 개발·파일럿용. 기본 off
    only_predicted: bool = False          # gold 를 예측이 있는 set_id 로 제한(M2b 수정 4) — 개발·파일럿 런이 4,000세트 전부를 FN 으로 세지 않도록. 기본 off = M1 동작


def _pdf_page_count(path: str) -> int:
    """PDF 페이지 수. OCR 비용이 페이지 비례라 추정의 정확도를 좌우한다.

    PyMuPDF(fitz) → pypdf → pdfinfo 순으로 시도한다. 전부 실패하면 1로 본다
    (과소추정이지만 비용 상한이 조기에 걸리는 것보다 낫다).
    """
    p = str(path)
    if not p.lower().endswith(".pdf"):
        return 1
    try:
        import fitz                                   # PyMuPDF
        with fitz.open(p) as doc:
            return max(1, doc.page_count)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        return max(1, len(PdfReader(p).pages))
    except Exception:
        pass
    try:
        out = subprocess.run(["pdfinfo", p], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if line.lower().startswith("pages:"):
                return max(1, int(line.split()[1]))
    except Exception:
        pass
    return 1


def _thinking_off(env_name: str) -> bool:
    """추출·중재의 생각(thinking) 모드가 꺼져 있는가 — 기본 끔 (module/consolidator/consolidation_agent.py 와 같은 규칙)."""
    return os.getenv(env_name, "off").strip().lower() in ("off", "0", "false", "no")


def _is_direct_text(resp: Optional[Dict[str, Any]]) -> bool:
    """응답이 PDF 텍스트층(direct_text)으로 처리됐는가 — 그러면 OCR 모델을 부르지 않았으므로 쪽당 OCR 요금이 없다.

    파이프라인(build_response)은 `ocr_provider` 에 **요청한** 공급자('alibaba')를 그대로 두고, 실제로 쓰인
    공급자는 `ocr_provider_used` 에 적는다. 예전에는 전자만 봐서 텍스트층 계약서에도 쪽당 OCR 요금이 붙었다
    (실측 T-001 1건 ₩58.3 vs 실제 ₩44.8 — 4,000건이면 ₩54,000 의 허수, --max-cost 도 일찍 걸린다).
    """
    r = resp or {}
    if r.get("ocr_provider_used") == "direct_text" or r.get("ocr_provider") == "direct_text":
        return True
    ocr = r.get("ocr_result")
    return isinstance(ocr, dict) and ocr.get("ocr_provider") == "direct_text"


def _estimate_cost(resp: Dict[str, Any], media: str, is_contract: bool,
                   pages: Optional[int] = None,
                   consolidated: Optional[bool] = None) -> float:
    """실측 단가 기반 추정치.

    `consolidated` 를 주면(요청한 값) 통합검증 요금 포함 여부를 그것으로 정한다; 안 주면 응답의
    `consolidate` 를 본다(없으면 예전처럼 포함). 텍스트층으로 처리된 계약서(ocr_provider == direct_text)
    에는 쪽당 OCR 요금을 더하지 않는다(M2 B4) — 4,000건이면 ₩54,000 의 허수였다.
    """
    # 실패한 처리는 실제로 모델을 부르지 않았다. 정상 요금을 매기면 비용 상한이
    # 헛돌고(예: 영상 P3 no-op 에 ₩37.41) 리포트의 비용도 부풀려진다.
    if resp is not None and resp.get("success") is False:
        ocr_len = len(str(resp.get("ocr_text") or ""))
        return COST_OCR_PER_PAGE if ocr_len else 0.0
    con = (resp or {}).get("consolidate") if consolidated is None else consolidated
    with_con = con is not False
    if is_contract or media == "text":
        n = pages if pages else 1
        for key in ("page_count", "pages", "total_pages"):
            if not pages and (resp or {}).get(key):
                try:
                    n = max(1, int(resp[key])); break
                except (TypeError, ValueError):
                    pass
        ocr = 0.0 if _is_direct_text(resp) else COST_OCR_PER_PAGE * n
        ext = COST_LLM_EXTRACT_NO_THINK if _thinking_off("EXTRACTION_THINKING") else COST_LLM_EXTRACT
        con_krw = COST_CONSOLIDATION_DOC_NO_THINK if _thinking_off("CONSOLIDATION_THINKING") else COST_CONSOLIDATION_DOC
        return ocr + ext + (con_krw if with_con else 0.0)
    vlm = COST_VLM_VIDEO if media == "video" else COST_VLM_IMAGE
    return vlm + (COST_CONSOLIDATION_IMG if with_con else 0.0)


def _nonnull(d: Dict[str, Any]) -> int:
    return sum(1 for v in (d or {}).values() if v not in (None, "", [], {}))


def _merge_preserving(con: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """통합 결과를 쓰되, 원본에만 있던 값은 되살린다.

    중재자는 LLM 과 NER 중 어느 값을 택할지 정하는 단계다. 한쪽에만 있는 필드를
    통째로 **지우는** 것은 중재가 아니라 손실이다. 실측 두 가지:
      · 계약서: 원본 44필드 → 통합 3필드 (success=True, error=None)
      · 영상  : 원본 9필드 → 통합 4필드. description·keyword·work_type·
        main_subjects·dominant_colors 가 통째로 사라졌다.
    두 번째는 비율(4/9=44%)로는 걸러지지 않는다. 그래서 비율 기준을 버리고
    '원본에 값이 있었는데 통합본에서 비었으면 되살린다' 로 바꾼다. 중재자가 값을
    **바꾸는** 것은 그대로 존중하고, 없애는 것만 되돌린다.
    """
    if not isinstance(con, dict) or not con:
        return raw if isinstance(raw, dict) else {}
    if not isinstance(raw, dict) or not raw:
        return con
    out = dict(con)
    restored = []
    for k, v in raw.items():
        if v in (None, "", [], {}):
            continue
        if out.get(k) in (None, "", [], {}):
            out[k] = v
            restored.append(k)
    if restored:
        logger.warning(f"통합에서 사라진 필드를 원본 추출로 복원했습니다: {restored}")
    return out


def _final_metadata(resp: Dict[str, Any]) -> Dict[str, Any]:
    """최종 메타데이터 — 통합 결과 우선, 단 통합이 지워버린 필드는 원본으로 채운다."""
    if not isinstance(resp, dict):
        return {}

    def pick(src: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(src, dict):
            return {}
        return _merge_preserving(src.get("consolidated_metadata") or {},
                                 src.get("metadata") or {})

    got = pick(resp)
    if got:
        return got
    res = resp.get("result")
    return pick(res) if isinstance(res, dict) else {}


# ---------------------------------------------------------------------------
# v4 — 23키 예측 기록 (M2 §2)
# ---------------------------------------------------------------------------

def _keyframe_summary(kf: Any) -> Optional[Dict[str, Any]]:
    """response["keyframes"] (frames 목록을 뺀 dict: kept·target·sampled·duration·schedule) → _meta 용 요약.

    예전 코드는 list 로 가정해 len() 을 쟀고, dict 인 실제 응답에서는 항상 None 이 적혔다.
    """
    if isinstance(kf, dict):
        return {k: kf.get(k) for k in ("kept", "target", "sampled", "duration", "schedule")}
    if isinstance(kf, list):                      # 옛 모양(프레임 목록)도 장수는 남긴다
        return {"kept": len(kf), "target": None, "sampled": None, "duration": None, "schedule": None}
    return None


def _leg_info(resp: Optional[Dict[str, Any]], elapsed_sec: float) -> Dict[str, Any]:
    """leg 1개의 요약 — request_id·소요시간·모델·OCR provider 등. _meta 와 results.jsonl 의 stages 에 같이 남긴다.

    `ocr_provider` 는 요청한 공급자, `ocr_provider_used` 는 실제로 쓰인 공급자(텍스트층이면 'direct_text') 다.
    """
    r = resp or {}
    return {
        "ok": bool(r.get("success")), "error": r.get("error"),
        "request_id": r.get("request_id"),
        "elapsed_sec": round(elapsed_sec, 2),
        "processing_time": r.get("processing_time"), "extraction_time": r.get("extraction_time"),
        "model_used": r.get("model_used"),
        "ocr_provider": r.get("ocr_provider"), "ocr_provider_used": r.get("ocr_provider_used"),
        "ocr_model": r.get("ocr_model"), "ocr_pages": r.get("ocr_pages"),
        "ner_model": r.get("ner_model_key"),
        "consolidate": r.get("consolidate"),
        "consolidation_model_used": r.get("consolidation_model_used"),
        "consolidation_fallback_used": r.get("consolidation_fallback_used"),
        "modality": r.get("modality"), "vlm_backend": r.get("vlm_backend"),
        "keyframes": _keyframe_summary(r.get("keyframes")),
        "file_name": r.get("file_name"), "file_hash": r.get("file_hash"),
    }


def _context_payload(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """규칙 파싱 결과 → VLM 에 넘길 {"title", "work_type"}. 제목·종별이 둘 다 비면 None.

    둘 다 비면 prompts.user_prompt_with_context 가 기본 프롬프트를 그대로 돌려주므로 '문맥을 썼다' 고
    기록하면 거짓이다 — _meta.prompt.contract_context_used 도 이 함수의 결과로 정한다.
    """
    if not isinstance(context, dict):
        return None
    title = context.get("title") or None
    work_type = context.get("work_type") or None
    if not title and not work_type:
        return None
    return {"title": title, "work_type": work_type}


def build_prediction_record(test: Optional[str], set_id: str, test_prediction: Optional[Dict[str, Any]], *,
                            file_name: Optional[str] = None,
                            contract_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """파이프라인의 test_prediction(23키) → 시험용 예측 레코드 (M2 §2, M1 §6).

    T-001: 1~16 은 응답 그대로, 17~23 은 **null 로 고정**(평가기가 비채점 키의 값을 경고로 보고한다),
           other_rights 는 전 계약서 상태 ③ 이므로 항상 null.
    T-002: 17~23 은 응답 그대로, 1~16 은 계약서 규칙 파싱의 제목·종별만 채우고 나머지 null(비채점).
    file_name 은 **원본 파일명**(basename, 공백 포함) — 파이프라인의 정제된 filename 이 아니다.
    """
    src = test_prediction if isinstance(test_prediction, dict) else {}
    rec: Dict[str, Any] = {"set_id": str(set_id), "test": test, **empty_prediction()}
    if test == "T-001":
        for k in T001_KEYS:
            rec[k] = src.get(k)
        rec["other_rights"] = None
    elif test == "T-002":
        for k in T002_KEYS:
            rec[k] = src.get(k)
        ctx = contract_context or {}
        rec["work_title"] = ctx.get("title") or None
        rec["work_type"] = ctx.get("work_type") or None
    if file_name:
        rec["file_name"] = file_name
    elif test == "T-002" and src.get("file_name"):   # T-002 는 file_name 이 채점 키 목록에 없어 따로 옮긴다
        rec["file_name"] = src.get("file_name")
    return rec


def _safe_set_filename(set_id: str) -> str:
    """<set_id>.json 의 파일명 — 평가기는 파일 stem 을 set_id 로 읽으므로 가능한 한 그대로 둔다."""
    s = str(set_id).strip() or "_"
    return s.replace(os.sep, "_").replace("/", "_")


class BatchRunner:
    def __init__(self, orchestrator, config: RunConfig, out_dir: str | Path,
                 progress_cb: Optional[Callable[[Dict], None]] = None):
        self.orch = orchestrator
        self.cfg = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out_dir / "results.jsonl"
        self.progress_cb = progress_cb
        self.cancel_event = threading.Event()

        self._lock = threading.Lock()
        self._cost = 0.0
        self._done = 0
        self._failed = 0
        self._cost_exceeded = False

        # v4 산출물 (M2 B3) — 채점 기준이 eval23 이 아니어도 test.log 는 남긴다(세트별 타이밍은 어디서나 유용하다)
        self.pred_dir = self.out_dir / "test_predictions"
        self.pred_jsonl = self.out_dir / "test_predictions.jsonl"
        self.log_path = self.out_dir / "test.log"
        self._log_handler: Optional[logging.Handler] = None
        self.log = self._attach_test_log()

    # -- test.log ----------------------------------------------------------
    def _attach_test_log(self) -> logging.Logger:
        """<out>/test.log — 실행기 전용 자식 로거에 FileHandler 를 단다.

        작업마다 로거를 따로 두는 이유: 웹 서버에서는 작업 여러 개가 한 프로세스에서 동시에 돌 수 있어
        모듈 로거 하나에 핸들러를 달면 서로의 로그가 섞인다. 부모로 전파는 그대로 두어 콘솔에도 보인다.
        """
        lg = logging.getLogger(f"{__name__}.run.{self.out_dir.name}.{uuid.uuid4().hex[:6]}")
        lg.setLevel(logging.INFO)
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        # 한 줄 형식 `HH:MM:SS LEVEL [set_id] [구성요소] 메시지` — 메시지 앞의 '[set_id] ' 는 세트 칸으로 옮기고, 비밀값은 가린다
        fh.setFormatter(LC.ConsoleLineFormatter())
        lg.addHandler(fh)
        self._log_handler = fh
        cfg = self.cfg
        lg.info("실행기 준비 — 채점 %s · test=%s · workers %d · 추출 %s · OCR %s/%s · NER %s · 통합검증 %s · "
                "저작물 중재자 %s · 이미지VLM %s · 프레임 %s · 출력 %s",
                cfg.scoring_set, cfg.test or "(엔트리별)", cfg.workers, cfg.model_name,
                cfg.ocr_provider, cfg.ocr_model or "(기본)", cfg.ner_model,
                cfg.consolidation_model if cfg.consolidate else "생략",
                "켬" if cfg.consolidate_work else "생략", cfg.vlm_prefer, cfg.frame_schedule, self.out_dir)
        return lg

    def write_log_record(self, record: logging.LogRecord) -> None:
        """다른 로거(파이프라인·모델 모듈)의 기록을 이 실행의 test.log 에 같은 형식으로 적는다 (시험 콘솔이 부른다).

        FileHandler.handle 로 넘겨 핸들러 잠금을 같이 쓴다 — 실행기 자신의 줄과 섞여도 줄이 깨지지 않는다.
        """
        h = self._log_handler
        if h is not None:
            h.handle(record)

    def close_log(self) -> None:
        """FileHandler 를 떼고 닫는다 — 작업이 끝난 뒤에도 핸들이 열려 있으면 파일이 늘어나고 핸들러가 누적된다."""
        h = self._log_handler
        if h is None:
            return
        try:
            self.log.removeHandler(h)
            h.close()
        finally:
            self._log_handler = None

    # -- 재개 ---------------------------------------------------------------
    def _completed_ids(self) -> set:
        done = set()
        if not (self.cfg.resume and self.results_path.exists()):
            return done
        with self.results_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 실패는 재시도 대상이므로 완료로 치지 않는다
                if r.get("ok"):
                    done.add(str(r.get("set_id")))
        return done

    def cancel(self):
        self.cancel_event.set()

    def _emit(self, event: Dict):
        if self.progress_cb:
            try:
                self.progress_cb(event)
            except Exception:  # 진행률 콜백 실패가 실행을 죽이면 안 된다
                pass

    # -- 세트 1건 -----------------------------------------------------------
    def _run_one(self, entry: ManifestEntry, gt: Dict[str, Any],
                 cost_box: Dict[str, float]) -> Dict[str, Any]:
        """세트 1건 처리 — 채점 기준이 eval23 이면 v4 다리-하나 경로, 아니면 v3 쌍 경로.

        `cost_box["v"]` 에 누적 비용을 즉시 기록한다 — 도중에 예외가 나도 이미 쓴 돈은
        회계에 남아야 한다. 반환값에만 담으면 실패 시 비용이 증발해 상한이 무력해진다.
        """
        if self.cfg.scoring_set == EVAL23:
            return self._run_one_eval23(entry, cost_box)
        return self._run_one_v3(entry, gt, cost_box)

    def _run_one_v3(self, entry: ManifestEntry, gt: Dict[str, Any],
                    cost_box: Dict[str, float]) -> Dict[str, Any]:
        """v3(plan/tta) — 계약서 leg → 저작물 leg → 상속 병합 → 채점. 저작물이 없으면 돈을 쓰기 전에 실패로 적는다."""
        t0 = time.perf_counter()
        out: Dict[str, Any] = {"set_id": entry.set_id, "media": entry.media,
                               "license_bucket": entry.license_bucket,
                               "contract": entry.contract, "work": entry.work}
        contract_meta = None
        stages: Dict[str, Any] = {}
        self.log.info("[%s] 시작 — 채점 %s · media=%s · 계약서=%s · 저작물=%s", entry.set_id, self.cfg.scoring_set,
                      entry.media, os.path.basename(entry.contract) if entry.contract else "-",
                      os.path.basename(entry.work) if entry.work else "-")

        # v3 채점은 저작물 leg 의 출력에서 읽는다 — 계약서 단독 세트는 채점할 수 없다.
        # 예전에는 계약서 leg 요금을 쓴 뒤 open("") 에서 죽었다(감사 §1.2); 호출 전에 끊는다.
        if not entry.work:
            msg = (f"저작물이 없는 세트는 v3 채점({self.cfg.scoring_set})을 할 수 없습니다 — "
                   f"계약서 단독 세트는 --test T-001 (eval23) 로 실행하세요")
            self.log.error("[%s] %s", entry.set_id, msg)
            out.update({"ok": False, "error": msg, "elapsed_sec": round(time.perf_counter() - t0, 2),
                        "cost_krw": 0.0, "stages": stages})
            return out

        # 1) 계약서
        if entry.contract:
            with open(entry.contract, "rb") as f:
                cbytes = f.read()
            cresp = self.orch.run(
                cbytes, os.path.basename(entry.contract),
                model_name=self.cfg.model_name,
                document_type=self.cfg.contract_document_type,
                ocr_provider=self.cfg.ocr_provider, ocr_model=self.cfg.ocr_model,
                ner_model=self.cfg.ner_model, consolidate=self.cfg.consolidate,
                consolidation_model=self.cfg.consolidation_model,
            )
            contract_meta = _final_metadata(cresp)
            cost_box["v"] += _estimate_cost(cresp, "text", True,
                                            pages=_pdf_page_count(entry.contract))
            # request_id 로 results/{id}/ 에 OCR·NER·LLM·통합검증 산출물이 저장돼 있다.
            # 이걸 남겨야 세트별로 파이프라인 원본 출력을 되찾을 수 있다.
            out["contract_request_id"] = cresp.get("request_id")
            stages["contract"] = {"ok": bool(cresp.get("success")) and bool(contract_meta),
                                  "error": cresp.get("error"),
                                  "fields": sum(1 for v in contract_meta.values()
                                                if v not in (None, "", []))}

        # 2) 저작물
        with open(entry.work, "rb") as f:
            wbytes = f.read()
        wresp = self.orch.run(
            wbytes, os.path.basename(entry.work),
            model_name=self.cfg.model_name,
            document_type=self.cfg.work_document_type,
            ocr_provider=self.cfg.ocr_provider, ocr_model=self.cfg.ocr_model,
            ner_model=self.cfg.ner_model, consolidate=self.cfg.consolidate,
            consolidation_model=self.cfg.consolidation_model,
            vlm_prefer=self.cfg.vlm_prefer,
        )
        cost_box["v"] += _estimate_cost(wresp, entry.media, False)
        out["work_request_id"] = wresp.get("request_id")

        # ⚠️ 저작물 처리가 실제로 성공했는지 반드시 확인한다.
        #    파이프라인은 실패해도 예외를 던지지 않고 success=False + metadata={} 를 돌려준다:
        #      · 영상/음성 → P3 가드로 미구현 (build_response(success=False))
        #      · .wmv/.swf → router 가 unknown 으로 보내 문서 경로 → OCR 공백 가드
        #    이걸 ok=True 로 기록하면 전 속성이 '오답'으로 집계되어 정확도가 통째로 무너지고,
        #    게다가 재개 시 완료로 간주돼 영영 재시도되지 않는다.
        work_ok = bool(wresp.get("success")) and bool(_final_metadata(wresp))
        stages["work"] = {"ok": work_ok, "error": wresp.get("error")}
        if not work_ok:
            out.update({
                "ok": False,
                "error": (f"저작물 처리 실패(media={entry.media}): "
                          f"{wresp.get('error') or 'success=False / 메타데이터 없음'}")[:300],
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "cost_krw": round(cost_box["v"], 2), "stages": stages,
            })
            self.log.warning("[%s] 종료 — FAIL · %.1fs · %s", entry.set_id, out["elapsed_sec"], out["error"])
            return out

        # 3) 상속 병합 — LLM 호출 없음
        if contract_meta:
            try:
                wresp = self.orch.apply_contract_inheritance(wresp, contract_meta)
                stages["inheritance"] = {"ok": True}
            except Exception as e:
                stages["inheritance"] = {"ok": False, "error": str(e)[:120]}

        extracted = _final_metadata(wresp)

        # 4) 채점
        scored = score_set(gt.get("attributes", {}), extracted, media=entry.media,
                           skip_tiers=self.cfg.skip_tiers,
                           attributes=attributes_for(self.cfg.scoring_set))

        out.update({
            "ok": True,
            # 어떤 기준으로 채점했는지 결과 줄 자체에 남긴다 — 작업 설정은 서버
            # 재시작 한 번에 사라지지만 results.jsonl 은 남는다. 리포트·모달이
            # 여기서 기준을 읽으면 11속성으로 되돌아가는 일이 없다.
            "scoring_set": self.cfg.scoring_set,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "cost_krw": round(cost_box["v"], 2),
            "stages": stages,
            "extracted": extracted,
            **scored,
        })
        acc = out.get("accuracy")
        self.log.info("[%s] 종료 — OK · %.1fs · ₩%.1f · 정확도 %s", entry.set_id, out["elapsed_sec"],
                      out["cost_krw"], f"{acc * 100:.0f}%" if acc is not None else "—")
        return out

    # -- v4: 다리 하나 + 23키 기록 ------------------------------------------
    def _keep_contract_copy(self, request_id: Optional[str], contract_path: str) -> None:
        """Te-2 짝 계약서를 저작물 leg 결과 폴더 source/contract.pdf 로 남긴다 — 원본 폴더가 사라져도 콘솔이 쪽을 보여 주게
        (Te-1 은 파이프라인이 source/document.pdf 로 남긴다). 계약서는 작아서(≈100 KB) 드라이브 줄에 서도 짧다."""
        base = getattr(self.orch, "results_dir", None)
        if not request_id or base is None or not str(contract_path).lower().endswith(".pdf"):
            return
        try:
            try:
                from module.local_limits import disk_slot
            except ImportError:
                from api.module.local_limits import disk_slot  # type: ignore
            leg = Path(base) / str(request_id)
            if not leg.is_dir():
                return
            dst = leg / "source" / "contract.pdf"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with disk_slot(contract_path):
                shutil.copy2(contract_path, dst)
        except Exception as e:  # noqa: BLE001 — 미리보기용 사본이 시험을 막지 않게
            self.log.warning("계약서 사본 저장 실패 (%s): %s", os.path.basename(str(contract_path)), e)

    def _contract_context(self, contract_path: str) -> Optional[Dict[str, Any]]:
        """계약서 PDF → {"title", "work_type", "source": "rules"} — 규칙 파싱만, 모델 호출 없음 (M2 A4)."""
        fn = _load_context_parser()
        if fn is None:
            return None
        try:
            ctx = fn(contract_path)
        except Exception as e:
            self.log.warning("계약서 문맥 파싱 실패 %s: %s", os.path.basename(contract_path), e)
            return None
        if not isinstance(ctx, dict):
            return None
        return {"title": ctx.get("title"), "work_type": ctx.get("work_type"),
                "source": ctx.get("source", "rules")}

    def _write_prediction(self, record: Dict[str, Any]) -> Path:
        """test_predictions/<set_id>.json 을 덮어쓰고 test_predictions.jsonl 에 한 줄 덧붙인다.

        재개하면 jsonl 에는 같은 set_id 가 여러 줄 남는다 — 평가기는 디렉터리(<set_id>.json, 최신본)를 읽는다.
        """
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        name = _safe_set_filename(record["set_id"])
        if name != str(record["set_id"]):
            self.log.warning("set_id %r 에 경로 문자가 있어 파일명을 %r 로 바꿔 씁니다 — 평가기 대조 시 set_id 가 어긋날 수 있습니다",
                             record["set_id"], name)
        path = self.pred_dir / f"{name}.json"
        with self._lock:
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            with self.pred_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def _meta(self, entry: ManifestEntry, test: Optional[str], ok: bool, error: Optional[str],
              elapsed: float, cost: float, legs: Dict[str, Any], context: Optional[Dict[str, Any]],
              resp: Dict[str, Any], pred_src: Any) -> Dict[str, Any]:
        """예측 레코드의 _meta — request id·leg 타이밍·모델·프롬프트 문맥·파일 sha256 (M2 B3)."""
        cfg = self.cfg
        return {
            "set_id": entry.set_id, "test": test, "media": entry.media, "license_bucket": entry.license_bucket,
            "ok": ok, "error": error, "elapsed_sec": elapsed, "cost_krw": round(cost, 2),
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "request_ids": {leg: info.get("request_id") for leg, info in legs.items() if info.get("request_id")},
            "legs": legs,
            "files": {"contract": entry.contract, "work": entry.work},
            "file_sha256": resp.get("file_hash"),
            "contract_context": context,
            "prompt": {
                "contract_context_used": _context_payload(context) is not None,
                "frame_schedule": cfg.frame_schedule if test == "T-002" else None,
                "consolidate_work": cfg.consolidate_work if test == "T-002" else None,
            },
            "models": {
                "extraction": cfg.model_name,
                "ocr": f"{cfg.ocr_provider}/{cfg.ocr_model or '(기본)'}",
                "ner": cfg.ner_model,
                "consolidation": (cfg.consolidation_model
                                  if (cfg.consolidate if test == "T-001" else cfg.consolidate_work) else None),
                "vlm_prefer": cfg.vlm_prefer if test == "T-002" else None,
                "model_used": resp.get("model_used"),
            },
            # 파이프라인 직렬화기가 남긴 비교용 값(예: model_copyright_holder, M2 A5)
            "pipeline_meta": pred_src.get("_meta") if isinstance(pred_src, dict) else None,
            "extra": entry.extra or None,
        }

    def _run_one_eval23(self, entry: ManifestEntry, cost_box: Dict[str, float]) -> Dict[str, Any]:
        """v4 — T-001 은 계약서 leg 만, T-002 는 저작물 leg 만(계약서는 규칙 문맥) 부르고 23키 레코드를 기록한다.

        실패해도 레코드는 남긴다(전부 null + _meta.error) — test_predictions/ 가 세트마다 한 파일씩 완결되고,
        results.jsonl 의 ok=False 로 재개 때 다시 시도된다(그때 파일은 덮어써진다).
        """
        t0 = time.perf_counter()
        test = effective_test(entry, self.cfg.test)
        out: Dict[str, Any] = {"set_id": entry.set_id, "media": entry.media,
                               "license_bucket": entry.license_bucket,
                               "contract": entry.contract, "work": entry.work,
                               "test": test, "scoring_set": EVAL23}
        legs: Dict[str, Any] = {}
        context: Optional[Dict[str, Any]] = None
        resp: Dict[str, Any] = {}
        pred_src: Any = None
        ok, error = False, None
        self.log.info("[%s] 시작 — test=%s · media=%s · 계약서=%s · 저작물=%s", entry.set_id, test, entry.media,
                      os.path.basename(entry.contract) if entry.contract else "-",
                      os.path.basename(entry.work) if entry.work else "-")
        try:
            if test == "T-001":
                if not entry.contract:
                    raise ValueError("T-001 세트에 계약서가 없습니다")
                t1 = time.perf_counter()
                resp = self.orch.run(
                    b"", os.path.basename(entry.contract), source_path=entry.contract,   # 메모리에 올리지 않는다 — setup() 이 작업 사본으로 복사하고 세트 끝에 지운다
                    model_name=self.cfg.model_name,
                    document_type=self.cfg.contract_document_type,
                    ocr_provider=self.cfg.ocr_provider, ocr_model=self.cfg.ocr_model,
                    ner_model=self.cfg.ner_model, consolidate=self.cfg.consolidate,
                    consolidation_model=self.cfg.consolidation_model,
                    # 파이프라인 쪽 test_prediction(llm_metadata.json)도 같은 시험 규칙(17~23 null)을 따르게 한다 —
                    # 최종 권위는 build_prediction_record 지만, 같은 leg 의 두 산출물이 서로 달라서는 안 된다
                    test=test,
                ) or {}
                cost_box["v"] += _estimate_cost(resp, "text", True, pages=_pdf_page_count(entry.contract),
                                                consolidated=self.cfg.consolidate)
                legs["contract"] = _leg_info(resp, time.perf_counter() - t1)
                out["contract_request_id"] = resp.get("request_id")
            elif test == "T-002":
                if not entry.work:
                    raise ValueError("T-002 세트에 저작물이 없습니다")
                if entry.contract:
                    t1 = time.perf_counter()
                    context = self._contract_context(entry.contract)
                    legs["context"] = {"ok": context is not None, "elapsed_sec": round(time.perf_counter() - t1, 3),
                                       "source": (context or {}).get("source"),
                                       "title": (context or {}).get("title"),
                                       "work_type": (context or {}).get("work_type")}
                    if context is not None and _context_payload(context) is not None:
                        self.log.info("[%s] 계약서 문맥 — 제목 %s · 종별 %s (규칙 파싱, %.2f초) — 저작물 leg 프롬프트에 참고 줄로 넣습니다",
                                      entry.set_id, repr(context.get("title")) if context.get("title") else "없음",
                                      context.get("work_type") or "없음", legs["context"]["elapsed_sec"],
                                      extra={"component": "문맥"})
                    # 계약서가 있는데 문맥을 못 얻었으면 세트마다 경고로 드러낸다 — 예전의 지연 import 경쟁(workers>1)은
                    # 바로 이 자리에서 None 을 조용히 넘겨 저작물 leg 가 문맥 없이 돌았다.
                    if context is None:
                        reason = "문맥 파서 미적재" if parse_contract_context is None else "규칙 파싱 실패"
                        self.log.warning("[%s] 계약서 %s 가 있지만 문맥(제목·종별)을 얻지 못했습니다(%s) — "
                                         "저작물 leg 를 계약서 문맥 없이 실행합니다",
                                         entry.set_id, os.path.basename(entry.contract), reason,
                                         extra={"component": "문맥"})
                    elif _context_payload(context) is None:
                        self.log.warning("[%s] 계약서 %s 규칙 파싱이 제목·종별을 하나도 읽지 못했습니다 — "
                                         "저작물 leg 를 계약서 문맥 없이 실행합니다",
                                         entry.set_id, os.path.basename(entry.contract), extra={"component": "문맥"})
                elif self.cfg.title_fallback and (entry.extra or {}).get("title", "").strip():
                    # 계약서 없음 — 매니페스트 제목을 문맥으로 쓴다 (--title-fallback; 본 시험에서는 모든 저작물에 계약서가 있다)
                    context = {"title": entry.extra["title"].strip(), "work_type": None, "source": "manifest_title"}
                    legs["context"] = {"ok": True, "elapsed_sec": 0.0, "source": "manifest_title",
                                       "title": context["title"], "work_type": None}
                t1 = time.perf_counter()
                resp = self.orch.run(
                    b"", os.path.basename(entry.work), source_path=entry.work,   # 영상(최대 8.6 GB)을 메모리에 올리지 않는다 — setup() 이 드라이브마다 하나씩 작업 사본으로 복사, 세트 끝에 지움
                    model_name=self.cfg.model_name,
                    document_type=self.cfg.work_document_type,
                    ocr_provider=self.cfg.ocr_provider, ocr_model=self.cfg.ocr_model,
                    ner_model=self.cfg.ner_model, consolidate=self.cfg.consolidate,
                    consolidation_model=self.cfg.consolidation_model,
                    vlm_prefer=self.cfg.vlm_prefer,
                    # --- v4 전용 인터페이스 (M2 A2·A3·A4·A6·A7; 구현자 A) ---
                    test=test,
                    consolidate_work=self.cfg.consolidate_work,
                    contract_context=_context_payload(context),
                    frame_schedule=self.cfg.frame_schedule,
                    media=entry.media,
                ) or {}
                cost_box["v"] += _estimate_cost(resp, entry.media, False, consolidated=self.cfg.consolidate_work)
                legs["work"] = _leg_info(resp, time.perf_counter() - t1)
                out["work_request_id"] = resp.get("request_id")
                if entry.contract:
                    self._keep_contract_copy(resp.get("request_id"), entry.contract)
            else:
                raise ValueError(f"시험 구분을 정할 수 없습니다 (test={test!r}; 엔트리 test 또는 --test 를 주세요)")

            pred_src = resp.get("test_prediction")
            if not resp.get("success"):
                error = (f"{'계약서' if test == 'T-001' else '저작물'} 처리 실패(media={entry.media}): "
                         f"{resp.get('error') or 'success=False'}")
            elif not isinstance(pred_src, dict):
                error = "응답에 test_prediction(23키) 이 없습니다 — 파이프라인 eval23 직렬화기(M2 A2) 미반영"
            else:
                ok = True
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            out["traceback"] = traceback.format_exc()[-800:]
            self.log.error("[%s] 예외 — %s", entry.set_id, error)

        src_file = entry.contract if test == "T-001" else entry.work
        file_name = resp.get("file_name") or (os.path.basename(src_file) if src_file else None)
        record = build_prediction_record(test, entry.set_id, pred_src, file_name=file_name,
                                         contract_context=context)
        elapsed = round(time.perf_counter() - t0, 2)
        record["_meta"] = self._meta(entry, test, ok, error, elapsed, cost_box["v"], legs, context, resp, pred_src)
        path = self._write_prediction(record)

        timing = " · ".join(f"{leg} {info.get('elapsed_sec')}s" for leg, info in legs.items()
                            if info.get("elapsed_sec") is not None)
        (self.log.info if ok else self.log.warning)(
            "[%s] 종료 — %s · %.1fs (%s) · ₩%.1f%s", entry.set_id, "OK" if ok else "FAIL", elapsed, timing or "-",
            cost_box["v"], f" · {error}" if error else "")
        out.update({
            "ok": ok,
            "error": error[:300] if error else None,
            "elapsed_sec": elapsed,
            "cost_krw": round(cost_box["v"], 2),
            "stages": legs,
            "extracted": {k: record.get(k) for k in KEY_NAMES},
            "prediction_path": str(path),
        })
        return out

    # -- 전체 실행 ----------------------------------------------------------
    def run(self, entries: List[ManifestEntry],
            ground_truth: Dict[str, Dict]) -> Dict[str, Any]:
        done_ids = self._completed_ids()
        todo = [e for e in entries if e.set_id not in done_ids]
        if self.cfg.limit:
            todo = todo[: self.cfg.limit]

        total = len(todo)
        t_start = time.perf_counter()
        self._emit({"type": "start", "total": total, "already_done": len(done_ids),
                    "workers": self.cfg.workers})
        self.log.info("배치 시작 — 대상 %d세트 (이미 완료 %d · 전체 %d) · workers %d",
                      total, len(done_ids), len(entries), self.cfg.workers)

        if not total:
            self.log.info("배치 종료 — 처리할 세트가 없습니다")
            if self.cfg.scoring_set != EVAL23:
                self.close_log()
            return {"total": 0, "already_done": len(done_ids), "ok": 0, "failed": 0,
                    "cost_krw": 0.0, "elapsed_sec": 0.0, "cancelled": False}

        fh = self.results_path.open("a", encoding="utf-8")

        def work(entry: ManifestEntry) -> Optional[Dict]:
            if self.cancel_event.is_set() or self._cost_exceeded:
                return None
            cost_box = {"v": 0.0}
            try:
                gt = ground_truth.get(entry.set_id, {})
                # 이 세트 처리 중에 남는 로그(파이프라인·모델 모듈 포함)를 세트에 묶는다 — 시험 콘솔이 실행 로그를 세트별로 가른다
                with LC.bind_set(entry.set_id):
                    res = self._run_one(entry, gt, cost_box)
            except Exception as e:
                res = {"set_id": entry.set_id, "ok": False,
                       "error": f"{type(e).__name__}: {e}"[:300],
                       "traceback": traceback.format_exc()[-800:],
                       "media": entry.media, "license_bucket": entry.license_bucket,
                       "cost_krw": round(cost_box["v"], 2)}
                self.log.error("[%s] 세트 예외 — %s", entry.set_id, res["error"])
            with self._lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                self._done += 1
                # 실패해도 이미 호출한 API 요금은 발생했다 — 상한 계산에서 빼면 안 된다.
                self._cost += res.get("cost_krw", 0.0)
                if not res.get("ok"):
                    self._failed += 1
                if (self.cfg.max_cost_krw is not None
                        and self._cost >= self.cfg.max_cost_krw
                        and not self._cost_exceeded):
                    self._cost_exceeded = True
                    self.cancel_event.set()
                    self._emit({"type": "cost_limit", "cost_krw": round(self._cost, 1),
                                "limit": self.cfg.max_cost_krw})
                    self.log.error("비용 상한 도달 ₩%.0f (상한 ₩%.0f) — 중단", self._cost, self.cfg.max_cost_krw)
                snapshot = {"type": "progress", "i": self._done, "total": total,
                            "set_id": res.get("set_id"), "ok": bool(res.get("ok")),
                            "accuracy": res.get("accuracy"),
                            "n_match": res.get("n_match"), "n_scored": res.get("n_scored"),
                            "elapsed_sec": res.get("elapsed_sec"),
                            "cost_krw_total": round(self._cost, 1),
                            "failed": self._failed,
                            "error": res.get("error")}
            self._emit(snapshot)
            return res

        try:
            with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
                # 문맥(실행 묶음)을 작업 스레드로 복사해 넘긴다 — 세트마다 새 사본
                futures = [LC.submit(ex, work, e) for e in todo]
                for _ in as_completed(futures):
                    pass
        finally:
            fh.close()

        summary = {
            "total": total, "already_done": len(done_ids),
            "ok": self._done - self._failed, "failed": self._failed,
            "cost_krw": round(self._cost, 2),
            "elapsed_sec": round(time.perf_counter() - t_start, 1),
            "cancelled": self.cancel_event.is_set(),
            "cost_exceeded": self._cost_exceeded,
            "results_path": str(self.results_path),
        }
        if self.cfg.scoring_set == EVAL23:
            summary["predictions_dir"] = str(self.pred_dir)
            summary["log_path"] = str(self.log_path)
        self._emit({"type": "done", **summary})
        self.log.info("배치 종료 — 성공 %d · 실패 %d · 추정비용 ₩%.0f · %.1fs%s",
                      summary["ok"], summary["failed"], summary["cost_krw"], summary["elapsed_sec"],
                      " · 취소됨" if summary["cancelled"] else "")
        if self.cfg.scoring_set != EVAL23:
            self.close_log()
        return summary

    # -- v4 마무리: M1 평가기 호출 -------------------------------------------
    def _list_predictions(self) -> List[Dict[str, Any]]:
        """test_predictions/*.json → [{set_id, test, ok}] (평가 대상 수·시험 구분 확인용)."""
        out: List[Dict[str, Any]] = []
        if not self.pred_dir.is_dir():
            return out
        for f in sorted(self.pred_dir.glob("*.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(r, dict):
                out.append({"set_id": r.get("set_id") or f.stem, "test": r.get("test"),
                            "ok": bool((r.get("_meta") or {}).get("ok"))})
        return out

    def _render_predictions_report(self, summary: Dict[str, Any]) -> str:
        """gold 없이(또는 평가 실패 시) 쓰는 report.md — 무엇을 몇 건 기록했고 어디서 지표를 낼 수 있는지."""
        from .report import load_results
        rows = load_results(self.results_path)
        failed = [r for r in rows if not r.get("ok")]
        L = [f"# v4 예측 기록 — {summary.get('test') or '(혼재)'}", "",
             f"- 예측 레코드 {summary.get('predictions', 0)}건 → `{self.pred_dir}`",
             f"- 세트 결과 {len(rows)}건 (실패 {len(failed)}) → `{self.results_path}`",
             f"- 로그 → `{self.log_path}`", ""]
        if summary.get("error"):
            L += [f"> ⚠️ 평가 실패: {summary['error']}", ""]
        if summary.get("warning"):
            L += [f"> ⚠️ {summary['warning']}", ""]
        if not self.cfg.gold:
            L += ["> gold 가 지정되지 않아 지표를 내지 않았습니다. 평가:", "",
                  f"    python -m api.module.evaluation.v4.evaluate --test {summary.get('test') or 'T-00?'} "
                  f"--gold <gold23.jsonl> --pred {self.pred_dir} --out {self.out_dir}", ""]
        if failed:
            L += ["## 실패 세트", "", "| set_id | 오류 |", "|---|---|"]
            for r in failed[:50]:
                L.append(f"| {r.get('set_id')} | {str(r.get('error'))[:100]} |")
            L.append("")
        return "\n".join(L)

    def finalize_eval23(self, test: Optional[str] = None) -> Dict[str, Any]:
        """예측 기록이 끝난 뒤 — gold 가 있으면 M1 평가기로 test_metrics.json · report.md 를 쓰고 요약을 돌려준다 (M2 B3).

        평가 실패가 예측 기록을 잃게 해서는 안 되므로 예외는 요약의 error 로만 남긴다. report.json 은 항상 쓴다
        (웹 /report?fmt=json 과 작업 복구가 이 파일을 본다).
        """
        preds = self._list_predictions()
        tests = sorted({p["test"] for p in preds if p.get("test")})
        test = test or self.cfg.test or (tests[0] if len(tests) == 1 else None)
        summary: Dict[str, Any] = {
            "scoring_set": EVAL23, "test": test, "tests_seen": tests,
            "predictions": len(preds), "predictions_ok": sum(1 for p in preds if p["ok"]),
            "pred_dir": str(self.pred_dir), "pred_jsonl": str(self.pred_jsonl), "log": str(self.log_path),
            "gold": self.cfg.gold, "tau": self.cfg.tau, "content_check": self.cfg.content_check,
            "embed_cache": self.cfg.embed_cache, "embed_revision": self.cfg.embed_revision,
            "n_boot": self.cfg.n_boot, "only_predicted": self.cfg.only_predicted, "evaluated": False,
        }
        if self.cfg.gold:
            if not test:
                summary["warning"] = ("시험 구분을 정할 수 없어(예측에 T-001·T-002 혼재 또는 없음) 평가를 건너뜁니다 — "
                                      "--test 를 주세요")
                self.log.warning(summary["warning"])
            else:
                try:
                    from .v4.evaluate import evaluate
                    self.log.info("평가 시작 — %s · gold %s · 예측 %d건 · τ=%.2f · 내용검사 %s · 임베딩캐시 %s · gold 범위 %s",
                                  test, self.cfg.gold, len(preds), self.cfg.tau,
                                  "on" if self.cfg.content_check else "off", self.cfg.embed_cache or "(없음)",
                                  "예측 있는 세트만(--only-predicted)" if self.cfg.only_predicted else "전체")
                    # τ 외의 평가기 옵션(M1 §5·§7)도 그대로 넘긴다 — 배치 CLI 에서 재평가할 때 임베딩 캐시를 다시 쓸 수 있어야 한다
                    eval_kw: Dict[str, Any] = dict(tau=self.cfg.tau, cells=True, content_check=self.cfg.content_check,
                                                   embed_cache=self.cfg.embed_cache,
                                                   embed_revision=self.cfg.embed_revision)
                    if self.cfg.n_boot is not None:
                        eval_kw["n_boot"] = int(self.cfg.n_boot)
                    if self.cfg.only_predicted:
                        # 개발·파일럿 런 — gold 를 예측이 있는 세트로 제한한다(M2b 수정 4). 기본값(False)은 M1 동작 그대로라 넘기지 않는다.
                        eval_kw["only_predicted"] = True
                    m = evaluate(test, self.cfg.gold, self.pred_dir, self.out_dir, **eval_kw)
                    summary.update({
                        "evaluated": True,
                        "n_sets": m.get("n_sets"), "n_predictions": m.get("n_predictions"),
                        "n_gold_dropped": m.get("n_gold_dropped"),
                        "macro_f1": m.get("macro_f1"), "micro_f1": m.get("micro_f1"), "passed": m.get("passed"),
                        "per_key_f1": {k: (v or {}).get("f1") for k, v in (m.get("keys") or {}).items()},
                        "excluded_keys": m.get("excluded_keys"),
                        "warning_counts": m.get("warning_counts"),
                        "outputs": m.get("outputs"),
                    })
                    mf, mi = m.get("macro_f1"), m.get("micro_f1")
                    self.log.info("평가 완료 — %s Macro-F1 %s · Micro-F1 %s → %s", test,
                                  f"{mf * 100:.1f}%" if mf is not None else "—",
                                  f"{mi * 100:.1f}%" if mi is not None else "—",
                                  "합격" if m.get("passed") else "불합격")
                except Exception as e:
                    summary["error"] = f"{type(e).__name__}: {e}"
                    self.log.error("평가 실패 — %s", summary["error"], exc_info=True)
        else:
            self.log.info("gold 미지정 — 예측만 기록했습니다 (%d건, %s)", len(preds), self.pred_dir)
        if not summary["evaluated"]:
            (self.out_dir / "report.md").write_text(self._render_predictions_report(summary), encoding="utf-8")
        (self.out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
        self.close_log()
        return summary
