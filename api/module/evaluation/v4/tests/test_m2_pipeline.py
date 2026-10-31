"""M2 파이프라인 측 단위 테스트 (docs/v4/M2_runner_spec.md §1.A A1~A8, §3.1) — 모델 호출·네트워크 없음.

다루는 것
  · A8  `import api` / `import api.module` 가 torch 를 올리지 않고 3초 안에 끝난다 (서브프로세스), evaluate --help 도.
  · A2  eval23_serializer: 재명명, 권리 허용/불허/null, 계약 유효기간, 키워드→해시태그, other_rights null,
        file_name/file_hash 통과, T-001 의 17~23 null, T-002 의 계약서 문맥 보충, 키 순서.
  · A5  party_rules: dataset/new_contracts 5건 본문층에서 이용자 = 한국저작권위원회, 저작자 = §(c) 표.
  · A4  contract_context: 같은 5건에서 관리번호 없는 제목·종별 어문; 합성 본문으로 체크 글리프·줄바꿈.
  · A6  keyframes 표 v3/v4.  · A7 router 의 .pptx 힌트, 렌더링 불가 시 명확한 실패.
  · A1/A3 pipeline: setup 의 원본 파일명·SHA-256, build_response 의 test_prediction/licensee_party,
        run(consolidate_work=False) 가 중재자를 부르지 않음, contract_context/frame_schedule 전달.

llm_extraction 패키지의 __init__ 은 torch 를 올리므로(~30초) party_rules·eval23_serializer 는 파일 경로로 적재하고,
파이프라인이 lazy-import 하는 `module.llm_extraction.*` 는 __path__ 만 가진 빈 부모 모듈로 대신한다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

from api.module.evaluation.v4.keys23 import KEY_NAMES, T001_KEYS, T002_KEYS

ROOT = Path(__file__).resolve().parents[5]
API = ROOT / "api"
NEW_CONTRACTS = ROOT / "dataset" / "new_contracts"


def _load_by_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, API / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


party_rules = _load_by_path("_m2_party_rules", "module/llm_extraction/party_rules.py")
eval23 = _load_by_path("_m2_eval23_serializer", "module/llm_extraction/schemas/eval23_serializer.py")

# 01_new_contracts_T001_keys.md §(b)(c): 기대값 (파일명 앞 5자리 → …)
EXPECTED_TITLES = {
    "00002": "운수 자판기",
    "00003": "F!RE 가사",
    "00004": "국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축 전략",
    "00005": "국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축 전략",
    "00006": "빛의 세상으로(희망가) 가사",
}
EXPECTED_AUTHORS = {
    "00002": ["김은혜"],
    "00003": ["양지웅", "이영지", "신혜진"],
    "00004": ["서동주"],
    "00005": ["서동주"],
    "00006": ["김재성", "송이진"],
}
LICENSEE = "한국저작권위원회"


def _pdfs() -> List[Path]:
    if not NEW_CONTRACTS.is_dir():
        pytest.skip(f"{NEW_CONTRACTS} 없음")
    pdfs = sorted(NEW_CONTRACTS.glob("*.pdf"))
    if len(pdfs) != 5:
        pytest.skip(f"new_contracts PDF 가 5건이 아님: {len(pdfs)}")
    return pdfs


def _pdf_text(path: Path) -> str:
    fitz = pytest.importorskip("fitz")
    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text("text") for page in doc)


# ============================================================ A8 — 가벼운 import
def _run_py(args: List[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)


def test_import_api_is_torch_free_and_fast():
    code = ("import sys, time, json; t=time.time(); import api, api.module; "
            "print(json.dumps({'torch': 'torch' in sys.modules, 'transformers': 'transformers' in sys.modules, "
            "'elapsed': round(time.time()-t, 3), 'names': sorted(set(api.__all__) & {'ner_predict', 'pdf_to_image'})}))")
    t0 = time.perf_counter()
    r = _run_py(["-c", code])
    wall = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr[-800:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["torch"] is False and info["transformers"] is False, info
    assert info["elapsed"] < 3.0 and wall < 3.0, (info, wall)
    assert info["names"] == ["ner_predict", "pdf_to_image"]      # 지연 속성 이름은 그대로 노출된다


def test_evaluate_cli_help_starts_fast():
    t0 = time.perf_counter()
    r = _run_py(["-m", "api.module.evaluation.v4.evaluate", "--help"])
    wall = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr[-800:]
    assert "usage" in r.stdout.lower()
    assert wall < 3.0, wall


# ============================================================ A2 — eval23 serializer
def _md(**over: Any) -> Dict[str, Any]:
    """실측 계약서 결과(api/web/results/20260909_180001_279187) 모양의 평면 metadata."""
    md: Dict[str, Any] = {
        "work_title": "13387752 운수 자판기",
        "author": ["양지웅 (저작물 3 건)", "이영지 (저작물 2 건)", "신혜진"],
        "copyright_holder": ["김은혜"],                 # 모델 값 — v4 출력에는 쓰지 않는다
        "work_type": "어문저작물",
        "reproduction_right": False,
        "public_performance_right": True,
        "public_transmission_right": "허용(방송권, 디지털음성송신권)",
        "exhibition_right": "O",
        "distribution_right": "X",
        "rental_right": None,
        "derivative_work_creation_right": True,
        "granted_rights": ["공연권", "전시권", "2차적저작물작성권"],
        "other_rights": "방송권",
        "license_start_date": "2009년 09월 27일",
        "license_end_date": "2033-04-14",
        "valid_period": "2009 년 09 월 27 일부터 2033 년 04 월 14 일까지",
        "digital_format": "PDF",
        "file_size": 97831,
        "resolution": None,
        "keyword": ["저작권", "이용허락", "어문저작물"],
        "description": "저작재산권 독점적 이용허락 계약서",
    }
    md.update(over)
    return md


RESP = {"file_name": "00002_13387752 운수 자판기 (1).pdf", "filename": "00002_13387752_운수_자판기_1.pdf",
        "file_hash": "a" * 64, "file_size_bytes": 97831}


def test_keys_in_keys23_order_then_meta():
    out = eval23.build(_md(), RESP, None)
    assert list(out)[:-1] == list(KEY_NAMES)
    assert list(out)[-1] == "_meta"
    assert out["_meta"]["serializer"] == "eval23"


def test_renames():
    out = eval23.build(_md(), RESP, None)
    assert out["creator"] == ["양지웅", "이영지", "신혜진"]          # author → creator, 주석 제거
    assert out["keywords"] == ["저작권", "이용허락", "어문저작물"]      # keyword → keywords
    assert out["file_format"] == "PDF"                               # digital_format → file_format
    assert out["performance_right"] == "허용"                        # public_performance_right → performance_right
    assert out["derivative_work_right"] == "허용"                    # derivative_work_creation_right → derivative_work_right
    assert out["contract_validity_period"] == "2009-09-27~2033-04-14"
    assert out["work_title"] == "운수 자판기"                         # 관리번호 접두 제거
    assert out["work_type"] == "어문"                                # 닫힌 집합
    assert out["description"] == "저작재산권 독점적 이용허락 계약서"
    for legacy in ("author", "keyword", "digital_format", "public_performance_right"):
        assert legacy not in out


def test_rights_allow_deny_null_forms():
    out = eval23.build(_md(), RESP, None)
    assert out["reproduction_right"] == "불허"             # False
    assert out["performance_right"] == "허용"              # True
    assert out["public_transmission_right"] == "허용"      # '허용(방송권, …)' — 뒤 괄호 제거
    assert out["exhibition_right"] == "허용"               # 'O'
    assert out["distribution_right"] == "불허"             # 'X'
    assert out["rental_right"] == "불허"                   # None 이지만 granted_rights 가 목록 모양 → 목록에 없음 = 불허
    assert eval23.build(_md(granted_rights=None), RESP, None)["rental_right"] is None   # 어디에도 없으면 null


def test_rights_fall_back_to_granted_rights_list_dict_and_glyphs():
    # 개별 필드가 비면 granted_rights 로 — 목록 모양: 있으면 허용, 없으면 불허
    md = _md(reproduction_right=None, public_performance_right=None, rental_right=None,
             granted_rights=["공연권", "대여권"])
    out = eval23.build(md, RESP, None)
    assert out["performance_right"] == "허용" and out["rental_right"] == "허용"
    assert out["reproduction_right"] == "불허"
    # dict 모양 + 체크 글리프(모델이 그대로 베낀 경우)도 허용/불허로
    md = _md(reproduction_right=None, exhibition_right="v", distribution_right="□",
             granted_rights={"복제권": "허락"})
    out = eval23.build(md, RESP, None)
    assert out["reproduction_right"] == "허용"
    assert out["exhibition_right"] == "허용" and out["distribution_right"] == "불허"
    # 해석 불가 → null + _meta.unparsed
    out = eval23.build(_md(exhibition_right="글쎄요", granted_rights=None), RESP, None)
    assert out["exhibition_right"] is None
    assert out["_meta"]["unparsed"]["exhibition_right"] == "글쎄요"


def test_contract_validity_period_sources():
    assert eval23.build(_md(), RESP, None)["contract_validity_period"] == "2009-09-27~2033-04-14"   # 14·15 에서
    out = eval23.build(_md(license_end_date=None), RESP, None)                                       # 한쪽이 비면 valid_period 정규화
    assert out["license_end_date"] is None
    assert out["contract_validity_period"] == "2009-09-27~2033-04-14"
    out = eval23.build(_md(license_start_date=None, license_end_date=None, valid_period=None), RESP, None)
    assert out["contract_validity_period"] is None
    out = eval23.build(_md(license_end_date=None, valid_period="계약 체결일로부터"), RESP, None)      # 해석 불가 → 원문 + unparsed
    assert out["contract_validity_period"] == "계약 체결일로부터"
    assert "contract_validity_period" in out["_meta"]["unparsed"]


def test_dates_normalised_and_tails_stripped():
    out = eval23.build(_md(license_start_date="2009년 09월 27일부터", license_end_date="2033.04.14"), RESP, None)
    assert out["license_start_date"] == "2009-09-27" and out["license_end_date"] == "2033-04-14"
    out = eval23.build(_md(license_start_date=None, effective_date="20090927"), RESP, None)          # effective_date 대체
    assert out["license_start_date"] == "2009-09-27"
    out = eval23.build(_md(license_start_date="미상"), RESP, None)
    assert out["license_start_date"] == "미상" and out["_meta"]["unparsed"]["license_start_date"] == "미상"


def test_hashtags_from_keywords_unless_model_gave_them():
    out = eval23.build(_md(), RESP, None)
    assert out["hashtags"] == ["#저작권", "#이용허락", "#어문저작물"]
    assert out["_meta"]["hashtags_source"] == "keywords"
    out = eval23.build(_md(hashtags=["#바다", "해변"]), RESP, None)
    assert out["hashtags"] == ["#바다", "#해변"] and out["_meta"]["hashtags_source"] == "model"
    out = eval23.build(_md(keyword=None), RESP, None)
    assert out["keywords"] is None and out["hashtags"] is None
    assert eval23.build(_md(keyword="바다, 해변|노을"), RESP, None)["keywords"] == ["바다", "해변", "노을"]


def test_other_rights_always_null():
    out = eval23.build(_md(other_rights=["방송권", "전송권"]), RESP, None)
    assert out["other_rights"] is None
    assert out["_meta"]["model_other_rights"] == ["방송권", "전송권"]


def test_file_name_and_hash_passthrough():
    out = eval23.build(_md(), RESP, None)
    assert out["file_name"] == "00002_13387752 운수 자판기 (1).pdf"      # 원본 이름(공백 유지), 정제 이름 아님
    assert out["file_hash"] == "a" * 64
    assert out["file_size"] == 97831
    out = eval23.build(_md(file_size=None), {"filename": "x_y.pdf", "file_size_bytes": "98,468"}, None)
    assert out["file_name"] == "x_y.pdf" and out["file_hash"] is None and out["file_size"] == 98468
    out = eval23.build({"_file_meta": {"sha256": "b" * 64}}, {}, None)
    assert out["file_hash"] == "b" * 64
    assert eval23.build({"digital_format": None}, {"file_name": "a.JPG"}, None)["file_format"] == "JPG"
    assert eval23.build({"resolution": "1920 x 1080"}, {}, None)["resolution"] == "1920x1080"


def test_t001_nulls_keys_17_to_23():
    out = eval23.build(_md(resolution="100x100", licensee_party=LICENSEE), RESP, "T-001")
    for k in T002_KEYS:
        assert out[k] is None, k
    for k in ("work_title", "file_name", "work_type", "creator", "copyright_holder", "license_start_date",
              "license_end_date", "contract_validity_period"):
        assert out[k] is not None, k
    assert out["_meta"]["test"] == "T-001"
    assert set(T001_KEYS) | set(T002_KEYS) == set(KEY_NAMES)


def test_t002_fills_title_and_type_from_contract_context():
    vlm = {"work_type": None, "description": "사진", "keyword": ["a"], "digital_format": "JPG", "file_size": 10}
    resp = {**RESP, "contract_context": {"title": "13387752 운수 자판기", "work_type": "어문", "source": "rules"}}
    out = eval23.build(vlm, resp, "T-002")
    assert out["work_title"] == "운수 자판기" and out["_meta"]["work_title_source"] == "contract_context"
    assert out["work_type"] == "어문" and out["_meta"]["work_type_source"] == "contract_context"
    assert out["_meta"]["contract_context"]["source"] == "rules"
    out = eval23.build({**vlm, "work_type": "사진저작물"}, resp, "T-002")     # VLM 값이 있으면 그대로(문맥은 빈 값만 보충)
    assert out["work_type"] == "사진" and out["_meta"]["work_type_source"] == "model"
    with pytest.raises(ValueError):
        eval23.build(vlm, resp, "T-003")


def test_copyright_holder_is_licensee_party_not_model_value():
    out = eval23.build(_md(licensee_party="한국저작권위원회"), RESP, None)
    assert out["copyright_holder"] == ["한국저작권위원회"]
    assert out["_meta"]["copyright_holder_source"] == "licensee_party"
    assert out["_meta"]["model_copyright_holder"] == ["김은혜"]
    out = eval23.build(_md(licensee_party="A기관, B기관"), RESP, None)
    assert out["copyright_holder"] == ["A기관", "B기관"]
    out = eval23.build(_md(user="합성이용기관 교육정보센터"), RESP, None)            # 규칙 값이 없으면 모델의 user
    assert out["copyright_holder"] == ["합성이용기관 교육정보센터"] and out["_meta"]["copyright_holder_source"] == "user"
    out = eval23.build(_md(), RESP, None)
    assert out["copyright_holder"] is None and out["_meta"]["copyright_holder_source"] is None


def test_creator_never_falls_back_and_title_year_prefix_kept():
    out = eval23.build(_md(author=None), RESP, None)
    assert out["creator"] is None
    assert eval23.build({"work_title": "2021 아리랑 응원가 가사"}, {}, None)["work_title"] == "2021 아리랑 응원가 가사"
    assert eval23.build({"author": "홍길동 (저작물 2 건)\n김철수"}, {}, None)["creator"] == ["홍길동", "김철수"]


# ============================================================ A5 — party_rules (5건 본문층)
def test_party_rules_on_the_five_new_contracts():
    for pdf in _pdfs():
        key = pdf.name[:5]
        text = _pdf_text(pdf)
        lic = party_rules.find_licensee(text)
        assert lic["name"] == LICENSEE, (key, lic)
        assert lic["source"] == "signature" and lic["preamble"] == LICENSEE and lic["agree"] is True, (key, lic)
        assert party_rules.extract_licensee(text) == LICENSEE
        authors = party_rules.find_author(text)
        assert party_rules.clean_names(authors) == EXPECTED_AUTHORS[key], (key, authors)
        if key == "00003":
            assert authors == ["양지웅 (저작물 3 건)", "이영지 (저작물 2 건)", "신혜진"]


def test_party_rules_preamble_only_with_wrapped_words():
    text = ("저작자 또는 저작재산권 이용허락자 김은혜(이하 “권리자”이라 함)와 저작재산권 이\n"
            "용자 한국저작권위원회 (이하 “이용\n자”라 함)는 아래 저작물 운수 자판기에 관한 저\n작재산권 …")
    lic = party_rules.find_licensee(text)
    assert lic == {"name": LICENSEE, "names": [LICENSEE], "source": "preamble", "signature": None,
                   "preamble": LICENSEE, "agree": None}
    old = '저작재산권 이용자 합성이용기관 교육정보센터(이하 "이용자"라 함)는'
    assert party_rules.find_licensee(old)["name"] == "합성이용기관 교육정보센터"


def test_party_rules_signature_variants():
    assert party_rules.find_licensee("권리자 : 김은혜\n이용자 : 한국저작권위원회\n성\n명(인) : 한국저작권위원회")["names"] == [LICENSEE]
    assert party_rules.find_licensee("이용자：\n재단법인 A\n주식회사 B\n생년월일 : 1")["names"] == ["재단법인 A", "주식회사 B"]
    assert party_rules.find_licensee("권리자 : 김은혜\n성\n명(인) : 김은혜") == {
        "name": None, "names": [], "source": None, "signature": None, "preamble": None, "agree": None}
    assert party_rules.find_licensee("")["name"] is None
    # 서명란과 전문이 다르면 agree=False, 서명란이 이긴다
    lic = party_rules.find_licensee('저작재산권 이용자 갑(이하 "이용자"라 함)\n이용자 : 을')
    assert lic["name"] == "을" and lic["agree"] is False


# ============================================================ A4 — contract_context (5건 + 합성)
def test_contract_context_on_the_five_new_contracts():
    from api.module.clip_extraction import contract_context as cc
    pytest.importorskip("fitz")
    for pdf in _pdfs():
        key = pdf.name[:5]
        ctx = cc.parse(pdf)
        assert ctx["title"] == EXPECTED_TITLES[key], (key, ctx)
        assert ctx["work_type"] == "어문" and ctx["source"] == "rules", (key, ctx)
        assert ctx["title_raw"].split(" ", 1)[0].isdigit() and ctx["title_raw"].endswith(ctx["title"])
        assert ctx["pdf"] == pdf.name and "error" not in ctx


def test_contract_context_parse_text_glyphs_and_wraps():
    from api.module.clip_extraction import contract_context as cc
    text = ("제2조 (계약의 대상) \n제목(제호) : 13385261 PPT 템플릿_여행계획용_001\n저작자 : 김모\n"
            "종별 : □어문저작물, □음악저작물, □연극저작물, □미술저작물, □건축저작물, ■사진\n"
            "저작물, □영상저작물, □도형저작물, □컴퓨터프로그램저작물, □기타(-)\n권리 : ■복제권, □공연권")
    assert cc.parse_text(text) == {"title": "PPT 템플릿_여행계획용_001", "work_type": "사진", "source": "rules",
                                   "title_raw": "13385261 PPT 템플릿_여행계획용_001", "work_type_ticked": ["사진"]}
    t2 = "제목(제호) : 2021 아리랑 응원가 가사\n저작자 : x\n종별 : □어문저작물, □사진저작물, V영상저작물, □기타(-)\n권리 : □복제권"
    assert cc.parse_text(t2)["title"] == "2021 아리랑 응원가 가사"           # 4자리 연도는 관리번호가 아니다
    assert cc.parse_text(t2)["work_type"] == "영상"
    t3 = "제목(제호) :\n긴 제목 첫 줄 \n둘째 줄\n저작자 : x\n종별 : □어문저작물, □기타(-)\n권리 : □복제권"
    assert cc.parse_text(t3)["title"] == "긴 제목 첫 줄 둘째 줄"              # 다음 줄 이어붙임, 라벨에서 멈춤
    assert cc.parse_text(t3)["work_type"] is None                            # 체크 없음
    assert cc.parse_text("종별 : □어문저작물, v기타(사진집)\n권리 : □복제권")["work_type"] == "기타"
    assert cc.parse_text("") == {"title": None, "work_type": None, "source": "rules", "title_raw": None,
                                 "work_type_ticked": []}
    missing = cc.parse(ROOT / "dataset" / "nope.pdf")
    assert missing["title"] is None and missing["work_type"] is None and "error" in missing


# ============================================================ A6 — 키프레임 표
@pytest.mark.parametrize("duration, v3, v4", [(5, 3, 3), (30, 5, 5), (120, 8, 8), (600, 12, 16), (1200, 16, 24)])
def test_frame_schedule_v3_vs_v4(duration, v3, v4):
    from api.module.clip_extraction.keyframes import target_frame_count
    assert target_frame_count(duration) == v3
    assert target_frame_count(duration, schedule="v3") == v3
    assert target_frame_count(duration, schedule="v4") == v4


def test_frame_schedule_unknown_is_rejected(tmp_path):
    from api.module.clip_extraction import keyframes
    with pytest.raises(ValueError):
        keyframes.target_frame_count(10, schedule="v9")
    with pytest.raises(ValueError):
        keyframes.extract_keyframes(str(tmp_path / "none.mp4"), str(tmp_path / "kf"), schedule="v9")
    assert set(keyframes.FRAME_SCHEDULES) == {"v3", "v4"}


# ============================================================ A4 — 프롬프트·추출기 문맥
def test_user_prompt_with_context():
    from api.module.clip_extraction.vlm import prompts as P
    assert P.user_prompt_with_context(P.USER_PROMPT) == P.USER_PROMPT
    assert P.user_prompt_with_context(P.USER_PROMPT, title=" ", work_type=None) == P.USER_PROMPT
    out = P.user_prompt_with_context(P.USER_PROMPT, title="운수  자판기", work_type="어문")
    assert out.startswith(P.USER_PROMPT.rstrip())
    assert out.endswith(P._CONTEXT_LINE.format(items="저작물명 '운수 자판기', 종별 '어문'"))
    assert P.user_prompt_with_context("x", work_type="사진").endswith(P._CONTEXT_LINE.format(items="종별 '사진'"))
    assert "저작물명이 알려 주는 주제어" in P._CONTEXT_LINE and "수어" in P._CONTEXT_LINE        # 2026-10-10: 제목 주제어 허용


class _FakeBackend:
    """VLMClient 모양의 가짜 — 받은 프롬프트를 기록하고 성공 결과를 돌려준다. 네트워크 없음."""

    model_label = "fake"

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def ping(self):
        return True, "fake ok"

    def _ok(self, image):
        from api.module.clip_extraction.vlm.client import VLMResult
        return VLMResult(model_label=self.model_label, image=str(image), ok=True, latency_s=0.01,
                         raw_text="{}", parsed={"work_type": "사진저작물"}, parse_ok=True)

    def extract(self, image_path, system_prompt, user_prompt, **kw):
        self.calls.append({"kind": "image", "user_prompt": user_prompt})
        return self._ok(image_path)

    def extract_frames(self, frame_paths, system_prompt, user_prompt, label="video", **kw):
        self.calls.append({"kind": "video", "user_prompt": user_prompt})
        return self._ok(label)


def test_extractor_passes_context_into_prompts():
    from api.module.clip_extraction.vlm.extractor import VLMExtractor
    fake = _FakeBackend()
    ex = VLMExtractor(backends=[fake])
    ctx = {"title": "운수 자판기", "work_type": "어문", "source": "rules"}
    r = ex.extract("x.jpg")
    assert r.ok and "참고(계약서)" not in fake.calls[-1]["user_prompt"] and r.prompt_context is None
    r = ex.extract("x.jpg", context=ctx)
    assert "참고(계약서): 저작물명 '운수 자판기', 종별 '어문'" in fake.calls[-1]["user_prompt"]
    assert r.prompt_context == ctx and r.backend_used == "fake"
    r = ex.extract_video(["f1.jpg", "f2.jpg"], [0.5, 3.0], label="v.mp4", context=ctx)
    p = fake.calls[-1]["user_prompt"]
    from api.module.clip_extraction.vlm import prompts as _P
    assert p.startswith("다음 2장은") and p.endswith(_P._CONTEXT_LINE.format(items="저작물명 '운수 자판기', 종별 '어문'"))
    r = ex.extract_video(["f1.jpg"], [0.5])
    assert "참고(계약서)" not in fake.calls[-1]["user_prompt"] and r.prompt_context is None
    with pytest.raises(TypeError):
        ex.extract("x.jpg", context="운수 자판기")


class _CountingBackend(_FakeBackend):
    """ping 호출 횟수를 세고, reachable 여부를 정할 수 있는 가짜 백엔드."""

    def __init__(self, label: str, reachable: bool = True):
        super().__init__()
        self.model_label = label
        self.reachable = reachable
        self.pings = 0

    def ping(self):
        self.pings += 1
        return (True, "ok") if self.reachable else (False, "APITimeoutError: Request timed out.")


def test_probe_chain_stops_at_first_reachable_backend():
    """회귀(RISK): 체인 전체를 두드리면 SYN 단계에서 멎은 자체 서버(GEMMA_URL) 하나에 다리마다 300s 를 기다렸다."""
    from api.module.clip_extraction.vlm.extractor import VLMExtractor
    a, b, c = _CountingBackend("openrouter"), _CountingBackend("vllm", reachable=False), _CountingBackend("qwen")
    ex = VLMExtractor(backends=[a, b, c])
    assert ex.active_backend() is a
    assert (a.pings, b.pings, c.pings) == (1, 0, 0)            # 첫 번째가 응답하면 뒤는 두드리지 않는다
    assert [p["label"] for p in ex.probe_log] == ["openrouter"]
    assert ex.extract("x.jpg").backend_used == "openrouter" and (b.pings, c.pings) == (0, 0)

    # 첫 번째가 죽어 있으면 다음으로 — 선택된 것까지만 두드린다
    a.reachable = False
    ex.refresh()
    assert ex.active_backend() is c
    assert (a.pings, b.pings, c.pings) == (2, 1, 1)
    assert [p["label"] for p in ex.probe_log] == ["openrouter", "vllm", "qwen"]

    # 전부 죽어 있으면 모두 두드린 뒤 RuntimeError (예전과 같다)
    c.reachable = False
    with pytest.raises(RuntimeError, match="Tried 3"):
        ex.refresh()


def test_vlm_ping_uses_short_timeout_without_retries(monkeypatch):
    """회귀(RISK): ping 이 추출용 timeout(300s)·재시도 1회를 그대로 쓰면 응답 없는 서버에 5분을 기다린다.

    연결은 받되 응답하지 않는 소켓(listen 만, accept 없음)으로 흉내 낸다 — SYN 단계 멈춤과 같은 증상.
    """
    import socket
    from api.module.clip_extraction.vlm import client as C
    from api.module.clip_extraction.vlm.client import VLMClient
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        monkeypatch.setenv("VLM_PING_TIMEOUT_SEC", "1")
        assert C._ping_timeout() == 1.0
        vc = VLMClient("hung", f"http://127.0.0.1:{port}/v1", "m", "not-needed", timeout=300.0)
        assert vc.client.timeout == 300.0 and vc.client.max_retries == 1  # 추출용 설정은 그대로
        t0 = time.perf_counter()
        ok, detail = vc.ping()
        dt = time.perf_counter() - t0
        assert ok is False and "Timeout" in detail and dt < 15, (detail, dt)
    finally:
        srv.close()
    monkeypatch.setenv("VLM_PING_TIMEOUT_SEC", "abc")
    assert C._ping_timeout() == C.PING_TIMEOUT_SEC                 # 잘못된 값은 기본값으로
    monkeypatch.setenv("VLM_PING_TIMEOUT_SEC", "-3")
    assert C._ping_timeout() == C.PING_TIMEOUT_SEC
    monkeypatch.delenv("VLM_PING_TIMEOUT_SEC")
    assert C._ping_timeout() == 10.0
    # 닫힌 포트는 즉시 거절된다 — timeout 과 무관하게 빠르다
    t0 = time.perf_counter()
    assert VLMClient("down", "http://127.0.0.1:9/v1", "m", "k").ping()[0] is False
    assert time.perf_counter() - t0 < 10


# ============================================================ A7 — 라우터 / 렌더링 불가
def test_router_pptx_follows_media_hint():
    from api.module.clip_extraction.router import route
    assert route("a.pptx")["modality"] == "document" and route("a.pptx")["render"] is None
    plan = route("PPT 템플릿.pptx", media_hint="image")
    assert plan["modality"] == "image" and plan["render"] == "pptx_slide1" and plan["extractors"] == ["vlm"]
    assert route("a.pptx", media_hint="video")["modality"] == "document"
    assert route("a.jpg", media_hint="image") == {**route("a.jpg"), "media_hint": "image"}
    assert route("a.docx", media_hint="image")["modality"] == "document"


def test_pptx_render_unavailable_is_a_clear_error(monkeypatch, tmp_path):
    import module.clip_extraction.pptx_render as pr   # pipeline 이 쓰는 import 경로(평면 module.*)
    src = tmp_path / "t.pptx"
    src.write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(pr, "powershell_path", lambda: None)
    assert pr.is_available() is False
    with pytest.raises(pr.PptxRenderError, match="powershell.exe"):
        pr.render_first_slide(src, tmp_path / "out.png")
    with pytest.raises(pr.PptxRenderError, match="원본 파일 없음"):
        pr.render_first_slide(tmp_path / "none.pptx", tmp_path / "out.png")


# ============================================================ A1/A3 — pipeline
@pytest.fixture
def light_llm_extraction():
    """pipeline 이 lazy-import 하는 module.llm_extraction.{party_rules, schemas.*} 를 패키지 __init__(torch) 없이.

    __path__ 만 가진 빈 부모 모듈을 끼우면 하위 모듈은 실제 파일에서 그대로 적재된다. 진짜 패키지가
    이미 올라와 있으면 그대로 둔다. 끝나면 이 테스트가 올린 module.llm_extraction* 항목을 치운다.
    """
    import module  # noqa: F401 — 편집 가능 설치(api/ 가 sys.path)로 평면 패키지가 보여야 한다
    added_stub = False
    if "module.llm_extraction" not in sys.modules:
        stub = types.ModuleType("module.llm_extraction")
        stub.__path__ = [str(API / "module" / "llm_extraction")]  # type: ignore[attr-defined]
        sys.modules["module.llm_extraction"] = stub
        added_stub = True
    before = set(sys.modules)
    yield
    if added_stub:
        for name in [n for n in sys.modules if n == "module.llm_extraction" or n.startswith("module.llm_extraction.")]:
            if name not in before or name == "module.llm_extraction":
                sys.modules.pop(name, None)


def _orch(tmp_path: Path):
    from api.web.pipeline import PipelineOrchestrator
    return PipelineOrchestrator(llm_processor=None, ner_predict_fn=None, available_ner_models={},
                                upload_dir=tmp_path / "uploads", results_dir=tmp_path / "results")


def test_setup_keeps_original_name_and_sha256(tmp_path):
    orch = _orch(tmp_path)
    data = b"%PDF-1.4 fake"
    ctx = orch.setup(data, "00002_13387752 운수 자판기 (1).pdf")
    assert ctx["original_filename"] == "00002_13387752 운수 자판기 (1).pdf"
    assert ctx["filename"] == "00002_13387752_운수_자판기_1.pdf"            # 저장용 정제 이름은 그대로
    assert ctx["sha256"] == hashlib.sha256(data).hexdigest()
    assert ctx["file_size"] == len(data) and ctx["upload_path"].read_bytes() == data
    assert orch.setup(data, "/tmp/dir/13388087_1_태안꽃박람회 엽서제작.jpg")["original_filename"] == "13388087_1_태안꽃박람회 엽서제작.jpg"


def test_build_response_t001_contract(tmp_path, light_llm_extraction):
    pdfs = _pdfs()
    text = _pdf_text(pdfs[0])                 # 00002
    orch = _orch(tmp_path)
    ctx = orch.setup(pdfs[0].read_bytes(), pdfs[0].name)
    llm = {"success": True, "metadata": _md(), "confidence": 0.8, "model_used": "m", "extraction_time": 1.0}
    resp = orch.build_response(
        ctx, model_name="m", document_type="저작재산권 이용허락 계약서",
        ocr_text=text, ocr_provider="alibaba", ocr_model=None,
        llm_result=llm, ner_model=None, ner_result=None,
        consolidate=False, consolidation_model="c", consolidation_result=None,
        consolidation_success=False, consolidation_error=None,
        ocr_result={"ocr_provider": "direct_text", "total_pages": 5}, test="T-001",
        contract_context_rules=orch._contract_rules(ctx),          # run() 의 문서 분기가 넘기는 것과 같다 (pipeline.run)
    )
    assert resp["file_name"] == pdfs[0].name and resp["file_hash"] == hashlib.sha256(pdfs[0].read_bytes()).hexdigest()
    assert resp["file_size_bytes"] == pdfs[0].stat().st_size
    assert resp["ocr_provider"] == "alibaba" and resp["ocr_provider_used"] == "direct_text" and resp["ocr_pages"] == 5
    assert resp["licensee_party"] == LICENSEE and resp["licensee_party_detail"]["source"] == "signature"
    assert resp["metadata"]["licensee_party"] == LICENSEE
    pred = resp["test_prediction"]
    assert pred["copyright_holder"] == [LICENSEE] and pred["_meta"]["model_copyright_holder"] == ["김은혜"]
    assert pred["file_name"] == pdfs[0].name and pred["work_title"] == "운수 자판기" and pred["creator"] == ["양지웅", "이영지", "신혜진"]
    # M2b 수정 2: 제목·종별은 규칙 파싱(contract_context_rules)이 모델 값을 이긴다 — 모델 값은 _meta 에만
    assert pred["_meta"]["work_title_source"] == "rules" and pred["_meta"]["work_type_source"] == "rules"
    assert pred["_meta"]["model_work_title"] == "13387752 운수 자판기" and pred["_meta"]["model_work_type"] == "어문저작물"
    assert resp["contract_context_rules"]["title"] == "운수 자판기" and resp["contract_context_rules"]["work_type"] == "어문"
    assert all(pred[k] is None for k in T002_KEYS) and pred["_meta"]["test"] == "T-001"
    assert pred["_meta"]["licensee_rule"]["agree"] is True
    assert resp["tta_metadata"]["identification_information"]["work_title"] == "13387752 운수 자판기"   # 기존 TTA 출력 유지
    for legacy in ("filename", "metadata", "consolidated_metadata", "consolidation_decisions", "entities", "processing_time"):
        assert legacy in resp


def test_contract_rules_report_original_filename(tmp_path):
    """회귀(NIT): contract_context_rules['pdf'] 가 저장용 정제 이름('…_운수_자판기_1.pdf')이었다 — 응답 file_name 과 같아야 한다."""
    pdfs = _pdfs()
    orch = _orch(tmp_path)
    ctx = orch.setup(pdfs[0].read_bytes(), "00002_13387752 운수 자판기 (1).pdf")
    assert ctx["filename"] == "00002_13387752_운수_자판기_1.pdf"
    rules = orch._contract_rules(ctx)
    assert rules["pdf"] == "00002_13387752 운수 자판기 (1).pdf"
    assert rules["title"] == "운수 자판기" and rules["work_type"] == "어문" and rules["source"] == "rules"
    assert orch._contract_rules(orch.setup(b"x", "a.jpg")) is None


def _fake_vlm_result(**over: Any) -> Dict[str, Any]:
    md = {"work_type": "사진저작물", "description": "바다 위의 배", "keyword": ["바다", "배"],
          "digital_format": "JPG", "file_size": 3, "resolution": "10x10"}
    out = {"success": True, "metadata": md, "confidence": 0.7, "model_used": "fake", "extraction_time": 0.1,
           "_file_meta": {}, "_vlm_raw": {}, "_render": None, "_prompt_context": None}
    out.update(over)
    return out


def test_run_image_without_work_consolidation(tmp_path, light_llm_extraction, monkeypatch):
    orch = _orch(tmp_path)
    seen: Dict[str, Any] = {}

    def fake_vlm(ctx, prefer="gemma", context=None, render=None):
        seen.update(prefer=prefer, context=context, render=render)
        return _fake_vlm_result(_prompt_context=context)

    def no_consolidation(*a, **k):
        raise AssertionError("consolidate_work=False 인데 중재자가 불렸다")

    monkeypatch.setattr(orch, "run_vlm", fake_vlm)
    monkeypatch.setattr(orch, "run_consolidation", no_consolidation)
    cc = {"title": "13387752 운수 자판기", "work_type": "사진", "source": "rules"}
    data = b"\xff\xd8\xff fake jpg"
    resp = orch.run(data, "13388087_1_태안꽃박람회 엽서제작.jpg", consolidate_work=False, contract_context=cc,
                    test="T-002", media="image", vlm_prefer="qwen")
    assert seen == {"prefer": "qwen", "context": cc, "render": None}
    assert resp["consolidate"] is False and resp["consolidate_work"] is False and resp["consolidation_model"] is None
    assert resp["consolidated_metadata"] == resp["metadata"] and resp["work_consolidation"]["applied"] is False
    assert resp["file_name"] == "13388087_1_태안꽃박람회 엽서제작.jpg" and resp["filename"] == "13388087_1_태안꽃박람회_엽서제작.jpg"
    pred = resp["test_prediction"]
    assert pred["file_hash"] == hashlib.sha256(data).hexdigest() == resp["file_hash"]
    assert pred["file_name"] == "13388087_1_태안꽃박람회 엽서제작.jpg" and pred["file_format"] == "JPG"
    assert pred["keywords"] == ["바다", "배"] and pred["hashtags"] == ["#바다", "#배"] and pred["resolution"] == "10x10"
    assert pred["work_title"] == "운수 자판기" and pred["work_type"] == "사진"       # 제목은 계약서 문맥, 종별은 VLM
    assert pred["_meta"]["contract_context"] == cc and pred["_meta"]["test"] == "T-002"
    assert resp["test"] == "T-002" and resp["contract_context"] == cc
    assert (tmp_path / "results" / resp["request_id"] / "llm_metadata.json").is_file()
    assert json.loads((tmp_path / "results" / resp["request_id"] / "llm_metadata.json").read_text(encoding="utf-8"))["test_prediction"]["file_hash"] == resp["file_hash"]


def test_run_image_default_still_consolidates(tmp_path, light_llm_extraction, monkeypatch):
    orch = _orch(tmp_path)
    calls: List[Any] = []
    monkeypatch.setattr(orch, "run_vlm", lambda ctx, prefer="gemma", context=None, render=None: _fake_vlm_result())

    def fake_con(llm_result, ner_result, ocr_text, document_type, result_dir, consolidation_model):
        calls.append(consolidation_model)
        return ({"consolidated_metadata": dict(llm_result["metadata"]), "validation_report": {"decisions": [], "summary": {}},
                 "model_used": consolidation_model}, True, None)

    monkeypatch.setattr(orch, "run_consolidation", fake_con)
    resp = orch.run(b"img", "a.jpg")
    assert calls == ["alibaba-qwen3.5-122b-a10b"]
    assert resp["consolidate"] is True and resp["consolidate_work"] is True and resp["consolidation_success"] is True
    assert "work_consolidation" not in resp and resp["test"] is None
    assert resp["test_prediction"]["work_type"] == "사진" and resp["test_prediction"]["_meta"]["test"] is None
    assert resp["contract_context"] is None


def test_run_routes_pptx_as_image_when_media_hint_given(tmp_path, light_llm_extraction, monkeypatch):
    orch = _orch(tmp_path)
    seen: Dict[str, Any] = {}

    def fake_vlm(ctx, prefer="gemma", context=None, render=None):
        seen.update(render=render)
        return _fake_vlm_result(_render={"method": render, "ok": True, "png": "x"},
                                metadata={"work_type": "미술저작물", "digital_format": "PPTX", "file_size": 4, "resolution": None})

    monkeypatch.setattr(orch, "run_vlm", fake_vlm)
    resp = orch.run(b"PK\x03\x04", "13385261_1_PPT 템플릿_여행계획용_001.pptx", media="image",
                    consolidate_work=False, test="T-002")
    assert seen == {"render": "pptx_slide1"} and resp["modality"] == "image" and resp["render"]["method"] == "pptx_slide1"
    assert resp["test_prediction"]["file_format"] == "PPTX" and resp["test_prediction"]["resolution"] is None


def test_run_vlm_pptx_render_unavailable_returns_failure_not_exception(tmp_path, monkeypatch):
    import module.clip_extraction.pptx_render as pr
    monkeypatch.setattr(pr, "powershell_path", lambda: None)
    orch = _orch(tmp_path)
    ctx = orch.setup(b"PK\x03\x04", "t.pptx")
    res = orch.run_vlm(ctx, render="pptx_slide1")
    assert res["success"] is False and "PPTX" in res["error"] and res["_render"] == {"method": "pptx_slide1", "ok": False}
    assert res["metadata"] == {} and res["model_used"] == "none"


def test_run_video_gets_schedule_and_context(tmp_path, light_llm_extraction, monkeypatch):
    orch = _orch(tmp_path)
    seen: Dict[str, Any] = {}

    def fake_video(ctx, prefer="omni", context=None, frame_schedule="v3"):
        seen.update(prefer=prefer, context=context, frame_schedule=frame_schedule)
        return _fake_vlm_result(metadata={"work_type": "영상저작물", "description": "d", "keyword": ["k"],
                                          "digital_format": "MP4", "file_size": 3, "resolution": "640x360"},
                                _keyframes={"kept": 3, "duration": 7.0, "schedule": frame_schedule})

    monkeypatch.setattr(orch, "run_video", fake_video)
    monkeypatch.setattr(orch, "run_consolidation", lambda *a, **k: pytest.fail("중재자 호출"))
    cc = {"title": "광천 토굴 새우젓", "work_type": "영상"}
    resp = orch.run(b"\x00\x00\x00\x18ftyp", "87248.mp4", consolidate_work=False, frame_schedule="v4",
                    contract_context=cc, test="T-002")
    assert seen == {"prefer": "omni", "context": cc, "frame_schedule": "v4"}
    assert resp["frame_schedule"] == "v4" and resp["keyframes"]["schedule"] == "v4"
    assert resp["consolidated_metadata"]["work_type"] == "영상저작물" and resp["work_consolidation"]["applied"] is False
    assert resp["test_prediction"]["file_format"] == "MP4" and resp["test_prediction"]["resolution"] == "640x360"
    assert resp["test_prediction"]["work_title"] == "광천 토굴 새우젓"


def test_run_rejects_unknown_test(tmp_path):
    with pytest.raises(ValueError):
        _orch(tmp_path).run(b"x", "a.jpg", test="T-003")


# ============================================================ M2b 수정 2 — 제목·종별 우선순위 (규칙 파싱 > 모델)
RULES_00004 = {"title": "국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축 전략", "work_type": "어문", "source": "rules",
               "pdf": "00004_13333800 국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축 전략.pdf",
               "title_raw": "13333800 국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축 전략", "work_type_ticked": ["어문"]}


def test_rule_parsed_title_and_type_take_precedence_over_model():
    """회귀(M2b 수정 2): T-001 개발 런에서 LLM 은 공백을 흔들었고('구축 전략'→'구축전략', '세상으로(희망가)'→'세상으로 (희망가)')
    규칙 파서는 5/5 정확했다 — contract_context_rules 가 있으면 제목·종별은 그것을 쓰고 모델 값은 _meta 에만 남긴다."""
    md = _md(work_title="13333800 국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축전략", work_type="텍스트")
    out = eval23.build(md, {**RESP, "contract_context_rules": RULES_00004}, "T-001")
    assert out["work_title"] == RULES_00004["title"]
    assert out["_meta"]["work_title_source"] == "rules"
    assert out["_meta"]["model_work_title"] == "13333800 국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축전략"
    assert out["work_type"] == "어문" and out["_meta"]["work_type_source"] == "rules"
    assert out["_meta"]["model_work_type"] == "텍스트"
    assert "work_type" not in out["_meta"].get("unparsed", {})            # 출력에 실리지 않은 모델 값은 unparsed 가 아니다
    assert out["_meta"]["rule_title"] == RULES_00004["title"] and out["_meta"]["rule_work_type"] == "어문"   # 기존 기록 유지

    # 두 번째 드리프트 사례
    rules6 = {"title": "빛의 세상으로(희망가) 가사", "work_type": "어문", "source": "rules"}
    out = eval23.build(_md(work_title="13333560 빛의 세상으로 (희망가) 가사"), {**RESP, "contract_context_rules": rules6}, "T-001")
    assert out["work_title"] == "빛의 세상으로(희망가) 가사" and out["_meta"]["work_title_source"] == "rules"
    assert out["_meta"]["model_work_title"] == "13333560 빛의 세상으로 (희망가) 가사"

    # 규칙이 비어 있으면(제호 줄을 못 읽은 PDF) 모델 값 — 관리번호 제거는 그대로, 출처 'model'
    empty_rules = {"title": None, "work_type": None, "source": "rules", "title_raw": None, "work_type_ticked": []}
    out = eval23.build(_md(), {**RESP, "contract_context_rules": empty_rules}, "T-001")
    assert out["work_title"] == "운수 자판기" and out["_meta"]["work_title_source"] == "model"
    assert out["work_type"] == "어문" and out["_meta"]["work_type_source"] == "model"
    assert out["_meta"]["model_work_title"] == "13387752 운수 자판기" and out["_meta"]["model_work_type"] == "어문저작물"
    # 규칙이 한쪽만 있으면 그 키만 규칙 (공백 정규화는 같은 _title 을 거친다)
    out = eval23.build(_md(), {**RESP, "contract_context_rules": {"title": "운수  자판기", "work_type": None}}, "T-001")
    assert out["work_title"] == "운수 자판기" and out["_meta"]["work_title_source"] == "rules"
    assert out["work_type"] == "어문" and out["_meta"]["work_type_source"] == "model"
    # 규칙이 아예 없는 응답(저작물 leg)·T-002 도 같은 규칙: 모델 값 + 출처 'model'; 규칙이 있으면 T-002 에서도 규칙
    out = eval23.build(_md(), RESP, "T-002")
    assert out["work_title"] == "운수 자판기" and out["_meta"]["work_title_source"] == "model" and out["_meta"]["work_type_source"] == "model"
    out = eval23.build(_md(), {**RESP, "contract_context_rules": RULES_00004}, "T-002")
    assert out["work_title"] == RULES_00004["title"] and out["_meta"]["work_title_source"] == "rules"
    # 규칙 dict 가 아니면 무시
    out = eval23.build(_md(), {**RESP, "contract_context_rules": "rules"}, "T-001")
    assert out["work_title"] == "운수 자판기" and out["_meta"]["work_title_source"] == "model"
    # 둘 다 없으면 null + 출처 None
    out = eval23.build({}, {}, None)
    assert out["work_title"] is None and out["_meta"]["work_title_source"] is None and out["_meta"]["model_work_title"] is None
    assert out["work_type"] is None and out["_meta"]["work_type_source"] is None and out["_meta"]["model_work_type"] is None
    # 모델 값이 닫힌 집합 밖이고 규칙도 없으면 — 예전처럼 원문을 내보내고 unparsed 에 적는다
    out = eval23.build(_md(work_type="텍스트"), RESP, "T-001")
    assert out["work_type"] == "텍스트" and out["_meta"]["unparsed"]["work_type"] == "텍스트" and out["_meta"]["work_type_source"] == "model"
    # T-002 문맥 보충은 모델 값이 비어 있을 때만, 그 출처는 'contract_context' 그대로
    out = eval23.build({"description": "d"}, {**RESP, "contract_context": {"title": "13387752 운수 자판기", "work_type": "어문"}}, "T-002")
    assert out["work_title"] == "운수 자판기" and out["_meta"]["work_title_source"] == "contract_context"
    assert out["work_type"] == "어문" and out["_meta"]["work_type_source"] == "contract_context"


# ============================================================ M2b 수정 3 — VLM 해시태그 요청 + 통합 스키마 매핑
def _json_block(text: str) -> Dict[str, Any]:
    """프롬프트 안의 JSON 스키마 블록({ … }) 하나를 파싱한다 — 값은 전부 문자열/문자열 목록이라 그대로 JSON 이다."""
    return json.loads(text[text.index("{"): text.index("}") + 1])


def test_vlm_prompts_request_hashtags_and_keep_other_lines():
    from api.module.clip_extraction.vlm import prompts as P
    expected_keys = {"description", "work_type", "work_type_reason", "keywords", "hashtags", "main_subjects",
                     "dominant_colors", "text_in_image", "scene_type", "estimated_quality"}
    for text in (P.USER_PROMPT, P.video_user_prompt([0.5, 3.0])):
        block = _json_block(text)
        assert set(block) == expected_keys                                      # hashtags 만 늘었다
        assert isinstance(block["hashtags"], list) and len(block["hashtags"]) == 1
        spec = block["hashtags"][0]
        assert "5~10" in spec and "'#태그'" in spec and "띄어쓰기 없이" in spec
        for word in ("대상", "장소", "행위", "매체", "명사", "인명", "라이선스", "출처", "저작물명의 주제어"):
            assert word in spec, word
        lines = text.splitlines()
        i = next(idx for idx, l in enumerate(lines) if l.startswith('  "keywords"'))
        assert lines[i + 1].startswith('  "hashtags"')                           # keywords 줄 바로 다음
        if text is P.USER_PROMPT:                                               # 2026-10-10: 지침 정렬 키워드 규칙 (이미지)
            assert lines[i].startswith('  "keywords": ["키워드 5~7개 — 저작물명에 주제어')
        else:                                                                   # 2026-10-31: 영상은 아카이브 분류어
            assert lines[i].startswith('  "keywords": ["키워드 5~10개 — 아카이브 검색용 분류어')
            assert "사람 이름" in lines[i] and "'영상'을 넣고" in lines[i]
        d = block["description"]
        if text is P.USER_PROMPT:                                               # 2026-10-31: 이미지 설명 = 정답 초안 지침의 글쓰기 규칙
            assert "1~3문장, 공백 포함 30~200자" in d and "첫 문장 = 주 피사체" in d and "추측" in d
        else:                                                                   # 영상 설명은 그대로 (Te-2 확인에서 옛 영화가 떨어짐)
            assert d.startswith("영상 내용을 한국어 2~3문장으로 구체적으로")
        assert "키워드·해시태그 작성 예시" in text and text.rstrip().endswith("JSON 객체 하나만 출력하세요.")
    # 나머지 줄은 그대로: 규칙 문단·마무리 문장·영상용 머리말·계약서 문맥 줄
    assert "★ 매우 중요: work_type은 '무엇을 찍었는가(피사체)'가 아니라" in P.USER_PROMPT
    assert P.USER_PROMPT.endswith("JSON 객체 하나만 출력하세요.")
    assert P.video_user_prompt([1.0]).startswith("다음 1장은 하나의 영상에서 시간순으로 추출한 대표 프레임입니다(각 시점: 1초).")
    assert "★ 고유명사(인물명·작품명)는 확실하지 않으면" in P.video_user_prompt([1.0])
    assert P.user_prompt_with_context(P.USER_PROMPT, title="t").endswith(P._CONTEXT_LINE.format(items="저작물명 't'"))
    assert P.USER_PROMPT.count('"hashtags"') == 1 and P._VIDEO_USER_TEMPLATE.count('"hashtags"') == 1


def test_map_vlm_to_unified_carries_hashtags(light_llm_extraction):
    from api.module.clip_extraction.schema_mapping import _normalize_hashtags, map_vlm_to_unified
    from module.llm_extraction.schemas.document_schemas import DocumentSchemas
    props = DocumentSchemas.get_unified_schema()["properties"]
    assert "hashtags" in props and props["hashtags"]["type"] == ["array", "null"]   # 하드 규칙: 스키마에 있는 키만 emit
    vlm = {"description": "항구에 정박한 배", "work_type": "사진저작물", "keywords": ["바다", "배"],
           "hashtags": ["#바다", "배", " #항구 ", "#바다", "", None, "#노을 #하늘"]}
    u = map_vlm_to_unified(vlm)
    assert u["hashtags"] == ["#바다", "#배", "#항구", "#노을", "#하늘"]          # '#' 보장·공백/중복/빈 항목 제거
    assert u["keyword"] == ["바다", "배"] and u["description"] == "항구에 정박한 배"
    assert set(k for k in u if not k.startswith("_")) == set(props)               # 스키마-complete 그대로
    # 문자열로 온 경우도 쪼갠다; 없으면 스키마 초기값([]) → eval23 은 keywords 로 대체
    assert map_vlm_to_unified({"hashtags": "#바다 #배, 항구;노을"})["hashtags"] == ["#바다", "#배", "#항구", "#노을"]
    assert map_vlm_to_unified({"description": "d"})["hashtags"] == []
    assert map_vlm_to_unified({"hashtags": None})["hashtags"] == []
    assert map_vlm_to_unified({"hashtags": ["#", " "]})["hashtags"] == []
    assert _normalize_hashtags(("a", "b")) == ["#a", "#b"] and _normalize_hashtags("") == []
    # eval23 직렬화기: VLM 해시태그 → hashtags_source='model'; 없으면 keywords 대체 그대로
    out = eval23.build(u, {"file_name": "a.jpg"}, "T-002")
    assert out["hashtags"] == ["#바다", "#배", "#항구", "#노을", "#하늘"] and out["_meta"]["hashtags_source"] == "model"
    out = eval23.build(map_vlm_to_unified({"keywords": ["바다"]}), {"file_name": "a.jpg"}, "T-002")
    assert out["hashtags"] == ["#바다"] and out["_meta"]["hashtags_source"] == "keywords"


def test_run_image_vlm_hashtags_reach_test_prediction(tmp_path, light_llm_extraction, monkeypatch):
    """VLM 이 낸 hashtags 가 run_vlm → schema_mapping → metadata → eval23 까지 그대로 흘러 hashtags_source='model' 이 된다."""
    import importlib
    ext_mod = importlib.import_module("module.clip_extraction.vlm.extractor")      # run_vlm 이 쓰는 import 경로

    class _FakeExtractor:
        def __init__(self, *a, **k):
            pass

        def extract(self, image_path, context=None):
            return types.SimpleNamespace(ok=True, parse_ok=True, error=None, backend_used="fake",
                                         parsed={"description": "항구에 정박한 배", "work_type": "사진저작물",
                                                 "keywords": ["바다", "배"], "hashtags": ["#바다", "배", "#항구"],
                                                 "main_subjects": ["배"], "dominant_colors": ["파랑"]})

    monkeypatch.setattr(ext_mod, "VLMExtractor", _FakeExtractor)
    orch = _orch(tmp_path)
    resp = orch.run(b"\xff\xd8\xff fake jpg", "13388087_1_태안꽃박람회 엽서제작.jpg", consolidate_work=False, test="T-002")
    assert resp["success"] is True, resp.get("error")
    assert resp["metadata"]["hashtags"] == ["#바다", "#배", "#항구"] and resp["vlm_backend"] == "fake"
    pred = resp["test_prediction"]
    assert pred["hashtags"] == ["#바다", "#배", "#항구"] and pred["_meta"]["hashtags_source"] == "model"
    assert pred["keywords"] == ["바다", "배"] and pred["description"] == "항구에 정박한 배"


# ---------------------------------------------------------------- OCR 모델 교체 (2026-10-11): qwen3-vl-235b 퇴역 → qwen3.7-plus 생각 끔
def test_ocr_default_retired_mapping_thinking_and_fallback():
    import sys
    sys.path.insert(0, "api")
    from module.ocr.alibaba_ocr import AlibabaCloudOCRProvider as A
    assert A(api_key="x").model == "qwen3.7-plus"
    for old in ("qwen3-vl-235b-a22b-instruct", "qwen3-vl-30b-a3b-instruct", "qwen-vl-plus", "qwen-vl-max"):
        assert A(api_key="x", model=old).model == "qwen3.7-plus"          # 퇴역 이름은 실패 대신 주 모델로
    assert A.thinking_params("qwen3.7-plus") == {"extra_body": {"enable_thinking": False}}
    assert A.thinking_params(A.FALLBACK_MODEL) == {"extra_body": {"enable_thinking": False}}
    assert A.thinking_params("qwen3.8-flash") == {}                       # 측정된 설정 그대로 (플래그 없음)
    assert A.OCR_MODEL_FALLBACK["qwen3.7-plus"] == A.FALLBACK_MODEL and A.OCR_MODEL_FALLBACK[A.FALLBACK_MODEL] == "qwen3.7-plus"
    for m in A.OCR_MODEL_FALLBACK.values():
        assert m not in A.RETIRED_MODELS and m not in A.TEXT_ONLY_MODELS


def test_vlm_qwen_backend_uses_qwen37_with_thinking_off():
    from api.module.clip_extraction.vlm import extractor as X
    b = X.make_qwen_backend()
    assert b.model == "qwen3.7-plus" and b.extra_body == {"enable_thinking": False}


# ---------------------------------------------------------------- 2026-10-11: 영상 체인 · JSON 실패 폴백 · 추론 블록 · 연구실 Gemma 연결 제한
def test_video_chain_starts_with_qwen38_omni_flash_then_omni_plus():
    from api.module.clip_extraction.vlm import extractor as X
    chain = X.default_backends(prefer="omni")
    assert [b.model for b in chain[:3]] == ["qwen3.8-omni-flash", "qwen3.5-omni-plus", "qwen3.7-plus"]


def test_unparseable_answer_falls_back_to_next_backend():
    from api.module.clip_extraction.vlm import extractor as X
    from api.module.clip_extraction.vlm.client import VLMResult

    class B:
        def __init__(self, label, parse_ok):
            self.model_label, self.parse_ok, self.calls = label, parse_ok, 0
        def ping(self):
            return True, "ok"
        def extract_frames(self, *a, **k):
            self.calls += 1
            return VLMResult(model_label=self.model_label, image="v", ok=True, latency_s=0.1, raw_text="x",
                             parsed={"description": "d"} if self.parse_ok else None, parse_ok=self.parse_ok)
    first, second = B("first", False), B("second", True)
    ex = X.VLMExtractor(backends=[first, second])
    r = ex.extract_video(["f1.jpg"], [0.5])
    # 같은 백엔드로 한 번 더 부른 뒤(VLM_PARSE_RETRIES 기본 1) 폴백한다
    assert r.ok and r.parse_ok and first.calls == 2 and second.calls == 1 and r.backend_used == "second"


def test_thought_block_is_stripped_before_json():
    from api.module.clip_extraction.vlm.client import _extract_json
    txt = '<thought>draft {"description": "wrong"}</thought>```json\n{"description": "right"}\n```'
    assert _extract_json(txt) == {"description": "right"}


def test_lab_gemma_backend_has_short_connect_timeout_and_no_retry():
    from api.module.clip_extraction.vlm import extractor as X
    b = X.make_gemma_backend()
    assert b.client.max_retries == 0
    assert getattr(b.client.timeout, "connect", None) == 10.0


def test_image_chain_auto_order():
    from api.module.clip_extraction.vlm import extractor as X
    chain = X.default_backends(prefer="auto")
    assert [b.model for b in chain[:3]] == ["qwen3.8-omni-flash", "qwen3.7-plus", "gemma-4-31b-it"]
    assert X.VLMExtractor.__init__.__defaults__[1] == "auto"     # prefer 기본값
