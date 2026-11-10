"""계약서 PDF → VLM 참고 문맥 {title, work_type} — 규칙만 쓰고 모델은 부르지 않는다 (M2 spec A4).

v4 T-002(옵션 B)는 저작물 파일과 함께 계약서를 받지만, 계약서에서 쓰는 값은 저작물명과 종별
둘뿐이다 (docs/test_procedure_review_20261004.md §12.7(a)). T-001 의 LLM 결과를 재사용하면
T-001 의 오류가 T-002 로 번지고 두 시험이 서로 묶이므로, PyMuPDF 본문층을 정규식으로 읽는다.
어떤 값을 넣었는지는 호출자가 응답에 그대로 남긴다(출처 기록).

규칙 (새 서식 — dataset/new_contracts 5건에서 검증, 01_new_contracts_T001_keys.md §(a)(b)):
  · 제목   제2조 `제목(제호) : <관리번호> <제목>`. 긴 제목은 다음 줄로 넘어가므로(`… 구축 \\n전략`)
           새 항목(`저작자 :`)이 시작되기 전까지 이어 붙이고, 앞의 관리번호 토큰 `^\\d{5,8}\\s+` 을
           뗀다. 4자리 연도로 시작하는 제목(`2021 아리랑 응원가 가사`)은 건드리지 않는다.
  · 종별   `종별 :` 줄의 체크된 칸. 체크 글리프는 v(U+0076)·V(U+0056)·■(U+25A0) 셋이 실측됐고
           빈 칸은 □(U+25A1)·◻(U+25FB). `□사진\\n저작물` 처럼 라벨이 줄 끝에서 갈라지므로
           `권리 :` 앞까지를 한 줄로 합친 뒤 본다. 값은 평가기의 닫힌 집합(어문/음악/…/기타)으로 낸다.

사용:
    from module.clip_extraction.contract_context import parse
    ctx = parse("dataset/new_contracts/00002_13387752 운수 자판기 (1).pdf")
    # {'title': '운수 자판기', 'work_type': '어문', 'source': 'rules', 'title_raw': '13387752 운수 자판기', ...}
    orch.run(work_bytes, name, contract_context=ctx, ...)

CLI 점검: python -m api.module.clip_extraction.contract_context <pdf> [<pdf> ...]
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 평가기 keys23.WORK_TYPES 와 같은 닫힌 집합 — 여기서 import 하지 않는 것은 이 모듈을
# 평가 패키지 없이도(배치 러너·CLI) 가볍게 쓰기 위해서다. 바뀌면 두 곳을 함께 고친다.
WORK_TYPES: Tuple[str, ...] = ("어문", "음악", "연극", "미술", "건축", "사진", "영상", "도형", "컴퓨터프로그램", "기타")

TICKED_GLYPHS = "vVⅴ■◼☑☒✓✔√●"      # 체크된 칸 (실측: v / V / ■)
UNTICKED_GLYPHS = "□◻▫○◯☐"           # 빈 칸 (실측: □ / ◻ / ▫)

_ID_PREFIX = re.compile(r"^\d{5,8}\s+")
_TITLE_LABEL = re.compile(r"제\s*목\s*\(\s*제\s*호\s*\)\s*[:：]\s*(?P<rest>.*)")
# v5 계약서 B·C (2026-11-03): B `1. 제호 : <제목>` · C 표 `제호(제목)` 칸(값은 다음 줄)
_TITLE_LABEL_B = re.compile(r"^\s*(?:\d+\s*\.\s*)?제\s*호\s*[:：]\s*(?P<rest>.*)")
_TITLE_LABEL_C = re.compile(r"^\s*제\s*호\s*\(\s*제\s*목\s*\)\s*$")
_NUMBERED_ITEM = re.compile(r"^\s*\d+\s*\.\s+\S+\s*[:：]")   # B 의 다음 항목 `2. 저작자 :`
# C 표에서 제목 칸 다음에 오는 라벨 칸 — 제목 이어붙이기를 여기서 멈춘다
_TABLE_LABEL = re.compile(r"^\s*(?:저\s*작\s*자|저\s*작\s*재\s*산\s*권\s*자|저\s*작\s*물\s*의\s*종\s*류|"
                          r"이\s*용\s*허\s*락\s*(?:권\s*리|기\s*간)|이\s*용\s*료)\s*$")
# '라벨 : 값' · '제N조 (…)' · '(1)' 처럼 새 항목을 여는 줄 — 제목 이어붙이기를 여기서 멈춘다
_NEW_FIELD = re.compile(r"^\s*(?:제\s*\d+\s*조\b|\(\s*\d+\s*\)|[^\s:：]{1,12}\s*[:：])")
_TYPE_LABEL = re.compile(r"종\s*별\s*[:：]|저\s*작\s*물\s*의\s*종\s*류\s*[:：]?")   # A `종별 :` · B `저작물의 종류 :` · C 표 칸
_RIGHTS_LABEL = re.compile(r"권\s*리\s*[:：]|제\s*\d+\s*조|이\s*용\s*허\s*락\s*권\s*리")  # 종별 목록의 끝
_TYPE_ITEM = re.compile(
    r"(?P<glyph>[" + re.escape(TICKED_GLYPHS) + re.escape(UNTICKED_GLYPHS) + r"])\s*"
    r"(?P<label>" + "|".join(sorted(WORK_TYPES, key=len, reverse=True)) + r")"
)
_MAX_TITLE_CONTINUATION = 3    # 제목이 넘어갈 수 있는 최대 줄 수 (서식상 1줄이 넘는 경우는 없었다)
_TYPE_WINDOW = 400             # '종별 :' 뒤에서 읽는 최대 글자 수


def _title_start(line: str) -> Optional[str]:
    """제목 라벨 줄이면 라벨 뒤 글자(C 표 칸이면 ''), 아니면 None."""
    for pat in (_TITLE_LABEL, _TITLE_LABEL_B):
        m = pat.search(line) if pat is _TITLE_LABEL else pat.match(line)
        if m:
            return m.group("rest")
    return "" if _TITLE_LABEL_C.match(line) else None


def _extract_title(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """(관리번호를 뗀 제목, 인쇄된 원문) — 제목 라벨 줄(A `제목(제호) :` · B `1. 제호 :` · C `제호(제목)`)과 이어지는 줄에서."""
    for i, line in enumerate(lines):
        rest = _title_start(line)
        if rest is None:
            continue
        parts = [rest]
        for nxt in lines[i + 1: i + 1 + _MAX_TITLE_CONTINUATION]:
            if not nxt.strip() or _NEW_FIELD.match(nxt) or _TABLE_LABEL.match(nxt) or _NUMBERED_ITEM.match(nxt):
                break
            parts.append(nxt)
        # 줄 경계에 공백이 인쇄돼 있으면(`구축 ⏎전략`) 그대로 살고, 낱말 안에서 넘어갔으면 붙는다.
        raw = " ".join("".join(parts).split())
        if not raw:
            continue
        bare = _ID_PREFIX.sub("", raw).strip() or raw
        return bare, raw
    return None, None


def _extract_work_type(text: str) -> Tuple[Optional[str], List[str]]:
    """(체크된 종별, 체크된 라벨 전체) — `종별 :` 이후 `권리 :` 앞까지를 한 줄로 합쳐 읽는다."""
    m = _TYPE_LABEL.search(text)
    if not m:
        return None, []
    segment = text[m.end(): m.end() + _TYPE_WINDOW]
    r = _RIGHTS_LABEL.search(segment)
    if r:
        segment = segment[: r.start()]
    segment = segment.replace("\n", "")      # 줄 끝에서 갈라진 라벨(□사진 / 저작물)을 붙인다
    ticked = [it.group("label") for it in _TYPE_ITEM.finditer(segment)
              if it.group("glyph") in TICKED_GLYPHS]
    if not ticked:
        return None, []
    if len(ticked) > 1:
        logger.warning("종별 칸이 둘 이상 체크됨 %s — 첫 값을 쓴다", ticked)
    return ticked[0], ticked


# 권리 7종 — 체크 칸 바로 뒤의 권리 이름 (공백을 모두 지운 본문에서). 한 권리가 여러 번 나와도 표시가 모두 같을 때만 값.
# v5 시험 계약서 4,200건에서 7종 모두 원장과 일치 (2026-11-10). 못 읽으면 None → 직렬화기가 모델 값을 쓴다.
RIGHT_LABELS: Tuple[Tuple[str, str], ...] = (
    ("reproduction_right", "복제권"), ("performance_right", "공연권"), ("public_transmission_right", "공중송신권"),
    ("exhibition_right", "전시권"), ("distribution_right", "배포권"), ("rental_right", "대여권"),
    ("derivative_work_right", "2차적저작물"))


def _extract_rights(text: str) -> Dict[str, Optional[bool]]:
    flat = re.sub(r"\s+", "", text or "")
    glyphs = re.escape(TICKED_GLYPHS + UNTICKED_GLYPHS)
    out: Dict[str, Optional[bool]] = {}
    for key, label in RIGHT_LABELS:
        states = {g in TICKED_GLYPHS for g in re.findall(rf"([{glyphs}]){re.escape(label)}", flat)}
        out[key] = True if states == {True} else False if states == {False} else None
    return out


def parse_text(text: str) -> Dict[str, Any]:
    """계약서 본문(문자열)에서 {title, work_type, source='rules', title_raw, work_type_ticked} 를 읽는다."""
    text = text or ""
    title, title_raw = _extract_title(text.splitlines())
    work_type, ticked = _extract_work_type(text)
    return {
        "title": title,
        "work_type": work_type,
        "source": "rules",
        "title_raw": title_raw,
        "work_type_ticked": ticked,
        "rights": _extract_rights(text),
    }


def pdf_text(pdf_path) -> str:
    """PyMuPDF 본문층 — 쪽 텍스트를 개행으로 이어 붙인다 (OCR·모델 호출 없음)."""
    import fitz  # PyMuPDF — 지연 import (이 모듈을 import 만 해도 fitz 를 올리지 않게)

    with fitz.open(str(pdf_path)) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def parse(pdf_path) -> Dict[str, Any]:
    """계약서 PDF → {"title", "work_type", "source": "rules", ...}. 실패해도 예외 대신 error 를 적는다.

    title / work_type 은 못 읽으면 None. 호출자는 그대로 run(contract_context=...) 에 넘기면 된다
    (둘 다 None 이면 프롬프트는 바뀌지 않는다).
    """
    path = Path(pdf_path)
    out: Dict[str, Any] = {"title": None, "work_type": None, "source": "rules", "pdf": path.name}
    try:
        text = pdf_text(path)
    except ImportError as e:
        out["error"] = f"PyMuPDF 없음: {e}"
        logger.warning("계약서 문맥 파싱 불가 — %s", out["error"])
        return out
    except Exception as e:  # noqa: BLE001 — 깨진 PDF 한 건이 배치를 죽이면 안 된다
        out["error"] = f"{type(e).__name__}: {e}"
        logger.warning("계약서 문맥 파싱 실패 %s — %s", path.name, out["error"])
        return out
    out.update(parse_text(text))
    if out["title"] is None or out["work_type"] is None:
        logger.info("계약서 문맥 일부 미검출 %s — title=%r work_type=%r",
                    path.name, out["title"], out["work_type"])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="계약서 PDF 에서 저작물명·종별을 규칙으로 읽는다 (모델 호출 없음)")
    ap.add_argument("pdfs", nargs="+", help="계약서 PDF 경로")
    args = ap.parse_args(argv)
    for p in args.pdfs:
        print(json.dumps(parse(p), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
