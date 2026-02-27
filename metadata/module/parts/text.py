"""텍스트/토큰 정규화 및 스팬 유틸리티."""
import re
import unicodedata
from typing import List, Tuple


# -----------------------------------------------------------------------------
# 변수 선언 (없음)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# class 선언 (없음)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# function 선언 (private 없음) / export
# -----------------------------------------------------------------------------

def normalize_ocr_text(text: str) -> str:
    """OCR 결과 텍스트 정규화: 연속 공백·NBSP를 단일 공백으로."""
    return re.sub(r"\s+", " ", text.replace("\u00A0", " ")).strip()


def clean_token_text(text: str) -> str:
    """토큰 문자열 정규화: NFKC, 대시 통일, 연속 공백 제거."""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans({"－": "-", "—": "-", "–": "-", "‐": "-"}))
    return re.sub(r"\s+", " ", t).strip()


def join_tokens_with_spans(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """토큰 리스트를 공백으로 이어 붙이고, 각 토큰의 (start, end) 스팬 리스트 반환."""
    spans: List[Tuple[int, int]] = []
    parts: List[str] = []
    cur = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            cur += 1
        start = cur
        parts.append(tok)
        cur += len(tok)
        end = cur
        spans.append((start, end))
    return "".join(parts), spans


def span_to_token_indices(
    ent_start: int, ent_end: int, token_spans: List[Tuple[int, int]]
) -> List[int]:
    """엔티티 스팬 (ent_start, ent_end)과 겹치는 토큰 인덱스 목록 반환."""
    idxs: List[int] = []
    for i, (ts, te) in enumerate(token_spans):
        if te <= ent_start:
            continue
        if ts >= ent_end:
            break
        if ts < ent_end and te > ent_start:
            idxs.append(i)
    return idxs
