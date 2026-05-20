"""텍스트/토큰 정규화, 문장 분리, 스팬 유틸리티.

OCR·문서 텍스트의 전처리부터 토큰/문장 구조화, 엔티티 스팬 변환까지 텍스트 관련
유틸리티를 모아 둔다. 한국어는 kiwipiepy, 그 외는 BlingFire로 문장을 분리한다.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from blingfire import text_to_sentences
from kiwipiepy import Kiwi


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_KIWI: Kiwi | None = None
_KOREAN_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]")
_TOKENIZE_PUNCT_RE = re.compile(r"([:()·,\[\]])")


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------

def _get_kiwi() -> Kiwi | None:
    """Kiwi 지연 초기화 (실패 시 None 반환)."""
    global _KIWI
    if _KIWI is None:
        try:
            _KIWI = Kiwi()
        except Exception:
            pass
    return _KIWI


def _kiwi_split_sentences(s: str) -> List[str]:
    kiwi = _get_kiwi()
    if kiwi is None:
        return [s.strip()]
    out: List[str] = []
    for item in kiwi.split_into_sents(s):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            first_item = item[0]
            out.append(first_item if isinstance(first_item, str) else str(first_item))
    return [s.strip() for s in out if s.strip()]


# ---------------------------------------------------------------------------
# 정규화 / 검증
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """선행/후행 공백 제거 + 연속된 공백(+NBSP)을 단일 공백으로."""
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"[ \u00A0]+", " ", text.strip())


def is_valid_text(text: str, min_length: int = 1) -> bool:
    """최소 길이 이상의 문자열인가."""
    return bool(isinstance(text, str) and len(text.strip()) >= min_length)


def normalize_unicode(text: str) -> str:
    """NFKC 정규화."""
    return unicodedata.normalize("NFKC", text or "")


def normalize_ocr_text(text: str) -> str:
    """OCR 결과 텍스트 정규화: 연속 공백·NBSP를 단일 공백으로."""
    return re.sub(r"\s+", " ", text.replace("\u00A0", " ")).strip()


def clean_token_text(text: str) -> str:
    """토큰 문자열 정규화: NFKC, 대시 통일, 연속 공백 제거."""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans({"－": "-", "—": "-", "–": "-", "‐": "-"}))
    return re.sub(r"\s+", " ", t).strip()


def remove_uppercase(text: str) -> str:
    """제어 문자 제거 + \\r·\\t를 공백/개행으로 정규화."""
    text = text.replace("\r", "\n").replace("\t", " ")
    return re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", text)


def fix_hyphenation(text: str) -> str:
    """줄바꿈 끼인 하이픈 연결 복원."""
    return re.sub(r"(\w)[\-\u2010\u2011\u2012\u2013\u2014]\s*\n\s*(\w)", r"\1\2", text)


def fix_url_spacing(text: str) -> str:
    """OCR로 쪼개진 URL 재결합."""
    text = re.sub(r"\b(w\s*w\s*w)\s*\.\s*", "www.", text, flags=re.I)
    text = re.sub(r"h\s*t\s*t\s*p\s*s?\s*:\s*//", "http://", text, flags=re.I)
    text = text.replace("http://s://", "https://")
    text = re.sub(r"(\bhttps?://)\s+", r"\1", text)
    text = re.sub(r"(https?://\S+)\s+(\S)", r"\1\2", text)
    return text


def normalize_whitespace(text: str) -> str:
    """3개 이상 빈 줄 → 2개로, 2개 이상 공백 → 1개로 축소."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \u00A0]{2,}", " ", text)
    return clean_text(text)


# ---------------------------------------------------------------------------
# 문장 / 토큰 / 스팬
# ---------------------------------------------------------------------------

def split_sentences(clean_text_all: str) -> List[str]:
    """문장 분리 (한국어는 kiwipiepy, 그 외는 BlingFire)."""
    if not clean_text_all:
        return []
    if _KOREAN_RE.search(clean_text_all):
        return _kiwi_split_sentences(clean_text_all)
    return [s.strip() for s in text_to_sentences(clean_text_all).split("\n") if s.strip()]


def tokenize(s: str) -> List[str]:
    """단어 수준 토큰화 — 학습 데이터(.jsonl)와 동일한 분할 방식.

    학습 데이터는 공백 기반 단어 토큰(예: '저작인접권의', '서울특별시')으로 구성되어
    있으므로 예측 시에도 동일하게 단어 단위로 분할한다. 구두점(: , ( ) · [ ])은 인접
    텍스트에서 분리해 별도 토큰으로 만든다.
    """
    if not s:
        return []
    s = _TOKENIZE_PUNCT_RE.sub(r" \1 ", s)
    return [t for t in s.split() if t]


def tokenize_sentence(clean_text_all: str) -> Dict[str, List[dict]]:
    """전체 문자열을 받아 문장·토큰 구조로 반환.

    sentences: [{"text": str, "label": "none", "sent_id": int}]
    tokens   : [{"text": str, "label": "none", "sent_id": int, "tok_id": int}]
    """
    sents = split_sentences(clean_text_all)
    sentences = [{"text": s, "label": "none", "sent_id": i}
                 for i, s in enumerate(sents)]

    tokens: List[dict] = []
    for sid, s in enumerate(sents):
        toks = tokenize(s)
        for tid, t in enumerate(toks):
            tokens.append({"text": t, "label": "none", "sent_id": sid, "tok_id": tid})

    return {"sentences": sentences, "tokens": tokens}


def read_text(raw_text: str) -> Dict[str, List[dict]]:
    """원문 문자열 → {"sentences": [...], "tokens": [...]} (정규화 + 문장/토큰화)."""
    if not is_valid_text(raw_text):
        return {"sentences": [], "tokens": []}

    s = raw_text
    s = normalize_unicode(s)
    s = remove_uppercase(s)
    s = fix_hyphenation(s)
    s = fix_url_spacing(s)
    s = normalize_whitespace(s)

    return tokenize_sentence(s)


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
