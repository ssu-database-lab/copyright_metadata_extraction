"""텍스트 처리 유틸리티 (kiwipiepy + BlingFire, 토크나이저 포함)"""
import re
import unicodedata
from typing import Dict, List

from blingfire import text_to_sentences
from kiwipiepy import Kiwi


# -----------------------------------------------------------------------------
# 변수 선언
# -----------------------------------------------------------------------------

_KIWI = None
_KOREAN_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]")


# -----------------------------------------------------------------------------
# class 선언 (없음)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# function 선언 (private 먼저)
# -----------------------------------------------------------------------------

def _get_kiwi():
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
            if isinstance(first_item, str):
                out.append(first_item)
            else:
                out.append(str(first_item))
    return [s.strip() for s in out if s.strip()]


# -----------------------------------------------------------------------------
# export
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"[ \u00A0]+", " ", text.strip())


def is_valid_text(text: str, min_length: int = 1) -> bool:
    return bool(isinstance(text, str) and len(text.strip()) >= min_length)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def remove_uppercase(text: str) -> str:
    text = text.replace("\r", "\n").replace("\t", " ")
    return re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", text)


def fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w)[\-\u2010\u2011\u2012\u2013\u2014]\s*\n\s*(\w)", r"\1\2", text)


def fix_url_spacing(text: str) -> str:
    text = re.sub(r"(\w)\s*\.\s*(\w)", r"\1.\2", text)
    text = re.sub(r"\b(w\s*w\s*w)\s*\.\s*", "www.", text, flags=re.I)
    text = re.sub(r"h\s*t\s*t\s*p\s*s?\s*:\s*//", "http://", text, flags=re.I)
    text = text.replace("http://s://", "https://")
    text = re.sub(r"(\bhttps?://)\s+", r"\1", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \u00A0]{2,}", " ", text)
    return clean_text(text)


def split_sentences(clean_text_all: str) -> List[str]:
    """문장 분리 (한국어 우선, 영어는 BlingFire)"""
    if not clean_text_all:
        return []

    if _KOREAN_RE.search(clean_text_all):
        return _kiwi_split_sentences(clean_text_all)
    return [s.strip() for s in text_to_sentences(clean_text_all).split("\n") if s.strip()]


def tokenize(s: str) -> List[str]:
    """한국어 문장을 토큰화"""
    if not s:
        return []

    kiwi = _get_kiwi()
    if kiwi is None:
        return s.split()

    morphs = kiwi.analyze(s)
    tokens: List[str] = []
    for word in morphs:
        if isinstance(word, (list, tuple)) and len(word) > 0:
            token = word[0]
            if isinstance(token, str):
                tokens.append(token)
            elif hasattr(token, "form"):
                tokens.append(str(getattr(token, "form")))
            elif isinstance(token, (list, tuple)) and len(token) > 0:
                for item in token:
                    if isinstance(item, str):
                        tokens.append(item)
                    elif hasattr(item, "form"):
                        tokens.append(str(getattr(item, "form")))
                    else:
                        tokens.append(str(item))
            else:
                tokens.append(str(token))
        elif isinstance(word, str):
            tokens.append(word)
        elif hasattr(word, "form"):
            tokens.append(str(getattr(word, "form")))
        else:
            tokens.append(str(word))
    return [str(t).strip() for t in tokens if str(t).strip()]


def tokenize_sentence(clean_text_all: str) -> Dict[str, List[dict]]:
    """
    전체 문자열을 받아 문장/토큰을 생성해 통일된 스키마로 반환.
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
    """
    입력: 원문 문자열
    출력: {"sentences": [...], "tokens": [...]}
    """
    if not is_valid_text(raw_text):
        return {"sentences": [], "tokens": []}

    s = raw_text
    s = normalize_unicode(s)
    s = remove_uppercase(s)
    s = fix_hyphenation(s)
    s = fix_url_spacing(s)
    s = normalize_whitespace(s)

    return tokenize_sentence(s)
