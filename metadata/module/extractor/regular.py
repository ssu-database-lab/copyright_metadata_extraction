from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import dateparser
import spacy
import yaml
from urlextract import URLExtract
import re

from .types import Decision


LABELS_PATH = Path("configs/labels.yaml")
_DATETIME_HINTS: Dict[str, Sequence[str]] = {
    "created_date": ("작성", "게재"),
    "registration_date": ("등록",),
    "production_date": ("제작", "발행"),
    "valid_period": ("유효", "기간", "부터", "까지", "~"),
}
_NUMERIC_HINTS: Dict[str, Sequence[str]] = {
    "seq_number": ("순번", "번호", "no"),
    "quantity": ("수량", "총수량"),
    "video_count": ("영상", "비디오"),
    "photo_count": ("사진", "이미지"),
    "document_count": ("문서",),
}

_URL_EXTRACTOR = URLExtract()
_NLP = None

# 공통 패턴
_DATE_PATTERNS = [
    r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',
    r'\d{4}-\d{1,2}-\d{1,2}',
    r'\d{4}\.\d{1,2}\.\d{1,2}',
    r'\d{4}/\d{1,2}/\d{1,2}',
]


# ---------- 공통 유틸 ----------


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=1)
def _get_datetime_labels() -> Sequence[str]:
    return tuple(_load_config().get("datetime_labels") or [])


@lru_cache(maxsize=1)
def _get_numeric_labels() -> Sequence[str]:
    return tuple(_load_config().get("numeric_labels") or [])


@lru_cache(maxsize=1)
def _get_regex_labels() -> Dict[str, str]:
    return _load_config().get("regex_labels") or {}


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("ko_core_news_lg")
        except OSError:
            _NLP = False
    return _NLP


def _decision(label: str, value: str, sent_id: Optional[int], tok_id: Optional[int], 
              source: str, meta: Optional[Dict[str, Any]] = None) -> Decision:
    return Decision(label=label, value=value, sent_id=sent_id, tok_id=tok_id, 
                    source=source, meta=meta or {})


def _choose_label(text: str, hints: Dict[str, Sequence[str]], allowed: Sequence[str]) -> Optional[str]:
    lower = text.lower()
    for label, candidates in hints.items():
        if label in allowed and any(c.lower() in lower for c in candidates):
            return label
    return allowed[0] if len(allowed) == 1 else None


def _match_token_by_value(tokens: Sequence[Dict[str, Any]], value: str) -> Optional[int]:
    norm = value.replace(",", "")
    for token in tokens:
        if str(token.get("text", "")).replace(",", "") == norm:
            return token.get("tok_id")
    return None


def _group_tokens_by_sentence(tokens: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for token in tokens:
        if token and (sid := token.get("sent_id")) is not None:
            grouped.setdefault(int(sid), []).append(token)
    for tok_list in grouped.values():
        tok_list.sort(key=lambda x: x.get("tok_id") or 0)
    return grouped


# ---------- Regex ----------


def regex_extractor(*, tokens: List[Dict[str, Any]], sentences: List[Dict[str, Any]]) -> List[Decision]:
    regex_specs = _get_regex_labels()
    if not regex_specs:
        return []
    
    decisions: List[Decision] = []
    token_groups = _group_tokens_by_sentence(tokens)
    
    for sentence in sentences:
        if not sentence:
            continue
        sent_id = sentence.get("sent_id")
        text = str(sentence.get("text", ""))
        if not text or sent_id is None:
            continue
        
        sid = int(sent_id)
        tokens_in_sent = token_groups.get(sid, [])
        
        # regex 패턴 매칭
        for label, pattern_str in regex_specs.items():
            try:
                for match in re.compile(pattern_str).finditer(text):
                    if value := match.group(0):
                        tok_id = next((t.get("tok_id") for t in tokens_in_sent if value in str(t.get("text", ""))), None)
                        decisions.append(_decision(label, value, sid, tok_id, "regex"))
            except re.error:
                continue
        
        # URL 추출 (urlextract + 한국어 도메인 지원)
        if "url" not in regex_specs:
            for url in _extract_urls_enhanced(text):
                tok_id = next((t.get("tok_id") for t in tokens_in_sent if url in str(t.get("text", ""))), None)
                decisions.append(_decision("url", url, sid, tok_id, "regex"))
    
    return decisions


def _extract_urls_enhanced(text: str) -> List[str]:
    """URL 추출 (urlextract + 한국어 도메인 지원)"""
    urls = set()
    
    # urlextract로 기본 URL 추출
    try:
        urls.update(_URL_EXTRACTOR.find_urls(text))
    except Exception:
        pass
    
    # 한국어 도메인 지원 패턴
    patterns = [
        r'https?://[가-힣\w][가-힣\w.-]*\.[a-z가-힣]{2,}(?:/[^\s<>"\'{}|\\^`\[\]]*)?',
        r'www\.[가-힣\w][가-힣\w.-]*\.[a-z가-힣]{2,}(?:/[^\s<>"\'{}|\\^`\[\]]*)?',
        r'\b[가-힣\w][가-힣\w.-]*\.[a-z가-힣]{2,}(?:/[^\s<>"\'{}|\\^`\[\]]*)?',
    ]
    
    for pattern_str in patterns:
        try:
            for match in re.compile(pattern_str, re.I).finditer(text):
                url = match.group(0)
                if _is_valid_url(url):
                    urls.add(url)
        except re.error:
            continue
    
    # 중복 제거: 프로토콜 있는 버전 우선
    urls_by_domain = {}
    for url in urls:
        norm = re.sub(r'^https?://|^www\.', '', url, flags=re.I).split('/')[0]
        existing = urls_by_domain.get(norm)
        if not existing or (('://' in url or url.startswith('www.')) and '://' not in existing and not existing.startswith('www.')):
            urls_by_domain[norm] = url
    
    return sorted(urls_by_domain.values())


def _is_valid_url(url: str) -> bool:
    """URL 형식 검증"""
    if not url or len(url) < 4 or '.' not in url:
        return False
    
    if any(re.search(p, url) for p in [r'\.http', r'^[^가-힣\w]', r'[가-힣]\s+\.', r'\.\s+[가-힣]']):
        return False
    
    parts = url.split('.')
    return len(parts) >= 2 and len(parts[-1].split('/')[0]) >= 2


# ---------- Datetime ----------


def datetime_extractor(*, sentences: List[Dict[str, Any]], 
                           token_groups: Dict[int, List[Dict[str, Any]]]) -> List[Decision]:
    allowed = _get_datetime_labels()
    if not allowed:
        return []

    decisions: List[Decision] = []
    date_formats = ['%Y년 %m월 %d일', '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d']
    
    for sentence in sentences:
        if not sentence:
            continue
        sid = sentence.get("sent_id")
        text = str(sentence.get("text", ""))
        if not text or sid is None:
            continue
        
        sid = int(sid)
        label = _choose_label(text, _DATETIME_HINTS, allowed)
        if not label:
            continue
        
        # 날짜 패턴 매칭
        found_dates = []
        for pattern_str in _DATE_PATTERNS:
            for match in re.compile(pattern_str).finditer(text):
                if parsed := dateparser.parse(match.group(0), languages=["ko", "en"], date_formats=date_formats):
                    found_dates.append(parsed)
        
        # 패턴 실패 시 전체 문장 파싱
        if not found_dates:
            if parsed := dateparser.parse(text, languages=["ko", "en"]):
                found_dates.append(parsed)
        
        # 날짜 추가
        tokens_in_sent = token_groups.get(sid, [])
        for found_date in found_dates:
            date_value = found_date.strftime("%Y-%m-%d")
            tok_id = _match_token_by_value(tokens_in_sent, date_value.replace("-", ""))
            decisions.append(_decision(label, date_value, sid, tok_id, "datetime", {"raw": text}))
    
    return decisions


# ---------- Numeric ----------


def numeric_extractor(*, sentences: List[Dict[str, Any]], 
                         token_groups: Dict[int, List[Dict[str, Any]]]) -> List[Decision]:
    allowed = _get_numeric_labels()
    if not allowed:
        return []

    nlp = _get_nlp()
    decisions: List[Decision] = []
    pattern = re.compile(r"\d{1,3}(?:,\d{3})*|\d+")
    date_hints = {"년", "월", "일", "작성", "등록", "제작", "유효", "기간"}

    for sentence in sentences:
        if not sentence:
            continue
        sid = sentence.get("sent_id")
        text = str(sentence.get("text", ""))
        if not text or sid is None:
            continue
        
        sid = int(sid)
        label = _choose_label(text, _NUMERIC_HINTS, allowed)
        if not label:
            continue

        # 날짜 패턴 내 숫자 수집 (부분 문자열 포함)
        date_matches = set()
        for pattern_str in _DATE_PATTERNS:
            for match in re.compile(pattern_str).finditer(text):
                for num in re.findall(r'\d+', match.group(0)):
                    date_matches.update(num[i:j] for i in range(len(num)) for j in range(i + 1, len(num) + 1))

        # 숫자 추출
        if nlp:
            values = [token.text for token in nlp(text) if token.like_num]
        else:
            values = pattern.findall(text)

        tokens_in_sent = token_groups.get(sid, [])
        for value in values:
            norm = value.replace(",", "")
            
            # 날짜 숫자 제외
            if norm in date_matches:
                continue
            
            # 연도 제외 (1900-2100, 날짜 힌트 있을 때)
            if len(norm) == 4:
                try:
                    if 1900 <= int(norm) <= 2100 and any(h in text for h in date_hints):
                        continue
                except ValueError:
                    pass
            
            tok_id = _match_token_by_value(tokens_in_sent, norm)
            decisions.append(_decision(label, norm, sid, tok_id, "numeric", 
                                       {"raw": value, "nlp": bool(nlp)}))
    
    return decisions


# ---------- Entry Point ----------


def regular_extractor(*, sentences: List[Dict[str, Any]], tokens: List[Dict[str, Any]]) -> List[Decision]:
    token_groups = _group_tokens_by_sentence(tokens)

    regex_decisions = regex_extractor(tokens=tokens, sentences=sentences)
    used_labels = {d.label for d in regex_decisions}
    extracted_values = {d.value for d in regex_decisions if d.value}
    phone_sent_ids = {d.sent_id for d in regex_decisions if d.label == "phone" and d.sent_id is not None}

    datetime_decisions = [d for d in datetime_extractor(sentences=sentences, token_groups=token_groups)
                          if d.label not in used_labels]
    used_labels.update(d.label for d in datetime_decisions)

    numeric_decisions = [d for d in numeric_extractor(sentences=sentences, token_groups=token_groups)
                         if d.label not in used_labels and d.value not in extracted_values 
                         and d.sent_id not in phone_sent_ids]

    return regex_decisions + datetime_decisions + numeric_decisions
