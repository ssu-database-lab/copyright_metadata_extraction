from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

try:
    import spacy
except ImportError:
    spacy = None
import yaml
try:
    from urlextract import URLExtract
except ImportError:
    URLExtract = None
import re

from module.parts.types import Decision


LABELS_PATH = Path("configs/labels.yaml")
_NUMERIC_HINTS: Dict[str, Sequence[str]] = {
    "seq_number": ("순번", "번호", "no"),
    "quantity": ("수량", "총수량"),
    "video_count": ("영상", "비디오"),
    "photo_count": ("사진", "이미지"),
    "document_count": ("문서",),
}

_URL_EXTRACTOR = URLExtract() if URLExtract else None
_NLP = None


# ---------- 공통 유틸 ----------


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=1)
def _get_labels(label_type: str) -> Sequence[str] | Dict[str, str]:
    """
    라벨 타입에 따라 labels.yaml에서 라벨 목록을 가져옵니다.
    
    Args:
        label_type: "datetime_labels", "numeric_labels", "regex_labels" 중 하나
    
    Returns:
        datetime_labels, numeric_labels: Sequence[str]
        regex_labels: Dict[str, str]
    """
    config = _load_config()
    if label_type == "regex_labels":
        return config.get("regex_labels") or {}
    else:
        return tuple(config.get(label_type) or [])


def _get_labels_as_sequence(label_type: str) -> Sequence[str]:
    """라벨을 Sequence[str]로 반환 (datetime_labels, numeric_labels용)"""
    result = _get_labels(label_type)
    if isinstance(result, dict):
        return tuple()
    return result


def _get_labels_as_dict(label_type: str) -> Dict[str, str]:
    """라벨을 Dict[str, str]로 반환 (regex_labels용)"""
    result = _get_labels(label_type)
    if isinstance(result, dict):
        return result
    return {}


def _get_nlp():
    global _NLP
    if spacy is None:
        return False
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


# ---------- Regex Extractors (라벨별 분리) ----------


def _regex_pattern_extractor(*, label: str, pattern: str, tokens: List[Dict[str, Any]], 
                             sentences: List[Dict[str, Any]], 
                             token_groups: Dict[int, List[Dict[str, Any]]],
                             flags: int = 0) -> List[Decision]:
    """
    정규식 패턴 기반 추출기 (phone, email 등 공통 로직)
    
    Args:
        label: 라벨 이름 (예: "phone", "email")
        pattern: 정규식 패턴
        tokens: 토큰 리스트
        sentences: 문장 리스트
        token_groups: 문장별 토큰 그룹
        flags: re.compile 플래그
    """
    decisions: List[Decision] = []
    used_values = set()
    
    # 문장 단위로 추출 (토큰 분리로 인한 누락 방지)
    for sentence in sentences:
        if not sentence:
            continue
        sid = sentence.get("sent_id")
        text = str(sentence.get("text", ""))
        if not text or sid is None:
            continue
        
        sid = int(sid)
        try:
            for match in re.compile(pattern, flags).finditer(text):
                if value := match.group(0):
                    if value not in used_values:
                        # 문장의 첫 번째 토큰 ID 사용
                        tokens_in_sent = token_groups.get(sid, [])
                        tok_id = tokens_in_sent[0].get("tok_id") if tokens_in_sent else None
                        decisions.append(_decision(label, value, sid, tok_id, "regex"))
                        used_values.add(value)
        except re.error:
            continue
    
    return decisions


def phone_extractor(*, tokens: List[Dict[str, Any]], sentences: List[Dict[str, Any]], 
                    token_groups: Dict[int, List[Dict[str, Any]]]) -> List[Decision]:
    """전화번호 추출기"""
    regex_specs = _get_labels_as_dict("regex_labels")
    if "phone" not in regex_specs:
        return []
    
    return _regex_pattern_extractor(
        label="phone",
        pattern=regex_specs["phone"],
        tokens=tokens,
        sentences=sentences,
        token_groups=token_groups
    )


def email_extractor(*, tokens: List[Dict[str, Any]], sentences: List[Dict[str, Any]], 
                    token_groups: Dict[int, List[Dict[str, Any]]]) -> List[Decision]:
    """이메일 주소 추출기"""
    regex_specs = _get_labels_as_dict("regex_labels")
    if "email" not in regex_specs:
        return []
    
    return _regex_pattern_extractor(
        label="email",
        pattern=regex_specs["email"],
        tokens=tokens,
        sentences=sentences,
        token_groups=token_groups,
        flags=re.IGNORECASE
    )


def date_extractor(*, tokens: List[Dict[str, Any]], sentences: List[Dict[str, Any]], 
                    token_groups: Dict[int, List[Dict[str, Any]]]) -> List[Decision]:
    """날짜 추출기"""
    regex_specs = _get_labels_as_dict("regex_labels")
    if "date" not in regex_specs:
        return []
    
    decisions: List[Decision] = []
    date_pattern = regex_specs["date"]
    used_values = set()
    
    # 문장 단위로 추출 (토큰 분리로 인한 누락 방지)
    for sentence in sentences:
        if not sentence:
            continue
        sid = sentence.get("sent_id")
        text = str(sentence.get("text", ""))
        if not text or sid is None:
            continue
        
        sid = int(sid)
        try:
            for match in re.compile(date_pattern).finditer(text):
                if value := match.group(0):
                    if value not in used_values:
                        # 문장의 첫 번째 토큰 ID 사용
                        tokens_in_sent = token_groups.get(sid, [])
                        tok_id = tokens_in_sent[0].get("tok_id") if tokens_in_sent else None
                        decisions.append(_decision("date", value, sid, tok_id, "regex"))
                        used_values.add(value)
        except re.error:
            continue
    
    return decisions


def url_extractor(*, tokens: List[Dict[str, Any]], sentences: Optional[List[Dict[str, Any]]] = None) -> List[Decision]:
    """URL 추출기"""
    regex_specs = _get_labels_as_dict("regex_labels")
    # regex_specs에 "url"이 없으면 기본적으로 URL 추출 수행
    if regex_specs and "url" in regex_specs:
        return []  # regex_labels에 url이 정의되어 있으면 여기서는 처리하지 않음
    
    decisions: List[Decision] = []
    used_values = set()
    
    # tokens를 기반으로 처리
    for token in tokens:
        if not token:
            continue
        tok_id = token.get("tok_id")
        sent_id = token.get("sent_id")
        text = str(token.get("text", ""))
        
        if not text or sent_id is None:
            continue
        
        sid = int(sent_id)
        for url in _extract_urls_enhanced(text):
            if url in text and url not in used_values:
                decisions.append(_decision("url", url, sid, tok_id, "regex"))
                used_values.add(url)
    
    return decisions


def regex_extractor(*, tokens: List[Dict[str, Any]], sentences: Optional[List[Dict[str, Any]]] = None) -> List[Decision]:
    """
    Regex 추출기 통합 함수: 각 라벨별 extractor를 호출하여 통합
    
    Args:
        tokens: 토큰 리스트 (word 단위)
        sentences: 문장 리스트
    """
    if sentences is None:
        sentences = []
    
    token_groups = _group_tokens_by_sentence(tokens)
    regex_specs = _get_labels_as_dict("regex_labels")
    
    all_decisions: List[Decision] = []
    used_labels = set()
    
    # 각 라벨별 extractor 호출
    # phone_extractor
    if "phone" in regex_specs:
        phone_decisions = phone_extractor(tokens=tokens, sentences=sentences, token_groups=token_groups)
        all_decisions.extend(phone_decisions)
        used_labels.add("phone")
    
    # email_extractor
    if "email" in regex_specs:
        email_decisions = email_extractor(tokens=tokens, sentences=sentences, token_groups=token_groups)
        all_decisions.extend(email_decisions)
        used_labels.add("email")
    
    # date_extractor
    if "date" in regex_specs:
        date_decisions = date_extractor(tokens=tokens, sentences=sentences, token_groups=token_groups)
        all_decisions.extend(date_decisions)
        used_labels.add("date")
    
    # url_extractor
    url_decisions = url_extractor(tokens=tokens, sentences=sentences)
    all_decisions.extend(url_decisions)
    used_labels.add("url")
    
    # 기타 regex_labels에 정의된 패턴들 (phone, email, date, url 제외)
    for label, pattern_str in regex_specs.items():
        if label in used_labels:
            continue
        
        # 일반적인 패턴 매칭 (word 단위)
        used_values = set()
        for token in tokens:
            if not token:
                continue
            tok_id = token.get("tok_id")
            sent_id = token.get("sent_id")
            text = str(token.get("text", ""))
            
            if not text or sent_id is None:
                continue
            
            sid = int(sent_id)
            try:
                for match in re.compile(pattern_str).finditer(text):
                    if value := match.group(0):
                        if value not in used_values:
                            all_decisions.append(_decision(label, value, sid, tok_id, "regex"))
                            used_values.add(value)
            except re.error:
                continue
    
    return all_decisions


def _extract_urls_enhanced(text: str) -> List[str]:
    """URL 추출 (urlextract + 한국어 도메인 지원)"""
    urls = set()
    
    # urlextract로 기본 URL 추출
    if _URL_EXTRACTOR:
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




# ---------- Numeric ----------


def numeric_extractor(*, sentences: List[Dict[str, Any]], 
                         token_groups: Dict[int, List[Dict[str, Any]]]) -> List[Decision]:
    """
    숫자 추출기
    
    Args:
        sentences: 문장 리스트
        token_groups: 문장별 토큰 그룹
    """
    allowed = _get_labels_as_sequence("numeric_labels")
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

        # seq_number는 더 구체적인 패턴만 추출 (예: "1.", "순번: 1", "번호 1" 등)
        if label == "seq_number":
            seq_patterns = [
                r'순번\s*[:：]?\s*(\d+)',
                r'번호\s*[:：]?\s*(\d+)',
                r'\bno\.?\s*[:：]?\s*(\d+)',
                r'^(\d+)\s*[\.\)]',  # 줄 시작의 "1.", "1)" 형식
            ]
            for pattern_str in seq_patterns:
                for match in re.compile(pattern_str, re.IGNORECASE | re.MULTILINE).finditer(text):
                    if value := match.group(1):
                        tokens_in_sent = token_groups.get(sid, [])
                        tok_id = _match_token_by_value(tokens_in_sent, value)
                        decisions.append(_decision(label, value, sid, tok_id, "numeric", 
                                                   {"raw": text, "pattern": pattern_str}))
            continue  # seq_number는 여기서 처리 완료

        # 날짜 패턴 내 숫자 수집 (부분 문자열 포함)
        date_matches = set()
        regex_specs = _get_labels_as_dict("regex_labels")
        date_pattern = regex_specs.get("date", "")
        if date_pattern:
            try:
                for match in re.compile(date_pattern).finditer(text):
                    for num in re.findall(r'\d+', match.group(0)):
                        date_matches.update(num[i:j] for i in range(len(num)) for j in range(i + 1, len(num) + 1))
            except re.error:
                pass

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
    """
    Regular 추출기 통합 함수
    
    Args:
        sentences: 문장 리스트 (datetime, numeric 추출에 사용)
        tokens: 토큰 리스트 (regex 추출에 word만 사용)
    
    Returns:
        Decision 리스트
    """
    token_groups = _group_tokens_by_sentence(tokens)

    # Regex: word(tokens)만 사용, 단 전화번호는 문장 단위로도 처리
    regex_decisions = regex_extractor(tokens=tokens, sentences=sentences)
    used_labels = {d.label for d in regex_decisions}
    extracted_values = {d.value for d in regex_decisions if d.value}
    phone_sent_ids = {d.sent_id for d in regex_decisions if d.label == "phone" and d.sent_id is not None}

    # Numeric: sentences 사용 (기존 로직 유지)
    numeric_decisions = [d for d in numeric_extractor(sentences=sentences, token_groups=token_groups)
                         if d.label not in used_labels and d.value not in extracted_values 
                         and d.sent_id not in phone_sent_ids]

    return regex_decisions + numeric_decisions
