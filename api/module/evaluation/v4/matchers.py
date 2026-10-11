"""v4 평가 — 키별 일치 판정과 TP/FP/FN 산정 (docs/v4/M1_evaluator_spec.md §4 Matching rules, §1 match rule 열, §2 gold 상태).

슬롯 채우기 관례(§4 Scalar keys):
  gold 상태 value : 일치 → TP 1 / 불일치(해석 불가 포함) → FP 1 + FN 1 / 예측 null → FN 1
  gold 상태 empty : 예측 non-null → FP 1 / 예측 null → 집계 제외(TN)
  gold 상태 na    : 항상 제외
리스트 키(키워드·해시태그)는 원소 단위 TP/FP/FN — 포함 일치, 1:1 탐욕 매칭 (§4 Lists).
이름 키(저작자·저작권자)는 집합 동일성으로 키 전체를 한 번만 판정한다 (§4 Names).
other_rights(type 'na')는 gold 상태와 무관하게 항상 제외하고, 예측이 non-null 이면 경고만 남긴다 (§1 #13).
gold 상태가 value 인데 정규화 후 비면 '확정 빈값(empty)'으로 다시 해석하고 detail["state"]="empty" 로 남긴다 —
평가기는 이 값으로 상태 표(①/②/③)를 세어 채점과 상태 표가 어긋나지 않게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union

from . import normalize as N
from .keys23 import BY_NAME, Key

GOLD_STATES: tuple[str, ...] = ("value", "empty", "na")


class Judge(Protocol):
    """설명 판정기 인터페이스 — embed_judge.EmbedJudge 또는 테스트 스텁."""

    def decide(self, pred_text: str, gold_text: str) -> dict[str, Any]: ...


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched: Optional[bool] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "matched": self.matched, "detail": dict(self.detail)}


def set_f1(tp: int, fp: int, fn: int) -> Optional[float]:
    """2TP/(2TP+FP+FN); 판정이 하나도 없으면 None."""
    denom = 2 * tp + fp + fn
    return None if denom == 0 else 2 * tp / denom


def gold_state(gold_entry: Optional[dict[str, Any]]) -> str:
    """gold 항목의 상태. 항목이 없으면 na; state 가 없으면 value 유무로 정한다."""
    if not gold_entry:
        return "na"
    st = gold_entry.get("state")
    if st in GOLD_STATES:
        return st
    return "value" if not N.is_blank(gold_entry.get("value")) else "na"


def scalar_counts(state: str, pred_present: bool, is_match: Optional[bool]) -> MatchResult:
    """단일값 키의 TP/FP/FN (§4 Scalar keys)."""
    if state == "na":
        return MatchResult(detail={"skipped": "gold_na"})
    if state == "empty":
        if pred_present:
            return MatchResult(fp=1, matched=False, detail={"gold": "empty"})
        return MatchResult(detail={"skipped": "gold_empty_pred_null"})
    if not pred_present:
        return MatchResult(fn=1, matched=False, detail={"pred": "null"})
    if is_match:
        return MatchResult(tp=1, matched=True)
    return MatchResult(fp=1, fn=1, matched=False)


def _mark_empty(res: MatchResult) -> MatchResult:
    """'value' gold 가 정규화 후 비어 'empty' 로 다시 해석됐음을 남긴다 — 평가기가 상태 표에 ② 로 센다."""
    res.detail["state"] = "empty"
    return res


# ---------------------------------------------------------------------------
# 리스트 키 — 키워드·해시태그
# ---------------------------------------------------------------------------
def tag_match(a: str, b: str) -> bool:
    """정규화된 태그 두 개: 같거나, 한쪽(2자 이상)이 다른 쪽에 포함되면 일치."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 2 and short in long_


def match_tag_lists(pred: list[str], gold: list[str]) -> tuple[int, int, int, list[tuple[str, str]]]:
    """1:1 탐욕 매칭 — gold 는 긴 것부터; 후보 중 완전 일치 > 더 긴 예측값 > 사전순. 결정적."""
    order = sorted(range(len(gold)), key=lambda i: (-len(gold[i]), gold[i]))
    used: set[int] = set()
    pairs: list[tuple[str, str]] = []
    for gi in order:
        g = gold[gi]
        cands = [pi for pi in range(len(pred)) if pi not in used and tag_match(pred[pi], g)]
        if not cands:
            continue
        best = min(cands, key=lambda pi: (pred[pi] != g, -len(pred[pi]), pred[pi]))
        used.add(best)
        pairs.append((g, pred[best]))
    tp = len(pairs)
    return tp, len(pred) - tp, len(gold) - tp, pairs


def match_list(pred: Any, gold_entry: Optional[dict[str, Any]]) -> MatchResult:
    """키워드·해시태그: 원소 단위 TP/FP/FN (§4 Lists)."""
    state = gold_state(gold_entry)
    if state == "na":
        return MatchResult(detail={"skipped": "gold_na"})
    pred_tags = N.split_tags(pred)
    gold_tags = N.split_tags((gold_entry or {}).get("value")) if state == "value" else []
    reinterpreted = state == "value" and not gold_tags  # value 상태인데 정규화 후 비면 확정 빈값으로 본다
    if state == "empty" or reinterpreted:
        if pred_tags:
            res = MatchResult(fp=len(pred_tags), matched=False, detail={"gold": "empty", "unmatched_pred": pred_tags})
        else:
            res = MatchResult(detail={"skipped": "gold_empty_pred_null"})
        return _mark_empty(res) if reinterpreted else res
    if not pred_tags:
        return MatchResult(fn=len(gold_tags), matched=False,
                           detail={"pred": "null", "unmatched_gold": gold_tags})
    tp, fp, fn, pairs = match_tag_lists(pred_tags, gold_tags)
    matched_pred = {p for _, p in pairs}
    matched_gold = {g for g, _ in pairs}
    detail = {
        "pairs": [list(p) for p in pairs],
        "unmatched_pred": [p for p in pred_tags if p not in matched_pred],
        "unmatched_gold": [g for g in gold_tags if g not in matched_gold],
        "set_f1": set_f1(tp, fp, fn),
    }
    return MatchResult(tp, fp, fn, matched=(fp == 0 and fn == 0), detail=detail)


# ---------------------------------------------------------------------------
# 이름 키 — 저작자·저작권자 (집합 동일성)
# ---------------------------------------------------------------------------
def names_equal(pred: Any, gold: Any) -> bool:
    return set(N.split_names(pred)) == set(N.split_names(gold))


def match_names(pred: Any, gold_entry: Optional[dict[str, Any]]) -> MatchResult:
    state = gold_state(gold_entry)
    if state == "na":
        return MatchResult(detail={"skipped": "gold_na"})
    pred_names = N.split_names(pred)
    gold_names: list[str] = []
    reinterpreted = False
    if state == "value":
        gold_names = N.split_names((gold_entry or {}).get("value"))
        if not gold_names:
            state, reinterpreted = "empty", True
    is_match = set(pred_names) == set(gold_names) if state == "value" else None
    res = scalar_counts(state, bool(pred_names), is_match)
    res.detail.update({"pred_norm": pred_names, "gold_norm": gold_names})
    return _mark_empty(res) if reinterpreted else res


# ---------------------------------------------------------------------------
# 단일값 키 — 정규화 함수 표
# ---------------------------------------------------------------------------
_SCALAR_NORMALIZERS = {
    "category": N.norm_work_type,
    "rights": N.norm_rights,
    "date": N.norm_date,
    "period": N.norm_period,
    "format": N.norm_format,
    "int": N.norm_int_bytes,
    "resolution": N.norm_resolution,
    "hash": N.norm_hash,
}


def _norm_pred(fn: Any, pred: Any) -> tuple[Any, Optional[str]]:
    """(정규화값, 해석불가 사유). 해석 불가는 '값은 있으나 틀림'으로 센다."""
    try:
        return fn(pred), None
    except ValueError as e:
        return None, str(e)


def match_scalar(key: Key, pred: Any, gold_entry: Optional[dict[str, Any]]) -> MatchResult:
    state = gold_state(gold_entry)
    if state == "na":
        return MatchResult(detail={"skipped": "gold_na"})
    fn = _SCALAR_NORMALIZERS[key.type]
    p, err = _norm_pred(fn, pred)
    pred_present = p is not None or err is not None
    detail: dict[str, Any] = {}
    if err is not None:
        detail["unparseable"] = err
    elif p is not None:
        detail["pred_norm"] = p
    if state == "empty":
        res = scalar_counts("empty", pred_present, None)
        res.detail.update(detail)
        return res
    try:
        g = fn((gold_entry or {}).get("value"))
    except ValueError as e:
        return MatchResult(detail={"skipped": "gold_unparseable", "gold_unparseable": str(e)})
    if g is None:  # value 상태인데 gold 가 비어 있음 → 확정 빈값으로 본다
        res = scalar_counts("empty", pred_present, None)
        res.detail.update(detail)
        return _mark_empty(res)
    detail["gold_norm"] = g
    res = scalar_counts("value", pred_present, p is not None and p == g)
    res.detail.update(detail)
    return res


def _title_key(s: Any) -> Optional[str]:
    """저작물명 비교키 — 앞의 관리번호 토큰을 뗀 뒤 casefold (§1 #1). 예측 쪽에만 적용한다(gold 는 원장 제목 그대로)."""
    return N.text_key(N.strip_id_prefix(N.scalar_input(s)))


def match_text(key: Key, pred: Any, gold_entry: Optional[dict[str, Any]]) -> MatchResult:
    """work_title(관리번호 접두 제거) · file_name: norm_text 후 대소문자 무시 완전 일치."""
    state = gold_state(gold_entry)
    if state == "na":
        return MatchResult(detail={"skipped": "gold_na"})
    p = N.text_key(N.scalar_input(pred))
    if state == "empty":
        return scalar_counts("empty", p is not None, None)
    g = N.text_key((gold_entry or {}).get("value"))
    if g is None:
        return _mark_empty(scalar_counts("empty", p is not None, None))
    is_match = p is not None and p == g
    if not is_match and key.name == "work_title" and p is not None:
        is_match = _title_key(p) == g  # 관리번호 접두는 예측 쪽만 뗀다 (§1 #1)
    res = scalar_counts("value", p is not None, is_match)
    res.detail.update({"pred_norm": p, "gold_norm": g})
    return res


# ---------------------------------------------------------------------------
# 설명 — 임베딩 판정기 (§5)
# ---------------------------------------------------------------------------
_JUDGE_FIELDS = ("cosine", "shared_tokens", "tau", "content_check", "content_ok")


def match_description(pred: Any, gold_entry: Optional[dict[str, Any]], judge: Optional[Judge]) -> MatchResult:
    state = gold_state(gold_entry)
    if state == "na":
        return MatchResult(detail={"skipped": "gold_na"})
    p = N.norm_text(N.scalar_input(pred))
    if state == "empty":
        return scalar_counts("empty", p is not None, None)
    g = N.norm_text((gold_entry or {}).get("value"))
    if g is None:
        return _mark_empty(scalar_counts("empty", p is not None, None))
    if p is None:
        return scalar_counts("value", False, None)
    if judge is None:  # 판정기 없이 호출되면 결정적 대체 규칙: 완전 일치
        is_match = p.casefold() == g.casefold()
        detail: dict[str, Any] = {"judge": "exact"}
    else:
        verdict = judge.decide(p, g)
        is_match = bool(verdict.get("match"))
        detail = {k: verdict[k] for k in _JUDGE_FIELDS if k in verdict}
    res = scalar_counts("value", True, is_match)
    res.detail.update(detail)
    return res


# ---------------------------------------------------------------------------
# 디스패치
# ---------------------------------------------------------------------------
def match_key(key: Union[Key, str], pred: Any, gold_entry: Optional[dict[str, Any]],
              judge: Optional[Judge] = None) -> dict[str, Any]:
    """한 세트 × 한 키의 판정 → {tp, fp, fn, matched, detail}."""
    k = BY_NAME[key] if isinstance(key, str) else key
    if k.type == "na":
        res = MatchResult(detail={"skipped": "na_key", "unscored_nonnull": not N.is_blank(pred)})
    elif k.type == "str":
        res = match_text(k, pred, gold_entry)
    elif k.type == "name_list":
        res = match_names(pred, gold_entry)
    elif k.type == "tag_list":
        res = match_list(pred, gold_entry)
    elif k.type == "description":
        res = match_description(pred, gold_entry, judge)
    else:
        res = match_scalar(k, pred, gold_entry)
    return res.as_dict()


__all__ = [
    "GOLD_STATES", "Judge", "MatchResult", "set_f1", "gold_state", "scalar_counts",
    "tag_match", "match_tag_lists", "match_list", "names_equal", "match_names",
    "match_scalar", "match_text", "match_description", "match_key",
]
