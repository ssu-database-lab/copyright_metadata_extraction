"""v4 평가기 — 예측 적재·채점·집계·보고 (docs/v4/M1_evaluator_spec.md §6 예측 입력, §7 지표·출력).

흐름: load_gold → load_predictions → score_set(세트×키, matchers.match_key) → aggregate
      → test_metrics.json / report.md / per_set.jsonl.
지표: 키별 TP·FP·FN 합산 → P/R/F1; Macro-F1 = 평가 건수>0 인 키 F1 의 산술평균(최종 지표),
      Micro-F1 = TP·FP·FN 총합(참고); 95% CI = 세트 단위 부트스트랩 1,000회(seed 0).
출력은 결정적이다(키 정렬, 고정 seed, 타임스탬프 없음).
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .keys23 import BY_NAME, KEY_NAMES, KEYS, T002_KEYS, scored_keys
from .matchers import Judge, gold_state, match_key
from .normalize import is_blank

logger = logging.getLogger(__name__)

SPEC = "docs/v4/M1_evaluator_spec.md"
PASS_MARK = 0.85
N_BOOT = 1000
BOOT_SEED = 0
TEST_TITLES = {"T-001": "공유저작물 속성정보 정확도", "T-002": "멀티모달 저작물 속성정보 정확도"}
TEST_LABELS = {"T-001": "Te-1", "T-002": "Te-2"}   # 시험 절차서의 이름 (보고서 제목에 쓴다 — 데이터의 test 값은 T-001 · T-002 그대로)
KIND_KEYS = ("description", "hashtags")  # gold 종류(kind)별로도 보고하는 키


# ---------------------------------------------------------------------------
# 자료형
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """세트 × 키 하나의 판정 (per_set.jsonl 한 줄)."""
    set_id: str
    key: str
    pred: Any
    gold: Any
    state: str
    kind: Optional[str]
    tp: int
    fp: int
    fn: int
    matched: Optional[bool]
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id, "key": self.key, "pred": self.pred, "gold": self.gold,
            "state": self.state, "kind": self.kind, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "matched": self.matched, "detail": self.detail,
        }


@dataclass
class Warnings:
    missing_predictions: list[str] = field(default_factory=list)
    predictions_without_gold: list[str] = field(default_factory=list)
    duplicate_predictions: list[str] = field(default_factory=list)
    unreadable_predictions: list[str] = field(default_factory=list)
    ignored_files: list[str] = field(default_factory=list)                   # 예측 디렉터리의 .json 아닌 파일
    predictions_without_known_keys: list[str] = field(default_factory=list)  # 23키가 하나도 없는 예측(래핑·스키마 불일치)
    gold_test_mismatch: list[str] = field(default_factory=list)
    unscored_nonnull: list[dict[str, str]] = field(default_factory=list)
    unparseable: list[dict[str, str]] = field(default_factory=list)
    gold_unparseable: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "missing_predictions": sorted(self.missing_predictions),
            "predictions_without_gold": sorted(self.predictions_without_gold),
            "duplicate_predictions": sorted(self.duplicate_predictions),
            "unreadable_predictions": sorted(self.unreadable_predictions),
            "ignored_files": sorted(self.ignored_files),
            "predictions_without_known_keys": sorted(self.predictions_without_known_keys),
            "gold_test_mismatch": sorted(self.gold_test_mismatch),
            "unscored_nonnull": sorted(self.unscored_nonnull, key=lambda w: (w["set_id"], w["key"])),
            "unparseable": sorted(self.unparseable, key=lambda w: (w["set_id"], w["key"])),
            "gold_unparseable": sorted(self.gold_unparseable, key=lambda w: (w["set_id"], w["key"])),
        }

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.as_dict().items()}


# ---------------------------------------------------------------------------
# 적재
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    out: list[tuple[int, Any]] = []
    with path.open(encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append((no, json.loads(line)))
            except json.JSONDecodeError as e:
                logger.warning("%s:%d JSON 해석 실패 — 건너뜀 (%s)", path, no, e)
    return out


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    """gold23.jsonl → set_id 순 정렬 목록. 같은 set_id 가 여러 줄이면 마지막 줄만 남긴다."""
    by_id: dict[str, dict[str, Any]] = {}
    for no, rec in _read_jsonl(Path(path)):
        if not isinstance(rec, dict) or is_blank(rec.get("set_id")) or not isinstance(rec.get("keys"), dict):
            logger.warning("gold %s:%d set_id/keys 없음 — 건너뜀", path, no)
            continue
        sid = str(rec["set_id"])
        if sid in by_id:
            logger.warning("gold set_id 중복: %s (마지막 줄 사용)", sid)
        by_id[sid] = rec
    return [by_id[s] for s in sorted(by_id)]


def _pred_from_obj(obj: Any) -> Optional[dict[str, Any]]:
    return obj if isinstance(obj, dict) else None


def load_predictions(path: str | Path, warnings: Optional[Warnings] = None) -> dict[str, dict[str, Any]]:
    """예측 입력(§6): 디렉터리의 <set_id>.json 들, 또는 .jsonl(각 줄에 set_id 또는 _meta.set_id)."""
    w = warnings if warnings is not None else Warnings()
    p = Path(path)
    preds: dict[str, dict[str, Any]] = {}
    if p.is_dir():
        for f in sorted(x for x in p.iterdir() if x.is_file()):
            if f.suffix.lower() != ".json":  # 'X.JSON' 도 예측으로 읽고, 그 외 파일은 무시했다고 알린다 (조용히 누락 처리 금지)
                w.ignored_files.append(f.name)
                continue
            try:
                obj = _pred_from_obj(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                obj = None
            if obj is None:
                w.unreadable_predictions.append(f.name)
                continue
            if f.stem in preds:  # 'S1.json' 과 'S1.JSON' 이 함께 있는 경우
                w.duplicate_predictions.append(f.stem)
            preds[f.stem] = obj
        return preds
    if p.suffix.lower() == ".jsonl":
        for no, rec in _read_jsonl(p):
            obj = _pred_from_obj(rec)
            sid = None if obj is None else (obj.get("set_id") or (obj.get("_meta") or {}).get("set_id"))
            if obj is None or is_blank(sid):
                w.unreadable_predictions.append(f"{p.name}:{no}")
                continue
            sid = str(sid)
            if sid in preds:
                w.duplicate_predictions.append(sid)
            preds[sid] = obj
        return preds
    raise ValueError(f"예측 입력은 디렉터리 또는 .jsonl 이어야 합니다: {path}")


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def _short(v: Any, n: int = 80) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def score_set(test: str, gold_record: dict[str, Any], pred: dict[str, Any],
              judge: Optional[Judge], warnings: Warnings) -> list[Decision]:
    """한 세트의 채점 대상 키 전부를 판정한다 (상태 na 포함 — per_set.jsonl 에 남긴다)."""
    set_id = str(gold_record["set_id"])
    gold_keys = gold_record.get("keys") or {}
    out: list[Decision] = []
    for name in scored_keys(test):
        key = BY_NAME[name]
        entry = gold_keys.get(name) or {}
        raw = pred.get(name)
        res = match_key(key, raw, entry, judge=judge)
        detail = res["detail"]
        # matcher 가 'value' gold 를 정규화 후 비어 'empty' 로 다시 해석했으면(detail["state"]) 상태 표도 그 상태로 센다
        state = "na" if key.type == "na" else (detail.get("state") or gold_state(entry))
        if "unparseable" in detail:
            warnings.unparseable.append({"set_id": set_id, "key": name, "value": _short(raw), "error": detail["unparseable"]})
        if "gold_unparseable" in detail:
            warnings.gold_unparseable.append({"set_id": set_id, "key": name, "value": _short(entry.get("value")), "error": detail["gold_unparseable"]})
        if detail.get("unscored_nonnull"):
            warnings.unscored_nonnull.append({"set_id": set_id, "key": name})
        out.append(Decision(set_id, name, raw, entry.get("value"), state, entry.get("kind"),
                            res["tp"], res["fp"], res["fn"], res["matched"], detail))
    return out


def check_known_keys(set_id: str, pred: dict[str, Any], warnings: Warnings) -> None:
    """23키가 하나도 없는 예측(예: 필드가 'metadata' 아래에 감싸인 API 응답, 다른 스키마 버전)은 전부 null 로 채점되어
    gold 값이 모두 FN 이 된다 — 조용히 0점이 되지 않게 경고로 드러낸다(§6). 채점 자체는 바꾸지 않는다."""
    if not any(k in pred for k in KEY_NAMES):
        warnings.predictions_without_known_keys.append(set_id)


def check_unscored(test: str, set_id: str, pred: dict[str, Any], warnings: Warnings) -> None:
    """T-001 예측은 17~23 이 null 이어야 한다(§6). T-002 의 1~16 은 계약서 복사가 허용되므로 경고하지 않는다(§0)."""
    if test != "T-001":
        return
    for name in T002_KEYS:
        if not is_blank(pred.get(name)):
            warnings.unscored_nonnull.append({"set_id": set_id, "key": name})


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def prf(tp: int, fp: int, fn: int) -> dict[str, Optional[float]]:
    """정밀도·재현율(한쪽 분모만 0 → 0.0, sklearn 관례), F1 = 2TP/(2TP+FP+FN). 판정이 전혀 없으면 셋 다 None."""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return {"precision": None, "recall": None, "f1": None}
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * tp / denom}


def _stats(decisions: Sequence[Decision]) -> dict[str, Any]:
    tp = sum(d.tp for d in decisions)
    fp = sum(d.fp for d in decisions)
    fn = sum(d.fn for d in decisions)
    states = Counter(d.state for d in decisions)
    n_scored = states.get("value", 0) + states.get("empty", 0)
    return {"tp": tp, "fp": fp, "fn": fn, "n_scored": n_scored,
            "states": {s: states.get(s, 0) for s in ("value", "empty", "na")}, **prf(tp, fp, fn)}


def aggregate_keys(decisions: Sequence[Decision], keys: Sequence[str]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        by_key[d.key].append(d)
    out: dict[str, dict[str, Any]] = {}
    for name in keys:
        k = BY_NAME[name]
        out[name] = {"no": k.no, "label_ko": k.label_ko, "type": k.type, **_stats(by_key.get(name, []))}
    return out


def included_keys(key_stats: dict[str, dict[str, Any]], keys: Sequence[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Macro 평균에 드는 키 = 평가 건수 > 0 이고 판정이 하나라도 있는 키. 나머지는 사유와 함께 나열."""
    inc: list[str] = []
    exc: list[dict[str, str]] = []
    for name in keys:
        st = key_stats[name]
        if st["n_scored"] == 0:
            exc.append({"key": name, "reason": "n_scored=0"})
        elif st["f1"] is None:
            exc.append({"key": name, "reason": "no_decisions"})
        else:
            inc.append(name)
    return inc, exc


def macro_micro(key_stats: dict[str, dict[str, Any]], inc: Sequence[str]) -> dict[str, Any]:
    if inc:
        macro = {m: float(np.mean([key_stats[k][m] for k in inc])) for m in ("precision", "recall", "f1")}
    else:
        macro = {"precision": None, "recall": None, "f1": None}
    tp = sum(key_stats[k]["tp"] for k in inc)
    fp = sum(key_stats[k]["fp"] for k in inc)
    fn = sum(key_stats[k]["fn"] for k in inc)
    micro = {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
    return {"macro": {**macro, "n_keys": len(inc)}, "micro": micro}


def _f1_array(sums: np.ndarray) -> np.ndarray:
    tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), np.nan)


def _ci(values: np.ndarray) -> Optional[list[float]]:
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap(decisions: Sequence[Decision], set_ids: Sequence[str], keys: Sequence[str],
              inc: Sequence[str], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    """세트 단위 복원추출 n_boot 회 → 키별·Macro·Micro F1 의 2.5/97.5 백분위."""
    S, K = len(set_ids), len(keys)
    if S == 0 or n_boot <= 0:
        return {"keys": {k: None for k in keys}, "macro": None, "micro": None}
    si = {s: i for i, s in enumerate(set_ids)}
    ki = {k: i for i, k in enumerate(keys)}
    arr = np.zeros((S, K, 3), dtype=np.float64)
    for d in decisions:
        arr[si[d.set_id], ki[d.key]] += (d.tp, d.fp, d.fn)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(n_boot, S))
    weights = np.zeros((n_boot, S), dtype=np.float64)
    for b in range(n_boot):
        weights[b] = np.bincount(idx[b], minlength=S)
    sums = (weights @ arr.reshape(S, K * 3)).reshape(n_boot, K, 3)
    f1_keys = _f1_array(sums)                                   # (n_boot, K)
    inc_idx = [ki[k] for k in inc]
    with np.errstate(invalid="ignore"):
        macro = np.nanmean(f1_keys[:, inc_idx], axis=1) if inc_idx else np.full(n_boot, np.nan)
    micro = _f1_array(sums[:, inc_idx, :].sum(axis=1)) if inc_idx else np.full(n_boot, np.nan)
    return {"keys": {k: _ci(f1_keys[:, ki[k]]) for k in keys}, "macro": _ci(macro), "micro": _ci(micro),
            "n_resamples": n_boot, "seed": seed}


def by_kind(decisions: Sequence[Decision], key: str) -> dict[str, dict[str, Any]]:
    """설명·해시태그를 gold 종류(kind)별로 — 두 트랙(site vs generated) + combined (§2, §5)."""
    ds = [d for d in decisions if d.key == key]
    groups: dict[str, list[Decision]] = defaultdict(list)
    for d in ds:
        groups[d.kind or "none"].append(d)
    out = {kind: _stats(groups[kind]) for kind in sorted(groups)}
    out["combined"] = _stats(ds)
    return out


def _cell_of(rec: dict[str, Any]) -> str:
    return f"{rec.get('media') or '?'}/{rec.get('bucket') or '?'}"


def by_cell(decisions: Sequence[Decision], gold_records: Sequence[dict[str, Any]],
            keys: Sequence[str]) -> dict[str, dict[str, Any]]:
    """셀(매체 × 라이선스 구간)별 키 지표 + Macro/Micro (CI 없음)."""
    cell_of = {str(r["set_id"]): _cell_of(r) for r in gold_records}
    groups: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        groups[cell_of[d.set_id]].append(d)
    n_sets = Counter(cell_of.values())
    out: dict[str, dict[str, Any]] = {}
    for cell in sorted(groups):
        ks = aggregate_keys(groups[cell], keys)
        inc, exc = included_keys(ks, keys)
        mm = macro_micro(ks, inc)
        out[cell] = {"n_sets": n_sets[cell], "keys": ks, "excluded_keys": exc, **mm}
    return out


def aggregate(test: str, decisions: Sequence[Decision], gold_records: Sequence[dict[str, Any]],
              n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    keys = list(scored_keys(test))
    set_ids = [str(r["set_id"]) for r in gold_records]
    key_stats = aggregate_keys(decisions, keys)
    inc, exc = included_keys(key_stats, keys)
    mm = macro_micro(key_stats, inc)
    boot = bootstrap(decisions, set_ids, keys, inc, n_boot=n_boot, seed=seed)
    for name in keys:
        key_stats[name]["ci95_f1"] = boot["keys"].get(name)
    mm["macro"]["ci95_f1"] = boot["macro"]
    mm["micro"]["ci95_f1"] = boot["micro"]
    macro_f1 = mm["macro"]["f1"]
    return {
        "keys": key_stats,
        "included_keys": inc,
        "excluded_keys": exc,
        "macro": mm["macro"],
        "micro": mm["micro"],
        "macro_f1": macro_f1,
        "micro_f1": mm["micro"]["f1"],
        "pass_mark": PASS_MARK,
        "passed": (macro_f1 is not None and macro_f1 >= PASS_MARK),
        "by_kind": {k: by_kind(decisions, k) for k in KIND_KEYS if k in keys},
        "cells": by_cell(decisions, gold_records, keys),
        "bootstrap": {"n_resamples": n_boot, "seed": seed},
    }


# ---------------------------------------------------------------------------
# 보고서 (report.md) — HWP 표기와 같은 한글 헤더: 항목 / 정밀도 / 재현율 / F1-score / 최종 지표
# ---------------------------------------------------------------------------
def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def _ci_str(ci: Optional[Sequence[float]]) -> str:
    return "-" if not ci else f"[{_pct(ci[0])}, {_pct(ci[1])}]"


def _key_rows(key_stats: dict[str, dict[str, Any]], keys: Sequence[str], with_ci: bool) -> list[str]:
    head = "| # | 항목 | 평가 건수 | TP | FP | FN | 정밀도 | 재현율 | F1-score |" + (" 95% CI (F1) |" if with_ci else "")
    rule = "|---|---|---:|---:|---:|---:|---:|---:|---:|" + ("---|" if with_ci else "")
    rows = [head, rule]
    for name in keys:
        st = key_stats[name]
        row = (f"| {st['no']} | {st['label_ko']} (`{name}`) | {st['n_scored']} | {st['tp']} | {st['fp']} | {st['fn']} "
               f"| {_pct(st['precision'])} | {_pct(st['recall'])} | {_pct(st['f1'])} |")
        if with_ci:
            row += f" {_ci_str(st.get('ci95_f1'))} |"
        rows.append(row)
    return rows


def _final_rows(macro: dict[str, Any], micro: dict[str, Any]) -> list[str]:
    verdict = "-" if macro["f1"] is None else ("합격" if macro["f1"] >= PASS_MARK else "불합격")
    return [
        "| 최종 지표 | 정밀도 | 재현율 | F1-score | 95% CI (F1) | 합격 기준 (Macro-F1 ≥ 85%) |",
        "|---|---:|---:|---:|---|---|",
        f"| **Macro 평균** (항목 {macro['n_keys']}개) | {_pct(macro['precision'])} | {_pct(macro['recall'])} "
        f"| **{_pct(macro['f1'])}** | {_ci_str(macro.get('ci95_f1'))} | **{verdict}** |",
        f"| Micro 평균 (참고) | {_pct(micro['precision'])} | {_pct(micro['recall'])} | {_pct(micro['f1'])} "
        f"| {_ci_str(micro.get('ci95_f1'))} | - |",
    ]


def _kind_rows(by_kind_stats: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    rows = ["| 항목 | gold 종류 | 평가 건수 | TP | FP | FN | 정밀도 | 재현율 | F1-score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(by_kind_stats):
        label = BY_NAME[key].label_ko
        for kind, st in by_kind_stats[key].items():
            rows.append(f"| {label} (`{key}`) | {kind} | {st['n_scored']} | {st['tp']} | {st['fp']} | {st['fn']} "
                        f"| {_pct(st['precision'])} | {_pct(st['recall'])} | {_pct(st['f1'])} |")
    return rows


def _state_rows(key_stats: dict[str, dict[str, Any]], keys: Sequence[str]) -> list[str]:
    rows = ["| # | 항목 | ① 값 있음 | ② 확정 빈값 | ③ 미수집·비적용 |", "|---|---|---:|---:|---:|"]
    for name in keys:
        st = key_stats[name]
        s = st["states"]
        rows.append(f"| {st['no']} | {st['label_ko']} (`{name}`) | {s['value']} | {s['empty']} | {s['na']} |")
    return rows


def _warning_lines(w: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"- 예측 누락 세트: {len(w['missing_predictions'])}건 (gold 값이 있는 키는 모두 FN)"
                 + (f" — {', '.join(w['missing_predictions'][:20])}" if w["missing_predictions"] else ""))
    lines.append(f"- gold 없는 예측: {len(w['predictions_without_gold'])}건 (무시)")
    if w["predictions_without_known_keys"]:
        lines.append(f"- 23키가 하나도 없는 예측: {len(w['predictions_without_known_keys'])}건 "
                     f"(전부 null 로 채점됨 — 스키마·래핑 확인) — {', '.join(w['predictions_without_known_keys'][:20])}")
    lines.append(f"- 평가 대상 외 항목이 null 이 아님: {len(w['unscored_nonnull'])}건 (FP 아님, 경고만)")
    lines.append(f"- 해석 불가 예측값: {len(w['unparseable'])}건 (오답 FP+FN 또는 FP 로 산정)")
    for item in w["unparseable"][:20]:
        lines.append(f"  - {item['set_id']} · {item['key']}: `{item['value']}`")
    if w["gold_unparseable"]:
        lines.append(f"- 해석 불가 gold 값: {len(w['gold_unparseable'])}건 (판정 제외 — gold 를 고쳐야 함)")
    if w["duplicate_predictions"]:
        lines.append(f"- 중복 set_id 예측: {len(w['duplicate_predictions'])}건 (마지막 것 사용)")
    if w["unreadable_predictions"]:
        lines.append(f"- 읽을 수 없는 예측 파일/줄: {len(w['unreadable_predictions'])}건")
    if w["ignored_files"]:
        lines.append(f"- 예측 디렉터리에서 무시한 파일(.json 아님): {len(w['ignored_files'])}건 — {', '.join(w['ignored_files'][:20])}")
    if w["gold_test_mismatch"]:
        lines.append(f"- 시험 구분이 다른 gold 레코드: {len(w['gold_test_mismatch'])}건 (제외)")
    return lines


def render_report(m: dict[str, Any], cells: bool = False) -> str:
    test = m["test"]
    keys = list(scored_keys(test))
    s = m["settings"]
    L: list[str] = []
    L.append(f"# v4 평가 보고서 — {TEST_LABELS.get(test, test)} {TEST_TITLES.get(test, '')}".rstrip())
    L.append("")
    L.append(f"- 판정 규칙: `{SPEC}` · 채점 키: {keys[0]}~{keys[-1]} ({len(keys)}개)")
    L.append(f"- gold: `{m['inputs']['gold']}` · 예측: `{m['inputs']['pred']}`")
    L.append(f"- 평가 세트 {m['n_sets']}개 · 예측 {m['n_predictions']}개 · 예측 누락 {len(m['warnings']['missing_predictions'])}개"
             + (f" · gold 제외 {m.get('n_gold_dropped', 0)}개 (--only-predicted: 예측 없는 세트는 채점하지 않음)"
                if s.get("only_predicted") else ""))
    if s.get("embed"):
        e = s["embed"]
        L.append(f"- 설명 판정: 임베딩 `{e['model_id']}` (revision {e['revision'] or '미기록'}), "
                 f"τ = {e['tau']}, 내용 토큰 확인 {'on' if e['content_check'] else 'off'}")
    L.append(f"- 95% CI: 세트 단위 부트스트랩 {m['bootstrap']['n_resamples']}회, seed {m['bootstrap']['seed']}")
    L.append("")
    L.append("## 1. 최종 지표")
    L.append("")
    L.extend(_final_rows(m["macro"], m["micro"]))
    if m["excluded_keys"]:
        exc = ", ".join(f"{BY_NAME[e['key']].label_ko} (`{e['key']}`, {e['reason']})" for e in m["excluded_keys"])
        L.append("")
        L.append(f"Macro 평균에서 제외된 항목: {exc}")
    L.append("")
    L.append("## 2. 항목별 지표")
    L.append("")
    L.extend(_key_rows(m["keys"], keys, with_ci=True))
    L.append("")
    L.append("## 3. 항목별 gold 상태 건수")
    L.append("")
    L.extend(_state_rows(m["keys"], keys))
    if m["by_kind"]:
        L.append("")
        L.append("## 4. 설명·해시태그 — gold 종류별 (site vs generated 두 트랙)")
        L.append("")
        L.extend(_kind_rows(m["by_kind"]))
    L.append("")
    L.append("## 5. 셀별 (매체 × 라이선스 구간)")
    L.append("")
    L.append("| 셀 | 세트 수 | Macro-F1 | Micro-F1 | 항목 수 |")
    L.append("|---|---:|---:|---:|---:|")
    for cell, c in m["cells"].items():
        L.append(f"| {cell} | {c['n_sets']} | {_pct(c['macro']['f1'])} | {_pct(c['micro']['f1'])} | {c['macro']['n_keys']} |")
    if cells:
        for cell, c in m["cells"].items():
            L.append("")
            L.append(f"### 5.{cell} — 세트 {c['n_sets']}개")
            L.append("")
            L.extend(_key_rows(c["keys"], keys, with_ci=False))
    L.append("")
    L.append("## 6. 경고")
    L.append("")
    L.extend(_warning_lines(m["warnings"]))
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def _make_judge(tau: float, content_check: bool, embed_cache: Optional[str | Path],
                embed_revision: Optional[str], embed_model: Optional[str], device: Optional[str]) -> Judge:
    from .embed_judge import DEFAULT_MODEL_ID, EmbedJudge  # 모델은 첫 decide() 때 지연 적재

    return EmbedJudge(model_id=embed_model or DEFAULT_MODEL_ID, revision=embed_revision, device=device,
                      cache_path=embed_cache, tau=tau, content_check=content_check)


def _judge_info(judge: Optional[Judge]) -> Optional[dict[str, Any]]:
    info = getattr(judge, "info", None)
    return info() if callable(info) else None


def write_outputs(out_dir: str | Path, metrics: dict[str, Any], decisions: Sequence[Decision], cells: bool) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"metrics": out / "test_metrics.json", "report": out / "report.md", "per_set": out / "per_set.jsonl"}
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["report"].write_text(render_report(metrics, cells=cells), encoding="utf-8")
    with paths["per_set"].open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return {k: str(v) for k, v in paths.items()}


def evaluate(
    test: str,
    gold_path: str | Path,
    pred_path: str | Path,
    out_dir: str | Path,
    *,
    tau: float = 0.65,
    content_check: bool = True,
    cells: bool = False,
    embed_cache: Optional[str | Path] = None,
    embed_revision: Optional[str] = None,
    embed_model: Optional[str] = None,
    device: Optional[str] = None,
    judge: Optional[Judge] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    only_predicted: bool = False,
) -> dict[str, Any]:
    """§7 — 예측을 gold 와 대조해 test_metrics.json · report.md · per_set.jsonl 을 쓰고 지표 dict 를 돌려준다.

    only_predicted (기본 False = M1 동작 그대로: 예측 누락 세트는 gold 값이 있는 키마다 FN):
        True 면 gold 를 **예측이 있는 set_id** 로 제한한다 — 개발·파일럿 런(예측 20세트)을 4,000세트 gold 에 대조하면
        나머지 3,980세트가 전부 FN 이 되어 지표가 0 에 붙는다. 제외한 gold 세트 수는 로그와 metrics["n_gold_dropped"] 에 남긴다.
    """
    if test not in TEST_TITLES:
        raise ValueError(f"test 는 T-001 또는 T-002 이어야 합니다: {test}")
    warnings = Warnings()
    gold_all = load_gold(gold_path)
    gold_records = []
    for r in gold_all:
        if r.get("test") and r["test"] != test:
            warnings.gold_test_mismatch.append(str(r["set_id"]))
        else:
            gold_records.append(r)
    preds = load_predictions(pred_path, warnings)
    n_gold_dropped = 0
    if only_predicted:
        n_gold_all = len(gold_records)
        gold_records = [r for r in gold_records if str(r["set_id"]) in preds]
        n_gold_dropped = n_gold_all - len(gold_records)
        logger.info("%s --only-predicted: 예측이 있는 세트만 채점 — gold %d세트 중 %d세트 제외, %d세트 대조",
                    test, n_gold_all, n_gold_dropped, len(gold_records))
    if judge is None and "description" in scored_keys(test):
        judge = _make_judge(tau, content_check, embed_cache, embed_revision, embed_model, device)
    logger.info("%s 채점 시작: gold %d세트, 예측 %d개", test, len(gold_records), len(preds))

    decisions: list[Decision] = []
    for rec in gold_records:
        sid = str(rec["set_id"])
        pred = preds.get(sid)
        if pred is None:
            warnings.missing_predictions.append(sid)
            pred = {}
        else:
            check_unscored(test, sid, pred, warnings)
            check_known_keys(sid, pred, warnings)
        decisions.extend(score_set(test, rec, pred, judge, warnings))
    gold_ids = {str(r["set_id"]) for r in gold_records}
    warnings.predictions_without_gold.extend(s for s in preds if s not in gold_ids)

    agg = aggregate(test, decisions, gold_records, n_boot=n_boot, seed=seed)
    metrics: dict[str, Any] = {
        "test": test,
        "spec": SPEC,
        "inputs": {"gold": str(gold_path), "pred": str(pred_path)},
        "n_sets": len(gold_records),
        "n_predictions": len(preds),
        "n_gold_dropped": n_gold_dropped,     # --only-predicted 로 제외한 gold 세트 수 (기본 0)
        "settings": {"tau": tau, "content_check": content_check, "only_predicted": only_predicted,
                     "embed": _judge_info(judge)},
        "warnings": warnings.as_dict(),
        "warning_counts": warnings.counts(),
        **agg,
    }
    metrics["outputs"] = write_outputs(out_dir, metrics, decisions, cells)
    logger.info("%s Macro-F1 %s · Micro-F1 %s → %s", test, _pct(metrics["macro_f1"]), _pct(metrics["micro_f1"]),
                "합격" if metrics["passed"] else "불합격")
    return metrics


__all__ = [
    "Decision", "Warnings", "PASS_MARK", "N_BOOT", "BOOT_SEED", "SPEC",
    "load_gold", "load_predictions", "score_set", "check_unscored", "check_known_keys",
    "prf", "aggregate_keys", "included_keys", "macro_micro", "bootstrap", "by_kind", "by_cell", "aggregate",
    "render_report", "write_outputs", "evaluate",
]
