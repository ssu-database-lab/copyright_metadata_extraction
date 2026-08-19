"""results.jsonl → 집계 리포트.

지표 정의를 여기서 못박는다. "정확도 85%"가 무엇의 85%인지 모호하면
시험기관과 숫자를 맞출 수 없다.

  속성 정확도   = 적중 속성 수 / 채점된 속성 수      (전 세트 합산, micro)
  세트 평균     = 세트별 정확도의 평균               (macro — 세트마다 채점 속성 수가 달라도 동일 가중)
  속성별 정확도 = 해당 속성이 채점된 세트 중 적중 비율

계획서 §2-4는 "속성정보의 정확도"라고만 하므로, 기본 지표는 micro(속성 정확도)로
보고하고 macro를 함께 제시한다 — 어느 쪽으로 합의되든 숫자가 준비돼 있도록.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .scoring import ATTRIBUTES


def load_results(path: str | Path) -> List[Dict]:
    p = Path(path)
    out = []
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # 재개로 같은 set_id가 여러 번 있으면 마지막 성공분만 남긴다
    latest: Dict[str, Dict] = {}
    for r in out:
        sid = str(r.get("set_id"))
        if sid not in latest or r.get("ok"):
            latest[sid] = r
    return list(latest.values())


def aggregate(results: List[Dict]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    tot_scored = sum(r.get("n_scored", 0) for r in ok)
    tot_match = sum(r.get("n_match", 0) for r in ok)
    macro = [r["accuracy"] for r in ok if r.get("accuracy") is not None]

    per_attr = defaultdict(lambda: {"scored": 0, "match": 0, "na": 0, "no_gt": 0,
                                    "tier_skip": 0, "batch": 0, "batch_match": 0})
    by_bucket = defaultdict(lambda: {"n": 0, "scored": 0, "match": 0})
    by_media = defaultdict(lambda: {"n": 0, "scored": 0, "match": 0})

    for r in ok:
        b = by_bucket[r.get("license_bucket") or "-"]
        m = by_media[r.get("media") or "-"]
        for d in (b, m):
            d["n"] += 1
            d["scored"] += r.get("n_scored", 0)
            d["match"] += r.get("n_match", 0)
        for name, info in (r.get("per_attr") or {}).items():
            st = info.get("status")
            a = per_attr[name]
            if st == "scored":
                a["scored"] += 1
                a["match"] += bool(info.get("match"))
                if info.get("batch_level"):
                    a["batch"] += 1
                    a["batch_match"] += bool(info.get("match"))
            elif st == "not_applicable":
                a["na"] += 1
            elif st == "skipped_tier":
                a["tier_skip"] += 1
            else:
                a["no_gt"] += 1

    return {
        "n_sets": len(results), "n_ok": len(ok), "n_failed": len(failed),
        "micro_accuracy": (tot_match / tot_scored) if tot_scored else None,
        "macro_accuracy": (sum(macro) / len(macro)) if macro else None,
        "total_scored": tot_scored, "total_match": tot_match,
        "per_attr": dict(per_attr),
        "by_bucket": dict(by_bucket), "by_media": dict(by_media),
        "cost_krw": round(sum(r.get("cost_krw", 0) for r in ok), 1),
        "elapsed_mean_sec": round(sum(r.get("elapsed_sec", 0) for r in ok) / len(ok), 1) if ok else 0,
        "failures": [{"set_id": r.get("set_id"), "error": r.get("error")} for r in failed[:20]],
    }


def _pct(n, d):
    return f"{n/d*100:.1f}%" if d else "—"


def render_markdown(agg: Dict[str, Any], title: str = "속성정보 추출 평가 결과") -> str:
    L = [f"# {title}", ""]
    L.append(f"- 세트 {agg['n_sets']}건 (성공 {agg['n_ok']} · 실패 {agg['n_failed']})")
    mic = agg["micro_accuracy"]
    mac = agg["macro_accuracy"]
    L.append(f"- **속성 정확도(micro): {mic*100:.1f}%**" if mic is not None else "- 속성 정확도: —")
    L.append(f"- 세트 평균(macro): {mac*100:.1f}%" if mac is not None else "- 세트 평균: —")
    L.append(f"- 채점 {agg['total_match']}/{agg['total_scored']} 속성")
    L.append(f"- 추정 비용 ₩{agg['cost_krw']:,.0f} · 세트당 평균 {agg['elapsed_mean_sec']}초")
    L.append("")
    L.append("> 목표: **85%** (연구개발계획서 §2-4, 2단계)")
    L.append("")

    L += ["## 속성별", "",
          "| 속성 | 채점 | 적중 | 정확도 | 항목별 정확도 | 해당없음 | 정답없음 | 정의불일치 |",
          "|---|---|---|---|---|---|---|---|"]
    for a in ATTRIBUTES:
        d = agg["per_attr"].get(a.name)
        if not d:
            continue
        # 컬렉션 공통 태그를 뺀 '항목별' 정확도를 함께 보여준다 — 모델이 픽셀에서
        # 만들어낼 수 없는 배치 태그가 섞이면 수치가 왜곡된다.
        item_n = d["scored"] - d.get("batch", 0)
        item_h = d["match"] - d.get("batch_match", 0)
        item = _pct(item_h, item_n) if item_n else "—"
        L.append(f"| {a.name} | {d['scored']} | {d['match']} | "
                 f"{_pct(d['match'], d['scored'])} | {item} | {d['na']} | {d['no_gt']} | "
                 f"{d.get('tier_skip', 0)} |")
    L.append("")

    for label, key in (("권리유형별", "by_bucket"), ("미디어별", "by_media")):
        rows = agg.get(key) or {}
        if not rows:
            continue
        L += [f"## {label}", "", "| 구분 | 세트 | 채점 | 적중 | 정확도 |", "|---|---|---|---|---|"]
        for k, d in sorted(rows.items()):
            L.append(f"| {k} | {d['n']} | {d['scored']} | {d['match']} | "
                     f"{_pct(d['match'], d['scored'])} |")
        L.append("")

    if agg.get("failures"):
        L += ["## 실패 세트", "", "| set_id | 오류 |", "|---|---|"]
        for f in agg["failures"]:
            L.append(f"| {f['set_id']} | {str(f['error'])[:90]} |")
        L.append("")
    return "\n".join(L)
