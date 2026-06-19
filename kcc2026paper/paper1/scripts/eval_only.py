"""paper 1 sweep — 학습 산출 adapter 만 사용해 gold OOD 평가 재실행.

paper1.py 의 첫 sweep (20260427_143110) 에서 학습은 완료됐으나
TokenClassNER.predict_batch 부재로 평가가 crash. adapter 는 보존되어 있어
재학습 없이 평가만 수행.

사용:
    .venv/bin/python paper1/scripts/eval_only.py <stamp>
    .venv/bin/python paper1/scripts/eval_only.py 20260427_143110
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
METADATA_ROOT = ROOT.parent / "metadata"
for path in (ROOT, METADATA_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from paper_module.core.ner.token_cls import TokenClassNER       # noqa: E402
from paper_module.core.run_logger import (                       # noqa: E402
    open_log, log_section, close_log,
)


# paper1.py 와 동일 상수
MODEL = "klue/bert-base"
CONFIGS = [
    ("rule", "m1_answer"),
    ("rule", "m2_context"),
    ("rule", "m3_negatives"),
]
FORMAT_CLASS = {
    "format-regular": [
        "phone", "email", "date", "ri_data", "ri_period", "ri_money",
        "address", "copyright_url", "copyright_uci", "copyright_num",
        "copyright_idnum", "copyright_status", "copyright_quantity",
        "copyright_language",
    ],
    "format-semi-regular": [
        "copyright_Keyword", "copyright_kotitle", "ri_law_reference",
        "ri_info", "ri_contract_type", "ri_copyright",
    ],
    "format-free": [
        "name", "company", "department", "position",
        "copyright_description", "copyright_type",
    ],
}
LABEL_CLASS = {lb: k for k, v in FORMAT_CLASS.items() for lb in v}
NER_LABELS = sorted(LABEL_CLASS.keys())


def extract_span_bio(tokens, bio_labels, target_label):
    b_tag = f"B-{target_label}"
    i_tag = f"I-{target_label}"
    out = []
    in_span = False
    for tok, lab in zip(tokens, bio_labels):
        if lab == b_tag:
            if out:
                break
            out = [tok]
            in_span = True
        elif lab == i_tag and in_span:
            out.append(tok)
        elif in_span:
            break
    return " ".join(out)


def load_gold(label):
    gf = ROOT / "configs" / "integrated" / "gold" / f"{label}.jsonl"
    if not gf.exists():
        return []
    return [json.loads(l) for l in gf.read_text(encoding="utf-8").splitlines() if l.strip()]


def eval_one(source: str, mode: str, sweep_root: Path) -> Dict[str, Any]:
    cfg_id = f"{source}_{mode}"
    cfg_dir = sweep_root / cfg_id
    model_path = cfg_dir / "model"
    adapter_parent = model_path / MODEL.replace("/", "--")
    log_path = cfg_dir / "run.txt"

    if not (adapter_parent / "adapter" / "full_model_weights.pt").exists():
        return {"status": "no_adapter", "cfg_id": cfg_id}

    print(f"\n[{cfg_id}] adapter={adapter_parent}")
    print(f"  loading TokenClassNER...")
    ner = TokenClassNER(adapter_parent)
    ner.load()

    t0 = time.time()
    per_label: Dict[str, Dict[str, Any]] = {}
    per_source: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "hit": 0})
    )
    n_total = 0
    for label in NER_LABELS:
        recs = load_gold(label)
        if not recs:
            per_label[label] = {"n_gold": 0, "hit": 0, "accuracy": 0.0,
                                "class": LABEL_CLASS[label]}
            continue
        hit = 0
        for rec in recs:
            tokens = rec["text"].split()
            ans = rec["answer"].strip()
            src_key = rec.get("source", "?")
            try:
                bio = ner.predict([tokens], threshold=0.25)[0]
            except Exception:
                bio = ["O"] * len(tokens)
            pred_span = extract_span_bio(tokens, bio, label)
            ok = bool(pred_span) and ((ans in pred_span) or (pred_span in ans))
            if ok:
                hit += 1
            per_source[label][src_key]["n"] += 1
            per_source[label][src_key]["hit"] += int(ok)
        acc = round(hit / len(recs), 4) if recs else 0.0
        per_label[label] = {
            "n_gold": len(recs), "hit": hit, "accuracy": acc,
            "class": LABEL_CLASS[label],
        }
        n_total += len(recs)
        print(f"    {label:<25} n={len(recs):>5}  hit={hit:>5}  acc={acc:.4f}")
    eval_sec = time.time() - t0

    per_class: Dict[str, Dict[str, float]] = {}
    for cls, lbls in FORMAT_CLASS.items():
        accs = [per_label[lb]["accuracy"] for lb in lbls if lb in per_label]
        per_class[cls] = {"n": len(accs),
                          "mean_acc": round(sum(accs) / max(len(accs), 1), 4)}

    ps_clean: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for lb, smap in per_source.items():
        ps_clean[lb] = {sk: {"n": v["n"], "hit": v["hit"],
                              "accuracy": round(v["hit"] / max(v["n"], 1), 4)}
                        for sk, v in smap.items()}

    overall = round(
        sum(v["accuracy"] for v in per_label.values()) / max(len(per_label), 1), 4
    )

    fp = open_log(log_path)
    try:
        log_section(fp, "meta", {
            "source": source, "mode": mode, "cfg_id": cfg_id,
            "model": MODEL, "method": "full", "training_mode": "integrated",
            "n_labels": len(NER_LABELS),
            "adapter_path": str(adapter_parent),
            "note": "eval-only re-run (paper1.py predict_batch 버그 우회)",
        })
        log_section(fp, "eval_result", {
            "n_labels": len(per_label),
            "accuracy_overall": overall,
            "per_class": per_class,
            "per_label": per_label,
            "per_source": ps_clean,
            "eval_time_sec": round(eval_sec, 1),
        })
        log_section(fp, "final", {
            "accuracy": overall,
            "total_time_sec": round(eval_sec, 1),
        })
    finally:
        close_log(fp)

    print(f"  → overall acc={overall:.4f}  reg={per_class['format-regular']['mean_acc']:.4f}  "
          f"semi={per_class['format-semi-regular']['mean_acc']:.4f}  "
          f"free={per_class['format-free']['mean_acc']:.4f}  ({eval_sec/60:.1f}m)")

    return {
        "status": "ok", "cfg_id": cfg_id, "source": source, "mode": mode,
        "accuracy_overall": overall,
        "per_label": per_label, "per_class": per_class, "per_source": ps_clean,
        "eval_time_sec": round(eval_sec, 1),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    stamp = sys.argv[1]
    sweep_root = ROOT / "paper1" / "data" / "runs" / stamp
    if not sweep_root.is_dir():
        raise SystemExit(f"sweep dir not found: {sweep_root}")

    print(f"sweep: {sweep_root}")
    summary = []
    for source, mode in CONFIGS:
        r = eval_one(source, mode, sweep_root)
        summary.append(r)

    (sweep_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsummary: {sweep_root / 'summary.json'}")
    print(f"\n=== 결과 요약 ===")
    for r in summary:
        if r["status"] == "ok":
            cfg = r["cfg_id"]
            acc = r["accuracy_overall"]
            pc = r["per_class"]
            print(f"  {cfg:<22} overall={acc:.4f}  reg={pc['format-regular']['mean_acc']:.4f}  "
                  f"semi={pc['format-semi-regular']['mean_acc']:.4f}  "
                  f"free={pc['format-free']['mean_acc']:.4f}")
        else:
            print(f"  {r['cfg_id']:<22} {r['status']}")


if __name__ == "__main__":
    main()
