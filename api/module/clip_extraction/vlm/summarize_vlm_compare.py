#!/usr/bin/env python3
"""
Summarize a vlm_compare_*.json report into a Gemma-vs-Qwen agreement table,
joined with KOGL ground-truth labels (_labels.csv) when available.

Reports, per image: each model's work_type, whether the two models AGREE,
and the KOGL 정보유형 for reference (noting KOGL labels are coarse — see
project memory). Also prints aggregate: agreement rate, parse success,
avg latency/tokens per model.

Usage:
    python -m api.module.clip_extraction.vlm.summarize_vlm_compare <report.json> [labels.csv]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def summarize(report_path: str, labels_csv: str | None = None) -> dict:
    rep = json.load(open(report_path, encoding="utf-8"))
    models = rep["models"]
    by_image = rep["by_image"]

    labels = {}
    if labels_csv and Path(labels_csv).exists():
        for r in csv.DictReader(open(labels_csv, encoding="utf-8-sig")):
            labels[r["file"]] = r

    print(f"Report: {Path(report_path).name}")
    print(f"Models: {', '.join(models)}")
    print(f"Images: {rep['n_images']}\n")

    # Per-image work_type table
    two = len(models) >= 2
    hdr = f"{'image':14} " + " ".join(f"{m.split()[0]:14}" for m in models)
    if two:
        hdr += " agree?"
    hdr += "  KOGL유형  제목"
    print(hdr)
    print("-" * len(hdr))

    agree = 0
    n_both = 0
    for img, recs in by_image.items():
        wts = []
        for m in models:
            rec = recs.get(m) or {}
            p = rec.get("parsed") or {}
            wts.append(p.get("work_type", "—") if rec.get("ok") else "ERR")
        lab = labels.get(img, {})
        line = f"{img[:14]:14} " + " ".join(f"{w:14}" for w in wts)
        if two:
            both_ok = all(w not in ("—", "ERR") for w in wts[:2])
            a = (wts[0] == wts[1]) if both_ok else None
            if both_ok:
                n_both += 1
                agree += int(a)
            line += f" {'✓' if a else ('✗' if a is False else '·'):6}"
        line += f"  {lab.get('정보유형','?'):6}  {lab.get('제목','')[:24]}"
        print(line)

    # Aggregates
    print("\n--- aggregates ---")
    for m in models:
        recs = [by_image[i].get(m) or {} for i in by_image]
        ok = sum(1 for r in recs if r.get("ok"))
        parsed = sum(1 for r in recs if r.get("parse_ok"))
        lats = [r["latency_s"] for r in recs if r.get("ok") and r.get("latency_s")]
        toks = [r["usage"]["completion_tokens"] for r in recs
                if r.get("ok") and r.get("usage", {}).get("completion_tokens")]
        avg_lat = round(sum(lats) / len(lats), 2) if lats else None
        avg_tok = round(sum(toks) / len(toks)) if toks else None
        print(f"  {m}: ok={ok}/{rep['n_images']} parsed={parsed} "
              f"avg_latency={avg_lat}s avg_tokens={avg_tok}")
    if two and n_both:
        print(f"  work_type agreement (both parsed): {agree}/{n_both} "
              f"({100*agree/n_both:.0f}%)")
    return {"agree": agree, "n_both": n_both}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: summarize_vlm_compare.py <report.json> [labels.csv]")
        raise SystemExit(2)
    summarize(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
