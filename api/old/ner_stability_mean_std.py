#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_PATH = Path("1.00_kfold_summary.json")
OUT_PATH = "robustness_bar_mean_std_1.00.png"

def get_entry(data, tag: str):
    """키에 prefix가 붙어도(예: 1.00_pure_bert) 대응."""
    for k, v in data.items():
        if k.endswith(tag):
            return v
    return None

def get_stats(entry):
    """
    Returns (mean, std) of fold scores.
    If 'mean_f1' and 'std_f1' are already in entry, use them.
    Otherwise calculate from 'fold_scores'.
    """
    if not entry:
        return 0.0, 0.0
        
    if "mean_f1" in entry and "std_f1" in entry:
        return float(entry["mean_f1"]), float(entry["std_f1"])
        
    scores = entry.get("fold_scores") or []
    scores = [float(x) for x in scores]
    if not scores:
        return 0.0, 0.0
        
    return float(np.mean(scores)), float(np.std(scores))

def main():
    if not SUMMARY_PATH.exists():
        print(f"[Error] Summary file not found: {SUMMARY_PATH}")
        return

    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    pairs = [
        ("mBERT",        "pure_bert",    "crf_bert"),
        ("KLUE-RoBERTa", "pure_roberta", "crf_roberta"),
        ("XLM-R",        "pure_xlm",     "crf_xlm"),
    ]

    labels = []
    pure_means, pure_stds = [], []
    crf_means, crf_stds = [], []
    skipped = []

    for name, pure_tag, crf_tag in pairs:
        pure = get_entry(data, pure_tag)
        crf  = get_entry(data, crf_tag)

        if pure is None or crf is None:
            skipped.append(name)
            continue

        labels.append(name)
        
        pm, ps = get_stats(pure)
        cm, cs = get_stats(crf)
        
        pure_means.append(pm)
        pure_stds.append(ps)
        
        crf_means.append(cm)
        crf_stds.append(cs)

    if not labels:
        print("No comparable (Pure vs CRF) pairs found in JSON.")
        return

    # Y축 범위 자동 계산 (Mean - Std의 최소값 기준)
    all_lows = [m - s for m, s in zip(pure_means + crf_means, pure_stds + crf_stds)]
    min_val = min(all_lows) if all_lows else 0.0
    YMIN = max(0.0, min_val - 0.05)
    YMAX = 1.00
    
    print(f"Calculated Y-Range: {YMIN:.2f} ~ {YMAX:.2f}")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Bars with Error Bars (capsize adds the little horizontal lines at the ends)
    rects1 = ax.bar(x - width/2, pure_means, width, yerr=pure_stds, label="Pure", capsize=5, alpha=0.9)
    rects2 = ax.bar(x + width/2, crf_means,  width, yerr=crf_stds,  label="CRF",  capsize=5, alpha=0.9)

    ax.set_title("Stability & Performance (Mean F1 ± Std Dev) @100% Data")
    ax.set_ylabel("Mean F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(YMIN, YMAX)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # 막대 위에 수치 표시 (Mean)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    if skipped:
        ax.text(
            0.99, 0.02,
            "Skipped: " + ", ".join(skipped),
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {OUT_PATH}")

if __name__ == "__main__":
    main()
