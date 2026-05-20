#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

SUMMARY_PATH = Path("1.00_kfold_summary.json")
OUT_PATH = "robustness_bar_worst_foldF1_1.00_zoom_wavy.png"

def get_entry(data, tag: str):
    """키에 prefix가 붙어도(예: 1.00_pure_bert) 대응."""
    for k, v in data.items():
        if k.endswith(tag):
            return v
    return None

def worst_fold_f1(entry) -> float:
    """fold_scores에서 최소값(min)을 worst-fold F1으로 사용."""
    scores = entry.get("fold_scores") or []
    scores = [float(x) for x in scores]
    return float(min(scores)) if scores else float("nan")

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

    labels, pure_vals, crf_vals = [], [], []
    skipped = []
    all_vals = []

    for name, pure_tag, crf_tag in pairs:
        pure = get_entry(data, pure_tag)
        crf  = get_entry(data, crf_tag)

        if pure is None or crf is None:
            skipped.append(name)
            continue

        labels.append(name)
        p_val = worst_fold_f1(pure)
        c_val = worst_fold_f1(crf)
        
        pure_vals.append(p_val)
        crf_vals.append(c_val)
        
        all_vals.append(p_val)
        all_vals.append(c_val)

    if not labels:
        print("No comparable (Pure vs CRF) pairs found in JSON.")
        return

    # Y축 범위 자동 계산
    min_val = min(all_vals) if all_vals else 0.0
    YMIN = max(0.0, min_val - 0.05) # 최소값보다 0.05 낮게 설정
    YMAX = 1.00
    
    print(f"Calculated Y-Range: {YMIN:.2f} ~ {YMAX:.2f}")
    print(f"Worst F1 Scores: {all_vals}")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, pure_vals, width, label="Pure")
    ax.bar(x + width/2, crf_vals,  width, label="CRF")

    ax.set_title("Robustness (Worst-fold F1) @100% Data (Higher is Better)")
    ax.set_ylabel("Worst-fold F1 (min over 5 folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(YMIN, YMAX)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # ====== 물결표(~~~)로 '축 생략' 표시: y축 왼쪽 아래 ======
    # (물결표가 그래프를 가리거나 지저분해 보일 수 있어 주석 처리합니다. 필요시 주석 해제)
    # def add_wavy_axis_break(ax, x0=-0.02, x1=0.06, y=0.02, amp=0.015, waves=3.0, lw=2.0):
    #     xs = np.linspace(x0, x1, 200)
    #     ys = y + amp * np.sin(2*np.pi*waves*(xs - xs.min())/(xs.max()-xs.min()))
    #     verts = np.column_stack([xs, ys])
    #     codes = [MplPath.MOVETO] + [MplPath.LINETO]*(len(xs)-1)
    #     patch = PathPatch(
    #         MplPath(verts, codes),
    #         transform=ax.transAxes,
    #         clip_on=False,
    #         fill=False,
    #         lw=lw,
    #         color="black",
    #     )
    #     ax.add_patch(patch)

    # 겹물결로 좀 더 눈에 띄게
    # add_wavy_axis_break(ax, y=0.02,  amp=0.012, waves=3.0, lw=2.0)
    # add_wavy_axis_break(ax, y=0.045, amp=0.012, waves=3.0, lw=2.0)

    # (옵션) 막대 위에 수치 표시
    for xi, v in zip(x - width/2, pure_vals):
        ax.text(xi, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    for xi, v in zip(x + width/2, crf_vals):
        ax.text(xi, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    # (옵션) 스킵된 모델 안내(그림 안에 작게)
    if skipped:
        ax.text(
            0.99, 0.02,
            "Skipped (missing pair): " + ", ".join(skipped),
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