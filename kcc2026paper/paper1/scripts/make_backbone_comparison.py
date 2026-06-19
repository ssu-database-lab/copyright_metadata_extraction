"""3 backbone × 3 mode 비교 그래프 (paper 1 §24-22 backbone extension).

입력: 3 stamp 의 summary.json
  - KLUE      → paper/paper1/data/runs/20260427_143110/summary.json
  - mBERT     → paper/paper1/data/runs/20260430_paper1_mbert/summary.json
  - KoELECTRA → paper/paper1/data/runs/20260430_paper1_koelectra/summary.json

출력: paper/paper1/figures/G_backbone_comparison.png
  - X 축: M1 / M2 / M3
  - 각 mode 마다 3 bar (KLUE, mBERT, KoELECTRA)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

STAMPS = {
    "KLUE":      "20260427_143110",
    "mBERT":     "20260430_paper1_mbert",
    "KoELECTRA": "20260430_paper1_koelectra",
}
BB_COLORS = {"KLUE": "#1f77b4", "mBERT": "#ff7f0e", "KoELECTRA": "#2ca02c"}

CFG_TO_MODE = {
    "rule_m1_answer": "M1",
    "rule_m2_context": "M2",
    "rule_m3_negatives": "M3",
}
MODES = ["M1", "M2", "M3"]


def load_acc(stamp: str) -> dict:
    p = ROOT / "paper1" / "data" / "runs" / stamp / "summary.json"
    s = json.load(p.open())
    out = {}
    for r in s:
        m = CFG_TO_MODE.get(r["cfg_id"])
        if m and r.get("status") == "ok":
            out[m] = r["accuracy_overall"]
    return out


def main():
    data = {bb: load_acc(stamp) for bb, stamp in STAMPS.items()}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.25
    x_base = list(range(len(MODES)))

    for i, (bb, accs) in enumerate(data.items()):
        vals = [accs.get(m, 0) for m in MODES]
        xs = [x + (i - 1) * width for x in x_base]
        ax.bar(xs, vals, width=width, label=bb,
               color=BB_COLORS[bb], edgecolor="black")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x_base)
    ax.set_xticklabels(MODES)
    ax.set_xlabel("mode")
    ax.set_ylabel("Gold accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    out = ROOT / "paper1" / "figures" / "G_backbone_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved: {out}")

    # 콘솔 표
    print(f"\n{'mode':<6}", *(f"{bb:>11}" for bb in STAMPS))
    for m in MODES:
        print(f"{m:<6}", *(f"{data[bb].get(m, 0):>11.4f}" for bb in STAMPS))


if __name__ == "__main__":
    main()
