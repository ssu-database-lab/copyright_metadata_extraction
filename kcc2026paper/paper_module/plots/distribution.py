"""Distribution-comparison plots (B-series) for the 3-distribution evaluation paper.

Consumes the output of `scripts/eval_sweep_gold.py` — a directory containing one
JSON file per configuration with the shape produced by `_safe_eval`.

Adds plot codes B00~B06:

B00  3-분포 × 3-supervision 메인 결과 bar (E1-A/B/C × Distribution A/B/C)
B01  모델별 A vs B paired bar (같은 supervision 내에서 backbone/method 비교)
B02  라벨별 A vs B scatter (fragile vs robust label 식별)
B03  라벨별 A-B delta bar (정렬된 부호 있는 그림)
B04  Distribution B threshold curve (모델별 overlay)
B05  All-config × 50-label heatmap on Distribution B
B06  Supervision source × 라벨 heatmap on Distribution B (E1-A/B/C 비교)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paper_module.plots.research import _setup_mpl, _save, _shorten  # reuse styling
from module.parts.labels import (
    ALL_LABELS,
    COPYRIGHT_INFO_LABELS,
    AUTHOR_INFO_LABELS,
    RIGHTS_INFO_LABELS,
)

# ── color palette ─────────────────────────────────────────────────────────────
_A_COLOR = "#2563EB"   # blue — Distribution A
_B_COLOR = "#EA580C"   # orange — Distribution B
_C_COLOR = "#16A34A"   # green — Distribution C

_E1A_COLOR = "#2563EB"
_E1B_COLOR = "#EA580C"
_E1C_COLOR = "#DC2626"

_CATEGORY_ORDER = ("copyright_info", "author_info", "rights_info")


# ── IO ───────────────────────────────────────────────────────────────────────

def load_sweep(sweep_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all per-config JSONs from a gold_sweep_{stamp}/ directory.

    Returns: {config_label: eval_result_dict}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(sweep_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def _best_thr_accs(entry: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, float]]:
    """(best_thr, per_label_acc_at_best_thr) — defensive against missing fields."""
    if not entry.get("ok"):
        return None, {}
    best_thr = entry.get("best_thr")
    per_label = entry.get("per_label_acc", {})
    if best_thr is None or not per_label:
        return None, {}
    key = str(best_thr)
    label_acc: Dict[str, float] = {}
    for lbl, thr_acc in per_label.items():
        if key in thr_acc:
            label_acc[lbl] = float(thr_acc[key])
    return float(best_thr), label_acc


# ── B00: main 3-distribution matrix bar ──────────────────────────────────────

def plot_b00_main_matrix(
    out_path: Path,
    *,
    matrix: Dict[str, Dict[str, float]],
) -> None:
    """3-supervision × 3-distribution grouped bar.

    matrix shape: {supervision_label: {"A": float, "B": float, "C": float}}
    """
    plt = _setup_mpl()
    sups = list(matrix.keys())
    n_sup = len(sups)
    dists = ("A", "B", "C")
    colors = {"A": _A_COLOR, "B": _B_COLOR, "C": _C_COLOR}

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bar_w = 0.25
    import numpy as np
    x = np.arange(n_sup)
    for i, d in enumerate(dists):
        heights = [matrix[s].get(d, 0.0) for s in sups]
        bars = ax.bar(x + (i - 1) * bar_w, heights, bar_w,
                      label=f"Distribution {d}", color=colors[d], edgecolor="white", linewidth=0.7)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}" if h >= 0.01 else "—",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(sups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("B00 · Main Result: 3-Distribution × 3-Supervision (KLUE BERT Full-FT)")
    ax.legend(loc="upper right", frameon=True)
    _save(fig, out_path)


# ── B01: backbone/method paired bar on A vs B ────────────────────────────────

def plot_b01_backbone_paired(
    out_path: Path,
    *,
    configs: List[Tuple[str, float, float]],  # (label, A_acc, B_acc)
    title: str = "B01 · Backbone × Method (Distribution A vs B)",
) -> None:
    plt = _setup_mpl()
    import numpy as np
    if not configs:
        return
    labels = [c[0] for c in configs]
    a_vals = [c[1] for c in configs]
    b_vals = [c[2] for c in configs]
    x = np.arange(len(labels))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels) + 2.5), 4.8))
    ax.bar(x - bar_w / 2, a_vals, bar_w, label="Distribution A (template)", color=_A_COLOR, edgecolor="white", linewidth=0.7)
    ax.bar(x + bar_w / 2, b_vals, bar_w, label="Distribution B (gold)",      color=_B_COLOR, edgecolor="white", linewidth=0.7)
    for xi, v in zip(x - bar_w / 2, a_vals):
        ax.text(xi, v + 0.015, f"{v:.2f}" if v else "—", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + bar_w / 2, b_vals):
        ax.text(xi, v + 0.015, f"{v:.2f}" if v else "—", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (best threshold)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)
    _save(fig, out_path)


# ── B02: per-label A vs B scatter ────────────────────────────────────────────

def plot_b02_label_scatter(
    out_path: Path,
    *,
    a_by_label: Dict[str, float],
    b_by_label: Dict[str, float],
    title: str = "B02 · Per-Label A vs B (KLUE E1-A Full-FT)",
) -> None:
    plt = _setup_mpl()
    import numpy as np

    def _cat(lbl: str) -> str:
        if lbl in COPYRIGHT_INFO_LABELS: return "copyright_info"
        if lbl in AUTHOR_INFO_LABELS:    return "author_info"
        if lbl in RIGHTS_INFO_LABELS:    return "rights_info"
        return "other"

    cats = {
        "copyright_info": {"color": "#2563EB", "marker": "o"},
        "author_info":    {"color": "#16A34A", "marker": "s"},
        "rights_info":    {"color": "#EA580C", "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for cat, style in cats.items():
        xs, ys, names = [], [], []
        for lbl in ALL_LABELS:
            if _cat(lbl) != cat: continue
            if lbl in a_by_label and lbl in b_by_label:
                xs.append(a_by_label[lbl])
                ys.append(b_by_label[lbl])
                names.append(lbl)
        ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                   s=48, label=cat, alpha=0.85, edgecolor="white", linewidth=0.6)
        # annotate the outliers: big gap (A high, B low) or uniformly-poor
        for x, y, n in zip(xs, ys, names):
            if (x > 0.85 and y < 0.5) or (x < 0.4 and y < 0.4):
                ax.annotate(_shorten(n, 18), (x, y),
                            xytext=(4, 4), textcoords="offset points", fontsize=7, alpha=0.85)

    ax.plot([0, 1], [0, 1], "--", color="#94A3B8", linewidth=1.0, label="perfect transfer")
    ax.set_xlabel("Accuracy on Distribution A (template)")
    ax.set_ylabel("Accuracy on Distribution B (gold)")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.02, 1.03)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    _save(fig, out_path)


# ── B03: A→B delta per label ────────────────────────────────────────────────

def plot_b03_delta_bar(
    out_path: Path,
    *,
    a_by_label: Dict[str, float],
    b_by_label: Dict[str, float],
    title: str = "B03 · Per-Label Generalization Gap (A − B), KLUE E1-A",
    top_n: int = 50,
) -> None:
    plt = _setup_mpl()

    items = []
    for lbl in ALL_LABELS:
        a = a_by_label.get(lbl)
        b = b_by_label.get(lbl)
        if a is None or b is None: continue
        items.append((lbl, a - b))
    items.sort(key=lambda x: -x[1])  # biggest gap first
    items = items[:top_n]

    names = [x[0] for x in items]
    deltas = [x[1] for x in items]
    colors = [_B_COLOR if d > 0.05 else ("#94A3B8" if abs(d) <= 0.05 else _A_COLOR) for d in deltas]

    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.20 * len(names) + 1.5)))
    ax.barh(range(len(names)), deltas, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_xlabel("Accuracy Gap (A − B), positive = drops on B")
    ax.set_title(title)
    for i, d in enumerate(deltas):
        ax.text(d + (0.01 if d >= 0 else -0.01), i, f"{d:+.2f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7)
    _save(fig, out_path)


# ── B04: threshold curve (Distribution B) ──────────────────────────────────

def plot_b04_threshold_curve_B(
    out_path: Path,
    *,
    series: List[Tuple[str, Dict[str, float]]],  # [(label, {thr_str: acc})]
    title: str = "B04 · Distribution B — Accuracy vs Threshold",
) -> None:
    plt = _setup_mpl()
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (name, thr_acc) in enumerate(series):
        if not thr_acc: continue
        items = sorted(thr_acc.items(), key=lambda x: float(x[0]))
        xs = [float(t) for t, _ in items]
        ys = [v for _, v in items]
        ax.plot(xs, ys, marker="o", label=name, linewidth=1.4)
    ax.set_xlabel("Probability threshold τ")
    ax.set_ylabel("Mean accuracy (50-label)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=8)
    _save(fig, out_path)


# ── B05: all-config × 50-label heatmap on B ────────────────────────────────

def plot_b05_config_label_heatmap_B(
    out_path: Path,
    *,
    config_to_label_acc: Dict[str, Dict[str, float]],
    title: str = "B05 · All Configurations × 50 Labels (Distribution B, best thr)",
) -> None:
    plt = _setup_mpl()
    import numpy as np
    configs = list(config_to_label_acc.keys())
    labels = list(ALL_LABELS)
    if not configs:
        return
    mat = np.zeros((len(configs), len(labels)))
    for i, cfg in enumerate(configs):
        for j, lbl in enumerate(labels):
            mat[i, j] = config_to_label_acc[cfg].get(lbl, float("nan"))

    fig, ax = plt.subplots(figsize=(max(12.0, 0.22 * len(labels) + 2), 0.35 * len(configs) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Accuracy", fontsize=8)
    _save(fig, out_path)


# ── B06: supervision × label heatmap on B ──────────────────────────────────

def plot_b06_supervision_label_heatmap_B(
    out_path: Path,
    *,
    sup_to_label_acc: Dict[str, Dict[str, float]],
    title: str = "B06 · Supervision Source × Label (Distribution B, KLUE Full-FT)",
) -> None:
    plt = _setup_mpl()
    import numpy as np
    sups = list(sup_to_label_acc.keys())
    labels = list(ALL_LABELS)
    mat = np.zeros((len(sups), len(labels)))
    for i, s in enumerate(sups):
        for j, lbl in enumerate(labels):
            mat[i, j] = sup_to_label_acc[s].get(lbl, float("nan"))

    fig, ax = plt.subplots(figsize=(max(13.0, 0.24 * len(labels) + 2), 0.55 * len(sups) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(sups)))
    ax.set_yticklabels(sups, fontsize=9)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Accuracy", fontsize=8)
    _save(fig, out_path)


# ── driver: consume a sweep directory and emit all B-plots ─────────────────

def generate_all_b_plots(
    sweep_dir: Path,
    out_dir: Path,
    *,
    a_by_label_e1a: Optional[Dict[str, float]] = None,
    a_by_label_e1b: Optional[Dict[str, float]] = None,
    a_by_label_e1c: Optional[Dict[str, float]] = None,
) -> List[Path]:
    """Generate every B-plot we can from a sweep directory.

    a_by_label_e1{a,b,c}: per-label Distribution A accuracies (from prior runs
    or from new integrated_full_klue's A-set eval). When None,
    B00/B02/B03/B06 use gold-only view (A column blank) — still useful but less rich.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep(sweep_dir)
    written: List[Path] = []

    # collect (config, best_thr, best_acc, per_label_at_best)
    config_to_label_acc: Dict[str, Dict[str, float]] = {}
    config_best_b: Dict[str, float] = {}
    config_thr_curve: Dict[str, Dict[str, float]] = {}  # Config → {thr_str: acc}
    for cfg, entry in sweep.items():
        if not entry.get("ok"): continue
        _, label_acc = _best_thr_accs(entry)
        config_to_label_acc[cfg] = label_acc
        config_best_b[cfg] = float(entry.get("best_acc", 0.0))
        config_thr_curve[cfg] = {k: float(v) for k, v in entry.get("acc_by_thr", {}).items()}

    # --- B00 main matrix (requires the three E1 rows on A, B, C) -----------
    B_klue_e1a = config_best_b.get("integrated_full_klue", 0.0)
    B_klue_e1b = config_best_b.get("e1b_klue", 0.0)
    B_klue_e1c = config_best_b.get("e1c_klue", 0.0)
    # Legacy A/C matrix values from the 3-distribution experiment notes.
    matrix = {
        "E1-A (rule silver)":   {"A": 0.9900, "B": B_klue_e1a, "C": 0.474},
        "E1-B (LLM silver)":    {"A": 0.4635, "B": B_klue_e1b, "C": 0.167},
        "E1-C (LLM+filter)":    {"A": 0.4560, "B": B_klue_e1c, "C": 0.000},
    }
    p = out_dir / "B00_main_matrix.png"
    plot_b00_main_matrix(p, matrix=matrix); written.append(p)

    # --- B01 backbone × method bar (A hardcoded from §11-1 where we have it) ---
    A_NUM = {
        "integrated_full_klue":      0.9900,
        "integrated_full_mbert":     0.9605,
        "integrated_full_koelectra": 0.9866,
        "integrated_full_deberta":   0.9682,
        "integrated_lora_klue":      0.9654,
        "seperated_mbert_full":      0.9696,
        "seperated_mbert_lora":      0.8850,
        "seperated_distil_full":     0.9711,
        "seperated_distil_lora":     0.6132,
        "seperated_rf":              0.4808,
        "seperated_lr":              0.6568,
        "e1b_klue":                  0.4635,
        "e1c_klue":                  0.4560,
    }
    cfg_items = [(cfg, A_NUM.get(cfg, 0.0), config_best_b[cfg]) for cfg in config_best_b]
    p = out_dir / "B01_backbone_paired.png"
    plot_b01_backbone_paired(p, configs=cfg_items); written.append(p)

    # --- B02/B03 per-label scatter & delta (KLUE E1-A, uses a_by_label_e1a if given)
    klue_b = config_to_label_acc.get("integrated_full_klue", {})
    if a_by_label_e1a and klue_b:
        p = out_dir / "B02_label_scatter.png"
        plot_b02_label_scatter(p, a_by_label=a_by_label_e1a, b_by_label=klue_b); written.append(p)
        p = out_dir / "B03_delta_bar.png"
        plot_b03_delta_bar(p, a_by_label=a_by_label_e1a, b_by_label=klue_b); written.append(p)

    # --- B04 threshold curve (Distribution B only, multi-config overlay) ----
    series = [(cfg, config_thr_curve[cfg]) for cfg in sorted(config_thr_curve.keys())]
    p = out_dir / "B04_threshold_curve_B.png"
    plot_b04_threshold_curve_B(p, series=series); written.append(p)

    # --- B05 all configs × 50 labels heatmap on B -------------------------
    p = out_dir / "B05_config_label_heatmap_B.png"
    plot_b05_config_label_heatmap_B(p, config_to_label_acc=config_to_label_acc); written.append(p)

    # --- B06 supervision × label heatmap on B (E1-A/B/C) -------------------
    sup_mat = {}
    if "integrated_full_klue" in config_to_label_acc:
        sup_mat["E1-A"] = config_to_label_acc["integrated_full_klue"]
    if "e1b_klue" in config_to_label_acc:
        sup_mat["E1-B"] = config_to_label_acc["e1b_klue"]
    if "e1c_klue" in config_to_label_acc:
        sup_mat["E1-C"] = config_to_label_acc["e1c_klue"]
    if sup_mat:
        p = out_dir / "B06_supervision_label_heatmap_B.png"
        plot_b06_supervision_label_heatmap_B(p, sup_to_label_acc=sup_mat); written.append(p)

    return written
