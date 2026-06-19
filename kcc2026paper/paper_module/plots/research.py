"""
Publication-quality NER evaluation plots.

Standard NLP/AI paper metrics:
  Train  : loss curves · F1 progression · per-label P/R/F1 · confusion matrix
           · PR curve · FP/FN/TP analysis · label distribution
  Predict: threshold-accuracy · per-label accuracy bar · category breakdown
           · label-threshold heatmap
  Compare: integrated vs seperated ensemble · LoRA vs full fine-tuning

seperated 모드 = 3개 전문 모델 앙상블
  - copyright_info model  → cp_* labels
  - author_info model     → ch_co/ja/nr_* labels
  - rights_info model     → ri_* labels
  예측 시 3개 모델 각각 자기 분야를 예측하고 결과를 하나로 병합.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── colour palette (publication-friendly) ─────────────────────────────────────
_C_BLUE   = "#2563EB"
_C_RED    = "#DC2626"
_C_GREEN  = "#16A34A"
_C_ORANGE = "#EA580C"
_C_PURPLE = "#7C3AED"
_C_TEAL   = "#0891B2"
_C_GREY   = "#64748B"
_C_AMBER  = "#D97706"

CATEGORY_COLORS = {
    "copyright_info": _C_BLUE,
    "author_info":    _C_GREEN,
    "rights_info":    _C_ORANGE,
}

# ── matplotlib init ───────────────────────────────────────────────────────────

def _setup_mpl() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#E2E8F0",
        "grid.linewidth":    0.7,
        "grid.linestyle":    "--",
    })
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    return plt


def _save(fig: Any, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        log.info("saved: %s", path)
    except Exception as e:
        log.warning("save failed %s: %s", path, e)
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass


def _shorten(s: str, n: int = 20) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ── data extraction helpers ───────────────────────────────────────────────────

def _unpack(eval_info: Optional[Dict], debug_metrics: Optional[Dict]) -> Dict:
    """Normalize nested eval_info / debug_metrics structures.

    Handles two main sources:
    - ner_train_metrics.json : {full_metrics_summary, log_history_tail, data_split, ...}
    - debug_metrics.json     : {eval_info: {full_metrics_summary, ...}, full_log_history}
    - eval_info from api     : {evaluation: {full_metrics_summary, ...}}
    """
    ei = eval_info or {}
    dm = debug_metrics or {}

    # ner_train_metrics.json is passed as debug_metrics
    # it has full_metrics_summary at top level
    if "full_metrics_summary" in dm:
        full = dm["full_metrics_summary"]
        lh   = dm.get("log_history_full") or dm.get("full_log_history") or dm.get("log_history_tail") or []
        ds   = dm.get("data_split", {})
    else:
        # legacy debug_metrics.json or eval_info from api
        _dm_ei = dm.get("eval_info", {})
        full   = (
            ei.get("full_metrics_summary")
            or _dm_ei.get("full_metrics_summary")
            or ei
        )
        lh = (
            dm.get("log_history_full")
            or dm.get("full_log_history")
            or dm.get("log_history")
            or _dm_ei.get("log_history_tail")
            or ei.get("log_history_tail")
            or []
        )
        ds = ei.get("data_split", _dm_ei.get("data_split", {}))

    return {
        "report":   full.get("token_classification_report", {}),
        "span":     full.get("span_classification_report", {}),
        "cm":       full.get("confusion_matrix", {}),
        "fa":       full.get("false_alarm_stats", {}),
        "fp_lbl":   full.get("fp_per_label", {}),
        "fn_lbl":   full.get("fn_per_label", {}),
        "tp_lbl":   full.get("tp_per_label", {}),
        "conf":     full.get("per_label_confidence", {}),
        "log_hist": lh,
        "ds":       ds,
    }


def _load_trainer_state(model_path_str: Optional[str],
                        model_name: Optional[str]) -> List[Dict]:
    """Load log_history from the best/latest checkpoint trainer_state.json."""
    if not model_path_str or not model_name:
        return []
    mdname = model_name.replace("/", "--")
    adapter_dir = Path(model_path_str) / mdname / "adapter"
    if not adapter_dir.is_absolute():
        from pathlib import Path as P
        adapter_dir = P(__file__).parent.parent.parent / adapter_dir
    checkpoints = sorted(adapter_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    # prefer the last checkpoint (best/latest)
    for ckpt in reversed(checkpoints):
        ts_path = ckpt / "trainer_state.json"
        if ts_path.exists():
            try:
                return json.loads(ts_path.read_text(encoding="utf-8")).get("log_history", [])
            except Exception:
                pass
    return []


def _split_log(log_hist: List[Dict]):
    """Split log_history into train steps and eval epochs."""
    train = [e for e in log_hist if "loss" in e and "eval_loss" not in e]
    evals = [e for e in log_hist if "eval_f1" in e or "eval_loss" in e]
    return train, evals


def _label_metrics(report: Dict) -> List[Tuple[str, float, float, float]]:
    """Extract (label, precision, recall, f1) for entity labels (skip macro/weighted/O)."""
    skip = {"macro avg", "weighted avg", "accuracy", "O"}
    rows = []
    for lbl, vals in report.items():
        if lbl in skip or not isinstance(vals, dict):
            continue
        p = vals.get("precision", 0.0)
        r = vals.get("recall",    0.0)
        f = vals.get("f1-score",  0.0)
        rows.append((lbl, p, r, f))
    return sorted(rows, key=lambda x: x[3])   # sort by F1 asc


# ── category mapping (50 labels → 3 groups) ───────────────────────────────────

_COPYRIGHT_LABELS = {
    "copyright_uci", "copyright_num", "copyright_kotitle", "copyright_entitle",
    "copyright_idnum", "copyright_type", "copyright_status", "copyright_quantity",
    "copyright_description", "copyright_Pname", "copyright_url", "copyright_Keyword",
    "copyright_language", "date", "copyright_con_status", "copyright_id",
}
_AUTHOR_LABELS = {
    f"{role}_{base}"
    for role in ("ch_co", "ch_ja", "ch_nr")
    for base in ("address", "name", "company", "department", "email", "phone", "position")
}
_RIGHTS_LABELS = {
    "ri_info", "ri_data", "ri_cpcheck", "ri_uncopyright", "ri_workhire",
    "ri_consent_type", "ri_law_reference", "ri_contract_type", "ri_money",
    "ri_copyright", "ri_jch_conset", "ri_period", "ri_portrait",
}

def _category_of(label: str) -> str:
    # strip B-/I- prefix if present
    bare = label[2:] if label.startswith(("B-", "I-")) else label
    if bare in _COPYRIGHT_LABELS: return "copyright_info"
    if bare in _AUTHOR_LABELS:    return "author_info"
    if bare in _RIGHTS_LABELS:    return "rights_info"
    return "other"


def _draw_category_separators(ax: Any, ordered_labels: List[str], axis: str = "y") -> None:
    """Draw dashed separator lines between label categories on a heatmap axis."""
    cat_order = ["copyright_info", "author_info", "rights_info", "other"]
    # Build boundaries: list of (start, end, cat_name) in order
    boundaries: List[Tuple[int, int, str]] = []
    cur_cat = _category_of(ordered_labels[0]) if ordered_labels else None
    start = 0
    for i, lbl in enumerate(ordered_labels):
        c = _category_of(lbl)
        if c != cur_cat:
            boundaries.append((start, i, cur_cat))
            cur_cat = c
            start = i
    if cur_cat is not None:
        boundaries.append((start, len(ordered_labels), cur_cat))

    for start, end, _cat in boundaries:
        if start == 0:
            continue
        pos = start - 0.5
        if axis == "y":
            ax.axhline(pos, color="white", linewidth=2.0, linestyle="-")
        else:
            ax.axvline(pos, color="white", linewidth=2.0, linestyle="-")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN PLOTS  (out_dir/train/)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_dir(out_dir: Path) -> Path:
    """out_dir 자체를 학습 플롯 저장 디렉터리로 사용 (호출자가 경로 완성)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_loss_curves(plt, out_dir: Path, log_hist: List[Dict]) -> None:
    """T01 — Train / validation loss curves over epochs."""
    _, evals = _split_log(log_hist)
    train_steps = [e for e in log_hist if "loss" in e and "eval_loss" not in e]
    if not evals and not train_steps:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # left: train loss (step-level)
    if train_steps:
        steps  = [e.get("step", i) for i, e in enumerate(train_steps)]
        losses = [e["loss"] for e in train_steps]
        axes[0].plot(steps, losses, color=_C_BLUE, alpha=0.4, linewidth=0.8, label="Train loss (step)")
        # smoothed
        if len(losses) > 10:
            import statistics
            w = max(1, len(losses) // 40)
            smoothed = [statistics.mean(losses[max(0, i-w):i+1]) for i in range(len(losses))]
            axes[0].plot(steps, smoothed, color=_C_BLUE, linewidth=1.8, label="Train loss (smooth)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()

    # right: eval loss + eval F1 (epoch-level)
    if evals:
        epochs     = [e.get("epoch", i+1) for i, e in enumerate(evals)]
        eval_loss  = [e.get("eval_loss") for e in evals]
        eval_f1    = [e.get("eval_f1")   for e in evals]

        ax2 = axes[1]
        ax3 = ax2.twinx()
        if any(v is not None for v in eval_loss):
            vs = [v for v in eval_loss if v is not None]
            ep = [epochs[i] for i, v in enumerate(eval_loss) if v is not None]
            ax2.plot(ep, vs, color=_C_RED, linewidth=2, marker="o", markersize=4, label="Val loss")
        if any(v is not None for v in eval_f1):
            vs = [v for v in eval_f1 if v is not None]
            ep = [epochs[i] for i, v in enumerate(eval_f1) if v is not None]
            ax3.plot(ep, vs, color=_C_GREEN, linewidth=2, marker="s", markersize=4,
                     linestyle="--", label="Val F1")
            ax3.set_ylim(0, 1.05)
            ax3.set_ylabel("F1 Score", color=_C_GREEN)
            ax3.tick_params(axis="y", colors=_C_GREEN)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss", color=_C_RED)
        ax2.tick_params(axis="y", colors=_C_RED)
        ax2.set_title("Validation Loss & F1")

        # combined legend
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper right")

    fig.suptitle("Learning Curves", fontsize=13, fontweight="bold", y=1.01)
    _save(fig, _train_dir(out_dir) / "T01_loss_curves.png")


def _plot_f1_progression(plt, out_dir: Path, log_hist: List[Dict]) -> None:
    """T02 — Validation F1 convergence over epochs."""
    evals = [e for e in log_hist if "eval_f1" in e]
    if len(evals) < 2:
        return

    epochs = [e.get("epoch", i+1) for i, e in enumerate(evals)]
    f1s    = [e["eval_f1"] for e in evals]
    accs   = [e.get("eval_accuracy") for e in evals]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, f1s, color=_C_BLUE, linewidth=2.2, marker="o", markersize=5, label="Eval F1")
    if any(v is not None for v in accs):
        acc_vals = [v for v in accs if v is not None]
        acc_ep   = [epochs[i] for i, v in enumerate(accs) if v is not None]
        ax.plot(acc_ep, acc_vals, color=_C_ORANGE, linewidth=1.6, marker="^",
                markersize=4, linestyle="--", label="Eval Accuracy")

    best_idx = f1s.index(max(f1s))
    ax.axvline(epochs[best_idx], color=_C_RED, linewidth=1, linestyle=":", alpha=0.7)
    ax.annotate(f"Best F1={f1s[best_idx]:.4f}",
                xy=(epochs[best_idx], f1s[best_idx]),
                xytext=(+6, -12), textcoords="offset points",
                fontsize=7.5, color=_C_RED)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation F1 Convergence")
    ax.legend()
    _save(fig, _train_dir(out_dir) / "T02_f1_progression.png")


def _plot_prf_overall(plt, out_dir: Path, report: Dict) -> None:
    """T03 — Category-level P/R/F1 grouped bar (copyright / author / rights / overall)."""
    if not report:
        return

    import numpy as np

    rows = _label_metrics(report)   # [(label, P, R, F1), ...]

    # Aggregate per category
    cat_order   = ["copyright_info", "author_info", "rights_info"]
    cat_display = {"copyright_info": "Copyright", "author_info": "Author", "rights_info": "Rights"}
    cat_buckets: Dict[str, Dict[str, List[float]]] = {c: {"p": [], "r": [], "f": []} for c in cat_order}
    for lbl, p, r, f in rows:
        c = _category_of(lbl)
        if c in cat_buckets:
            cat_buckets[c]["p"].append(p)
            cat_buckets[c]["r"].append(r)
            cat_buckets[c]["f"].append(f)

    # Include overall macro avg
    macro = report.get("macro avg", {})
    group_labels = [cat_display[c] for c in cat_order] + ["Macro Avg"]
    group_p = [
        *[sum(cat_buckets[c]["p"]) / len(cat_buckets[c]["p"]) if cat_buckets[c]["p"] else 0 for c in cat_order],
        macro.get("precision", 0),
    ]
    group_r = [
        *[sum(cat_buckets[c]["r"]) / len(cat_buckets[c]["r"]) if cat_buckets[c]["r"] else 0 for c in cat_order],
        macro.get("recall", 0),
    ]
    group_f = [
        *[sum(cat_buckets[c]["f"]) / len(cat_buckets[c]["f"]) if cat_buckets[c]["f"] else 0 for c in cat_order],
        macro.get("f1-score", 0),
    ]

    x     = np.arange(len(group_labels))
    width = 0.25
    bar_colors = [CATEGORY_COLORS["copyright_info"], CATEGORY_COLORS["author_info"],
                  CATEGORY_COLORS["rights_info"], _C_GREY]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (metric, vals, marker_c) in enumerate(zip(
        ["Precision", "Recall", "F1"],
        [group_p, group_r, group_f],
        [_C_BLUE, _C_GREEN, _C_RED],
    )):
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.9,
                      label=metric, color=marker_c, alpha=0.82, edgecolor="white")
        for j, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=bar_colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Category-level Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _save(fig, _train_dir(out_dir) / "T03_prf_overall.png")


def _plot_per_label_f1(plt, out_dir: Path, report: Dict) -> None:
    """T04 — Per-label F1 bar grouped by category (copyright / author / rights)."""
    rows = _label_metrics(report)
    if not rows:
        return

    # Group by category, within each category sort by F1 descending
    import numpy as np
    cat_order = ["copyright_info", "author_info", "rights_info", "other"]
    cat_map = {r[0]: _category_of(r[0]) for r in rows}

    grouped: Dict[str, list] = {c: [] for c in cat_order}
    for r in rows:
        grouped[cat_map[r[0]]].append(r)
    for c in cat_order:
        grouped[c].sort(key=lambda x: x[3], reverse=True)

    ordered_rows = []
    cat_boundaries: List[Tuple[int, int, str]] = []   # (start_idx, end_idx, cat_name)
    for c in cat_order:
        if not grouped[c]:
            continue
        start = len(ordered_rows)
        ordered_rows.extend(grouped[c])
        cat_boundaries.append((start, len(ordered_rows), c))

    raw_labels = [r[0] for r in ordered_rows]
    labels     = [_shorten(r[0], 22) for r in ordered_rows]
    ps         = [r[1] for r in ordered_rows]
    rs         = [r[2] for r in ordered_rows]
    fs         = [r[3] for r in ordered_rows]
    bar_colors = [CATEGORY_COLORS.get(cat_map[l], _C_GREY) for l in raw_labels]

    y     = np.arange(len(labels))
    h     = 0.22
    fig_h = max(6, len(labels) * 0.36)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    # P/R/F1 bars — same category color, different alpha
    for i, (p, r, f, c) in enumerate(zip(ps, rs, fs, bar_colors)):
        ax.barh(i + h,   p, h * 0.9, color=c, alpha=0.40)
        ax.barh(i,       r, h * 0.9, color=c, alpha=0.65)
        ax.barh(i - h,   f, h * 0.9, color=c, alpha=0.95)
        # F1 value annotation
        ax.text(f + 0.01, i - h, f"{f:.2f}", va="center", fontsize=6.5,
                fontweight="bold", color=c)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("Per-Label Precision / Recall / F1  (grouped by category)", fontsize=12, fontweight="bold")

    # Category separator lines + right-side labels
    for start, end, cat_name in cat_boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color="#CBD5E1", linewidth=1.5, linestyle="--")
        mid = (start + end - 1) / 2
        ax.annotate(
            cat_name.replace("_", " ").title(),
            xy=(1.17, mid), xycoords=("axes fraction", "data"),
            fontsize=7, color=CATEGORY_COLORS.get(cat_name, _C_GREY),
            fontweight="bold", ha="right", va="center", rotation=0,
        )

    # legend: P / R / F1 alpha explanation
    for lbl, alpha in [("Precision", 0.40), ("Recall", 0.65), ("F1", 0.95)]:
        ax.barh([], [], color=_C_GREY, alpha=alpha, label=lbl)
    ax.legend(loc="lower right", fontsize=8)

    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    _save(fig, _train_dir(out_dir) / "T04_per_label_prf.png")


def _plot_confusion_matrix(plt, out_dir: Path, cm_data: Dict) -> None:
    """T05 — Normalized confusion matrix (top-N labels by frequency)."""
    matrix = cm_data.get("matrix")
    lbls   = cm_data.get("labels")
    if not matrix or not lbls:
        return

    import numpy as np
    mat = np.array(matrix, dtype=float)

    # keep top 20 most frequent (by row sum) to stay readable
    row_sums = mat.sum(axis=1)
    top_idx  = np.argsort(row_sums)[-20:][::-1]
    mat      = mat[np.ix_(top_idx, top_idx)]
    lbls     = [_shorten(lbls[i], 14) for i in top_idx]

    # normalize
    row_s = mat.sum(axis=1, keepdims=True)
    norm  = np.divide(mat, np.where(row_s == 0, 1, row_s))

    fig_s = max(6, len(lbls) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_s, fig_s * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")

    ax.set_xticks(range(len(lbls)))
    ax.set_yticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(lbls, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix (normalized, top-20 labels)")

    for i in range(len(lbls)):
        for j in range(len(lbls)):
            v = norm[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        color="white" if (v < 0.35 or v > 0.75) else "black")
    _save(fig, _train_dir(out_dir) / "T05_confusion_matrix.png")


def _plot_pr_curve(plt, out_dir: Path, report: Dict) -> None:
    """T06 — Precision vs Recall scatter per label (P-R plot)."""
    rows = _label_metrics(report)
    if not rows:
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 5))

    # Color by category
    cat_to_color = {
        "copyright_info": _C_BLUE,
        "author_info":    _C_GREEN,
        "rights_info":    _C_ORANGE,
        "other":          _C_GREY,
    }

    for lbl, p, r, f in rows:
        cat   = _category_of(lbl)
        color = cat_to_color[cat]
        ax.scatter(r, p, color=color, s=50 + f * 100, alpha=0.75,
                   edgecolors="white", linewidths=0.4)

    # iso-F1 contours
    f_scores = [0.2, 0.4, 0.6, 0.8, 0.9]
    rc = np.linspace(0.01, 1.0, 200)
    for fs in f_scores:
        pc = fs * rc / (2 * rc - fs + 1e-9)
        pc = np.where((pc >= 0) & (pc <= 1), pc, np.nan)
        ax.plot(rc, pc, color=_C_GREY, linewidth=0.7, linestyle=":", alpha=0.5)
        valid = ~np.isnan(pc)
        if valid.any():
            idx = valid.nonzero()[0][len(valid.nonzero()[0])//2]
            ax.annotate(f"F1={fs}", xy=(rc[idx], pc[idx]),
                        fontsize=6, color=_C_GREY, alpha=0.7)

    # legend for categories
    for cat, color in cat_to_color.items():
        if cat == "other":
            continue
        ax.scatter([], [], color=color, s=60, label=cat.replace("_", " ").title())

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Plot per Label\n(bubble size ∝ F1)")
    ax.legend(loc="lower left", framealpha=0.8)
    _save(fig, _train_dir(out_dir) / "T06_pr_plot.png")


def _plot_fp_fn_analysis(plt, out_dir: Path,
                         fp_lbl: Dict, fn_lbl: Dict, tp_lbl: Dict) -> None:
    """T07 — TP / FP / FN per label (error analysis bar chart)."""
    labels = sorted(set(list(fp_lbl) + list(fn_lbl) + list(tp_lbl)))
    if not labels:
        return

    # top 25 by error count
    labels = sorted(labels, key=lambda l: fp_lbl.get(l, 0) + fn_lbl.get(l, 0), reverse=True)[:25]

    import numpy as np
    y     = np.arange(len(labels))
    h     = 0.25
    fig_h = max(4, len(labels) * 0.32)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    tps = [tp_lbl.get(l, 0) for l in labels]
    fps = [fp_lbl.get(l, 0) for l in labels]
    fns = [fn_lbl.get(l, 0) for l in labels]

    ax.barh(y + h,   tps, h, label="TP", color=_C_GREEN,  alpha=0.85)
    ax.barh(y,       fps, h, label="FP", color=_C_ORANGE, alpha=0.85)
    ax.barh(y - h,   fns, h, label="FN", color=_C_RED,    alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels([_shorten(l) for l in labels], fontsize=7)
    ax.set_xlabel("Count")
    ax.set_title("TP / FP / FN per Label  (top-25 by error count)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _save(fig, _train_dir(out_dir) / "T07_fp_fn_analysis.png")


def _plot_label_distribution(plt, out_dir: Path, train_dir: Optional[Path]) -> None:
    """T08 — Training data label distribution (entity count per label)."""
    if train_dir is None or not train_dir.exists():
        return

    counts: Dict[str, int] = {}
    for jf in train_dir.glob("*.jsonl"):
        label = jf.stem
        n = 0
        try:
            for line in jf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    n += sum(1 for t in rec.get("labels", []) if t.startswith("B-"))
        except Exception:
            pass
        if n:
            counts[label] = n

    if not counts:
        return

    import numpy as np
    items  = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [_shorten(k) for k, _ in items]
    vals   = [v for _, v in items]
    colors = [CATEGORY_COLORS.get(_category_of(k), _C_GREY) for k, _ in items]

    fig_h = max(4, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Entity Count")
    ax.set_title("Training Data — Entity Distribution per Label")

    # category legend
    for cat, color in CATEGORY_COLORS.items():
        ax.barh([], [], color=color, label=cat.replace("_", " ").title())
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _save(fig, _train_dir(out_dir) / "T08_label_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT PLOTS  (저장 경로는 호출자가 out_dir로 완성해서 전달)
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_dir(out_dir: Path) -> Path:
    """out_dir 자체를 예측 플롯 저장 디렉터리로 사용 (호출자가 경로 완성)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_threshold_accuracy(plt, out_dir: Path,
                             acc_by_thr: Dict[float, float],
                             mode: str, method: str) -> None:
    """P01 — Average accuracy vs confidence threshold (line chart)."""
    if not acc_by_thr:
        return
    thrs = sorted(acc_by_thr.keys())
    accs = [acc_by_thr[t] for t in thrs]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(thrs, accs, color=_C_BLUE, linewidth=2.2, marker="o", markersize=6)

    best_t = thrs[accs.index(max(accs))]
    best_a = max(accs)
    ax.axvline(best_t, color=_C_RED, linewidth=1.2, linestyle=":", alpha=0.7)
    ax.annotate(f"Best: {best_a:.4f}\n@ thr={best_t}",
                xy=(best_t, best_a),
                xytext=(8, -20), textcoords="offset points",
                fontsize=8, color=_C_RED,
                arrowprops=dict(arrowstyle="->", color=_C_RED, lw=1))

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy vs Threshold  [{mode} / {method}]")
    _save(fig, _predict_dir(out_dir) / "P01_threshold_accuracy.png")


def _plot_per_label_accuracy(plt, out_dir: Path,
                             per_label_acc: Dict[str, Dict[float, float]],
                             mode: str) -> None:
    """P02 — Per-label accuracy at best threshold (horizontal bar, category-colored)."""
    if not per_label_acc:
        return

    # best threshold = highest avg accuracy across labels
    all_thrs = sorted({t for la in per_label_acc.values() for t in la})
    if not all_thrs:
        return
    avg_by_thr = {t: sum(la.get(t, 0) for la in per_label_acc.values()) / len(per_label_acc)
                  for t in all_thrs}
    best_thr = max(avg_by_thr, key=avg_by_thr.get)

    items  = sorted(per_label_acc.items(), key=lambda x: x[1].get(best_thr, 0))
    labels = [_shorten(k) for k, _ in items]
    vals   = [v.get(best_thr, 0) for _, v in items]
    colors = [CATEGORY_COLORS.get(_category_of(k), _C_GREY) for k, _ in items]

    fig_h = max(4, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    bars = ax.barh(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(min(val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=6.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Per-Label Accuracy @ thr={best_thr}  [{mode}]")

    for cat, color in CATEGORY_COLORS.items():
        ax.barh([], [], color=color, label=cat.replace("_", " ").title())
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _save(fig, _predict_dir(out_dir) / "P02_per_label_accuracy.png")


def _plot_category_accuracy(plt, out_dir: Path,
                            per_label_acc: Dict[str, Dict[float, float]],
                            mode: str) -> None:
    """P03 — Accuracy by category (copyright / author / rights) per threshold.

    For seperated mode: visualizes each specialized model's contribution.
    """
    if not per_label_acc:
        return
    all_thrs = sorted({t for la in per_label_acc.values() for t in la})
    if not all_thrs:
        return

    cats = ["copyright_info", "author_info", "rights_info"]
    cat_labels_map = {
        "copyright_info": _COPYRIGHT_LABELS,
        "author_info":    _AUTHOR_LABELS,
        "rights_info":    _RIGHTS_LABELS,
    }
    cat_display = {
        "copyright_info": "Copyright Info\n(16 labels)",
        "author_info":    "Author Info\n(21 labels)",
        "rights_info":    "Rights Info\n(13 labels)",
    }

    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 4))

    for cat in cats:
        cat_set = cat_labels_map[cat]
        cat_data = {k: v for k, v in per_label_acc.items() if k in cat_set}
        if not cat_data:
            continue
        avg_by_t = [
            sum(la.get(t, 0) for la in cat_data.values()) / len(cat_data)
            for t in all_thrs
        ]
        ax.plot(all_thrs, avg_by_t,
                color=CATEGORY_COLORS[cat], linewidth=2.2, marker="o", markersize=5,
                label=cat_display[cat])

    # overall
    overall = [sum(la.get(t, 0) for la in per_label_acc.values()) / len(per_label_acc)
               for t in all_thrs]
    ax.plot(all_thrs, overall,
            color="black", linewidth=2.0, linestyle="--", marker="^", markersize=5,
            label="Overall (50 labels)")

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    title_suffix = " [Ensemble: 3 Specialized Models]" if mode == "seperated" else ""
    ax.set_title(f"Accuracy by Category{title_suffix}  [{mode}]")
    ax.legend(loc="upper right")

    if mode == "seperated":
        ax.text(0.01, 0.03,
                "Seperated: each model predicts its own domain, results merged",
                transform=ax.transAxes, fontsize=7, color=_C_GREY, style="italic")

    _save(fig, _predict_dir(out_dir) / "P03_category_accuracy.png")


def _plot_accuracy_heatmap(plt, out_dir: Path,
                           per_label_acc: Dict[str, Dict[float, float]],
                           mode: str) -> None:
    """P04 — Label × Threshold accuracy heatmap."""
    if not per_label_acc:
        return
    all_thrs = sorted({t for la in per_label_acc.values() for t in la})
    if not all_thrs:
        return

    # group by category for ordered display
    ordered_labels = []
    for cat_set in [_COPYRIGHT_LABELS, _AUTHOR_LABELS, _RIGHTS_LABELS]:
        ordered_labels.extend(
            sorted(l for l in per_label_acc if l in cat_set)
        )
    ordered_labels.extend(l for l in per_label_acc if l not in
                          (_COPYRIGHT_LABELS | _AUTHOR_LABELS | _RIGHTS_LABELS))

    import numpy as np
    mat = np.array([[per_label_acc[l].get(t, 0.0) for t in all_thrs]
                    for l in ordered_labels])

    fig_h = max(8, len(ordered_labels) * 0.42)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Accuracy")
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(all_thrs)))
    ax.set_xticklabels([f"{t:.2f}" for t in all_thrs], fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_yticklabels([_shorten(l, 22) for l in ordered_labels], fontsize=8)
    ax.set_xlabel("Confidence Threshold", fontsize=10)
    ax.set_title(f"Per-Label Accuracy Heatmap  [{mode}]", fontsize=12, fontweight="bold", pad=10)

    # text annotations — white on dark cells (v<0.35 or v>0.75), black on mid
    for i in range(len(ordered_labels)):
        for j in range(len(all_thrs)):
            v = mat[i, j]
            txt_color = "white" if (v < 0.35 or v > 0.75) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=txt_color)

    # category separators
    sep_positions = []
    n_cp = sum(1 for l in ordered_labels if l in _COPYRIGHT_LABELS)
    n_au = sum(1 for l in ordered_labels if l in _AUTHOR_LABELS)
    if n_cp:
        sep_positions.append(n_cp - 0.5)
    if n_au:
        sep_positions.append(n_cp + n_au - 0.5)
    for sp in sep_positions:
        ax.axhline(sp, color="white", linewidth=2.5)

    _save(fig, _predict_dir(out_dir) / "P04_accuracy_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison_integrated_vs_seperated(
    plt: Any,
    out_dir_int: Path,
    out_dir_sep: Path,
    int_acc: Dict[float, float],
    sep_acc: Dict[float, float],
    method: str,
    display: str,
) -> None:
    """C01 — Integrated (single model) vs Seperated (3-model ensemble) accuracy comparison."""
    if not int_acc and not sep_acc:
        return

    all_thrs = sorted(set(list(int_acc.keys()) + list(sep_acc.keys())))

    fig, ax = plt.subplots(figsize=(7, 4))

    if int_acc:
        thrs = sorted(int_acc)
        ax.plot(thrs, [int_acc[t] for t in thrs],
                color=_C_BLUE, linewidth=2.2, marker="o", markersize=6,
                label="Integrated (single model, 50 labels)")
    if sep_acc:
        thrs = sorted(sep_acc)
        ax.plot(thrs, [sep_acc[t] for t in thrs],
                color=_C_ORANGE, linewidth=2.2, marker="s", markersize=6,
                linestyle="--",
                label="Seperated (3-model ensemble)")

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Integrated vs Seperated Ensemble  [{display} / {method}]")
    ax.legend(loc="lower left")

    ax.text(0.01, 0.02,
            "Seperated: copyright model + author model + rights model → merged predictions",
            transform=ax.transAxes, fontsize=7, color=_C_GREY, style="italic")

    for d in (out_dir_int, out_dir_sep):
        d.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir_int / f"C01_integrated_vs_seperated_{method}.png")
    # also save copy to seperated dir
    try:
        import shutil
        shutil.copy(out_dir_int / f"C01_integrated_vs_seperated_{method}.png",
                    out_dir_sep / f"C01_integrated_vs_seperated_{method}.png")
    except Exception:
        pass


def plot_method_comparison(
    plt: Any,
    out_dir: Path,
    lora_acc: Dict[float, float],
    full_acc: Dict[float, float],
) -> None:
    """C02 — LoRA vs Full fine-tuning accuracy comparison (line + bar)."""
    if not lora_acc and not full_acc:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # left: line chart (threshold sensitivity)
    ax = axes[0]
    if lora_acc:
        thrs = sorted(lora_acc)
        ax.plot(thrs, [lora_acc[t] for t in thrs],
                color=_C_BLUE, linewidth=2.2, marker="o", markersize=5, label="LoRA")
    if full_acc:
        thrs = sorted(full_acc)
        ax.plot(thrs, [full_acc[t] for t in thrs],
                color=_C_RED, linewidth=2.2, marker="s", markersize=5,
                linestyle="--", label="Full FT")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("LoRA vs Full FT — Threshold Sensitivity")
    ax.legend()

    # right: best accuracy bar
    ax2 = axes[1]
    import numpy as np
    methods = []
    bests   = []
    colors  = []
    if lora_acc:
        methods.append("LoRA")
        bests.append(max(lora_acc.values()))
        colors.append(_C_BLUE)
    if full_acc:
        methods.append("Full FT")
        bests.append(max(full_acc.values()))
        colors.append(_C_RED)

    bars = ax2.bar(methods, bests, color=colors, alpha=0.85, width=0.45,
                   edgecolor="white")
    for bar, val in zip(bars, bests):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Best Accuracy")
    ax2.set_title("Best Accuracy Comparison")

    fig.suptitle("Fine-tuning Method Comparison: LoRA vs Full", fontsize=12,
                 fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "C02_lora_vs_full.png")


def plot_efficiency_scatter(
    plt: Any,
    summary: Dict,
    out_dir: Path,
) -> None:
    """G01 — Efficiency-Performance Scatter (training time vs. best validation accuracy).

    Each point = (model, method, mode) combination.
    Colours distinguish models; markers distinguish method (LoRA ● / Full FT ▲);
    mode (integrated vs seperated) is encoded by marker edge style.
    Pareto frontier (lower time, higher accuracy) is drawn as a step line.
    """
    import numpy as np

    # ── short display labels ─────────────────────────────────────────────────
    MODEL_SHORT: Dict[str, str] = {
        "google-bert/bert-base-multilingual-cased": "mBERT",
        "klue/bert-base":                           "KLUE-BERT",
        "microsoft/deberta-v3-base":                "DeBERTa-v3",
        "monologg/koelectra-base-v3-discriminator": "KoELECTRA",
    }
    MODEL_COLORS: Dict[str, str] = {
        "google-bert/bert-base-multilingual-cased": _C_BLUE,
        "klue/bert-base":                           _C_GREEN,
        "microsoft/deberta-v3-base":                _C_RED,
        "monologg/koelectra-base-v3-discriminator": _C_PURPLE,
    }
    METHOD_MARKER = {"lora": "o", "full": "^"}
    MODE_EDGE     = {"integrated": "none", "seperated": "black"}   # filled vs edged

    points: List[Tuple[float, float, str, str, str, str]] = []  # (time, acc, model, method, mode, label)

    for model_name, method_data in summary.items():
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        for method, mode_data in method_data.items():
            for mode, res in mode_data.items():
                t = res.get("train_time_seconds", 0) or 0
                acc_raw = res.get("validation_accuracy_by_threshold", {})
                if not acc_raw or t <= 0:
                    continue
                best_acc = max(float(v) for v in acc_raw.values())
                label_str = f"{short}\n{method}/{mode}"
                points.append((t, best_acc, model_name, method, mode, label_str))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # ── scatter ──────────────────────────────────────────────────────────────
    plotted_model_handles: Dict[str, Any] = {}
    plotted_method_handles: Dict[str, Any] = {}

    for (t, acc, model_name, method, mode, lbl) in points:
        color  = MODEL_COLORS.get(model_name, _C_GREY)
        marker = METHOD_MARKER.get(method, "o")
        edge   = MODE_EDGE.get(mode, "none")
        size   = 120

        sc = ax.scatter(
            t, acc,
            c=color, marker=marker, s=size,
            edgecolors=edge, linewidths=1.2,
            zorder=3, alpha=0.90,
        )
        # annotation offset to avoid overlap
        ax.annotate(
            lbl, xy=(t, acc),
            xytext=(6, 4), textcoords="offset points",
            fontsize=6.5, color=color,
        )
        short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
        if short not in plotted_model_handles:
            plotted_model_handles[short] = plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color, markersize=8, label=short,
            )
        if method not in plotted_method_handles:
            plotted_method_handles[method] = plt.Line2D(
                [0], [0], marker=METHOD_MARKER[method], color=_C_GREY,
                markersize=8, linestyle="none",
                label=f"{'LoRA' if method=='lora' else 'Full FT'}",
            )

    # ── Pareto frontier ───────────────────────────────────────────────────────
    # Lower time + higher accuracy = Pareto-optimal
    sorted_pts = sorted(points, key=lambda p: p[0])  # sort by time ascending
    pareto: List[Tuple[float, float]] = []
    best_acc_so_far = -1.0
    for (t, acc, *_rest) in sorted_pts:
        if acc > best_acc_so_far:
            pareto.append((t, acc))
            best_acc_so_far = acc

    if len(pareto) >= 2:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.step(px, py, where="post", color=_C_AMBER, linewidth=1.8,
                linestyle="--", zorder=2, label="Pareto frontier")
        ax.scatter(px, py, color=_C_AMBER, s=60, zorder=4, marker="*")

    # ── axes & legend ─────────────────────────────────────────────────────────
    ax.set_xlabel("Training Time (seconds)", fontsize=11)
    ax.set_ylabel("Best Validation Accuracy", fontsize=11)
    ax.set_title("Efficiency–Performance Trade-off\n(lower-left = faster; upper-right = more accurate)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(bottom=0)

    # mode legend proxy
    mode_handles = [
        plt.Line2D([0], [0], marker="o", color=_C_GREY, markersize=8,
                   linestyle="none", markerfacecolor=_C_GREY,
                   markeredgecolor="none", label="integrated (no edge)"),
        plt.Line2D([0], [0], marker="o", color=_C_GREY, markersize=8,
                   linestyle="none", markerfacecolor=_C_GREY,
                   markeredgecolor="black", linewidth=1.2, label="seperated (black edge)"),
    ]

    legend1 = ax.legend(
        handles=list(plotted_model_handles.values()),
        title="Model", loc="lower right", fontsize=7, title_fontsize=8,
    )
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=list(plotted_method_handles.values()) + mode_handles,
        title="Method / Mode", loc="upper right", fontsize=7, title_fontsize=8,
    )
    ax.add_artist(legend2)
    if len(pareto) >= 2:
        ax.legend(
            handles=[plt.Line2D([0], [0], color=_C_AMBER, linewidth=1.8,
                                linestyle="--", label="Pareto frontier")],
            loc="upper left", fontsize=7,
        )

    ax.grid(True, linestyle="--", alpha=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "G01_efficiency_scatter.png")


def plot_delta_heatmap(
    plt: Any,
    out_dir: Path,
    lora_per_label: Dict[str, Dict[float, float]],
    full_per_label: Dict[str, Dict[float, float]],
    thresholds: List[float],
) -> None:
    """G02 — Delta Heatmap: LoRA − Full FT per-label accuracy difference.

    Rows = 50 labels (category-grouped), Columns = threshold values.
    Diverging colormap: blue = LoRA better, red = Full better, white = tied.
    """
    import numpy as np

    cat_order = ["copyright_info", "author_info", "rights_info", "other"]
    all_labels: List[str] = []
    for cat in cat_order:
        cat_labels = sorted(
            [l for l in set(list(lora_per_label) + list(full_per_label))
             if _category_of(l) == cat]
        )
        all_labels.extend(cat_labels)

    if not all_labels or not thresholds:
        return

    thrs_sorted = sorted(thresholds)
    data = np.zeros((len(all_labels), len(thrs_sorted)))
    for i, lbl in enumerate(all_labels):
        for j, thr in enumerate(thrs_sorted):
            l_acc = lora_per_label.get(lbl, {}).get(thr, float("nan"))
            f_acc = full_per_label.get(lbl, {}).get(thr, float("nan"))
            if not (np.isnan(l_acc) or np.isnan(f_acc)):
                data[i, j] = l_acc - f_acc
            else:
                data[i, j] = float("nan")

    n = len(all_labels)
    fig_h = max(8, n * 0.42)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    vmax = max(0.3, float(np.nanmax(np.abs(data))))
    im = ax.imshow(data, aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(thrs_sorted)))
    ax.set_xticklabels([f"{t:.2f}" for t in thrs_sorted],
                        fontsize=9, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([_shorten(l, 22) for l in all_labels], fontsize=7.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Label")
    ax.set_title("LoRA − Full FT Accuracy Delta\n(blue = LoRA better, red = Full better)",
                 fontsize=12, fontweight="bold")

    # cell annotations
    for i in range(n):
        for j in range(len(thrs_sorted)):
            v = data[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=txt_color)

    # category separator lines
    _draw_category_separators(ax, all_labels, axis="y")

    fig.colorbar(im, ax=ax, shrink=0.6, label="LoRA − Full Δ accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "G02_delta_heatmap.png")


def plot_lora_rank_curve(
    plt: Any,
    out_dir: Path,
    summary: Dict[str, Any],
) -> None:
    """G03 — LoRA Rank Curve: lora_r vs best validation accuracy.

    summary 구조 (model_name → method → mode → payload):
      mode 키가 "integrated_r{rank}" 형태인 항목에서 데이터 추출.
      full FT의 integrated 결과를 baseline 수평선으로 표시.

    out_dir / "G03_lora_rank_curve.png" 저장.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = False

    for model_name, method_data in summary.items():
        lora_data = method_data.get("lora", {})
        full_data  = method_data.get("full", {})

        # LoRA rank sweep points
        rank_acc: Dict[int, float] = {}
        for mode_key, payload in lora_data.items():
            if not mode_key.startswith("integrated_r"):
                continue
            try:
                rank = int(mode_key[len("integrated_r"):])
            except ValueError:
                continue
            thr_acc = payload.get("validation_accuracy_by_threshold", {})
            if not thr_acc:
                continue
            best = max(float(v) for v in thr_acc.values())
            rank_acc[rank] = best

        if not rank_acc:
            continue

        ranks = sorted(rank_acc)
        accs  = [rank_acc[r] for r in ranks]
        short = model_name.split("/")[-1]

        ax.plot(ranks, accs, marker="o", linewidth=1.8, label=short)

        # Full FT baseline (integrated)
        full_int = full_data.get("integrated", {}).get("validation_accuracy_by_threshold", {})
        if full_int:
            full_best = max(float(v) for v in full_int.values())
            ax.axhline(
                full_best,
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label=f"{short} (full FT)",
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("LoRA Rank (r)")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("G03 — LoRA Rank vs. Performance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.85)
    ax.set_xticks(sorted({int(mode_key[len("integrated_r"):])
                          for md in summary.values()
                          for mode_key in md.get("lora", {})
                          if mode_key.startswith("integrated_r")
                          and mode_key[len("integrated_r"):].isdigit()}))

    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "G03_lora_rank_curve.png")


def plot_e1_threshold_comparison(
    plt: Any,
    out_dir: Path,
    summary: Dict[str, Any],
    *,
    model_name: str = "klue/bert-base",
    method: str = "full",
) -> None:
    """G04 — E1 데이터 소스 비교: threshold vs 전체 정확도 곡선 3조건 중첩.

    summary 구조: {model_name: {method: {mode: payload}}}
    mode e1a / e1b / e1c 가 있는 model+method 조합을 탐색.

    out_dir / "G04_e1_threshold_comparison.png" 저장.
    """
    E1_LABELS = {"e1a": "E1-A (Rule Silver)", "e1b": "E1-B (LLM Silver)", "e1c": "E1-C (LLM+Filter)"}
    E1_COLORS = {"e1a": _C_BLUE, "e1b": _C_GREEN, "e1c": _C_RED}
    E1_STYLES = {"e1a": "-", "e1b": "--", "e1c": "-."}

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False

    model_data = summary.get(model_name, {})
    method_data = model_data.get(method, {})

    for mode, label in E1_LABELS.items():
        payload = method_data.get(mode)
        if payload is None:
            continue
        thr_acc = payload.get("validation_accuracy_by_threshold", {})
        if not thr_acc:
            continue
        thrs = sorted(float(k) for k in thr_acc)
        accs = [float(thr_acc[str(t) if str(t) in thr_acc else t]) for t in thrs]
        ax.plot(
            thrs, accs,
            marker="o", linewidth=2, linestyle=E1_STYLES[mode],
            color=E1_COLORS[mode], label=label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("G04 — E1: Silver 생성 방식별 정확도 비교", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "G04_e1_threshold_comparison.png")


def plot_e1_label_heatmap(
    plt: Any,
    out_dir: Path,
    summary: Dict[str, Any],
    *,
    model_name: str = "klue/bert-base",
    method: str = "full",
    threshold: float = 0.25,
) -> None:
    """G05 — E1 라벨별 정확도 heatmap (조건 × 라벨).

    E1-A / E1-B / E1-C 3개 행, 최대 50개 라벨 열.
    out_dir / "G05_e1_label_heatmap.png" 저장.
    """
    import numpy as np

    E1_MODES = ["e1a", "e1b", "e1c"]
    E1_LABELS_DISPLAY = ["E1-A\n(Rule)", "E1-B\n(LLM)", "E1-C\n(LLM+Filter)"]
    THR_KEY = str(threshold)

    model_data = summary.get(model_name, {})
    method_data = model_data.get(method, {})

    # 공통 라벨 수집
    all_labels: List[str] = []
    label_accs: Dict[str, List[float]] = {}  # mode → [acc per label]

    ref_payload = None
    for mode in E1_MODES:
        p = method_data.get(mode)
        if p and p.get("per_label_accuracy"):
            ref_payload = p
            break
    if ref_payload is None:
        return

    ordered_labels = list(ref_payload["per_label_accuracy"].keys())
    matrix = []
    for mode in E1_MODES:
        payload = method_data.get(mode)
        row = []
        for lbl in ordered_labels:
            if payload and "per_label_accuracy" in payload:
                thr_map = payload["per_label_accuracy"].get(lbl, {})
                acc = float(thr_map.get(THR_KEY, thr_map.get(threshold, 0.0)))
            else:
                acc = 0.0
            row.append(acc)
        matrix.append(row)

    mat = np.array(matrix, dtype=float)
    n_labels = len(ordered_labels)
    fig_w = max(14, n_labels * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(ordered_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(E1_MODES)))
    ax.set_yticklabels(E1_LABELS_DISPLAY, fontsize=9)
    ax.set_title(
        f"G05 — E1 라벨별 정확도 비교 (thr={threshold})",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # 카테고리 구분선
    _draw_category_separators(ax, ordered_labels, axis="x")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, out_dir / "G05_e1_label_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_train_plots(
    out_dir: Path,
    eval_info: Optional[Dict],
    debug_metrics: Optional[Dict],
    train_time_s: Optional[float],
    train_dir: Optional[Path],
    mode: str,
    method: str,
    *,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Generate all training-phase plots → out_dir/ (호출자가 .../train/ 경로를 포함해 전달)

    T01 loss curves · T02 F1 convergence · T03 P/R/F1 overall
    T04 per-label P/R/F1 · T05 confusion matrix · T06 P-R plot
    T07 FP/FN analysis · T08 label distribution
    """
    plt = _setup_mpl()
    d   = _unpack(eval_info, debug_metrics)

    # Supplement log_history from trainer_state.json if missing or only a short tail
    if (model_path or model_name):
        trainer_lh = _load_trainer_state(model_path, model_name)
        if len(trainer_lh) > len(d["log_hist"]):
            d["log_hist"] = trainer_lh

    count = 0

    if d["log_hist"]:
        _plot_loss_curves(plt, out_dir, d["log_hist"])
        count += 1
        _plot_f1_progression(plt, out_dir, d["log_hist"])
        count += 1

    if d["report"]:
        _plot_prf_overall(plt, out_dir, d["report"])
        count += 1
        _plot_per_label_f1(plt, out_dir, d["report"])
        count += 1
        _plot_pr_curve(plt, out_dir, d["report"])
        count += 1

    if d["cm"] and d["cm"].get("matrix"):
        _plot_confusion_matrix(plt, out_dir, d["cm"])
        count += 1

    if d["fp_lbl"] or d["fn_lbl"]:
        _plot_fp_fn_analysis(plt, out_dir, d["fp_lbl"], d["fn_lbl"], d["tp_lbl"])
        count += 1

    _plot_label_distribution(plt, out_dir, train_dir)
    count += 1

    print(f"  [plots/train] {count}개 저장 → {out_dir}/")


def generate_predict_plots(
    out_dir: Path,
    acc_by_thr: Dict[float, float],
    per_label_acc: Optional[Dict[str, Dict[float, float]]],
    mode: str,
    method: str,
) -> None:
    """Generate all prediction-phase plots → out_dir/ (호출자가 .../predict/ 경로를 포함해 전달)

    P01 threshold accuracy · P02 per-label accuracy bar
    P03 category breakdown (ensemble view) · P04 accuracy heatmap
    """
    plt   = _setup_mpl()
    count = 0

    if acc_by_thr:
        _plot_threshold_accuracy(plt, out_dir, acc_by_thr, mode, method)
        count += 1

    if per_label_acc:
        _plot_per_label_accuracy(plt, out_dir, per_label_acc, mode)
        count += 1
        _plot_category_accuracy(plt, out_dir, per_label_acc, mode)
        count += 1
        _plot_accuracy_heatmap(plt, out_dir, per_label_acc, mode)
        count += 1

    print(f"  [plots/predict] {count}개 저장 → {out_dir}/")


def generate_all_plots(
    out_dir: Path,
    eval_info: Optional[Dict],
    debug_metrics: Optional[Dict],
    train_time_s: Optional[float],
    train_dir: Optional[Path],
    mode: str,
    method: str,
    acc_by_thr: Optional[Dict[float, float]] = None,
    per_label_acc: Optional[Dict[str, Dict[float, float]]] = None,
    *,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    """Run both train and predict plot generation."""
    generate_train_plots(out_dir, eval_info, debug_metrics, train_time_s,
                         train_dir, mode, method,
                         model_path=model_path, model_name=model_name)
    if acc_by_thr is not None:
        generate_predict_plots(out_dir, acc_by_thr, per_label_acc, mode, method)
