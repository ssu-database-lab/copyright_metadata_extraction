"""학습 곡선 시각화 — TokenClassNER.train(save_plots=True) 시만 호출."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PLOTS_SUBDIR = "plots"


def _generate_plots(
    adapter_dir: Path,
    *,
    log_history: List[Dict[str, Any]],
    full_metrics: Optional[Dict[str, Any]] = None,
    grad_norms: Optional[List[Dict[str, Any]]] = None,
    epoch_times: Optional[List[float]] = None,
    weights_before: Optional[Dict[str, Any]] = None,
    weights_after: Optional[Dict[str, Any]] = None,
    method: str = "unknown",
    model_name: str = "model",
    debug: bool = False,
) -> List[str]:
    """논문 작성용 그래프 생성. adapter_dir/plots/ 에 저장. 저장된 파일 경로 목록 반환."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        log.warning("matplotlib 없음 — 그래프 생성 스킵")
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    plots_dir = adapter_dir / PLOTS_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    title_prefix = f"{model_name.split('/')[-1]} [{method}]"

    def _savefig(name: str, fig: Any) -> None:
        path = plots_dir / name
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
        if debug:
            print(f"    그래프 저장: {path}")

    # ── 1. Training Curves (loss + val accuracy) ─────────────────────
    try:
        step_entries   = [e for e in log_history if "loss" in e and "eval_loss" not in e]
        eval_entries   = [e for e in log_history if "eval_loss" in e]

        if step_entries or eval_entries:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            if step_entries:
                steps  = [e.get("step", i) for i, e in enumerate(step_entries)]
                losses = [e["loss"] for e in step_entries]
                ax1.plot(steps, losses, color="#2196F3", linewidth=1.2, alpha=0.8, label="Train Loss")

            if eval_entries:
                eval_steps = [e.get("step", i) for i, e in enumerate(eval_entries)]
                eval_loss  = [e["eval_loss"] for e in eval_entries if "eval_loss" in e]
                eval_acc   = [e["eval_accuracy"] for e in eval_entries if "eval_accuracy" in e]
                if eval_loss:
                    ax1.plot(eval_steps, eval_loss, color="#F44336", linewidth=1.5,
                             marker="o", markersize=4, label="Val Loss")
                if eval_acc:
                    ax2.plot(eval_steps, eval_acc, color="#4CAF50", linewidth=1.5,
                             marker="s", markersize=4, linestyle="--", label="Val Accuracy")

            ax1.set_xlabel("Step")
            ax1.set_ylabel("Loss")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0, 1.05)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            fig.suptitle(f"{title_prefix} — Training Curves")
            fig.tight_layout()
            _savefig("01_training_curves.png", fig)
    except Exception as ex:
        log.warning("training_curves 그래프 실패: %s", ex)

    # ── 2. Learning Rate Schedule ─────────────────────────────────────
    try:
        lr_entries = [e for e in log_history if "learning_rate" in e]
        if lr_entries:
            steps = [e.get("step", i) for i, e in enumerate(lr_entries)]
            lrs   = [e["learning_rate"] for e in lr_entries]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, lrs, color="#9C27B0", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.set_title(f"{title_prefix} — LR Schedule")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _savefig("02_lr_schedule.png", fig)
    except Exception as ex:
        log.warning("lr_schedule 그래프 실패: %s", ex)

    # ── 3. Per-Label F1 Bar Chart ─────────────────────────────────────
    try:
        if full_metrics and "token_classification_report" in full_metrics:
            report = full_metrics["token_classification_report"]
            labels_f1 = {
                k: v["f1-score"]
                for k, v in report.items()
                if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")
            }
            if labels_f1:
                sorted_items = sorted(labels_f1.items(), key=lambda x: x[1])
                lbls = [i[0] for i in sorted_items]
                f1s  = [i[1] for i in sorted_items]
                supports = [report[l].get("support", 0) for l in lbls]

                cmap   = plt.get_cmap("RdYlGn")
                colors = [cmap(v) for v in f1s]

                n = len(lbls)
                fig_h = max(6, n * 0.28)
                fig, ax = plt.subplots(figsize=(10, fig_h))
                bars = ax.barh(range(n), f1s, color=colors, edgecolor="white", linewidth=0.5)
                ax.set_yticks(range(n))
                ax.set_yticklabels(lbls, fontsize=8)
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("F1-Score")
                ax.set_title(f"{title_prefix} — Per-Label Token F1")
                ax.axvline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="F1=0.9")

                # support 수치 주석
                for i, (bar, sup) in enumerate(zip(bars, supports)):
                    ax.text(bar.get_width() + 0.01, i, f"n={sup}", va="center", fontsize=7)

                ax.legend(fontsize=8)
                fig.tight_layout()
                _savefig("03_per_label_f1.png", fig)
    except Exception as ex:
        log.warning("per_label_f1 그래프 실패: %s", ex)

    # ── 4. Per-Label Precision / Recall / F1 Grouped Bar ─────────────
    try:
        if full_metrics and "token_classification_report" in full_metrics:
            report = full_metrics["token_classification_report"]
            items = [
                (k, v) for k, v in report.items()
                if isinstance(v, dict) and k not in ("accuracy", "macro avg", "weighted avg")
            ]
            if items:
                lbls = [i[0] for i in items]
                prec = [i[1]["precision"] for i in items]
                rec  = [i[1]["recall"] for i in items]
                f1   = [i[1]["f1-score"] for i in items]

                x = np.arange(len(lbls))
                w = 0.26
                n = len(lbls)
                fig_w = max(12, n * 0.4)
                fig, ax = plt.subplots(figsize=(fig_w, 5))
                ax.bar(x - w, prec, w, label="Precision", color="#42A5F5", alpha=0.85)
                ax.bar(x,     rec,  w, label="Recall",    color="#66BB6A", alpha=0.85)
                ax.bar(x + w, f1,   w, label="F1",        color="#FFA726", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Score")
                ax.set_title(f"{title_prefix} — Precision / Recall / F1 per Label")
                ax.legend()
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                _savefig("04_per_label_prf.png", fig)
    except Exception as ex:
        log.warning("per_label_prf 그래프 실패: %s", ex)

    # ── 5. Confusion Matrix ───────────────────────────────────────────
    try:
        if full_metrics and "confusion_matrix" in full_metrics:
            cm_data = full_metrics["confusion_matrix"]
            cm_arr  = np.array(cm_data["matrix"], dtype=float)
            cm_lbls = cm_data["labels"]

            # 행 정규화 (recall 기준)
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_norm  = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

            n = len(cm_lbls)
            fig_sz = max(8, n * 0.45)
            fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.9))

            try:
                import seaborn as sns
                sns.heatmap(
                    cm_norm, ax=ax,
                    xticklabels=cm_lbls, yticklabels=cm_lbls,
                    cmap="Blues", vmin=0, vmax=1,
                    linewidths=0.3, linecolor="white",
                    annot=(n <= 20), fmt=".2f",
                    cbar_kws={"label": "Recall (row-normalized)"},
                )
            except ImportError:
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(n))
                ax.set_xticklabels(cm_lbls, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(n))
                ax.set_yticklabels(cm_lbls, fontsize=7)
                fig.colorbar(im, ax=ax, label="Recall (row-normalized)")

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{title_prefix} — Confusion Matrix (row-normalized recall)")
            plt.xticks(fontsize=7, rotation=45, ha="right")
            plt.yticks(fontsize=7)
            fig.tight_layout()
            _savefig("05_confusion_matrix.png", fig)
    except Exception as ex:
        log.warning("confusion_matrix 그래프 실패: %s", ex)

    # ── 6. Confidence Distribution Histogram ─────────────────────────
    try:
        if full_metrics and "confidence_flat" in full_metrics:
            confs = np.array(full_metrics["confidence_flat"])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(confs, bins=50, color="#26C6DA", edgecolor="white", linewidth=0.4)
            if "confidence_stats" in full_metrics:
                cs = full_metrics["confidence_stats"]
                ax.axvline(cs["mean"], color="#F44336", linestyle="--",
                           linewidth=1.5, label=f"mean={cs['mean']:.3f}")
                ax.axvline(cs["p50"],  color="#FF9800", linestyle=":",
                           linewidth=1.5, label=f"median={cs['p50']:.3f}")
            ax.set_xlabel("Max Softmax Probability")
            ax.set_ylabel("Token Count")
            ax.set_title(f"{title_prefix} — Prediction Confidence Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _savefig("06_confidence_distribution.png", fig)
    except Exception as ex:
        log.warning("confidence_distribution 그래프 실패: %s", ex)

    # ── 7. Per-Label Confidence (box-like with percentiles) ───────────
    try:
        if full_metrics and "confidence_per_label" in full_metrics:
            plc = full_metrics["confidence_per_label"]
            items = sorted(plc.items(), key=lambda x: x[1]["p50"])
            if items:
                lbls   = [i[0] for i in items]
                p25    = [i[1]["p25"] for i in items]
                p50    = [i[1]["p50"] for i in items]
                p75    = [i[1]["p75"] for i in items]
                means  = [i[1]["mean"] for i in items]
                n = len(lbls)
                fig_h = max(6, n * 0.28)
                fig, ax = plt.subplots(figsize=(10, fig_h))
                y = np.arange(n)
                ax.barh(y, [h - l for l, h in zip(p25, p75)], left=p25,
                        height=0.5, color="#42A5F5", alpha=0.6, label="IQR (p25-p75)")
                ax.scatter(p50,   y, color="#1565C0", s=30, zorder=5, label="Median")
                ax.scatter(means, y, color="#F44336", s=20, marker="D", zorder=5, label="Mean")
                ax.set_yticks(y)
                ax.set_yticklabels(lbls, fontsize=8)
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("Max Softmax Probability")
                ax.set_title(f"{title_prefix} — Per-Label Confidence (IQR)")
                ax.legend(fontsize=8)
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                _savefig("07_per_label_confidence.png", fig)
    except Exception as ex:
        log.warning("per_label_confidence 그래프 실패: %s", ex)

    # ── 8. Gradient Norm Curve ────────────────────────────────────────
    try:
        if grad_norms:
            steps = [g["step"] for g in grad_norms]
            # total grad norm = sqrt(sum of all layer norm^2)
            total_norms = [
                float(np.sqrt(sum(v ** 2 for v in g["layer_norms"].values())))
                for g in grad_norms
            ]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, total_norms, color="#FF5722", linewidth=1.0, alpha=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel("Total Gradient Norm")
            ax.set_title(f"{title_prefix} — Gradient Norm over Training")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _savefig("08_grad_norm_curve.png", fig)
    except Exception as ex:
        log.warning("grad_norm 그래프 실패: %s", ex)

    # ── 9. Epoch Timing Bar ───────────────────────────────────────────
    try:
        if epoch_times:
            epochs_idx = list(range(1, len(epoch_times) + 1))
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(epochs_idx, epoch_times, color="#8D6E63", alpha=0.8, edgecolor="white")
            for bar, t in zip(bars, epoch_times):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, f"{t:.0f}s",
                        ha="center", va="bottom", fontsize=9)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"{title_prefix} — Epoch Training Time")
            ax.set_xticks(epochs_idx)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            _savefig("09_epoch_timing.png", fig)
    except Exception as ex:
        log.warning("epoch_timing 그래프 실패: %s", ex)

    # ── 10. Weight Change Magnitude (before vs after) ─────────────────
    try:
        if weights_before and weights_after:
            common_keys = [k for k in weights_before if k in weights_after]
            if common_keys:
                # mean absolute delta per layer
                deltas = {
                    k: abs(weights_after[k]["mean"] - weights_before[k]["mean"])
                    for k in common_keys
                }
                # 상위 30개 레이어만 (너무 많으면 그래프 불가)
                top_items = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:30]
                top_items = list(reversed(top_items))  # ascending for barh

                names  = [i[0].split(".")[-2] + "." + i[0].split(".")[-1] for i in top_items]
                values = [i[1] for i in top_items]

                fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
                ax.barh(range(len(names)), values, color="#AB47BC", alpha=0.8)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=8)
                ax.set_xlabel("|mean(after) - mean(before)|")
                ax.set_title(f"{title_prefix} — Top 30 Weight Change Magnitude")
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                _savefig("10_weight_delta.png", fig)
    except Exception as ex:
        log.warning("weight_delta 그래프 실패: %s", ex)

    # ── 11. Span F1 vs Token F1 comparison (if seqeval available) ─────
    try:
        if full_metrics and "span_classification_report" in full_metrics and "token_classification_report" in full_metrics:
            span_r  = full_metrics["span_classification_report"]
            token_r = full_metrics["token_classification_report"]

            # base entity names (without B-/I- prefix)
            span_labels = [k for k in span_r if isinstance(span_r[k], dict) and k not in ("micro avg", "macro avg", "weighted avg")]
            if span_labels:
                span_f1  = [span_r[l]["f1-score"] for l in span_labels]
                # find corresponding token F1 for B-label
                token_f1 = []
                for l in span_labels:
                    b_key = f"B-{l}"
                    if b_key in token_r and isinstance(token_r[b_key], dict):
                        token_f1.append(token_r[b_key]["f1-score"])
                    else:
                        token_f1.append(0.0)

                x = np.arange(len(span_labels))
                w = 0.35
                n = len(span_labels)
                fig_w = max(10, n * 0.5)
                fig, ax = plt.subplots(figsize=(fig_w, 5))
                ax.bar(x - w / 2, span_f1,  w, label="Entity F1 (seqeval)", color="#26A69A", alpha=0.85)
                ax.bar(x + w / 2, token_f1, w, label="Token F1 (B- label)",  color="#EF5350", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(span_labels, rotation=45, ha="right", fontsize=8)
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("F1-Score")
                ax.set_title(f"{title_prefix} — Span F1 vs Token F1")
                ax.legend()
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                _savefig("11_span_vs_token_f1.png", fig)
    except Exception as ex:
        log.warning("span_vs_token_f1 그래프 실패: %s", ex)

    # ── 12. F1 / Precision / Recall over Epochs (span-level) ──────────
    try:
        eval_entries = [e for e in log_history if "eval_f1" in e]
        if eval_entries:
            epoch_nums = [e.get("epoch", i + 1) for i, e in enumerate(eval_entries)]
            f1_vals    = [e["eval_f1"]        for e in eval_entries]
            prec_vals  = [e.get("eval_precision", float("nan")) for e in eval_entries]
            rec_vals   = [e.get("eval_recall",    float("nan")) for e in eval_entries]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(epoch_nums, f1_vals,   color="#F44336", linewidth=2.0,
                    marker="o", markersize=5, label="Span F1")
            ax.plot(epoch_nums, prec_vals, color="#42A5F5", linewidth=1.5,
                    marker="s", markersize=4, linestyle="--", label="Precision")
            ax.plot(epoch_nums, rec_vals,  color="#66BB6A", linewidth=1.5,
                    marker="^", markersize=4, linestyle=":",  label="Recall")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{title_prefix} — Span F1 / Precision / Recall over Epochs")
            fig.tight_layout()
            _savefig("12_f1_pr_over_epochs.png", fig)
    except Exception as ex:
        log.warning("f1_pr_over_epochs 그래프 실패: %s", ex)

    # ── 13. F1 Convergence: Best-so-far + Derivative ───────────────────
    try:
        eval_entries = [e for e in log_history if "eval_f1" in e]
        if len(eval_entries) >= 2:
            epoch_nums = [e.get("epoch", i + 1) for i, e in enumerate(eval_entries)]
            f1_vals    = [e["eval_f1"] for e in eval_entries]

            # running max (best-so-far)
            best_so_far = []
            best = -1.0
            for v in f1_vals:
                best = max(best, v)
                best_so_far.append(best)

            # derivative ΔF1 per epoch
            deltas = [0.0] + [f1_vals[i] - f1_vals[i - 1] for i in range(1, len(f1_vals))]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

            ax1.plot(epoch_nums, f1_vals,    color="#F44336", linewidth=1.5,
                     marker="o", markersize=4, label="F1", alpha=0.8)
            ax1.plot(epoch_nums, best_so_far, color="#1565C0", linewidth=2.0,
                     linestyle="--", label="Best-so-far F1")
            ax1.set_ylabel("Span F1")
            ax1.set_ylim(0, 1.05)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"{title_prefix} — F1 Convergence")

            colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
            ax2.bar(epoch_nums, deltas, color=colors, alpha=0.75, edgecolor="white")
            ax2.axhline(0, color="gray", linewidth=0.8)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("ΔF1")
            ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax2.grid(True, axis="y", alpha=0.3)

            fig.tight_layout()
            _savefig("13_f1_convergence.png", fig)
    except Exception as ex:
        log.warning("f1_convergence 그래프 실패: %s", ex)

    # ── 14. Precision-Recall Curve with AP ─────────────────────────────
    try:
        if full_metrics and "pr_curve" in full_metrics:
            prc = full_metrics["pr_curve"]
            prec_arr = prc["precision"]
            rec_arr  = prc["recall"]
            ap       = prc.get("average_precision", float("nan"))

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.step(rec_arr, prec_arr, where="post", color="#1976D2", linewidth=2.0,
                    label=f"AP = {ap:.3f}")
            ax.fill_between(rec_arr, prec_arr, alpha=0.15, color="#1976D2", step="post")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{title_prefix} — Precision-Recall Curve (entity vs O)")
            fig.tight_layout()
            _savefig("14_pr_curve.png", fig)
    except Exception as ex:
        log.warning("pr_curve 그래프 실패: %s", ex)

    # ── 15. Summary Panel ──────────────────────────────────────────────
    try:
        eval_entries = [e for e in log_history if "eval_f1" in e]
        lines: List[str] = [
            f"Model : {model_name}",
            f"Method: {method}",
            "",
            "── Final Evaluation (test set) ──",
        ]

        if full_metrics and "span_classification_report" in full_metrics:
            sr = full_metrics["span_classification_report"]
            for key in ("micro avg", "macro avg", "weighted avg"):
                if key in sr and isinstance(sr[key], dict):
                    r = sr[key]
                    lines.append(
                        f"  {key:<14} P={r.get('precision', 0):.3f}"
                        f"  R={r.get('recall', 0):.3f}"
                        f"  F1={r.get('f1-score', 0):.3f}"
                    )
        elif full_metrics and "token_classification_report" in full_metrics:
            tr = full_metrics["token_classification_report"]
            for key in ("macro avg", "weighted avg"):
                if key in tr and isinstance(tr[key], dict):
                    r = tr[key]
                    lines.append(
                        f"  {key:<14} P={r.get('precision', 0):.3f}"
                        f"  R={r.get('recall', 0):.3f}"
                        f"  F1={r.get('f1-score', 0):.3f}"
                    )

        if full_metrics and "pr_curve" in full_metrics:
            ap = full_metrics["pr_curve"].get("average_precision", float("nan"))
            lines.append(f"  Average Precision (AP) = {ap:.4f}")

        if full_metrics and "false_alarm_stats" in full_metrics:
            fa = full_metrics["false_alarm_stats"]
            far = fa.get("false_alarm_rate", float("nan"))
            fp_o = fa.get("false_positives_on_O", 0)
            tot_o = fa.get("total_O_gold", 0)
            lines.append(f"  False Alarm Rate (FAR) = {far:.4f}  ({fp_o}/{tot_o})")

        lines.append("")
        lines.append("── Training Progress ──")
        if eval_entries:
            best_f1_val = max(e["eval_f1"] for e in eval_entries)
            best_ep = eval_entries[[e["eval_f1"] for e in eval_entries].index(best_f1_val)].get("epoch", "?")
            last_f1 = eval_entries[-1]["eval_f1"]
            lines.append(f"  Best F1 : {best_f1_val:.4f} @ epoch {best_ep}")
            lines.append(f"  Last F1 : {last_f1:.4f}")
            lines.append(f"  Epochs  : {len(eval_entries)}")

        if epoch_times:
            lines.append(f"  Total training time : {sum(epoch_times):.0f}s")
            lines.append(f"  Avg per epoch       : {sum(epoch_times)/len(epoch_times):.0f}s")

        if full_metrics and "confidence_stats" in full_metrics:
            cs = full_metrics["confidence_stats"]
            lines.append("")
            lines.append("── Confidence Stats ──")
            lines.append(f"  mean={cs['mean']:.3f}  std={cs['std']:.3f}"
                         f"  p50={cs['p50']:.3f}  p95={cs['p95']:.3f}")

        # render as text figure
        fig, ax = plt.subplots(figsize=(9, max(4, len(lines) * 0.35 + 1)))
        ax.axis("off")
        text_body = "\n".join(lines)
        ax.text(
            0.03, 0.97, text_body,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F5F5F5", "edgecolor": "#BDBDBD"},
        )
        fig.suptitle(f"{title_prefix} — Summary", fontsize=12, fontweight="bold")
        fig.tight_layout()
        _savefig("15_summary_panel.png", fig)
    except Exception as ex:
        log.warning("summary_panel 그래프 실패: %s", ex)

    # ── 16. Overfitting Analysis: train vs val loss per epoch ──────────
    try:
        eval_entries = [e for e in log_history if "eval_loss" in e]
        step_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
        if eval_entries and step_entries:
            # 에폭별 평균 학습 loss 계산
            import math
            epoch_train_loss: dict = {}
            for e in step_entries:
                ep = int(math.floor(float(e.get("epoch", 0))))
                if ep not in epoch_train_loss:
                    epoch_train_loss[ep] = []
                epoch_train_loss[ep].append(e["loss"])
            avg_train_loss = {
                ep: sum(v) / len(v) for ep, v in epoch_train_loss.items() if v
            }

            val_epochs = [e.get("epoch", i + 1) for i, e in enumerate(eval_entries)]
            val_losses = [e["eval_loss"] for e in eval_entries]
            val_f1s = [e.get("eval_f1", float("nan")) for e in eval_entries]

            # 에폭 정수로 정렬된 train loss
            tr_epochs = sorted(avg_train_loss.keys())
            tr_losses = [avg_train_loss[ep] for ep in tr_epochs]

            fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

            # 서브플롯 1: Train vs Val Loss
            ax = axes[0]
            ax.plot(tr_epochs, tr_losses, color="#42A5F5", linewidth=1.5,
                    marker="o", markersize=3, label="Train Loss (avg per epoch)")
            ax.plot(val_epochs, val_losses, color="#F44336", linewidth=2.0,
                    marker="s", markersize=4, label="Val Loss")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{title_prefix} — Overfitting Analysis")

            # 서브플롯 2: 일반화 격차 (Generalization Gap)
            ax2 = axes[1]
            # val_epochs와 tr_epochs 정렬 맞추기 (정수 에폭으로 align)
            val_ep_int = [int(round(ep)) for ep in val_epochs]
            gaps = []
            gap_epochs = []
            for i, ep in enumerate(val_ep_int):
                if ep in avg_train_loss:
                    gap = val_losses[i] - avg_train_loss[ep]
                    gaps.append(gap)
                    gap_epochs.append(val_epochs[i])
            if gaps:
                gap_colors = ["#F44336" if g > 0.05 else "#4CAF50" for g in gaps]
                ax2.bar(gap_epochs, gaps, color=gap_colors, alpha=0.7, width=0.6,
                        label="Val Loss − Train Loss")
                ax2.axhline(0, color="gray", linewidth=0.8)
                ax2.axhline(0.05, color="#FF9800", linestyle="--", linewidth=1.0,
                            alpha=0.7, label="threshold=0.05")
            ax2.set_ylabel("Gap (Val − Train)")
            ax2.legend(fontsize=9)
            ax2.grid(True, axis="y", alpha=0.3)

            # 서브플롯 3: Val F1 추이 (있으면)
            ax3 = axes[2]
            valid_f1 = [(ep, f) for ep, f in zip(val_epochs, val_f1s)
                        if not (isinstance(f, float) and f != f)]
            if valid_f1:
                f1_ep, f1_val = zip(*valid_f1)
                ax3.plot(f1_ep, f1_val, color="#AB47BC", linewidth=2.0,
                         marker="D", markersize=4, label="Val Span F1")
                best_f1 = max(f1_val)
                best_ep_f1 = f1_ep[list(f1_val).index(best_f1)]
                ax3.axvline(best_ep_f1, color="#AB47BC", linestyle=":",
                            alpha=0.6, linewidth=1.2)
                ax3.set_ylim(0, 1.05)
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Val F1")
            ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

            fig.tight_layout()
            _savefig("16_overfitting_analysis.png", fig)
    except Exception as ex:
        log.warning("overfitting_analysis 그래프 실패: %s", ex)

    # ── 17. Per-label False Positive / False Negative Analysis ────────
    try:
        if full_metrics and "fp_per_label" in full_metrics:
            fp_data = full_metrics["fp_per_label"]
            fn_data = full_metrics["fn_per_label"]
            tp_data = full_metrics["tp_per_label"]
            labels_sorted = sorted(fp_data.keys(), key=lambda l: fp_data[l] + fn_data.get(l, 0), reverse=True)

            if labels_sorted:
                n = len(labels_sorted)
                fp_vals = [fp_data[l] for l in labels_sorted]
                fn_vals = [fn_data.get(l, 0) for l in labels_sorted]
                tp_vals = [tp_data.get(l, 0) for l in labels_sorted]
                prec_vals = [tp / (tp + fp) if (tp + fp) > 0 else 0.0
                             for tp, fp in zip(tp_vals, fp_vals)]
                rec_vals = [tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            for tp, fn in zip(tp_vals, fn_vals)]

                fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 0.35 + 2), 10))

                x = range(n)
                # Top: FP and FN counts stacked
                ax_fp = axes[0]
                bar_w = 0.38
                ax_fp.bar([i - bar_w/2 for i in x], fp_vals, width=bar_w,
                          color="#EF5350", alpha=0.85, label="False Positives (FP)")
                ax_fp.bar([i + bar_w/2 for i in x], fn_vals, width=bar_w,
                          color="#FFA726", alpha=0.85, label="False Negatives (FN)")
                ax_fp.set_xticks(list(x))
                ax_fp.set_xticklabels(labels_sorted, rotation=55, ha="right", fontsize=8)
                ax_fp.set_ylabel("Token Count")
                ax_fp.legend(fontsize=9)
                ax_fp.grid(True, axis="y", alpha=0.3)
                ax_fp.set_title(f"{title_prefix} — Per-label FP / FN Count (token-level)")

                # Bottom: Precision and Recall per label
                ax_pr = axes[1]
                ax_pr.bar([i - bar_w/2 for i in x], prec_vals, width=bar_w,
                          color="#42A5F5", alpha=0.85, label="Precision")
                ax_pr.bar([i + bar_w/2 for i in x], rec_vals, width=bar_w,
                          color="#66BB6A", alpha=0.85, label="Recall")
                ax_pr.axhline(0.8, color="red", linestyle="--", linewidth=0.8,
                              alpha=0.6, label="P=0.8 threshold")
                ax_pr.set_xticks(list(x))
                ax_pr.set_xticklabels(labels_sorted, rotation=55, ha="right", fontsize=8)
                ax_pr.set_ylabel("Score")
                ax_pr.set_ylim(0, 1.05)
                ax_pr.legend(fontsize=9)
                ax_pr.grid(True, axis="y", alpha=0.3)
                ax_pr.set_title(f"{title_prefix} — Per-label Precision / Recall (token-level)")

                fig.tight_layout()
                _savefig("17_false_positive_analysis.png", fig)

                # False alarm rate on O tokens
                fa = full_metrics.get("false_alarm_stats", {})
                if fa:
                    total_O = fa.get("total_O_gold", 0)
                    fp_O = fa.get("false_positives_on_O", 0)
                    far = fa.get("false_alarm_rate", 0.0)
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.bar(["False Positives\non O-labeled tokens",
                             "Correct O\npredictions"],
                            [fp_O, total_O - fp_O],
                            color=["#EF5350", "#66BB6A"], alpha=0.85)
                    ax2.set_ylabel("Token Count")
                    ax2.set_title(
                        f"{title_prefix} — False Alarm Rate on Background (O) tokens\n"
                        f"FAR = {far:.4f}  ({fp_O}/{total_O} O-tokens mis-tagged as entity)"
                    )
                    ax2.grid(True, axis="y", alpha=0.3)
                    for bar, val in zip(ax2.patches, [fp_O, total_O - fp_O]):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_O*0.01,
                                 f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                    fig2.tight_layout()
                    _savefig("18_false_alarm_rate.png", fig2)
    except Exception as ex:
        log.warning("false_positive_analysis 그래프 실패: %s", ex)

    return saved


# ═══════════════════════════════════════════════════════════════════════
# 데이터 로딩
