"""학습/평가 메트릭 헬퍼 — 가중치 통계, GPU 메모리, 라벨별 정밀 메트릭."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

log = logging.getLogger(__name__)


def _weight_stats(model: Any) -> Dict[str, Any]:
    """requires_grad 파라미터의 통계 (mean/std/min/max) 수집."""
    import numpy as np

    stats: Dict[str, Any] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            arr = param.detach().cpu().float().numpy().ravel()
            stats[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "numel": int(arr.size),
            }
    return stats


def _mem_stats() -> Dict[str, float]:
    """현재 프로세스의 CPU/GPU 메모리 사용량."""
    result: Dict[str, float] = {}
    try:
        import os

        import psutil

        result["cpu_rss_mb"] = psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        pass
    if torch is not None and torch.cuda.is_available():
        result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
        result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
    return result


def _np_softmax(x: Any) -> Any:
    """numpy 배열에 softmax 적용 (마지막 축 기준)."""
    import numpy as np

    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _compute_full_metrics(
    preds_raw: Any,   # (N, seq_len, num_labels) numpy array
    gold: Any,        # (N, seq_len) numpy array
    id2label: Dict[int, str],
    label_list: List[str],
) -> Dict[str, Any]:
    """test set 예측으로부터 포괄적 메트릭 계산.

    반환 딕셔너리 키:
        flat_preds_names, flat_golds_names  - 유효 토큰 예측/실제 라벨 리스트
        token_classification_report         - sklearn per-label precision/recall/F1
        span_classification_report          - seqeval entity-level (설치 시)
        confusion_matrix                    - {"matrix": 2D list, "labels": list}
        confidence_stats                    - 전체 confidence 분포 통계
        confidence_per_label                - 라벨별 confidence 통계 (percentiles)
        confidence_flat                     - 유효 토큰 max softmax prob 전체 목록
    """
    import numpy as np

    preds_ids = np.argmax(preds_raw, axis=-1)
    valid_mask = gold != -100

    # ── 1. flat 라벨 리스트 ───────────────────────────────────────────
    flat_preds = [id2label.get(int(p), "O") for p, g in zip(preds_ids.ravel(), gold.ravel()) if g != -100]
    flat_golds = [id2label.get(int(g), "O") for g in gold.ravel() if g != -100]

    result: Dict[str, Any] = {
        "flat_preds_names": flat_preds,
        "flat_golds_names": flat_golds,
    }

    # ── 2. token-level classification report (sklearn) ───────────────
    try:
        from sklearn.metrics import classification_report as sk_report, confusion_matrix as sk_cm
        report_dict = sk_report(flat_golds, flat_preds, output_dict=True, zero_division=0)
        result["token_classification_report"] = report_dict

        # confusion matrix - use all labels present
        present_labels = sorted({l for l in flat_golds + flat_preds if l != "O"})
        if present_labels:
            cm = sk_cm(flat_golds, flat_preds, labels=present_labels)
            result["confusion_matrix"] = {
                "matrix": cm.tolist(),
                "labels": present_labels,
            }
    except ImportError:
        log.debug("sklearn 없음 — token classification report 스킵")
    except Exception as ex:
        log.warning("classification_report 실패: %s", ex)

    # ── 3. seqeval entity-level F1 ────────────────────────────────────
    try:
        from seqeval.metrics import classification_report as seq_report

        # 시퀀스별로 재구성 (subword -100 스킵)
        true_seqs: List[List[str]] = []
        pred_seqs: List[List[str]] = []
        for i in range(gold.shape[0]):
            t_seq = [id2label.get(int(l), "O") for l in gold[i] if l != -100]
            p_seq = [id2label.get(int(p), "O") for p, g in zip(preds_ids[i], gold[i]) if g != -100]
            true_seqs.append(t_seq)
            pred_seqs.append(p_seq)

        span_report = seq_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
        result["span_classification_report"] = span_report
    except ImportError:
        log.debug("seqeval 없음 — span F1 스킵")
    except Exception as ex:
        log.warning("seqeval 평가 실패: %s", ex)

    # ── 4. confidence 분포 ────────────────────────────────────────────
    probs = _np_softmax(preds_raw)
    max_probs = np.max(probs, axis=-1)
    flat_confs = max_probs[valid_mask].ravel()
    result["confidence_flat"] = flat_confs.tolist()

    if len(flat_confs) > 0:
        result["confidence_stats"] = {
            "mean":  float(flat_confs.mean()),
            "std":   float(flat_confs.std()),
            "min":   float(flat_confs.min()),
            "max":   float(flat_confs.max()),
            "p25":   float(np.percentile(flat_confs, 25)),
            "p50":   float(np.percentile(flat_confs, 50)),
            "p75":   float(np.percentile(flat_confs, 75)),
            "p90":   float(np.percentile(flat_confs, 90)),
            "p95":   float(np.percentile(flat_confs, 95)),
        }

    # ── 5. 라벨별 confidence 분포 통계 ───────────────────────────────
    per_label_conf: Dict[str, Any] = {}
    for label_id, label_name in id2label.items():
        mask = gold == label_id
        if mask.any():
            lc = max_probs[mask].ravel()
            per_label_conf[label_name] = {
                "mean": float(lc.mean()),
                "std":  float(lc.std()),
                "p50":  float(np.percentile(lc, 50)),
                "p25":  float(np.percentile(lc, 25)),
                "p75":  float(np.percentile(lc, 75)),
                "count": int(len(lc)),
            }
    result["confidence_per_label"] = per_label_conf

    # ── 6. PR curve (entity vs O) ─────────────────────────────────────
    # score = 1 - P(O) per token (higher → more likely an entity)
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score

        o_id = None
        for k, v in id2label.items():
            if v == "O":
                o_id = k
                break

        if o_id is not None:
            # binary: entity=1 if gold != O
            y_true_bin = np.array([1 if g != "O" else 0 for g in flat_golds], dtype=int)
            # score = 1 - P(O)
            o_probs = _np_softmax(preds_raw)[:, :, o_id]
            flat_o_probs = np.array(
                [float(o_probs[i, j]) for i in range(gold.shape[0]) for j in range(gold.shape[1]) if gold[i, j] != -100]
            )
            entity_scores = 1.0 - flat_o_probs

            if y_true_bin.sum() > 0:
                prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true_bin, entity_scores)
                ap = float(average_precision_score(y_true_bin, entity_scores))
                result["pr_curve"] = {
                    "precision": prec_arr.tolist(),
                    "recall": rec_arr.tolist(),
                    "thresholds": thr_arr.tolist(),
                    "average_precision": ap,
                }
    except ImportError:
        log.debug("sklearn 없음 — PR curve 스킵")
    except Exception as ex:
        log.warning("PR curve 계산 실패: %s", ex)

    # ── 7. per-label false positive analysis ─────────────────────────
    try:
        from collections import Counter
        # FP per label: token predicted as label X but gold is not label X
        # FN per label: token gold is label X but predicted as something else
        # True entity labels (exclude O)
        entity_labels = sorted({l for l in flat_golds + flat_preds if l != "O"})
        fp_per_label: Dict[str, int] = {}
        fn_per_label: Dict[str, int] = {}
        tp_per_label: Dict[str, int] = {}
        for lbl in entity_labels:
            tp = sum(1 for g, p in zip(flat_golds, flat_preds) if g == lbl and p == lbl)
            fp = sum(1 for g, p in zip(flat_golds, flat_preds) if g != lbl and p == lbl)
            fn = sum(1 for g, p in zip(flat_golds, flat_preds) if g == lbl and p != lbl)
            tp_per_label[lbl] = tp
            fp_per_label[lbl] = fp
            fn_per_label[lbl] = fn
        result["fp_per_label"] = fp_per_label
        result["fn_per_label"] = fn_per_label
        result["tp_per_label"] = tp_per_label
        # False positive rate on tokens that were predicted as entity but are O in gold
        total_O_gold = sum(1 for g in flat_golds if g == "O")
        false_positive_on_O = sum(1 for g, p in zip(flat_golds, flat_preds) if g == "O" and p != "O")
        result["false_alarm_stats"] = {
            "total_O_gold": total_O_gold,
            "false_positives_on_O": false_positive_on_O,
            "false_alarm_rate": false_positive_on_O / total_O_gold if total_O_gold > 0 else 0.0,
        }
    except Exception as ex:
        log.warning("FP analysis 실패: %s", ex)

    return result

