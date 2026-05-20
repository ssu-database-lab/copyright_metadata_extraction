"""HuggingFace Trainer 콜백 — epoch timer, grad-norm, 전체 학습 이벤트 로깅."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


class _EpochTimerCallback(TrainerCallback):
    """Epoch별 소요 시간 기록."""

    def __init__(self) -> None:
        self.epoch_times: List[float] = []
        self._epoch_start: Optional[float] = None

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._epoch_start is not None:
            self.epoch_times.append(time.perf_counter() - self._epoch_start)
            self._epoch_start = None


class _GradNormCallback(TrainerCallback):
    """Step별 gradient norm 기록 (log_interval 스텝마다)."""

    def __init__(self, log_interval: int = 50) -> None:
        self.log_interval = log_interval
        self.grad_norms: List[Dict[str, Any]] = []

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if model is None or state.global_step % self.log_interval != 0:
            return
        layer_norms: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_norms[name] = float(param.grad.norm().item())
        if layer_norms:
            self.grad_norms.append({"step": state.global_step, "layer_norms": layer_norms})


class _FullLoggingCallback(TrainerCallback):
    """HF Trainer 의 모든 이벤트를 training.log 에 실시간 기록 (line-buffered).

    기록 항목:
      on_train_begin   — 환경·TrainingArguments·모델·데이터셋 스냅샷
      on_epoch_begin   — 에폭 시작 시각
      on_epoch_end     — 에폭 소요 시간
      on_log           — step 단위 loss / learning_rate / grad_norm / 임의 metric
      on_evaluate      — eval_*  metric 일체
      on_save          — checkpoint 저장 + best_metric
      on_step_end      — global_step 마다 간단 heartbeat (500 step 간격)
      on_train_end     — best_metric · best_model_checkpoint · 총 스텝 수
    """

    def __init__(self, log_path: Path, extra_info: Dict[str, Any]):
        self.log_path = Path(log_path)
        self.extra_info = extra_info
        self.fp: Optional[Any] = None
        self._epoch_start: Optional[float] = None
        self._train_start: Optional[float] = None

    def _ensure_open(self) -> None:
        if self.fp is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.fp = self.log_path.open("a", encoding="utf-8", buffering=1)

    def _write(self, event: str, data: Dict[str, Any]) -> None:
        self._ensure_open()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        assert self.fp is not None
        self.fp.write(f"\n[{ts}] {event}\n")
        for k, v in data.items():
            try:
                sval = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                sval = repr(v)
            self.fp.write(f"  {k} = {sval}\n")

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:
        import sys as _sys

        self._train_start = time.perf_counter()

        env: Dict[str, Any] = {
            "python": _sys.version.split()[0],
            "platform": _sys.platform,
        }
        try:
            import transformers as _tr  # noqa
            env["transformers"] = _tr.__version__
        except Exception:
            pass
        try:
            import torch as _t
            env["torch"] = _t.__version__
            env["cuda_available"] = _t.cuda.is_available()
            if _t.cuda.is_available():
                env["cuda_version"] = _t.version.cuda
                env["gpu_name"] = _t.cuda.get_device_name(0)
                env["gpu_vram_gb"] = round(
                    _t.cuda.get_device_properties(0).total_memory / 1e9, 2
                )
        except Exception:
            pass
        try:
            import peft as _peft  # noqa
            env["peft"] = _peft.__version__
        except Exception:
            pass
        try:
            import numpy as _np
            env["numpy"] = _np.__version__
        except Exception:
            pass

        # TrainingArguments 전량 덤프
        try:
            args_dict = args.to_dict()
        except Exception:
            args_dict = {k: getattr(args, k, None) for k in dir(args)
                         if not k.startswith("_") and not callable(getattr(args, k, None))}

        # 모델 정보
        model_info: Dict[str, Any] = {}
        if model is not None:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model_info["total_params"] = total_params
                model_info["trainable_params"] = trainable
                model_info["trainable_ratio"] = round(trainable / total_params, 6) if total_params else 0
                model_info["model_class"] = model.__class__.__name__
                if hasattr(model, "config"):
                    cfg = model.config
                    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                              "vocab_size", "max_position_embeddings", "model_type"):
                        if hasattr(cfg, k):
                            model_info[f"config.{k}"] = getattr(cfg, k)
            except Exception as ex:
                model_info["error"] = str(ex)

        # 훈련 프로토콜 명시 (k-fold / XGBoost / early_stopping 여부)
        protocol = {
            "k_fold": False,
            "k_fold_n_splits": None,
            "xgboost": False,
            "random_forest": False,
            "boosting": False,
            "sklearn_classifier": False,
            "neural_backbone": "transformer (HF AutoModelForTokenClassification)",
            "task_head": "token_classification (BIO 3-way per label)",
            "holdout_split": "train/val/test 3-way (8/12, 2/12, 2/12), SPLIT_SEED=42",
            "early_stopping": args.load_best_model_at_end,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "load_best_model_at_end": args.load_best_model_at_end,
            "fp16": args.fp16,
            "optimizer": args.optim,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        }

        self._write("train_begin", {
            "extra_info": self.extra_info,
            "environment": env,
            "training_arguments": args_dict,
            "training_protocol": protocol,
            "model": model_info,
            "state_initial": {
                "num_train_epochs": state.num_train_epochs,
                "max_steps": state.max_steps,
                "global_step": state.global_step,
            },
        })

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self._epoch_start = time.perf_counter()
        self._write("epoch_begin", {
            "epoch": state.epoch,
            "global_step": state.global_step,
        })

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        dur = time.perf_counter() - self._epoch_start if self._epoch_start else None
        self._write("epoch_end", {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "duration_sec": round(dur, 2) if dur else None,
        })

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if state.global_step > 0 and state.global_step % 500 == 0:
            self._write("step_heartbeat", {
                "global_step": state.global_step,
                "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            })

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        if not logs:
            return
        entry = {"global_step": state.global_step, "epoch": state.epoch}
        entry.update(logs)
        self._write("log", entry)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        if metrics:
            self._write("evaluate", {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "metrics": metrics,
            })

    def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self._write("save_checkpoint", {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
        })

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        total = time.perf_counter() - self._train_start if self._train_start else None
        self._write("train_end", {
            "total_time_sec": round(total, 2) if total else None,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "log_history_length": len(state.log_history) if state.log_history else 0,
        })
        # 전체 log_history 덤프 (post-hoc 분석용)
        try:
            self._write("log_history_full", {"history": state.log_history})
        except Exception:
            pass
        if self.fp:
            self.fp.close()
            self.fp = None


# ═══════════════════════════════════════════════════════════════════════
