"""매우 상세한 학습 로거 — 어디서든 부착해서 모든 내부 변화를 기록.

기존 `_FullLoggingCallback` (token_cls.py) 가 캡처하지 못한 항목들을 보충:
  (a) 파라미터별 weight 통계 (step 마다 layer 별 mean/std/L2/min/max)
  (b) 파라미터별 gradient 통계  (on_pre_optimizer_step 시점 — grad 살아있는 유일한 hook)
  (c) GPU memory · CUDA cache step 별 스냅샷 (+pynvml utilization 가능 시)
  (d) LR scheduler 곡선 (scalars 안에 learning_rate 가 들어감)
  (e) [skipped v1] Data loader 샘플링 분포
  (f) Tokenizer · model.config 별도 dump
  (g) Random state (python·numpy·torch·cuda) step 별 해시
  (h) step 별 wall-clock duration (on_step_begin → on_step_end)
  (i) gradient accumulation sub-step counter (gradient_accumulation_steps>1 시 의미)
  (j) save 시 어댑터 파일 fingerprint (path / size / sha256-head)

출력 디렉터리 구조 (out_dir 아래):
    env.json            — 환경·라이브러리·GPU·OS·pip freeze (1회)
    config.json         — TrainingArguments 100+ 필드·model.config·tokenizer·param_count (1회)
    scalars.jsonl       — step 마다 loss·learning_rate·grad_norm·eval_*  (TensorBoard scalar 대응)
    params.jsonl        — step 마다 layer 별 weight/grad 통계
    gpu.jsonl           — step 마다 GPU memory/cache (+util)
    events.jsonl        — Trainer 모든 callback 이벤트 (init·train_begin·epoch·log·evaluate·save·train_end)
    random_state.jsonl  — step 마다 RNG state hash
    log_history.json    — 학습 종료 시 state.log_history 전체 (1회)

부착 방법 1 — HF Trainer:
    from module.extractor.ner.full_logger import FullLogger
    flog = FullLogger(out_dir="data/out/run_2026/log")
    trainer.add_callback(flog.callback())

부착 방법 2 — 환경변수 (token_cls.py 가 자동 감지):
    os.environ["PAPER1_LOG_DIR"] = "data/out/run_2026/log"
    # ner_train(...) 호출 시 자동 부착

수동 호출:
    flog.log_env()
    flog.log_config(model, tokenizer, training_args)
    flog.log_scalar(step=100, loss=0.5, learning_rate=4e-5)
    flog.log_params(step=100, model=model)
    flog.log_gpu(step=100)
    flog.log_event("custom_event", anything=42)
    flog.log_random_state(step=100)
    flog.close()

검색 / 분석:
    # 특정 step의 모든 정보:
    grep '"step": 200' data/out/.../log/scalars.jsonl
    grep '"step": 200' data/out/.../log/params.jsonl
    # GPU memory peak:
    jq '.gpus[0].max_memory_allocated_gb' data/out/.../log/gpu.jsonl | sort -n | tail -1
    # encoder.layer.6 의 weight L2 변화:
    jq 'select(.step|.>=0) | .groups.bert[].weight_l2' data/out/.../log/params.jsonl
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from transformers.trainer_callback import TrainerCallback  # type: ignore
    _HAS_HF = True
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass
    _HAS_HF = False


# ════════════════════════════════════════════════════════════════════════
# FullLogger — 단일 인스턴스가 모든 jsonl/json 파일을 관리.
# ════════════════════════════════════════════════════════════════════════


class FullLogger:
    """매우 상세한 학습 로거.

    Args:
        out_dir: 로그 디렉터리 (자동 생성).
        log_params_every: step 몇 번마다 layer 별 weight/grad 통계 기록 (default 50).
        log_gpu_every: step 몇 번마다 GPU memory 기록 (default 10).
        log_random_every: step 몇 번마다 RNG state 기록 (default 100).
        max_pip_freeze: pip freeze 상위 N개만 env.json 에 기록 (default 200).
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        log_params_every: int = 50,
        log_gpu_every: int = 10,
        log_random_every: int = 100,
        max_pip_freeze: int = 200,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_params_every = max(1, int(log_params_every))
        self.log_gpu_every = max(1, int(log_gpu_every))
        self.log_random_every = max(1, int(log_random_every))
        self.max_pip_freeze = max_pip_freeze

        self._scalars = (self.out_dir / "scalars.jsonl").open("a", encoding="utf-8", buffering=1)
        self._params = (self.out_dir / "params.jsonl").open("a", encoding="utf-8", buffering=1)
        self._gpu = (self.out_dir / "gpu.jsonl").open("a", encoding="utf-8", buffering=1)
        self._events = (self.out_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self._random = (self.out_dir / "random_state.jsonl").open("a", encoding="utf-8", buffering=1)

        self._env_logged = False
        self._config_logged = False
        self._closed = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        for fp in (self._scalars, self._params, self._gpu, self._events, self._random):
            try:
                fp.close()
            except Exception:
                pass
        self._closed = True

    @staticmethod
    def _ts() -> str:
        return datetime.now().isoformat(timespec="microseconds")

    def _write(self, fp, payload: Dict[str, Any]) -> None:
        try:
            fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            fp.write(json.dumps({"_log_write_error": str(e), "ts": self._ts()}) + "\n")

    # ── one-time dumps ─────────────────────────────────────────────────

    def log_env(self) -> None:
        """환경·라이브러리·GPU·OS·pip freeze 1회 dump → env.json"""
        if self._env_logged:
            return
        self._env_logged = True
        env: Dict[str, Any] = {
            "ts": self._ts(),
            "platform": platform.platform(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "platform_processor": platform.processor(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
            "argv": list(sys.argv),
            "pid": os.getpid(),
            "env_subset": {
                k: os.environ.get(k, "")
                for k in (
                    "TOKENIZERS_PARALLELISM",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "HF_HOME",
                    "TRANSFORMERS_CACHE",
                    "PAPER1_CONFIG",
                    "PAPER1_LOG_DIR",
                    "PAPER1_TRAINING_LOG",
                )
            },
        }
        for lib in ("torch", "transformers", "datasets", "peft", "accelerate",
                    "numpy", "scipy", "sklearn", "tokenizers", "safetensors", "huggingface_hub"):
            try:
                m = __import__(lib)
                env[f"{lib}_version"] = getattr(m, "__version__", "?")
            except ImportError:
                env[f"{lib}_version"] = None

        if _HAS_TORCH and torch is not None:
            env["torch_cuda_available"] = bool(torch.cuda.is_available())
            env["torch_cuda_version"] = torch.version.cuda
            try:
                env["torch_cudnn_version"] = (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available() else None
                )
            except Exception:
                env["torch_cudnn_version"] = None
            env["torch_num_threads"] = torch.get_num_threads()
            env["torch_default_dtype"] = str(torch.get_default_dtype())
            if torch.cuda.is_available():
                env["gpu_count"] = torch.cuda.device_count()
                gpus = []
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "name": p.name,
                        "total_memory_gb": round(p.total_memory / 1e9, 3),
                        "major": p.major,
                        "minor": p.minor,
                        "multi_processor_count": p.multi_processor_count,
                    })
                env["gpus"] = gpus

        try:
            import subprocess
            r = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=20,
            )
            pkgs = sorted(line.strip() for line in r.stdout.splitlines() if line.strip())
            env["pip_packages_count"] = len(pkgs)
            env["pip_packages"] = pkgs[: self.max_pip_freeze]
        except Exception as e:
            env["pip_freeze_error"] = str(e)

        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            env["cpu_count_physical"] = psutil.cpu_count(logical=False)
            env["cpu_count_logical"] = psutil.cpu_count(logical=True)
            env["memory_total_gb"] = round(vm.total / 1e9, 2)
            env["memory_available_gb"] = round(vm.available / 1e9, 2)
        except ImportError:
            pass

        (self.out_dir / "env.json").write_text(
            json.dumps(env, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def log_config(
        self,
        model: Any = None,
        tokenizer: Any = None,
        training_args: Any = None,
    ) -> None:
        """TrainingArguments + model.config + tokenizer + param count 1회 dump."""
        if self._config_logged:
            return
        self._config_logged = True
        cfg: Dict[str, Any] = {"ts": self._ts()}

        if training_args is not None:
            try:
                cfg["training_args"] = (
                    training_args.to_dict() if hasattr(training_args, "to_dict")
                    else dict(training_args.__dict__)
                )
            except Exception as e:
                cfg["training_args_error"] = str(e)

        if model is not None:
            try:
                if hasattr(model, "config"):
                    cfg["model_config"] = model.config.to_dict()
                cfg["model_class"] = model.__class__.__name__
                cfg["model_arch_repr"] = str(model)[:8000]
                if hasattr(model, "parameters"):
                    total = sum(p.numel() for p in model.parameters())
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    cfg["model_total_params"] = total
                    cfg["model_trainable_params"] = trainable
                    cfg["model_trainable_ratio"] = round(trainable / max(total, 1), 6)
                if hasattr(model, "named_parameters"):
                    cfg["model_param_groups"] = [
                        {
                            "name": n,
                            "shape": list(p.shape),
                            "numel": p.numel(),
                            "dtype": str(p.dtype),
                            "requires_grad": bool(p.requires_grad),
                        }
                        for n, p in model.named_parameters()
                    ]
            except Exception as e:
                cfg["model_error"] = str(e)

        if tokenizer is not None:
            try:
                cfg["tokenizer_class"] = tokenizer.__class__.__name__
                cfg["tokenizer_vocab_size"] = getattr(tokenizer, "vocab_size", None)
                cfg["tokenizer_model_max_length"] = getattr(tokenizer, "model_max_length", None)
                cfg["tokenizer_is_fast"] = getattr(tokenizer, "is_fast", None)
                cfg["tokenizer_padding_side"] = getattr(tokenizer, "padding_side", None)
                cfg["tokenizer_truncation_side"] = getattr(tokenizer, "truncation_side", None)
                cfg["tokenizer_special_tokens"] = {
                    k: getattr(tokenizer, k, None)
                    for k in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token", "bos_token", "eos_token")
                }
                init_kwargs = getattr(tokenizer, "init_kwargs", {})
                cfg["tokenizer_init_kwargs"] = {
                    k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
                    for k, v in (init_kwargs or {}).items()
                }
            except Exception as e:
                cfg["tokenizer_error"] = str(e)

        (self.out_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── per-step streams ───────────────────────────────────────────────

    def log_scalar(self, step: int, **kwargs: Any) -> None:
        """step 마다 loss·lr·grad_norm·eval_* 기록."""
        payload: Dict[str, Any] = {"ts": self._ts(), "step": int(step)}
        for k, v in kwargs.items():
            if v is None:
                continue
            payload[k] = v
        self._write(self._scalars, payload)

    def log_params(self, step: int, model: Any) -> None:
        """step 마다 layer 별 weight/grad 통계 (mean·std·L2·min·max)."""
        if not _HAS_TORCH or model is None or torch is None:
            return
        groups: Dict[str, List[Dict[str, Any]]] = {}
        try:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    top = name.split(".", 1)[0] if "." in name else name
                    pf = p.detach().float()
                    stats: Dict[str, Any] = {
                        "name": name,
                        "shape": list(p.shape),
                        "numel": p.numel(),
                        "dtype": str(p.dtype),
                        "weight_mean": float(pf.mean().item()),
                        "weight_std": float(pf.std().item()) if p.numel() > 1 else 0.0,
                        "weight_l2": float(pf.norm().item()),
                        "weight_min": float(pf.min().item()),
                        "weight_max": float(pf.max().item()),
                        "weight_abs_mean": float(pf.abs().mean().item()),
                    }
                    if p.grad is not None:
                        gf = p.grad.detach().float()
                        stats.update({
                            "grad_mean": float(gf.mean().item()),
                            "grad_std": float(gf.std().item()) if gf.numel() > 1 else 0.0,
                            "grad_l2": float(gf.norm().item()),
                            "grad_min": float(gf.min().item()),
                            "grad_max": float(gf.max().item()),
                            "grad_abs_mean": float(gf.abs().mean().item()),
                        })
                    groups.setdefault(top, []).append(stats)
        except Exception as e:
            self._write(self._params, {"ts": self._ts(), "step": int(step), "_error": str(e)})
            return
        self._write(self._params, {"ts": self._ts(), "step": int(step), "groups": groups})

    def log_gpu(self, step: int) -> None:
        """step 마다 GPU memory/cache (+pynvml util 가능 시)."""
        if not _HAS_TORCH or torch is None or not torch.cuda.is_available():
            return
        gpus = []
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "index": i,
                "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 4),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 4),
                "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 4),
                "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved(i) / 1e9, 4),
            })
        payload: Dict[str, Any] = {"ts": self._ts(), "step": int(step), "gpus": gpus}
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            utils = []
            for i in range(torch.cuda.device_count()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                utils.append({
                    "index": i,
                    "gpu_util_pct": int(u.gpu),
                    "memory_util_pct": int(u.memory),
                    "memory_used_gb": round(m.used / 1e9, 4),
                    "memory_free_gb": round(m.free / 1e9, 4),
                    "memory_total_gb": round(m.total / 1e9, 4),
                })
            payload["nvml"] = utils
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        except Exception:
            pass
        self._write(self._gpu, payload)

    def log_event(self, event: str, **kwargs: Any) -> None:
        """임의 callback 이벤트 기록."""
        payload: Dict[str, Any] = {"ts": self._ts(), "event": event}
        payload.update(kwargs)
        self._write(self._events, payload)

    def log_random_state(self, step: int) -> None:
        """RNG state hash (재현성용)."""
        payload: Dict[str, Any] = {"ts": self._ts(), "step": int(step)}
        try:
            payload["python_random_hash"] = hash(repr(random.getstate()))
        except Exception as e:
            payload["python_random_error"] = str(e)
        if _HAS_NUMPY and np is not None:
            try:
                st = np.random.get_state()
                payload["numpy_random_kind"] = st[0]
                payload["numpy_random_pos"] = int(st[2]) if len(st) > 2 else None
                payload["numpy_random_head_hash"] = hash(bytes(st[1][:16].tobytes())) if len(st) > 1 else None
            except Exception as e:
                payload["numpy_random_error"] = str(e)
        if _HAS_TORCH and torch is not None:
            try:
                rng = torch.get_rng_state()
                payload["torch_rng_hash"] = int(rng.sum().item())
                if torch.cuda.is_available():
                    payload["cuda_rng_hashes"] = [
                        int(s.sum().item())
                        for s in torch.cuda.get_rng_state_all()
                    ]
            except Exception as e:
                payload["torch_rng_error"] = str(e)
        self._write(self._random, payload)

    def log_log_history(self, history: List[Dict[str, Any]]) -> None:
        """학습 종료 시 state.log_history 전체 dump."""
        (self.out_dir / "log_history.json").write_text(
            json.dumps({"ts": self._ts(), "history": history}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── HF Trainer 부착 ────────────────────────────────────────────────

    def callback(self) -> "_FullLoggerCallback":
        """HF Trainer.add_callback() 에 넣을 수 있는 callback 반환."""
        return _FullLoggerCallback(self)


# ════════════════════════════════════════════════════════════════════════
# HF Trainer Callback wrapper
# ════════════════════════════════════════════════════════════════════════


class _FullLoggerCallback(TrainerCallback):  # type: ignore[misc]
    """FullLogger 의 모든 hook 을 HF Trainer 의 callback 으로 자동 호출.

    grad 캡처: on_step_end 시점엔 optimizer.zero_grad() 로 grad 가 사라짐.
    on_pre_optimizer_step (transformers>=4.40) 가 backward 직후·zero_grad 직전 시점이라
    여기서 log_params 호출해야 grad 가 살아있음.
    """

    def __init__(self, flog: FullLogger):
        self.flog = flog
        self._model_ref: Any = None
        self._tokenizer_ref: Any = None
        self._params_logged_at: set = set()  # step 중복 방지
        self._step_t0: Optional[float] = None  # on_step_begin 의 wall-clock
        self._substep_count: int = 0           # gradient accumulation sub-step counter
        self._first_step_config_logged: bool = False  # train_begin 에 model None 일 때 fallback

    def on_init_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.flog.log_event("on_init_end", step=int(getattr(state, "global_step", 0) or 0))

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:
        if model is not None:
            self._model_ref = model
        if "tokenizer" in kw and kw["tokenizer"] is not None:
            self._tokenizer_ref = kw["tokenizer"]
        elif "processing_class" in kw and kw["processing_class"] is not None:
            self._tokenizer_ref = kw["processing_class"]
        self.flog.log_env()
        # model 이 아직 None 이면 log_config 를 보류 → on_step_end 에서 model 캡처 후 다시 시도.
        if self._model_ref is not None:
            self.flog.log_config(self._model_ref, self._tokenizer_ref, args)
            self._first_step_config_logged = True
        self.flog.log_event(
            "on_train_begin",
            step=int(state.global_step or 0),
            epoch=float(state.epoch or 0.0),
            max_steps=int(state.max_steps or 0),
            num_train_epochs=float(getattr(state, "num_train_epochs", 0.0) or 0.0),
            model_ref_captured=self._model_ref is not None,
        )

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.flog.log_event("on_epoch_begin", step=int(state.global_step), epoch=float(state.epoch or 0.0))

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.flog.log_event("on_epoch_end", step=int(state.global_step), epoch=float(state.epoch or 0.0))

    def on_step_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        """step 시작 wall-clock 마크 (on_step_end 에서 duration 계산용)."""
        import time as _t
        self._step_t0 = _t.perf_counter()
        self._substep_count = 0

    def on_substep_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        """gradient accumulation sub-step (gradient_accumulation_steps > 1 일 때 의미)."""
        self._substep_count += 1

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:
        """backward 직후·zero_grad 직전 — grad 가 살아있는 유일한 시점."""
        if model is not None and self._model_ref is None:
            self._model_ref = model
        step = int(state.global_step or 0) + 1  # global_step 은 optimizer.step() 후에 증가
        if step <= 0:
            return
        if step % self.flog.log_params_every == 0 and self._model_ref is not None:
            self.flog.log_params(step, self._model_ref)
            self._params_logged_at.add(step)

    def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> None:
        if model is not None and self._model_ref is None:
            self._model_ref = model
        # train_begin 시점에 model 누락이었으면 첫 step 에서 보충.
        if not self._first_step_config_logged and self._model_ref is not None:
            self.flog.log_config(self._model_ref, self._tokenizer_ref, args)
            self._first_step_config_logged = True
        step = int(state.global_step or 0)
        if step <= 0:
            return
        # step duration (on_step_begin → on_step_end 간 wall-clock).
        if self._step_t0 is not None:
            import time as _t
            dur = _t.perf_counter() - self._step_t0
            self.flog.log_scalar(
                step=step, step_duration_sec=round(dur, 6),
                substeps=self._substep_count or 1,
            )
            self._step_t0 = None
        if step % self.flog.log_gpu_every == 0:
            self.flog.log_gpu(step)
        # log_params 는 on_pre_optimizer_step 에서 처리됨 — fallback (이전 transformers 호환)
        if (step % self.flog.log_params_every == 0 and self._model_ref is not None
                and step not in self._params_logged_at):
            self.flog.log_params(step, self._model_ref)
        if step % self.flog.log_random_every == 0:
            self.flog.log_random_state(step)

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        logs = dict(logs or {})
        logs.pop("step", None)  # log_scalar 의 명시 step= 와 충돌 방지
        logs.setdefault("epoch", float(state.epoch or 0.0))
        self.flog.log_scalar(step=int(state.global_step or 0), **logs)
        self.flog.log_event("on_log", step=int(state.global_step or 0), logs=logs)

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        metrics = dict(metrics or {})
        metrics.pop("step", None)
        metrics.setdefault("epoch", float(state.epoch or 0.0))
        metrics["kind"] = "eval"
        self.flog.log_scalar(step=int(state.global_step or 0), **metrics)
        self.flog.log_event("on_evaluate", step=int(state.global_step or 0), metrics=metrics)

    def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        """save 이벤트 + 어댑터 디렉터리 fingerprint (어떤 step 의 어떤 가중치인지 확증)."""
        artifacts = self._fingerprint_save_dir(args, state)
        self.flog.log_event(
            "on_save",
            step=int(state.global_step or 0),
            best_metric=getattr(state, "best_metric", None),
            best_model_checkpoint=getattr(state, "best_model_checkpoint", None),
            artifacts=artifacts,
        )

    @staticmethod
    def _fingerprint_save_dir(args: Any, state: Any) -> List[Dict[str, Any]]:
        """최근 저장된 checkpoint 디렉터리의 핵심 파일별 size·sha256 (head 64KB)."""
        import hashlib
        out: List[Dict[str, Any]] = []
        try:
            base = Path(getattr(args, "output_dir", "."))
            ckpt = state.best_model_checkpoint if getattr(state, "best_model_checkpoint", None) else None
            roots: List[Path] = []
            if ckpt:
                roots.append(Path(ckpt))
            # 마지막 step 번호 기반 추측 경로도 포함.
            step = int(getattr(state, "global_step", 0) or 0)
            roots.append(base / f"checkpoint-{step}")
            roots.append(base)  # output_dir 자체 (full save 케이스)
            for root in roots:
                if not root.is_dir():
                    continue
                for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "config.json"):
                    p = root / name
                    if not p.is_file():
                        continue
                    sz = p.stat().st_size
                    try:
                        with p.open("rb") as f:
                            head = f.read(64 * 1024)
                        sha = hashlib.sha256(head).hexdigest()
                    except Exception:
                        sha = None
                    out.append({"path": str(p), "size_bytes": sz, "sha256_head64k": sha})
                if out:
                    break  # 첫 매칭 root 만 기록
        except Exception as e:
            out.append({"_fingerprint_error": str(e)})
        return out

    def on_predict(self, args: Any, state: Any, control: Any, metrics: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
        self.flog.log_event("on_predict", step=int(state.global_step or 0), metrics=metrics or {})

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        history = list(getattr(state, "log_history", []) or [])
        self.flog.log_log_history(history)
        self.flog.log_event(
            "on_train_end",
            step=int(state.global_step or 0),
            log_history_len=len(history),
            best_metric=getattr(state, "best_metric", None),
            best_model_checkpoint=getattr(state, "best_model_checkpoint", None),
        )


# ════════════════════════════════════════════════════════════════════════
# 편의 함수
# ════════════════════════════════════════════════════════════════════════


def attach_to_trainer(
    trainer: Any,
    out_dir: Union[str, Path],
    **kwargs: Any,
) -> FullLogger:
    """HF Trainer 에 FullLogger 부착하고 인스턴스 반환."""
    flog = FullLogger(out_dir, **kwargs)
    trainer.add_callback(flog.callback())
    return flog


def from_env() -> Optional[FullLogger]:
    """환경변수 PAPER1_LOG_DIR 가 설정되어 있으면 FullLogger 인스턴스 생성, 아니면 None.

    추가 환경변수:
        PAPER1_LOG_PARAMS_EVERY  — params.jsonl 기록 step 간격 (default 50)
        PAPER1_LOG_GPU_EVERY     — gpu.jsonl   기록 step 간격 (default 10)
        PAPER1_LOG_RANDOM_EVERY  — random_state.jsonl 기록 step 간격 (default 100)
    """
    out_dir = os.environ.get("PAPER1_LOG_DIR")
    if not out_dir:
        return None

    def _ienv(k: str, default: int) -> int:
        v = os.environ.get(k)
        try:
            return int(v) if v else default
        except Exception:
            return default

    return FullLogger(
        out_dir,
        log_params_every=_ienv("PAPER1_LOG_PARAMS_EVERY", 50),
        log_gpu_every=_ienv("PAPER1_LOG_GPU_EVERY", 10),
        log_random_every=_ienv("PAPER1_LOG_RANDOM_EVERY", 100),
    )
