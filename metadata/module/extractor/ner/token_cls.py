"""BERT 계열 token classification 엔진 — 학습 + 예측 (TokenClassNER).

silver BIO jsonl 로 BERT 를 학습하고 학습된 모델로 텍스트를 predict 한다.
predict.py / train.py 는 이 엔진을 호출하는 얇은 wrapper.

이 파일에 남는 것:
    - GPU 최적화 env 변수, ``_amp_kwargs`` (bf16/fp16 자동 선택)
    - LoRA target_modules 매핑 (``_lora_target_modules``)
    - ``TokenClassNER`` 클래스: ``has_adapter``, ``load``, ``train``, ``predict``,
      ``predict_at_thresholds``

분리된 보조 모듈:
    - ``_callbacks.py`` — HF Trainer 콜백 3종 (epoch timer, grad norm, full logging)
    - ``_metrics.py``   — 가중치/메모리 스냅샷, 라벨별 정밀 메트릭
    - ``_plots.py``     — 학습 곡선 png (``save_plots=True`` 일 때만)
    - ``_dataload.py``  — silver jsonl 로드 + 라벨 맵 빌드

외부 연결:
    - ``_runtime.py`` → ``_token_cls_cache`` 가 TokenClassNER 인스턴스 캐싱
    - ``predict.py``  → ``ner_predict`` 안에서 ``TokenClassNER.predict`` 호출
    - ``train.py``    → ``ner_train`` 안에서 ``TokenClassNER(model_dir).train`` 호출
    - ``full_logger.py`` → ``PAPER1_LOG_DIR`` env 지정 시 부착 (research only)
"""
from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
except ImportError:
    AutoModelForTokenClassification = None  # type: ignore[assignment,misc]
    TrainerCallback = object  # type: ignore[assignment,misc]

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore[assignment,misc]
    PeftModel = None  # type: ignore[assignment,misc]

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore[assignment,misc]

ADAPTER_SUBDIR = "adapter"
LABEL_MAP_FILE = "label_map.json"
TRAIN_METHOD_FILE = "train_method.json"
TRAIN_CONFIG_FILE = "train_config.json"
DEBUG_METRICS_FILE = "debug_metrics.json"
PLOTS_SUBDIR = "plots"
HOLDOUT_PREDICT_MAX_ROWS = int(os.environ.get("NER_HOLDOUT_PREDICT_MAX_ROWS", "20000"))
EVAL_MAX_ROWS = int(os.environ.get("NER_EVAL_MAX_ROWS", "20000"))
EVAL_ACCUMULATION_STEPS = int(os.environ.get("NER_EVAL_ACCUMULATION_STEPS", "16"))
# GPU 최적화 — env 변수로 override (저-VRAM 호스트는 GRADIENT_CHECKPOINTING=1 등 보수 모드).
GRADIENT_CHECKPOINTING = os.environ.get("NER_GRADIENT_CHECKPOINTING", "0") == "1"
EVAL_BATCH_MULT = int(os.environ.get("NER_EVAL_BATCH_MULT", "3"))
# TRAIN_BATCH_MULT>1: VRAM 추가 활용. frozen-eval 비교 시 1 유지.
TRAIN_BATCH_MULT = int(os.environ.get("NER_TRAIN_BATCH_MULT", "1"))
DATALOADER_NUM_WORKERS = int(os.environ.get("NER_DATALOADER_NUM_WORKERS", "4"))
DATALOADER_PIN_MEMORY = os.environ.get("NER_DATALOADER_PIN_MEMORY", "1") == "1"

if torch is not None:
    # TF32 matmul on Ampere+ (Blackwell included). ~10-15% speedup, no
    # measurable hit on NER accuracy. Safe to leave on globally.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:  # pragma: no cover - older torch
        pass


def _amp_kwargs() -> Dict[str, bool]:
    """Return TrainingArguments AMP kwargs. Prefer bf16 on Ampere+ (no loss scaler, no NaN)."""
    if torch is None or not torch.cuda.is_available():
        return {}
    try:
        if torch.cuda.is_bf16_supported():
            return {"bf16": True}
    except Exception:
        pass
    return {"fp16": True}

# 모델별 LoRA target_modules 매핑
# PEFT가 자동 감지하지 못하는 아키텍처에 명시 필요
_LORA_TARGET_MODULES_MAP: Dict[str, List[str]] = {
    # DistilBERT 계열
    "distilbert":   ["q_lin", "v_lin"],
    # ALBERT 계열
    "albert":       ["query", "value"],
    # XLNet 계열
    "xlnet":        ["q", "v"],
    # ELECTRA 계열 (KoELECTRA 포함) — 자동 감지 실패 시 대비
    "electra":      ["query", "value"],
    # DeBERTa-v2/v3 계열
    "deberta":      ["query_proj", "value_proj"],
}
_LORA_TARGET_MODULES_DEFAULT: List[str] = ["query", "value"]  # BERT 계열 기본값


def _lora_target_modules(model: Any) -> List[str]:
    """모델 클래스 이름 기반으로 LoRA target_modules 반환.

    PEFT의 자동 감지가 실패하는 아키텍처(DistilBERT 등)를 명시적으로 처리.
    매핑에 없으면 BERT 기본값(query/value)을 사용.
    """
    cls_name = type(model).__name__.lower()
    for key, modules in _LORA_TARGET_MODULES_MAP.items():
        if key in cls_name:
            return modules
    # 모델 내부 레이어를 직접 검사해 실제 존재하는 이름 찾기
    try:
        named = {n.split(".")[-1] for n, _ in model.named_modules()}
        for candidate in ("query", "q_proj", "q_lin", "query_proj"):
            if candidate in named:
                val = candidate.replace("query", "value").replace("q_", "v_").replace("q_proj", "v_proj")
                if val in named:
                    return [candidate, val]
    except Exception:
        pass
    return _LORA_TARGET_MODULES_DEFAULT


# ---------------------------------------------------------------------------
# 분리된 보조 모듈에서 사용 함수 import (구조 단순화)
# ---------------------------------------------------------------------------

from module.extractor.ner._callbacks import (
    _EpochTimerCallback,
    _GradNormCallback,
    _FullLoggingCallback,
)
from module.extractor.ner._metrics import (
    _weight_stats,
    _mem_stats,
    _np_softmax,
    _compute_full_metrics,
)
from module.extractor.ner._plots import _generate_plots
from module.extractor.ner._dataload import _load_records_by_label, build_label_map



class TokenClassNER:
    """Token Classification NER (학습 + 예측).

    fine_tuning_method:
        - "lora"  : PEFT LoRA (기본값, 어댑터 저장)
        - "full"  : 전체 파라미터 학습 (Full Fine-Tuning)
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.adapter_dir = model_dir / ADAPTER_SUBDIR
        self.model: Any = None
        self.tokenizer: Any = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.device = "cpu"
        self._loaded = False
        self.adapter_load_path: Optional[str] = None

    @property
    def has_adapter(self) -> bool:
        lm = self.adapter_dir / LABEL_MAP_FILE
        if not lm.exists():
            return False
        # LoRA: adapter_config.json 존재
        if (self.adapter_dir / "adapter_config.json").exists():
            return True
        # 기타 방법: train_method.json 존재
        return (self.adapter_dir / TRAIN_METHOD_FILE).exists()

    def _load_method(self) -> str:
        """저장된 학습 방법 읽기. 미기록 시 legacy LoRA로 간주."""
        p = self.adapter_dir / TRAIN_METHOD_FILE
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get(
                    "fine_tuning_method", "lora"
                )
            except Exception:
                pass
        if (self.adapter_dir / "adapter_config.json").exists():
            return "lora"
        return "lora"

    def _latest_full_weight_dir(self) -> Optional[Path]:
        """Return the newest HF full-model save dir under adapter/checkpoints."""
        candidates: List[Tuple[float, Path]] = []
        dirs = [self.adapter_dir]
        dirs.extend(p for p in self.adapter_dir.glob("checkpoint-*") if p.is_dir())
        for d in dirs:
            existing = [
                p for p in (d / "model.safetensors", d / "pytorch_model.bin")
                if p.exists()
            ]
            if existing:
                candidates.append((max(p.stat().st_mtime for p in existing), d))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _to_device(self) -> None:
        if torch is not None and torch.cuda.is_available() and self.model is not None:
            self.model = self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"

    # ------------------------------------------------------------------
    # 로드
    # ------------------------------------------------------------------

    def load(self) -> None:
        """베이스 모델 + 학습 방법별 가중치 로드."""
        if self._loaded:
            return
        if AutoModelForTokenClassification is None:
            raise RuntimeError("transformers 패키지가 필요합니다.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), use_fast=True,
        )

        if self.has_adapter:
            label_map_path = self.adapter_dir / LABEL_MAP_FILE
            if not label_map_path.exists():
                log.warning("라벨 맵이 없습니다: %s", label_map_path)
                self.model = None
                self._loaded = True
                return

            label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
            self.id2label = {int(k): v for k, v in label_map["id2label"].items()}
            self.label2id = label_map["label2id"]

            method = self._load_method()

            # full 방식: base 로드 스킵, full adapter 에서 직접 로드 (분류 헤드 포함).
            # 이렇게 하면 base 로드 시점의 "classifier MISSING" 경고가 안 나옴.
            if method == "full":
                base_model = None
            else:
                base_model = AutoModelForTokenClassification.from_pretrained(
                    str(self.model_dir),
                    num_labels=len(self.id2label),
                    id2label=self.id2label,
                    label2id=self.label2id,
                    ignore_mismatched_sizes=True,
                )

            if method == "lora":
                if PeftModel is not None:
                    self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
                    self.adapter_load_path = str(self.adapter_dir.resolve())
                    log.info("LoRA 어댑터 로드: %s", self.adapter_load_path)
                else:
                    log.warning("peft 패키지 없음 — LoRA 어댑터를 로드할 수 없습니다.")
                    self.model = None
                    self.adapter_load_path = None

            elif method == "full":
                # 1순위: 최신 HF safetensors (adapter/ 또는 adapter/checkpoint-*/).
                # 2순위 (legacy): adapter/full_model_weights.pt.
                full_dir = self._latest_full_weight_dir()
                weights_path = self.adapter_dir / "full_model_weights.pt"

                loaded = False
                if full_dir is not None:
                    full_weight = (
                        full_dir / "model.safetensors"
                        if (full_dir / "model.safetensors").exists()
                        else full_dir / "pytorch_model.bin"
                    )
                    legacy_mtime = weights_path.stat().st_mtime if weights_path.exists() else -1
                    if full_weight.stat().st_mtime > legacy_mtime:
                        self.model = AutoModelForTokenClassification.from_pretrained(
                            str(full_dir),
                            num_labels=len(self.id2label),
                            id2label=self.id2label,
                            label2id=self.label2id,
                        )
                        self.adapter_load_path = str(full_dir.resolve())
                        log.info("full 가중치 로드 (HF safetensors): %s", self.adapter_load_path)
                        loaded = True

                if not loaded and weights_path.exists() and torch is not None:
                    # legacy .pt: base 모델 로드 후 state_dict 덮어쓰기.
                    base_model = AutoModelForTokenClassification.from_pretrained(
                        str(self.model_dir),
                        num_labels=len(self.id2label),
                        id2label=self.id2label,
                        label2id=self.label2id,
                        ignore_mismatched_sizes=True,
                    )
                    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                    base_model.load_state_dict(state, strict=True)
                    self.model = base_model
                    self.adapter_load_path = str(weights_path.resolve())
                    log.info("full 가중치 로드 (legacy .pt): %s", self.adapter_load_path)
                    loaded = True

                if not loaded:
                    log.warning(
                        "full 가중치 파일 없음 (HF: %s, legacy: %s) — 학습된 NER 없이 동작.",
                        full_dir, weights_path,
                    )
                    self.model = None
                    self.adapter_load_path = None

            else:
                log.warning("알 수 없는 학습 방법 '%s' — 베이스 가중치만 사용.", method)
                self.model = base_model
                self.adapter_load_path = None

        else:
            log.info("학습된 어댑터가 없습니다 (%s). 학습 필요.", self.model_dir.name)
            self.model = None
            self.adapter_load_path = None

        if self.model is not None:
            self.model.eval()
            self._to_device()
        self._loaded = True

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def train(
        self,
        train_dir: Path,
        epochs: int = 5,
        batch_size: int = 8,
        lr: float = 2e-5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        fine_tuning_method: str = "lora",
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.01,
        debug: bool = False,
        train_ratio: float = 8 / 12,
        val_ratio: float = 2 / 12,
        test_ratio: float = 2 / 12,
        split_seed: int = 42,
        save_plots: bool = False,
        early_stopping_patience: int = 0,
        extra_input_dirs: Optional[List[Path]] = None,
        negative_input_dirs: Optional[List[Path]] = None,
        max_per_label: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """NER 학습.

        fine_tuning_method:
            "lora"  — PEFT LoRA (기본값)
            "full"  — 전체 파라미터 학습
        """
        # ── method 정규화 및 검증 ──────────────────────────────────────
        method = fine_tuning_method.lower().strip()
        if method == "full_finetuning":
            method = "full"
        if method not in ("lora", "full"):
            log.error("알 수 없는 fine_tuning_method: %r (지원: lora, full)", fine_tuning_method)
            return False, {}

        if AutoModelForTokenClassification is None:
            log.error("transformers 패키지가 필요합니다.")
            return False, {}
        if Dataset is None:
            log.error("datasets 패키지가 필요합니다.")
            return False, {}

        # LoRA 전용 peft 가드 (다른 방법은 peft 불필요)
        if method == "lora" and LoraConfig is None:
            log.error("peft 패키지가 필요합니다 (fine_tuning_method='lora').")
            return False, {}

        if debug:
            from module.extractor.ner._runtime import configure_ner_debug, ner_debug_print

            configure_ner_debug(True)
            ner_debug_print(
                f"[NER debug][token_cls.train] model_dir={self.model_dir} "
                f"train_dir={train_dir} method={method} epochs={epochs} "
                f"batch={batch_size} lr={lr} warmup_ratio={warmup_ratio} "
                f"weight_decay={weight_decay}"
            )

        # ── 라벨별 데이터 로드 ────────────────────────────────────────
        # 1) 기본 silver 디렉터리 로드
        all_silver_dirs = [train_dir]
        # extra_input_dirs: 추가 silver (다른 카테고리 포함 가능, 라벨 검증 없음)
        if extra_input_dirs:
            all_silver_dirs.extend(extra_input_dirs)

        pos_by_label = _load_records_by_label(all_silver_dirs)
        if not pos_by_label:
            log.warning("학습 데이터가 없습니다: %s", all_silver_dirs)
            return False, {}

        total_extra_loaded = 0
        if extra_input_dirs:
            extra_by_label = _load_records_by_label(extra_input_dirs)
            total_extra_loaded = sum(len(v) for v in extra_by_label.values())
            print(f"  추가 silver {len(extra_by_label)}개 라벨 그룹, 총 {total_extra_loaded}건")

        # 2) 라벨별 레코드 수 상한 (max_per_label): 병합 후 적용
        if max_per_label is not None and max_per_label > 0:
            import random as _rnd
            _rng = _rnd.Random(split_seed)
            before = sum(len(v) for v in pos_by_label.values())
            for lbl, recs in pos_by_label.items():
                if len(recs) > max_per_label:
                    pos_by_label[lbl] = _rng.sample(recs, max_per_label)
            after = sum(len(v) for v in pos_by_label.values())
            print(f"  [max_per_label={max_per_label}] {before}건 → {after}건 "
                  f"(삭제 {before - after}건)")

        # 3) negative 디렉터리 로드
        neg_by_label: Dict[str, List[Dict[str, Any]]] = {}
        if negative_input_dirs:
            neg_by_label = _load_records_by_label(negative_input_dirs)
            if debug:
                from module.extractor.ner._runtime import ner_debug_print
                total_neg = sum(len(v) for v in neg_by_label.values())
                ner_debug_print(
                    f"[NER debug][token_cls.train] negative by_label={list(neg_by_label.keys())} "
                    f"total={total_neg}"
                )

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            log.error(
                "train_ratio + val_ratio + test_ratio 합이 1이어야 합니다. "
                "현재: %s + %s + %s",
                train_ratio,
                val_ratio,
                test_ratio,
            )
            return False, {}

        # ── 라벨별 비율 분할 → 취합 ──────────────────────────────────
        rng = random.Random(split_seed)
        all_label_groups = sorted(set(pos_by_label) | set(neg_by_label))

        train_records: List[Dict[str, Any]] = []
        _val_list: List[Dict[str, Any]] = []
        _test_list: List[Dict[str, Any]] = []

        per_label_mode: Dict[str, str] = {}
        total_silver_rows = 0

        for lbl_grp in all_label_groups:
            pos_recs = pos_by_label.get(lbl_grp, [])
            neg_recs = neg_by_label.get(lbl_grp, [])
            combined = pos_recs + neg_recs
            total_silver_rows += len(combined)
            if not combined:
                continue

            order   = list(range(len(combined)))
            rng.shuffle(order)
            total_n = len(combined)

            if total_n < 10:
                train_records.extend(combined[i] for i in order)
                per_label_mode[lbl_grp] = "no_split_too_small"
            elif total_n < 30:
                n_tr = max(1, int(total_n * train_ratio))
                train_records.extend(combined[i] for i in order[:n_tr])
                _val_list.extend(combined[i] for i in order[n_tr:])
                per_label_mode[lbl_grp] = "train_val_only"
            else:
                n_tr = int(total_n * train_ratio)
                n_v  = int(total_n * val_ratio)
                train_records.extend(combined[i] for i in order[:n_tr])
                _val_list.extend(combined[i] for i in order[n_tr : n_tr + n_v])
                _test_list.extend(combined[i] for i in order[n_tr + n_v :])
                per_label_mode[lbl_grp] = "silver_split"

        val_records:  Optional[List[Dict[str, Any]]] = _val_list  if _val_list  else None
        test_records: Optional[List[Dict[str, Any]]] = _test_list if _test_list else None

        if not train_records:
            log.warning("학습 데이터가 없습니다 (분할 후 train=0)")
            return False, {}

        total_n = len(train_records) + len(_val_list) + len(_test_list)
        data_split: Dict[str, Any] = {
            "train_ratio":    train_ratio,
            "val_ratio":      val_ratio,
            "test_ratio":     test_ratio,
            "split_seed":     split_seed,
            "total_rows":     total_n,
            "silver_rows":    total_silver_rows,
            "extra_rows":     total_extra_loaded,
            "mode":           "silver_split",
            "per_label_mode": per_label_mode,
            "train_rows":     len(train_records),
            "val_rows":       len(_val_list),
            "test_rows":      len(_test_list),
        }

        eval_cap: Dict[str, Any] = {"max_rows": EVAL_MAX_ROWS}
        if EVAL_MAX_ROWS > 0 and val_records and len(val_records) > EVAL_MAX_ROWS:
            val_records = random.Random(split_seed + 1001).sample(val_records, EVAL_MAX_ROWS)
            eval_cap["val_rows_used"] = len(val_records)
        if EVAL_MAX_ROWS > 0 and test_records and len(test_records) > EVAL_MAX_ROWS:
            test_records = random.Random(split_seed + 1002).sample(test_records, EVAL_MAX_ROWS)
            eval_cap["test_rows_used"] = len(test_records)
        if len(eval_cap) > 1:
            data_split["eval_cap"] = eval_cap

        label_list, label2id = build_label_map(
            train_records + (val_records or []) + (test_records or []),
        )
        id2label = {v: k for k, v in label2id.items()}
        print(
            f"  라벨 {len(label_list)}개, 학습 {len(train_records)}개"
            + (f", 검증 {len(val_records)}" if val_records else "")
            + (f", 평가(홀드아웃) {len(test_records)}" if test_records else "")
        )
        if debug:
            from module.extractor.ner._runtime import ner_debug_print

            ner_debug_print(
                f"[NER debug][token_cls.train] labels={label_list[:20]}"
                f"{'...' if len(label_list) > 20 else ''}"
            )

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        base_model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_dir),
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        # ── Fine-Tuning 방법별 모델 준비 ─────────────────────────────
        if method == "lora":
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                bias="none",
                target_modules=_lora_target_modules(base_model),
                modules_to_save=["classifier"],
            )
            model = get_peft_model(base_model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            print(f"  LoRA: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

        else:  # full
            model = base_model
            for param in model.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters())
            total = trainable
            print(f"  full: {trainable:,} / {total:,} (100.0%)")

        # ── Tokenize ──────────────────────────────────────────────────
        def tokenize_and_align(examples: Dict[str, Any]) -> Dict[str, Any]:
            tokenized = tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                padding=False,
            )
            aligned = []
            for i, labels in enumerate(examples["labels"]):
                word_ids = tokenized.word_ids(batch_index=i)
                ids = []
                prev = None
                for wid in word_ids:
                    if wid is None:
                        ids.append(-100)
                    elif wid != prev:
                        ids.append(label2id.get(labels[wid], 0))
                    else:
                        ids.append(-100)
                    prev = wid
                aligned.append(ids)
            tokenized["labels"] = aligned
            return tokenized

        dataset = Dataset.from_list(train_records)
        tokenized_ds = dataset.map(
            tokenize_and_align, batched=True,
            remove_columns=["tokens", "labels"],
        )

        eval_ds = None
        test_ds = None
        if val_records:
            eval_dset = Dataset.from_list(val_records)
            eval_ds = eval_dset.map(
                tokenize_and_align, batched=True,
                remove_columns=["tokens", "labels"],
            )
        if test_records:
            test_dset = Dataset.from_list(test_records)
            test_ds = test_dset.map(
                tokenize_and_align, batched=True,
                remove_columns=["tokens", "labels"],
            )

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        train_config = {
            "fine_tuning_method": method,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "lora_r": lora_r if method == "lora" else None,
            "lora_alpha": lora_alpha if method == "lora" else None,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": split_seed,
            "save_plots": save_plots,
            "eval_max_rows": EVAL_MAX_ROWS,
            "holdout_predict_max_rows": HOLDOUT_PREDICT_MAX_ROWS,
            "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
        }
        (self.adapter_dir / LABEL_MAP_FILE).write_text(
            json.dumps({"id2label": id2label, "label2id": label2id}, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.adapter_dir / TRAIN_METHOD_FILE).write_text(
            json.dumps({"fine_tuning_method": method}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.adapter_dir / TRAIN_CONFIG_FILE).write_text(
            json.dumps(train_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if debug:
            try:
                import transformers as _transformers

                _transformers.logging.set_verbosity_debug()
            except Exception:
                log.debug("transformers.logging.set_verbosity_debug 실패", exc_info=True)

        import numpy as np

        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            """Epoch마다 token accuracy + seqeval span F1/P/R 계산."""
            predictions, labels = eval_pred
            pred_ids = np.argmax(predictions, axis=-1)
            mask = labels != -100
            if not np.any(mask):
                return {"eval_accuracy": 0.0}
            correct = (pred_ids == labels) & mask
            acc = float(correct.sum() / mask.sum())
            metrics: Dict[str, float] = {"eval_accuracy": acc}

            # seqeval span-level F1/P/R (학습 중 epoch마다 측정)
            try:
                from seqeval.metrics import (
                    f1_score as seq_f1,
                    precision_score as seq_prec,
                    recall_score as seq_rec,
                )
                true_seqs: List[List[str]] = []
                pred_seqs: List[List[str]] = []
                for i in range(labels.shape[0]):
                    t = [id2label[int(l)] for l in labels[i] if l != -100]
                    p = [id2label[int(pr)] for pr, l in zip(pred_ids[i], labels[i]) if l != -100]
                    true_seqs.append(t)
                    pred_seqs.append(p)
                metrics["eval_f1"]        = seq_f1(true_seqs,   pred_seqs, zero_division=0)
                metrics["eval_precision"] = seq_prec(true_seqs, pred_seqs, zero_division=0)
                metrics["eval_recall"]    = seq_rec(true_seqs,  pred_seqs, zero_division=0)
            except Exception:
                pass

            return metrics

        use_eval = eval_ds is not None and len(eval_ds) > 0

        # ── debug: 학습 전 수집 ───────────────────────────────────────
        trainable_params_info: List[Dict[str, Any]] = []
        weights_before: Dict[str, Any] = {}
        mem_before: Dict[str, float] = {}
        if debug:
            trainable_params_info = [
                {
                    "name": name,
                    "shape": list(param.shape),
                    "numel": param.numel(),
                }
                for name, param in model.named_parameters()
                if param.requires_grad
            ]
            weights_before = _weight_stats(model)
            mem_before = _mem_stats()

        # ── 콜백 설정 ─────────────────────────────────────────────────
        callbacks = []
        epoch_timer = _EpochTimerCallback()
        grad_norm_cb = _GradNormCallback(log_interval=50)
        if debug:
            callbacks = [epoch_timer, grad_norm_cb]

        # Legacy PAPER1_TRAINING_LOG 환경변수로 지정된 경로에 모든 Trainer
        # 이벤트를 실시간 텍스트로 기록한다.
        _full_log_path = os.environ.get("PAPER1_TRAINING_LOG")
        if _full_log_path:
            _extra_info = {
                "PAPER1_CONFIG": os.environ.get("PAPER1_CONFIG", ""),
                "dataset_size_train": len(tokenized_ds) if tokenized_ds is not None else None,
                "dataset_size_eval": len(eval_ds) if eval_ds is not None else None,
                "label_list": sorted(self.label2id.keys()) if self.label2id else [],
                "num_labels": len(self.label2id) if self.label2id else 0,
                "fine_tuning_method": method,
                "lora_r": lora_r if method == "lora" else None,
                "lora_alpha": lora_alpha if method == "lora" else None,
                "split_seed": split_seed,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
            }
            callbacks.append(_FullLoggingCallback(Path(_full_log_path), _extra_info))

        # Legacy PAPER1_LOG_DIR 환경변수가 지정되면 FullLogger 를 부착.
        # env.json·config.json·scalars/params/gpu/events/random_state.jsonl 자동 생성.
        try:
            from module.extractor.ner.full_logger import from_env as _full_logger_from_env
            _flog = _full_logger_from_env()
            if _flog is not None:
                callbacks.append(_flog.callback())
                print(f"  FullLogger 부착: {_flog.out_dir}")
        except Exception as _flog_err:
            log.warning("FullLogger 부착 실패: %s", _flog_err)

        if early_stopping_patience > 0 and use_eval:
            try:
                from transformers import EarlyStoppingCallback
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                ))
                print(f"  조기종료 활성화: patience={early_stopping_patience}")
            except Exception as _es_err:
                log.warning("EarlyStoppingCallback 등록 실패: %s", _es_err)

        use_early_stopping = early_stopping_patience > 0 and use_eval
        training_kwargs = dict(
            output_dir=str(self.adapter_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size * TRAIN_BATCH_MULT,
            per_device_eval_batch_size=batch_size * EVAL_BATCH_MULT,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            save_strategy="epoch" if use_early_stopping else "no",
            save_total_limit=1 if use_early_stopping else None,
            load_best_model_at_end=use_early_stopping,
            metric_for_best_model="eval_f1" if use_early_stopping else None,
            greater_is_better=True if use_early_stopping else None,
            eval_strategy="epoch" if use_eval else "no",
            eval_accumulation_steps=EVAL_ACCUMULATION_STEPS if use_eval else None,
            logging_steps=1 if debug else 50,
            report_to="none",
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            save_safetensors=True,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            dataloader_pin_memory=DATALOADER_PIN_MEMORY,
            optim="adamw_torch_fused" if (torch is not None and torch.cuda.is_available()) else "adamw_torch",
            **_amp_kwargs(),
        )
        if "save_only_model" in inspect.signature(TrainingArguments.__init__).parameters:
            training_kwargs["save_only_model"] = True
        training_args = TrainingArguments(**training_kwargs)

        if debug:
            from module.extractor.ner._runtime import ner_debug_print

            ner_debug_print(f"[NER debug][token_cls.train] tokenized_rows={len(tokenized_ds)}")
            ner_debug_print(f"[NER debug][token_cls.train] TrainingArguments={training_args}")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics if use_eval else None,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            processing_class=tokenizer,
            callbacks=callbacks if callbacks else None,
        )

        if debug:
            from module.extractor.ner._runtime import ner_debug_print

            ner_debug_print("[NER debug][token_cls.train] Trainer.train() 시작")
        train_out = trainer.train()
        if debug:
            from module.extractor.ner._runtime import ner_debug_print

            ner_debug_print("[NER debug][token_cls.train] Trainer.train() 종료")

        train_metrics_dict: Dict[str, Any] = {}
        if hasattr(train_out, "metrics") and train_out.metrics:
            train_metrics_dict = dict(train_out.metrics)

        # ── 홀드아웃 평가 (항상 predict()로 전체 메트릭 수집) ────────
        test_metrics_holdout: Dict[str, Any] = {}
        full_metrics_data: Optional[Dict[str, Any]] = None

        def _stream_holdout_metrics() -> Dict[str, Any]:
            if torch is None:
                return {"error": "torch not available"}
            was_training = bool(getattr(model, "training", False))
            model.eval()
            total_tokens = 0
            correct_tokens = 0
            dataloader = trainer.get_eval_dataloader(test_ds)
            for step, inputs in enumerate(dataloader, 1):
                inputs = trainer._prepare_inputs(inputs)
                labels = inputs.get("labels")
                if labels is None:
                    continue
                with torch.no_grad():
                    if torch.cuda.is_available():
                        _amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        with torch.cuda.amp.autocast(dtype=_amp_dtype):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                preds = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                correct_tokens += int(((preds == labels) & mask).sum().item())
                total_tokens += int(mask.sum().item())
                del outputs, preds, labels, inputs, mask
                if step % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "streaming": True,
                "test_samples": len(test_ds),
                "test_tokens": total_tokens,
                "test_token_accuracy": (
                    correct_tokens / total_tokens if total_tokens else 0.0
                ),
            }

        if test_ds is not None and len(test_ds) > 0:
            try:
                if len(test_ds) >= HOLDOUT_PREDICT_MAX_ROWS:
                    test_metrics_holdout = _stream_holdout_metrics()
                    test_metrics_holdout["full_metrics_skipped"] = (
                        f"test rows >= {HOLDOUT_PREDICT_MAX_ROWS}; "
                        "streamed token accuracy to avoid storing logits"
                    )
                else:
                    pred_output = trainer.predict(test_ds)
                    test_metrics_holdout = dict(pred_output.metrics) if pred_output.metrics else {}
                    preds_raw = pred_output.predictions
                    gold_ids  = pred_output.label_ids

                    full_metrics_data = _compute_full_metrics(
                        preds_raw, gold_ids, id2label, label_list
                    )
                print(f"  홀드아웃 평가(test): {test_metrics_holdout}")
            except Exception as ex:
                test_metrics_holdout = {"error": str(ex)}
                log.exception("token_cls 홀드아웃 평가 실패")

        # ── 저장: 방법별 분기 ─────────────────────────────────────────
        if method == "lora":
            model.save_pretrained(str(self.adapter_dir))
        else:  # full
            trainer.save_model(str(self.adapter_dir))

        tokenizer.save_pretrained(str(self.adapter_dir))

        # ── 공통 메타데이터 저장 ──────────────────────────────────────
        (self.adapter_dir / LABEL_MAP_FILE).write_text(
            json.dumps({"id2label": id2label, "label2id": label2id}, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.adapter_dir / TRAIN_METHOD_FILE).write_text(
            json.dumps({"fine_tuning_method": method}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        train_config = {
            "fine_tuning_method": method,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "lora_r": lora_r if method == "lora" else None,
            "lora_alpha": lora_alpha if method == "lora" else None,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": split_seed,
            "save_plots": save_plots,
            "eval_max_rows": EVAL_MAX_ROWS,
            "holdout_predict_max_rows": HOLDOUT_PREDICT_MAX_ROWS,
            "eval_accumulation_steps": EVAL_ACCUMULATION_STEPS,
        }
        (self.adapter_dir / TRAIN_CONFIG_FILE).write_text(
            json.dumps(train_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── metrics 파일 저장 ─────────────────────────────────────────
        metrics_payload: Dict[str, Any] = {
            "data_split": data_split,
            "train_metrics": train_metrics_dict,
            "test_metrics_holdout": test_metrics_holdout,
        }
        if full_metrics_data:
            # 요약만 저장 (flat list 제외 - 너무 큼)
            summary = {k: v for k, v in full_metrics_data.items() if k != "confidence_flat" and k != "flat_preds_names" and k != "flat_golds_names"}
            metrics_payload["full_metrics_summary"] = summary
        if trainer.state.log_history:
            metrics_payload["log_history_full"] = trainer.state.log_history

        metrics_path = self.adapter_dir / "ner_train_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  학습/평가 메트릭 저장: {metrics_path}")

        # ── debug_metrics.json 저장 ───────────────────────────────────
        if debug:
            from module.extractor.ner._runtime import ner_debug_print

            weights_after = _weight_stats(model)
            mem_after = _mem_stats()
            debug_metrics: Dict[str, Any] = {
                "fine_tuning_method": method,
                "trainable_params": trainable_params_info,
                "trainable_param_count": sum(p["numel"] for p in trainable_params_info),
                "total_param_count": sum(p.numel() for p in model.parameters()),
                "full_log_history": trainer.state.log_history,
                "epoch_times_seconds": epoch_timer.epoch_times,
                "grad_norms": grad_norm_cb.grad_norms,
                "memory_before_training": mem_before,
                "memory_after_training": mem_after,
                "weights_before_training": weights_before,
                "weights_after_training": weights_after,
            }
            if full_metrics_data:
                debug_metrics["full_metrics"] = full_metrics_data

            debug_metrics_path = self.adapter_dir / DEBUG_METRICS_FILE
            debug_metrics_path.write_text(
                json.dumps(debug_metrics, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"  디버그 메트릭 저장: {debug_metrics_path}")
            ner_debug_print(
                f"[NER debug][token_cls.train] debug_metrics 저장: {debug_metrics_path}"
            )

        # ── 그래프 생성 ──────────────────────────────────────────────
        if save_plots:
            try:
                plot_data = {}
                if full_metrics_data:
                    plot_data = full_metrics_data
                saved_plots = _generate_plots(
                    self.adapter_dir,
                    log_history=trainer.state.log_history,
                    full_metrics=plot_data,
                    grad_norms=grad_norm_cb.grad_norms if debug else None,
                    epoch_times=epoch_timer.epoch_times if debug else None,
                    weights_before=weights_before if debug else None,
                    weights_after=weights_after if debug else None,
                    method=method,
                    model_name=self.model_dir.name,
                    debug=debug,
                )
                if saved_plots:
                    print(f"  그래프 {len(saved_plots)}개 저장: {self.adapter_dir / PLOTS_SUBDIR}")
            except Exception as ex:
                log.warning("그래프 생성 실패: %s", ex)

        # ── 인스턴스 상태 업데이트 ────────────────────────────────────
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id
        self._loaded = True
        self.model.eval()
        self._to_device()

        print(f"  [{method}] 어댑터/가중치 저장: {self.adapter_dir}")
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, metrics_payload

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def predict(
        self,
        texts: List[List[str]],
        threshold: float = 0.55,
    ) -> List[List[str]]:
        """토큰 리스트 → BIO 태그 리스트."""
        if not self._loaded:
            self.load()
        if self.model is None:
            return [["O"] * len(tokens) for tokens in texts]

        all_bio: List[List[str]] = []

        for tokens in texts:
            if not tokens:
                all_bio.append([])
                continue

            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding=True,
            )
            word_ids = encoded.word_ids(batch_index=0)
            inputs = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)[0]
            preds = torch.argmax(probs, dim=-1)
            confs = torch.max(probs, dim=-1).values

            bio = ["O"] * len(tokens)
            seen: set[int] = set()
            for idx, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                if word_id < len(bio):
                    label = self.id2label.get(preds[idx].item(), "O")
                    bio[word_id] = label if confs[idx].item() >= threshold else "O"

            all_bio.append(bio)

        return all_bio

    def predict_at_thresholds(
        self,
        texts: List[List[str]],
        thresholds: List[float],
    ) -> List[Dict[float, List[str]]]:
        """inference 1회 실행 → 여러 threshold에 일괄 적용.

        Returns:
            List[Dict[float, List[str]]] — texts와 같은 길이.
            각 원소: {threshold → BIO 리스트}
        """
        if not self._loaded:
            self.load()
        if self.model is None:
            return [{thr: ["O"] * len(t) for thr in thresholds} for t in texts]

        results: List[Dict[float, List[str]]] = []

        for tokens in texts:
            if not tokens:
                results.append({thr: [] for thr in thresholds})
                continue

            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding=True,
            )
            word_ids = encoded.word_ids(batch_index=0)
            inputs = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)[0]
            preds = torch.argmax(probs, dim=-1)
            confs = torch.max(probs, dim=-1).values

            # word_id → (best_label, confidence)
            word_preds: Dict[int, Tuple[str, float]] = {}
            seen: set = set()
            for idx, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                if wid < len(tokens):
                    word_preds[wid] = (
                        self.id2label.get(preds[idx].item(), "O"),
                        float(confs[idx].item()),
                    )

            # threshold별 BIO 적용
            thr_map: Dict[float, List[str]] = {}
            for thr in thresholds:
                bio = ["O"] * len(tokens)
                for wid, (lbl, conf) in word_preds.items():
                    bio[wid] = lbl if conf >= thr else "O"
                thr_map[thr] = bio
            results.append(thr_map)

        return results


# ═══════════════════════════════════════════════════════════════════════
# ▼▼▼  레거시 코드 (비활성화)  ▼▼▼
# head_only / bitfit / role-free 정규화 — 2026-04-05 이후 미사용
# ═══════════════════════════════════════════════════════════════════════

# def _normalize_bio_label(label: str) -> str:
#     """ch_co_/ch_ja_/ch_nr_ 역할 접두사 제거 (role-free 학습 형식으로 정규화).
#     B-ch_co_address → B-address  /  I-ch_nr_name → I-name
#     ⚠️ _load_records_by_label에서 이 함수를 제거했음 — role-specific 라벨 보존이 목적.
#     """
#     for bio in ("B-", "I-"):
#         if label.startswith(bio):
#             rest = label[len(bio):]
#             for prefix in ("ch_co_", "ch_ja_", "ch_nr_"):
#                 if rest.startswith(prefix):
#                     return bio + rest[len(prefix):]
#     return label


# def load_bio_data(train_dirs):
#     """레거시 로더: role-free 정규화 포함. _load_records_by_label로 대체됨."""
#     if isinstance(train_dirs, Path):
#         train_dirs = [train_dirs]
#     records = []
#     for train_dir in train_dirs:
#         for p in sorted(train_dir.glob("*.jsonl")):
#             for line in p.read_text(encoding="utf-8").splitlines():
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     obj = json.loads(line)
#                     tokens = obj.get("tokens", [])
#                     labels = obj.get("labels", [])
#                     if tokens and len(tokens) == len(labels):
#                         norm_labels = [_normalize_bio_label(l) for l in labels]
#                         records.append({"tokens": tokens, "labels": norm_labels})
#                 except Exception:
#                     continue
#     return records


# --- head_only / bitfit 로드 (_load_model 내) ---
# elif method == "head_only":
#     weights_path = self.adapter_dir / "head_only_weights.pt"
#     if weights_path.exists() and torch is not None:
#         state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
#         base_model.load_state_dict(state, strict=False)
#         self.model = base_model
#         self.adapter_load_path = str(weights_path.resolve())
#     else:
#         self.model = None
# elif method == "bitfit":
#     weights_path = self.adapter_dir / "bitfit_weights.pt"
#     if weights_path.exists() and torch is not None:
#         state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
#         base_model.load_state_dict(state, strict=False)
#         self.model = base_model
#         self.adapter_load_path = str(weights_path.resolve())
#     else:
#         self.model = None


# --- head_only / bitfit 학습 준비 (train 내) ---
# elif method == "head_only":
#     model = base_model
#     for param in model.parameters():
#         param.requires_grad = False
#     for name, param in model.named_parameters():
#         if "classifier" in name:
#             param.requires_grad = True
# else:  # bitfit
#     model = base_model
#     for name, param in model.named_parameters():
#         param.requires_grad = name.endswith(".bias")


# --- head_only / bitfit 저장 (train 내) ---
# elif method == "head_only":
#     head_state = {k: v for k, v in model.state_dict().items() if "classifier" in k}
#     torch.save(head_state, str(self.adapter_dir / "head_only_weights.pt"))
# else:  # bitfit
#     bias_state = {k: v for k, v in model.state_dict().items() if k.endswith(".bias")}
#     torch.save(bias_state, str(self.adapter_dir / "bitfit_weights.pt"))
