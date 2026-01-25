"""
어댑터 기반 NER 학습 로직 (라벨별 어댑터 학습)
- 데이터: configs/training/ner_labels/{label}.jsonl (각 라벨별 파일)
- 저장: models/ner/adapters/{ner_<label>}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import random
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification
)

try:
    from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer
except ImportError as e:
    raise RuntimeError("adapters를 설치하세요: pip install -U adapters") from e

from seqeval.metrics import precision_score, recall_score, f1_score

from ..config import (
    BIO_LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    CONFIG_PATH,
    TRAINING_DATA_DIR,
)
import yaml


# -------------------------
# 데이터 포맷
# -------------------------
# jsonl 각 라인:
# {"id":"...","tokens":[...],"labels":[...]}  (labels는 BIO_LABELS 원소)


@dataclass
class Sample:
    tokens: List[str]
    labels: List[str]


def _iter_jsonl_files(data_dir: Path) -> List[Path]:
    """*.jsonl 파일 목록 반환"""
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.glob("*.jsonl") if p.is_file()])


def _iter_json_files(data_dir: Path) -> List[Path]:
    """*.json 파일 목록 반환 (training_data.json 등)"""
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.glob("*.json") if p.is_file()])


def _load_jsonl_samples(data_dir: Path) -> List[Sample]:
    """JSONL 파일에서 샘플 로드 (각 라인이 하나의 JSON 객체)"""
    import json

    samples: List[Sample] = []
    for fp in _iter_jsonl_files(data_dir):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tokens = obj.get("tokens") or []
                labels = obj.get("labels") or []
                if not tokens or not labels:
                    continue
                if len(tokens) != len(labels):
                    continue
                samples.append(Sample(tokens=tokens, labels=labels))
    return samples


def _load_json_samples(data_dir: Path) -> List[Sample]:
    """JSON 파일에서 샘플 로드 ({"data": [{"tokens": [...], "labels": [...]}, ...]} 형식)"""
    import json

    samples: List[Sample] = []
    for fp in _iter_json_files(data_dir):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            # {"data": [...]} 형식 또는 직접 배열 형식
            if isinstance(data, dict) and "data" in data:
                items = data["data"]
            elif isinstance(data, list):
                items = data
            else:
                continue
            
            for item in items:
                tokens = item.get("tokens") or []
                labels = item.get("labels") or []
                if not tokens or not labels:
                    continue
                if len(tokens) != len(labels):
                    continue
                samples.append(Sample(tokens=tokens, labels=labels))
    return samples


def _load_samples_for_label(data_dir: Path, label: str) -> List[Sample]:
    """특정 라벨의 학습 데이터 로드 (ner_labels/{label}.jsonl 파일)"""
    label_file = data_dir / "ner_labels" / f"{label}.jsonl"
    if not label_file.exists():
        return []
    
    import json
    samples: List[Sample] = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = obj.get("tokens") or []
            labels = obj.get("labels") or []
            if not tokens or not labels:
                continue
            if len(tokens) != len(labels):
                continue
            samples.append(Sample(tokens=tokens, labels=labels))
    return samples


def _load_all_samples(data_dir: Path) -> List[Sample]:
    """모든 라벨의 학습 데이터 로드 (하위 호환성을 위해 유지)"""
    samples = []
    samples.extend(_load_jsonl_samples(data_dir))
    samples.extend(_load_json_samples(data_dir))
    return samples


def _split_train_val(
    samples: List[Sample], train_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample]]:
    rnd = random.Random(seed)
    idx = list(range(len(samples)))
    rnd.shuffle(idx)
    cut = int(len(idx) * train_ratio)
    train = [samples[i] for i in idx[:cut]]
    val = [samples[i] for i in idx[cut:]]
    return train, val


def _filter_samples_for_label(samples: List[Sample], target_label: str) -> List[Sample]:
    # target_label: "address" -> keep those containing B-address / I-address
    b = f"B-{target_label}"
    i = f"I-{target_label}"
    out = []
    for s in samples:
        if any(tag == b or tag == i for tag in s.labels):
            out.append(s)
    return out


# -------------------------
# Dataset (word-level BIO -> token-level align)
# -------------------------
class NERDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # word-level tokenize
        # padding은 DataCollator에서 처리하므로 여기서는 하지 않음
        encoded = self.tokenizer(
            s.tokens,
            is_split_into_words=True,
            padding=False,  # DataCollator에서 처리
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # 리스트 반환
        )

        word_ids = encoded.word_ids()
        input_ids = encoded["input_ids"]
        
        # aligned_labels를 word_ids 길이에 맞춰 생성 (padding 전 실제 길이)
        aligned_labels: List[int] = []
        prev_word_idx = None

        # word_ids를 순회하면서 labels 정렬
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP] 등은 -100 (padding은 DataCollator에서 처리)
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                # 새로운 단어의 첫 번째 subword
                if word_idx < len(s.labels):
                    tag = s.labels[word_idx]
                    aligned_labels.append(int(LABEL_TO_ID.get(tag, 0)))
                else:
                    # word_idx가 labels 범위를 벗어나면 "O"
                    aligned_labels.append(int(LABEL_TO_ID.get("O", 0)))
            else:
                # 같은 단어의 추가 subword는 -100
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        # word_ids와 aligned_labels는 같은 길이여야 함
        assert len(aligned_labels) == len(word_ids), f"aligned_labels 길이({len(aligned_labels)}) != word_ids 길이({len(word_ids)})"

        encoded["labels"] = aligned_labels
        return encoded


def _compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_preds = []
    true_labels = []
    for p_seq, l_seq in zip(preds, labels):
        p_tags = []
        l_tags = []
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            p_tags.append(ID_TO_LABEL.get(int(p), "O"))
            l_tags.append(ID_TO_LABEL.get(int(l), "O"))
        true_preds.append(p_tags)
        true_labels.append(l_tags)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def _load_ner_labels_from_yaml() -> List[str]:
    if not Path(CONFIG_PATH).exists():
        return []
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ner_config = config.get("ner", {})
    labels = ner_config.get("labels", []) if isinstance(ner_config, dict) else []
    return list(labels) if labels else []


# -------------------------
# Main train
# -------------------------
def train_adapter_ner(
    model_name: str = "bert-base-multilingual-cased",
    model_path: Optional[str] = None,
    adapter_dir: str = "models/ner/adapters",
    train_data_path: Optional[str] = None,  # None이면 configs/training
    train_ratio: float = 0.8,
    random_seed: int = 42,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    reduction_factor: int = 16,
) -> Dict[str, Any]:

    # 모델 경로: model_downloaded/{model_name} 우선, 없으면 HF
    if model_path is None:
        local = Path("model_downloaded") / model_name
        model_path = str(local) if local.exists() else model_name
    else:
        mp = Path(model_path)
        model_path = str(mp) if mp.exists() else model_name

    # 데이터 경로: 사용자 요청대로 configs/training 기본
    data_dir = Path(train_data_path) if train_data_path else TRAINING_DATA_DIR
    if not data_dir.exists():
        raise ValueError(f"학습 데이터 디렉토리가 없습니다: {data_dir}")

    ner_labels = _load_ner_labels_from_yaml()
    if not ner_labels:
        raise ValueError("configs/labels.yaml에서 ner.labels를 찾을 수 없습니다.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoAdapterModel.from_pretrained(model_path)

    # head (token classification용)
    # add_tagging_head는 token classification을 위한 전용 메서드
    model.add_tagging_head(
        "ner",
        num_labels=len(BIO_LABELS),
        id2label={i: label for i, label in enumerate(BIO_LABELS)},
    )

    adapter_cfg = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)

    adapter_dir_path = Path(adapter_dir)
    adapter_dir_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "model_name": model_name,
        "model_path": model_path,
        "adapter_dir": str(adapter_dir_path),
        "trained_adapters": [],
    }

    for label in ner_labels:
        adapter_name = f"ner_{label}"

        # 라벨 포함 샘플만 학습(“라벨별 어댑터”)
        # 해당 라벨을 포함하는 샘플만 먼저 필터링
        # 각 라벨별 파일에서 직접 로드 ({label}.jsonl)
        label_samples = _load_samples_for_label(data_dir, label)

        if not label_samples:
            # 이 라벨 데이터가 아직 없다면 스킵(=점진적 weak-supervision)
            print(f"[스킵] {adapter_name}: 학습 데이터가 없습니다. ({data_dir}/ner_labels/{label}.jsonl 파일이 없음)")
            continue

        # 각 라벨별로 train/val split 수행
        # (전체 split이 아닌 라벨별 split으로, 샘플이 적어도 학습 가능)
        train_samples, val_samples = _split_train_val(
            label_samples, train_ratio=train_ratio, seed=random_seed
        )

        # 학습 샘플이 없으면 검증 샘플을 학습에 사용 (샘플이 매우 적을 때)
        if not train_samples and val_samples:
            print(f"[경고] {adapter_name}: 학습 샘플이 없어 검증 샘플을 학습에 사용합니다.")
            train_samples = val_samples
            val_samples = []

        # 학습 샘플이 여전히 없으면 스킵
        if not train_samples:
            print(f"[스킵] {adapter_name}: 학습할 샘플이 없습니다.")
            continue

        # 어댑터 추가/활성화
        if adapter_name not in model.adapters_config.adapters:
            model.add_adapter(adapter_name, config=adapter_cfg)
        model.train_adapter(adapter_name)
        model.set_active_adapters(adapter_name)

        train_ds = NERDataset(train_samples, tokenizer, max_length=max_length)
        val_ds = NERDataset(val_samples, tokenizer, max_length=max_length) if val_samples else None

        # DataCollator 사용 (배치 처리 시 올바른 shape 보장)
        # padding="max_length"는 truncation이 없을 때 경고를 발생시킬 수 있으므로
        # padding=True로 변경하고 max_length는 TrainingArguments에서 처리
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,  # 배치 내 최대 길이로 padding
        )

        out_dir = adapter_dir_path / adapter_name
        args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir=str(adapter_dir_path / "logs"),
            logging_steps=20,
            save_strategy="epoch",
            eval_strategy="epoch" if val_ds else "no",
            report_to=[],  # wandb 등 로깅 서비스 사용 안 함
            load_best_model_at_end=False,  # 어댑터만 저장하므로 체크포인트 로드 불필요
        )

        trainer = AdapterTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=_compute_metrics if val_ds else None,
        )

        train_result = trainer.train()
        
        # 학습 결과 출력
        print(f"\n{'='*60}")
        print(f"[학습 완료] {adapter_name} (라벨: {label})")
        print(f"{'='*60}")
        print(f"  목적: {label} 라벨 위주로 검출하도록 특화된 어댑터")
        print(f"  학습 샘플 수: {len(train_samples)}개")
        print(f"  검증 샘플 수: {len(val_samples) if val_samples else 0}개")
        print(f"  최종 학습 손실: {train_result.training_loss:.4f}")
        if val_ds and hasattr(train_result, 'metrics'):
            metrics = train_result.metrics
            print(f"  검증 메트릭:")
            for key, value in metrics.items():
                if key not in ['train_runtime', 'train_samples_per_second', 'train_steps_per_second', 'epoch']:
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.4f}")
                    else:
                        print(f"    - {key}: {value}")
        print(f"  어댑터 저장 경로: {out_dir}")
        print(f"{'='*60}\n")

        # 어댑터 저장(요청대로 models/ 밑)
        model.save_adapter(str(out_dir), adapter_name)
        
        # 결과에 메트릭 추가
        adapter_result = {
            "label": label,
            "adapter_name": adapter_name,
            "save_path": str(out_dir),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "training_loss": train_result.training_loss,
        }
        if val_ds and hasattr(train_result, 'metrics'):
            adapter_result["metrics"] = {k: v for k, v in train_result.metrics.items() 
                                         if k not in ['train_runtime', 'train_samples_per_second', 'train_steps_per_second']}
        results["trained_adapters"].append(adapter_result)

        # 다음 라벨 학습을 위해 active 해제
        try:
            model.set_active_adapters(None)
        except Exception:
            pass

    # 전체 학습 결과 요약 출력
    print(f"\n{'='*60}")
    print(f"[전체 학습 완료]")
    print(f"{'='*60}")
    print(f"  모델: {model_name}")
    print(f"  모델 경로: {model_path}")
    print(f"  어댑터 저장 디렉토리: {adapter_dir_path}")
    print(f"  학습 방식: 라벨별 독립 어댑터 (각 어댑터는 특정 라벨 위주로 검출)")
    print(f"  학습된 어댑터 수: {len(results['trained_adapters'])}개")
    print(f"  스킵된 라벨 수: {len(ner_labels) - len(results['trained_adapters'])}개 (데이터 없음)")
    for adapter_info in results["trained_adapters"]:
        print(f"\n  - {adapter_info['adapter_name']} (라벨: {adapter_info['label']})")
        print(f"    학습 샘플: {adapter_info['train_samples']}개")
        print(f"    검증 샘플: {adapter_info['val_samples']}개")
        loss = adapter_info.get('training_loss')
        if isinstance(loss, float):
            print(f"    학습 손실: {loss:.4f}")
        else:
            print(f"    학습 손실: {loss}")
        if 'metrics' in adapter_info:
            print(f"    검증 메트릭:")
            for key, value in adapter_info['metrics'].items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.4f}")
                else:
                    print(f"      {key}: {value}")
    print(f"{'='*60}\n")
    
    # 결과를 JSON 파일로 저장
    import json
    results_file = adapter_dir_path / "training_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"학습 결과가 저장되었습니다: {results_file}")

    return results
