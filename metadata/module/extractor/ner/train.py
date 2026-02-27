# train.py
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from module.extractor.ner.base import (
    bio_to_ner_spans,
    get_adapter_dir,
    get_adapters_dir,
    get_gliner_train_dir,
    get_optimized_dir,
    get_training_data_signature,
    list_adapter_runs,
    load_labels_from_predict,
    get_next_run_id,
    write_train_state,
)

try:
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
except ImportError as e:
    raise RuntimeError("GLiNER2가 필요합니다.") from e


def _line_to_gliner_record(obj: dict) -> Optional[dict]:
    """한 줄 JSON (tokens, labels BIO) → GLiNER2 학습 형식 {"input": str, "output": {"entities": {label: [mention, ...]}}}."""
    tokens = obj.get("tokens", [])
    labels = obj.get("labels", [])
    if not tokens or len(tokens) != len(labels):
        return None
    ner = bio_to_ner_spans(labels)
    entities: dict = {}
    for start, end, label in ner:
        mention = " ".join(tokens[int(start) : int(end) + 1])
        if label not in entities:
            entities[label] = []
        entities[label].append(mention)
    if not entities:
        return None
    return {"input": " ".join(tokens), "output": {"entities": entities}}


def _merge_train_dir_to_jsonl(train_dir: Path) -> Path:
    """라벨별 .jsonl 을 predict 라벨만 병합해 GLiNER2 형식 임시 파일로."""
    labels, _ = load_labels_from_predict()
    allowed = set(labels) if labels else None
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="gliner_train_")
    with open(fd, "w", encoding="utf-8") as out:
        for p in sorted(train_dir.glob("*.jsonl")):
            if allowed is not None and p.stem not in allowed:
                continue
            for line in open(p, "r", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _line_to_gliner_record(json.loads(line))
                    if rec:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    continue
    return Path(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="fastino/gliner2-large-v1")
    p.add_argument("--train_jsonl", default=None, help="학습 JSONL (--train_dir 없을 때 필수)")
    p.add_argument("--train_dir", default=None, help="라벨별 .jsonl 디렉터리 (기본: configs/gliner/train)")
    p.add_argument("--valid_jsonl", default=None)
    p.add_argument("--out_dir", default=None, help="어댑터 저장 경로 (기본: configs/gliner/train/adapter)")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--task_lr", type=float, default=5e-4)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target", nargs="+", default=["encoder"])

    return p.parse_args()


def _make_config(args: argparse.Namespace, out_dir: Path) -> "TrainingConfig":
    return TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target,
        save_adapter_only=True,
    )


def _run_optimization(args: argparse.Namespace, train_dir: Path) -> None:
    """전체 데이터로 재학습해 optimized/에 저장 후 run_* 삭제."""
    merged_path = _merge_train_dir_to_jsonl(train_dir)
    opt_dir = get_optimized_dir()
    opt_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = GLiNER2.from_pretrained(args.base_model)
        config = _make_config(args, opt_dir)
        trainer = GLiNER2Trainer(model=model, config=config)
        trainer.train(train_data=str(merged_path))
        for run_path in list_adapter_runs():
            shutil.rmtree(run_path, ignore_errors=True)
    finally:
        Path(merged_path).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    train_dir_path = Path(args.train_dir) if args.train_dir else get_gliner_train_dir()
    has_dir_data = train_dir_path.exists() and any(train_dir_path.glob("*.jsonl"))

    if has_dir_data:
        merged = _merge_train_dir_to_jsonl(train_dir_path)
        train_jsonl = str(merged)
        signature = get_training_data_signature(train_dir_path)
        out_dir = get_adapters_dir() / f"run_{get_next_run_id():06d}"
    else:
        train_jsonl = args.train_jsonl
        signature = ""
        out_dir = Path(args.out_dir) if args.out_dir else get_adapter_dir()

    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_jsonl:
        print("[안내] 학습 데이터가 없습니다. --train_dir 또는 --train_jsonl 을 지정하세요.")
        return

    try:
        model = GLiNER2.from_pretrained(args.base_model)
        config = _make_config(args, out_dir)
        trainer = GLiNER2Trainer(model=model, config=config)
        if args.valid_jsonl:
            trainer.train(train_data=train_jsonl, valid_data=args.valid_jsonl)
        else:
            trainer.train(train_data=train_jsonl)
        if signature:
            write_train_state(signature, out_dir)
        print(f"[OK] Done. Adapter saved under: {out_dir}")
        if has_dir_data and len(list_adapter_runs()) >= 5:
            print("[OK] Adapter 5개 이상 → 최적화 수행 중...")
            _run_optimization(args, train_dir_path)
            write_train_state(signature, get_optimized_dir())
            print("[OK] 최적화 완료. optimized/ 사용.")
    finally:
        if has_dir_data:
            Path(train_jsonl).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
