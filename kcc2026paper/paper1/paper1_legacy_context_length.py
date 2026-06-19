"""§24-2 논문 1 — NER Context-Length 한계 검증 (단일 파일).

**실행 정책** (사용자 확정):
  - Integrated 전용 (seperated 는 논문 4).
  - Gold 는 최종 OOD 평가 전용, 훈련·튜닝에 미참여.
  - 4 BERT × 11 labels × 5 case levels = 220 configurations.
  - Silver per-label cap = 10,000 (seed=42 random sample).
  - Train split 유지 (8/12), epochs=5, batch=24.

사용:
  python3 paper1/paper1_legacy_context_length.py                        # 기본 = all (build → run, --skip-existing)
  python3 paper1/paper1_legacy_context_length.py build                  # 데이터만 재생성
  python3 paper1/paper1_legacy_context_length.py run                    # sweep 만 실행
  python3 paper1/paper1_legacy_context_length.py run --models 0         # 1개 모델만
  python3 paper1/paper1_legacy_context_length.py run --labels name --case case2   # 1 config
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA_ROOT = ROOT.parent / "metadata"
for path in (ROOT, METADATA_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from paper_module.core.ner.train import ner_train            # noqa: E402
from paper_module.core.ner.token_cls import TokenClassNER    # noqa: E402
from paper_module.core.run_logger import (                    # noqa: E402
    open_log, log_section, close_log,
)


# ============================================================
# 실험 설정
# ============================================================

MODELS = [
    "klue/bert-base",                                   # 0: KLUE (한국어 특화)
    "monologg/koelectra-base-v3-discriminator",         # 1: KoELECTRA
    "google-bert/bert-base-multilingual-cased",         # 2: mBERT (다국어)
    "distilbert-base-multilingual-cased",               # 3: DistilBERT (경량)
]

LABELS = [
    # 짧은 span
    "name", "phone", "email", "date", "ri_period",
    # 긴 span
    "address", "company",
    "copyright_url", "copyright_Keyword",
    "copyright_kotitle", "copyright_description",
]

CASE_LEVELS = [0, 1, 2, 3, None]  # None = full (원본 그대로)


def case_name(n):
    return "full" if n is None else str(n)


CASE_STRS = [f"case{case_name(n)}" for n in CASE_LEVELS]

MAX_SILVER_PER_LABEL = 10000

# 하이퍼파라미터 (§24-19 paper 6 grid sweep 기반 — per-label 시나리오)
#   문헌 표준 (Devlin lr=2e-5, Mosbach epochs=10) → paper 6 도메인 검증 결과로 갱신
EPOCHS = 15
BATCH_SIZE = 16
LR = 4e-5
EARLY_STOPPING_PATIENCE = 3
TRAIN_RATIO = 8 / 12
VAL_RATIO = 2 / 12
TEST_RATIO = 2 / 12
SPLIT_SEED = 42


# ============================================================
# 데이터 빌드 (case 슬라이싱)
# ============================================================

def shrink_gold(record, case_n):
    text = record["text"]
    answer = record["answer"]
    if case_n is None:
        return dict(record)
    words = text.split()
    ans_words = answer.split()
    if not ans_words:
        return None
    start_idx = None
    for i in range(len(words) - len(ans_words) + 1):
        if words[i:i + len(ans_words)] == ans_words:
            start_idx = i
            break
    if start_idx is None:
        return None
    end_idx = start_idx + len(ans_words) - 1
    lo = max(0, start_idx - case_n)
    hi = min(len(words), end_idx + case_n + 1)
    return {
        "text": " ".join(words[lo:hi]),
        "answer": answer,
        "source": record.get("source", ""),
    }


def shrink_silver(record, label, case_n):
    tokens = record.get("tokens") or []
    labels = record.get("labels") or []
    if len(tokens) != len(labels) or not tokens:
        return None
    b_tag = f"B-{label}"
    i_tag = f"I-{label}"
    try:
        b_idx = labels.index(b_tag)
    except ValueError:
        return None
    e_idx = b_idx
    while e_idx + 1 < len(labels) and labels[e_idx + 1] == i_tag:
        e_idx += 1
    if case_n is None:
        return {"tokens": list(tokens), "labels": list(labels)}
    lo = max(0, b_idx - case_n)
    hi = min(len(tokens), e_idx + case_n + 1)
    return {"tokens": tokens[lo:hi], "labels": labels[lo:hi]}


def read_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_case_test():
    src = ROOT / "configs" / "integrated"
    out_root = ROOT / "paper1" / "configs" / "case_test"

    # 기존 case_test 전부 제거 (case 레벨 변경으로 인한 orphan 방지)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_silver = out_root / "silver"
    out_gold = out_root / "gold"

    rng = random.Random(SPLIT_SEED)

    for label in LABELS:
        silver_all = read_jsonl(src / "silver" / f"{label}.jsonl")
        gold_all = read_jsonl(src / "gold" / f"{label}.jsonl")

        # Silver per-label cap
        if len(silver_all) > MAX_SILVER_PER_LABEL:
            silver_all = rng.sample(silver_all, MAX_SILVER_PER_LABEL)

        print(f"\n[{label}] silver={len(silver_all)} (cap {MAX_SILVER_PER_LABEL}), gold={len(gold_all)}")

        for case_n in CASE_LEVELS:
            sub = f"{label}_case{case_name(case_n)}"

            s_shrunk = [s for s in (shrink_silver(r, label, case_n) for r in silver_all) if s]
            write_jsonl(out_silver / sub / f"{label}.jsonl", s_shrunk)

            g_shrunk = [g for g in (shrink_gold(r, case_n) for r in gold_all) if g]
            write_jsonl(out_gold / sub / f"{label}.jsonl", g_shrunk)

            print(f"  case{case_name(case_n):>4}: silver={len(s_shrunk):>6}, gold={len(g_shrunk):>6}")

    print("\n데이터 빌드 완료.")


# ============================================================
# 훈련 + 평가 (1 config)
# ============================================================

def extract_span(tokens, bio_labels, target_label):
    b_tag = f"B-{target_label}"
    i_tag = f"I-{target_label}"
    out_tokens = []
    in_span = False
    for tok, lab in zip(tokens, bio_labels):
        if lab == b_tag:
            if out_tokens:
                break
            out_tokens = [tok]
            in_span = True
        elif lab == i_tag and in_span:
            out_tokens.append(tok)
        elif in_span:
            break
    return " ".join(out_tokens)


def run_one_config(model, label, case, out_root, *, skip_existing):
    import os
    model_display = model.replace("/", "--")
    cfg_dir = out_root / model_display / f"{label}_{case}"
    log_path = cfg_dir / "run.txt"
    training_log_path = cfg_dir / "training.log"
    model_path = cfg_dir / "model"

    # HF Trainer 의 모든 이벤트를 training.log 에 실시간 기록하도록 token_cls 에 신호.
    os.environ["PAPER1_TRAINING_LOG"] = str(training_log_path)
    os.environ["PAPER1_CONFIG"] = f"{model_display}/{label}_{case}"

    # skip 조건: [final] 있는 완료 로그
    if skip_existing and log_path.exists():
        txt = log_path.read_text(encoding="utf-8")
        if "[final]" in txt:
            acc = 0.0
            in_final = False
            for line in txt.splitlines():
                if line.strip() == "[final]":
                    in_final = True
                    continue
                if in_final and line.startswith("[") and line.endswith("]"):
                    break
                if in_final and line.startswith("accuracy = "):
                    try:
                        acc = float(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
            return {"status": "skipped", "accuracy": acc}
        print("  (불완전 run.txt 감지 — 재실행)")

    silver_dir = ROOT / "paper1" / "configs" / "case_test" / "silver" / f"{label}_{case}"
    gold_dir = ROOT / "paper1" / "configs" / "case_test" / "gold" / f"{label}_{case}"
    gold_file = gold_dir / f"{label}.jsonl"

    if not (silver_dir / f"{label}.jsonl").exists():
        return {"status": "error", "error": f"silver missing: {silver_dir}"}
    if not gold_file.exists():
        return {"status": "error", "error": f"gold missing: {gold_file}"}

    t_total = time.time()
    fp = open_log(log_path)
    try:
        log_section(fp, "meta", {
            "model": model, "label": label, "case": case,
            "silver_dir": str(silver_dir), "gold_dir": str(gold_dir),
        })
        log_section(fp, "hparams", {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "train_ratio": "8/12", "val_ratio": "2/12", "test_ratio": "2/12",
            "split_seed": SPLIT_SEED,
            "max_silver_per_label": MAX_SILVER_PER_LABEL,
        })

        # ───── 훈련 ─────
        t0 = time.time()
        ner_train(
            model_name=model,
            input_path=str(silver_dir),
            model_path=str(model_path),
            fine_tuning_method="full",
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
            split_seed=SPLIT_SEED,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            force=True, debug=False,
        )
        train_sec = time.time() - t0
        log_section(fp, "train_result", {"train_time_sec": round(train_sec, 1)})

        # ───── Gold 평가 ─────
        adapter_parent = model_path / model_display
        assert (adapter_parent / "adapter" / "label_map.json").exists(), \
            f"학습 산출물 불완전: {adapter_parent}"
        ner = TokenClassNER(adapter_parent)
        ner.load()

        records = [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        t1 = time.time()
        hit = 0
        for rec in records:
            tokens = rec["text"].split()
            bio = ner.predict([tokens], threshold=0.25)[0]
            pred = extract_span(tokens, bio, label)
            ans = rec["answer"]
            if pred and (ans in pred or pred in ans):
                hit += 1
        eval_sec = time.time() - t1
        acc = hit / len(records) if records else 0.0

        log_section(fp, "eval_result", {
            "n_gold": len(records), "hit": hit,
            "accuracy": round(acc, 4), "eval_time_sec": round(eval_sec, 1),
        })

        total_sec = time.time() - t_total
        log_section(fp, "final", {
            "accuracy": round(acc, 4),
            "total_time_sec": round(total_sec, 1),
        })
        return {"status": "ok", "accuracy": acc, "total_sec": total_sec}
    except Exception as ex:
        log_section(fp, "error", {
            "type": type(ex).__name__,
            "message": str(ex),
            "traceback_first_line": traceback.format_exc().splitlines()[-1],
        })
        return {"status": "error", "error": str(ex)}
    finally:
        close_log(fp)
        # 디스크 절약
        if model_path.exists():
            shutil.rmtree(model_path, ignore_errors=True)


# ============================================================
# sweep
# ============================================================

class _Tee:
    """stdout/stderr 를 콘솔 + 파일로 동시 기록 (line-buffered)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run_sweep(*, models, labels, cases, skip_existing, stamp=None):
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "paper1" / "data" / "runs" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    # 터미널 stdout/stderr 를 out_root/terminal.log 에 tee.
    # 학습 루프의 진행 바·HF 경고·PyTorch 메시지까지 전부 보존.
    terminal_log = open(out_root / "terminal.log", "a", encoding="utf-8", buffering=1)
    print(f"[terminal log] {out_root}/terminal.log", flush=True)
    sys.stdout = _Tee(sys.__stdout__, terminal_log)
    sys.stderr = _Tee(sys.__stderr__, terminal_log)

    total = len(models) * len(labels) * len(cases)
    print("\n===== paper1 sweep =====")
    print(f"  models ({len(models)}): {models}")
    print(f"  labels ({len(labels)}): {labels}")
    print(f"  cases  ({len(cases)}):   {cases}")
    print(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, silver_cap={MAX_SILVER_PER_LABEL}")
    print(f"  총 {total} configs → {out_root}")
    print("=" * 60)

    idx = 0
    t_global = time.time()
    summary = []

    for model in models:
        for label in labels:
            for case in cases:
                idx += 1
                tag = f"[{idx}/{total}] {model.split('/')[-1]} / {label} / {case}"
                print(f"\n{tag}")
                result = run_one_config(model, label, case, out_root, skip_existing=skip_existing)
                elapsed = time.time() - t_global
                eta_min = (elapsed / idx) * (total - idx) / 60 if idx > 0 else 0
                status = result["status"]
                acc = result.get("accuracy", 0.0)
                print(f"  → {status}  acc={acc:.4f}  (global {elapsed/60:.1f}m, ETA {eta_min:.0f}m)")
                summary.append({"model": model, "label": label, "case": case, **result})

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    total_min = (time.time() - t_global) / 60
    print("\n===== 완료 =====")
    print(f"  총 소요: {total_min:.1f}분")
    print(f"  요약: {summary_path}")


# ============================================================
# CLI
# ============================================================

def cmd_build(_args):
    build_case_test()


def cmd_run(args):
    models = MODELS if args.models is None else [MODELS[int(i)] for i in args.models.split(",")]
    labels = LABELS if args.labels is None else args.labels.split(",")
    cases = CASE_STRS if args.case is None else args.case.split(",")
    run_sweep(
        models=models, labels=labels, cases=cases,
        skip_existing=args.skip_existing, stamp=args.stamp,
    )


def cmd_all(args):
    cmd_build(args)
    # 기본 skip_existing=True (중단 후 재개 안전)
    if not hasattr(args, "skip_existing"):
        args.skip_existing = True
    cmd_run(args)


def main():
    if len(sys.argv) == 1:
        sys.argv.extend(["all", "--skip-existing"])

    ap = argparse.ArgumentParser(description="§24-2 논문 1 — Context-Length 한계 검증")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="paper1/configs/case_test/ 재생성")

    for name in ("run", "all"):
        p = sub.add_parser(name, help=f"{name} 명령")
        p.add_argument("--models", default=None,
                       help=f"모델 인덱스 쉼표구분 (기본: 전체 {len(MODELS)}개)")
        p.add_argument("--labels", default=None,
                       help=f"라벨 이름 쉼표구분 (기본: 전체 {len(LABELS)}개)")
        p.add_argument("--case", default=None,
                       help=f"case 레벨 쉼표구분 (기본: {','.join(CASE_STRS)})")
        p.add_argument("--skip-existing", action="store_true",
                       help="완료된 run.txt 건너뜀")
        p.add_argument("--stamp", default=None)

    args = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
