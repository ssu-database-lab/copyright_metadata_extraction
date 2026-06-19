"""§24-22 논문 1 — "노이즈의 성질과 학습방식에 따른 모델 성능 차이 분석".

26 NER 라벨 × 2 silver source × 3 학습 입력 mode = **6 configs**.

  Source S:
    rule  - configs/integrated/silver/      (§24-14 규칙 silver v2, AI 5-mode 노이즈 증강)
    llm   - configs/llm/silver/             (scripts/generate_llm_silver.py 로 생성)

  Mode M:
    m1_answer    - BIO 라벨 토큰만 (앞뒤 텍스트 0)
    m2_context   - BIO + 자연 문맥 (silver 원본 형태)
    m3_negatives - m2 + all-O negative 샘플 25% 추가

학습: KLUE BERT × Full FT × Integrated (paper 5 §24-19 HP — lr=4e-5, ep=3, bs=32, max_per_label=10000).
평가: configs/integrated/gold OOD (라벨별 + class별 + source별).

산출:
  paper/paper1/configs/{rule,llm}/{m1_answer,m2_context,m3_negatives}/<label>.jsonl
  paper/paper1/data/runs/<stamp>/<config_id>/{run.txt, training.log, log/, model/}
  paper/paper1/data/runs/<stamp>/summary.json

사용:
  python3 paper1/paper1.py                                      # build (가능한 source) + run + analyze
  python3 paper1/paper1.py build                                # silver 디렉터리 빌드
  python3 paper1/paper1.py build --source rule                  # rule 만
  python3 paper1/paper1.py build --source llm                   # llm 만 (configs/llm/silver 필요)
  python3 paper1/paper1.py run                                  # 6 configs sweep
  python3 paper1/paper1.py run --configs rule_m1,rule_m2        # 일부만
  python3 paper1/paper1.py analyze <stamp>                      # 결과 집계
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
METADATA_ROOT = ROOT.parent / "metadata"
for path in (ROOT, METADATA_ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from paper_module.core.ner.train import ner_train               # noqa: E402
from paper_module.core.ner.token_cls import TokenClassNER       # noqa: E402
from paper_module.core.run_logger import (                       # noqa: E402
    open_log, log_section, close_log,
)
from module.parts.labels import REGEX_LABEL_SET, NER_LABEL_SET  # noqa: E402


# ============================================================
# 실험 상수
# ============================================================

MODEL = "klue/bert-base"  # 기본값 — main() 에서 --model 로 override 가능
BACKBONES = {  # paper 4 ranking 기준 상위 3
    "klue": "klue/bert-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "koelectra": "monologg/koelectra-base-v3-discriminator",
}
PAPER_LABEL_SET = REGEX_LABEL_SET | NER_LABEL_SET
NER_LABELS = sorted(PAPER_LABEL_SET)  # 26 historical paper labels

# paper 5 §24-19 HP (integrated)
EPOCHS = 3
BATCH_SIZE = 32
LR = 4e-5
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
TRAIN_RATIO = 8 / 12
VAL_RATIO = 2 / 12
TEST_RATIO = 2 / 12
MAX_PER_LABEL = 10000
SEED = 42

NEG_RATIO = 0.25  # M3 의 negative 비율

SOURCES = ("rule", "llm")
MODES = ("m1_answer", "m2_context", "m3_negatives")
CONFIGS = [(s, m) for s in SOURCES for m in MODES]

SILVER_BASE = ROOT / "paper1" / "configs"
RULE_SILVER_SRC = ROOT / "configs" / "integrated" / "silver"
LLM_SILVER_SRC = ROOT / "configs" / "llm" / "silver"

# Format-regularity (paper 5 와 동일)
FORMAT_CLASS = {
    "format-regular": [
        "phone", "email", "date", "ri_data", "ri_period", "ri_money",
        "address", "copyright_url", "copyright_uci", "copyright_num",
        "copyright_idnum", "copyright_status", "copyright_quantity",
        "copyright_language",
    ],
    "format-semi-regular": [
        "copyright_Keyword", "copyright_kotitle", "ri_law_reference",
        "ri_info", "ri_contract_type", "ri_copyright",
    ],
    "format-free": [
        "name", "company", "department", "position",
        "copyright_description", "copyright_type",
    ],
}
LABEL_CLASS = {lb: k for k, v in FORMAT_CLASS.items() for lb in v}


# ============================================================
# 공통 유틸
# ============================================================


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(p: Path, recs: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_span_bio(tokens, bio_labels, target_label):
    b_tag = f"B-{target_label}"
    i_tag = f"I-{target_label}"
    out = []
    in_span = False
    for tok, lab in zip(tokens, bio_labels):
        if lab == b_tag:
            if out:
                break
            out = [tok]
            in_span = True
        elif lab == i_tag and in_span:
            out.append(tok)
        elif in_span:
            break
    return " ".join(out)


def load_gold(label):
    gf = ROOT / "configs" / "integrated" / "gold" / f"{label}.jsonl"
    return _read_jsonl(gf)


# ============================================================
# Mode transforms
# ============================================================


def transform_m1_answer(recs: List[Dict]) -> List[Dict]:
    """답만 — 각 record 에서 contiguous B-X·I-X span 만 추출.

    한 record 에 여러 span 이 있으면 각각 분리. O 토큰은 모두 제거.
    """
    out: List[Dict] = []
    for rec in recs:
        tokens, labels = rec["tokens"], rec["labels"]
        span_tok: List[str] = []
        span_lab: List[str] = []
        in_span = False
        for tok, lab in zip(tokens, labels):
            if lab.startswith("B-"):
                if span_tok:
                    out.append({"tokens": span_tok, "labels": span_lab})
                span_tok, span_lab = [tok], [lab]
                in_span = True
            elif lab.startswith("I-") and in_span:
                span_tok.append(tok)
                span_lab.append(lab)
            else:
                if span_tok:
                    out.append({"tokens": span_tok, "labels": span_lab})
                    span_tok, span_lab = [], []
                in_span = False
        if span_tok:
            out.append({"tokens": span_tok, "labels": span_lab})
    return out


def transform_m2_context(recs: List[Dict]) -> List[Dict]:
    """문장 positive — silver 원본 그대로."""
    return [dict(r) for r in recs]


def _make_true_negative(rec: Dict) -> Dict | None:
    """record 에서 entity span (B-/I-) 토큰을 *제거* 하여 entity-free 문장 생성.

    label noise (같은 문장에 모순 라벨 부여) 가 아닌 진짜 negative.
    예: ["저작자",":","이서연","서명함"] / ["O","O","B-name","O"]
        → ["저작자",":","서명함"]      / ["O","O","O"]

    Returns None 인 경우:
      - keep 토큰 0개 (모든 토큰이 entity)
      - keep 토큰 < 2 (너무 짧으면 학습 신호 없음)
    """
    tokens = rec.get("tokens", [])
    labels = rec.get("labels", [])
    if not tokens or not labels:
        return None
    keep_idx = [i for i, lab in enumerate(labels) if lab == "O"]
    if len(keep_idx) < 2:
        return None
    return {
        "tokens": [tokens[i] for i in keep_idx],
        "labels": ["O"] * len(keep_idx),
    }


def transform_m3_negatives(recs: List[Dict], rng: random.Random) -> List[Dict]:
    """M3 — 75% positives + 25% true negatives, 총량 = M2 (substitutive).

    True negative = M2 record 에서 entity span 을 *제거* 한 entity-free 문장
    (label noise 가 아니라 정말로 entity 가 없는 자연스러운 문장).

    M2 와 동일한 총 record 수 → 데이터 양 confound 제거, 통제 비교 가능.
    """
    n_total = len(recs)
    n_neg = int(n_total * NEG_RATIO)
    n_pos = n_total - n_neg

    shuffled = list(recs)
    rng.shuffle(shuffled)
    positives = [dict(r) for r in shuffled[:n_pos]]

    # negative pool = recs 에서 entity 제거. 실패 record (전부 entity 등) skip.
    neg_pool: List[Dict] = []
    for r in shuffled:
        neg = _make_true_negative(r)
        if neg is not None:
            neg_pool.append(neg)
        if len(neg_pool) >= n_neg * 2:  # 충분히 모이면 중단
            break

    if len(neg_pool) >= n_neg:
        negatives = rng.sample(neg_pool, n_neg)
    else:
        # 부족 시 가능한 만큼만 + 단순 oversample
        negatives = list(neg_pool)
        if neg_pool:
            while len(negatives) < n_neg:
                negatives.append(dict(rng.choice(neg_pool)))

    out = positives + negatives
    rng.shuffle(out)
    return out


MODE_FN = {
    "m1_answer": lambda recs, rng: transform_m1_answer(recs),
    "m2_context": lambda recs, rng: transform_m2_context(recs),
    "m3_negatives": lambda recs, rng: transform_m3_negatives(recs, rng),
}


# ============================================================
# Silver builder
# ============================================================


def build_one(source: str, mode: str, *, force: bool = False) -> Tuple[bool, Dict[str, int]]:
    """source × mode 조합으로 26 라벨 silver 생성.

    Returns: (success, per_label_record_counts)
    """
    src_dir = RULE_SILVER_SRC if source == "rule" else LLM_SILVER_SRC
    if not src_dir.exists():
        return False, {}

    out_dir = SILVER_BASE / source / mode
    if out_dir.exists() and not force and any(out_dir.glob("*.jsonl")):
        counts = {lb: len(_read_jsonl(out_dir / f"{lb}.jsonl"))
                  for lb in NER_LABELS if (out_dir / f"{lb}.jsonl").exists()}
        return True, counts

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    counts: Dict[str, int] = {}
    for label in NER_LABELS:
        src = src_dir / f"{label}.jsonl"
        recs = _read_jsonl(src)
        if not recs:
            counts[label] = 0
            continue
        if len(recs) > MAX_PER_LABEL:
            recs = rng.sample(recs, MAX_PER_LABEL)
        out = MODE_FN[mode](recs, rng)
        _write_jsonl(out_dir / f"{label}.jsonl", out)
        counts[label] = len(out)
    return True, counts


def build_all(*, source: str = "both", force: bool = False) -> Dict[str, Dict[str, int]]:
    sources = SOURCES if source == "both" else (source,)
    summary: Dict[str, Dict[str, int]] = {}
    for s in sources:
        for m in MODES:
            ok, counts = build_one(s, m, force=force)
            total = sum(counts.values())
            print(f"  build {s}/{m}: {'OK' if ok else 'SKIPPED'}  total_records={total}")
            if ok:
                summary[f"{s}/{m}"] = counts
            else:
                src_dir = RULE_SILVER_SRC if s == 'rule' else LLM_SILVER_SRC
                print(f"    (소스 없음: {src_dir})")
    return summary


# ============================================================
# 단일 config 학습 + 평가
# ============================================================


def run_one(source: str, mode: str, out_root: Path, *, skip_existing: bool) -> Dict[str, Any]:
    cfg_id = f"{source}_{mode}"
    cfg_dir = out_root / cfg_id
    log_path = cfg_dir / "run.txt"
    training_log_path = cfg_dir / "training.log"
    full_log_dir = cfg_dir / "log"
    model_path = cfg_dir / "model"

    silver_dir = SILVER_BASE / source / mode
    if not silver_dir.exists() or not any(silver_dir.glob("*.jsonl")):
        return {"status": "skipped_no_silver", "cfg_id": cfg_id,
                "reason": f"silver 없음 — `python3 paper1/paper1.py build --source {source}` 먼저 실행"}

    os.environ["PAPER1_TRAINING_LOG"] = str(training_log_path)
    os.environ["PAPER1_LOG_DIR"] = str(full_log_dir)
    os.environ["PAPER1_CONFIG"] = f"paper1/{cfg_id}"

    if skip_existing and log_path.exists():
        txt = log_path.read_text(encoding="utf-8")
        if "[final]" in txt:
            return _parse_completed(log_path, cfg_id, source, mode)

    cfg_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    fp = open_log(log_path)
    try:
        n_records_per_label = {
            lb: len(_read_jsonl(silver_dir / f"{lb}.jsonl")) for lb in NER_LABELS
        }
        log_section(fp, "meta", {
            "source": source, "mode": mode, "cfg_id": cfg_id,
            "model": MODEL, "method": "full", "training_mode": "integrated",
            "n_labels": len(NER_LABELS),
            "silver_dir": str(silver_dir),
            "silver_total_records": sum(n_records_per_label.values()),
            "silver_per_label_min": min(n_records_per_label.values()),
            "silver_per_label_max": max(n_records_per_label.values()),
        })
        log_section(fp, "hparams", {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
            "train_ratio": "8/12", "val_ratio": "2/12", "test_ratio": "2/12",
            "split_seed": SEED,
            "max_per_label": MAX_PER_LABEL,
            "neg_ratio_m3": NEG_RATIO,
        })

        t0 = time.time()
        ner_train(
            model_name=MODEL,
            input_path=str(silver_dir),
            model_path=str(model_path),
            fine_tuning_method="full",
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
            split_seed=SEED,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            force=False, debug=False,  # 기존 adapter 있으면 재학습 skip
        )
        train_sec = time.time() - t0
        log_section(fp, "train_result", {"train_time_sec": round(train_sec, 1)})

        adapter_parent = model_path / MODEL.replace("/", "--")
        ner = TokenClassNER(adapter_parent)
        ner.load()

        t1 = time.time()
        per_label: Dict[str, Dict[str, Any]] = {}
        per_source: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"n": 0, "hit": 0})
        )
        for label in NER_LABELS:
            recs = load_gold(label)
            if not recs:
                per_label[label] = {"n_gold": 0, "hit": 0, "accuracy": 0.0,
                                    "class": LABEL_CLASS[label]}
                continue
            hit = 0
            for rec in recs:
                tokens = rec["text"].split()
                ans = rec["answer"].strip()
                src_key = rec.get("source", "?")
                bio = ner.predict([tokens], threshold=0.25)[0]
                pred_span = extract_span_bio(tokens, bio, label)
                ok = bool(pred_span) and ((ans in pred_span) or (pred_span in ans))
                if ok:
                    hit += 1
                per_source[label][src_key]["n"] += 1
                per_source[label][src_key]["hit"] += int(ok)
            acc = round(hit / len(recs), 4) if recs else 0.0
            per_label[label] = {
                "n_gold": len(recs), "hit": hit, "accuracy": acc,
                "class": LABEL_CLASS[label],
            }
        eval_sec = time.time() - t1

        per_class: Dict[str, Dict[str, float]] = {}
        for cls, lbls in FORMAT_CLASS.items():
            accs = [per_label[lb]["accuracy"] for lb in lbls if lb in per_label]
            per_class[cls] = {"n": len(accs),
                              "mean_acc": round(sum(accs) / max(len(accs), 1), 4)}

        ps_clean = {}
        for lb, smap in per_source.items():
            ps_clean[lb] = {sk: {"n": v["n"], "hit": v["hit"],
                                  "accuracy": round(v["hit"] / max(v["n"], 1), 4)}
                            for sk, v in smap.items()}

        overall = round(
            sum(v["accuracy"] for v in per_label.values()) / max(len(per_label), 1), 4
        )
        log_section(fp, "eval_result", {
            "n_labels": len(per_label),
            "accuracy_overall": overall,
            "per_class": per_class,
            "per_label": per_label,
            "per_source": ps_clean,
            "eval_time_sec": round(eval_sec, 1),
        })
        log_section(fp, "final", {
            "accuracy": overall,
            "total_time_sec": round(time.time() - t_total, 1),
        })
        return {
            "status": "ok", "cfg_id": cfg_id, "source": source, "mode": mode,
            "accuracy_overall": overall,
            "per_label": per_label, "per_class": per_class, "per_source": ps_clean,
            "train_time_sec": round(train_sec, 1),
            "eval_time_sec": round(eval_sec, 1),
            "total_time_sec": round(time.time() - t_total, 1),
            "silver_total_records": sum(n_records_per_label.values()),
        }
    except Exception as ex:
        log_section(fp, "error", {
            "exc": str(ex), "trace": traceback.format_exc(),
        })
        return {"status": "error", "error": str(ex), "cfg_id": cfg_id,
                "source": source, "mode": mode}
    finally:
        close_log(fp)


def _parse_completed(log_path: Path, cfg_id: str, source: str, mode: str) -> Dict[str, Any]:
    """기존 완료된 run.txt 에서 결과 회수 (skip_existing 용)."""
    txt = log_path.read_text(encoding="utf-8")
    acc = 0.0
    for line in txt.splitlines():
        if line.startswith("accuracy = "):
            try:
                acc = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return {"status": "skipped", "cfg_id": cfg_id, "source": source, "mode": mode,
            "accuracy_overall": acc}


# ============================================================
# Sweep
# ============================================================


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # transformers 5.x loading-report 가 sys.stdout.isatty() 를 호출 → 실제 스트림에 위임
        s0 = self.streams[0] if self.streams else None
        return bool(s0.isatty()) if hasattr(s0, "isatty") else False

    def __getattr__(self, name):
        # write/flush/isatty 외 부가 속성(fileno·encoding 등) 조회는 실제 스트림에 위임
        return getattr(self.streams[0], name)


def make_out_root(stamp=None) -> Path:
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "paper1" / "data" / "runs" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    t = open(out_root / "terminal.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, t)
    sys.stderr = _Tee(sys.__stderr__, t)
    print(f"[terminal log] {out_root}/terminal.log", flush=True)
    return out_root


def run_sweep(configs: List[Tuple[str, str]], out_root: Path, *, skip_existing: bool):
    total = len(configs)
    print(f"\n===== paper 1 §24-22 sweep ({total} configs × KLUE BERT × Full FT × Integrated) =====")
    print(f"  HP: lr={LR}, epochs={EPOCHS}, batch={BATCH_SIZE} (paper 5 §24-19)")
    print(f"  configs: {[f'{s}_{m}' for s, m in configs]}")
    print("=" * 60)

    summary = []
    t0 = time.time()
    for i, (source, mode) in enumerate(configs, 1):
        print(f"\n[{i}/{total}] {source}/{mode}")
        r = run_one(source, mode, out_root, skip_existing=skip_existing)
        elapsed = time.time() - t0
        eta = (elapsed / i) * (total - i) / 60 if i > 0 else 0
        acc = r.get("accuracy_overall", 0)
        print(f"  → {r['status']}  acc={acc:.4f}  (global {elapsed/60:.1f}m, ETA {eta:.0f}m)")
        summary.append(r)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n===== 완료 =====\n  총 소요: {(time.time()-t0)/60:.1f}분")
    print(f"  요약: {out_root / 'summary.json'}")


# ============================================================
# CLI
# ============================================================


def cmd_build(args):
    print(f"=== paper 1 silver build (source={args.source}) ===")
    summary = build_all(source=args.source, force=args.force)
    out = ROOT / "paper1" / "configs" / "build_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  요약: {out}")


def _parse_configs(spec: str) -> List[Tuple[str, str]]:
    """rule_m1,llm_m2 형태 → [("rule","m1_answer"), ("llm","m2_context")]."""
    short_to_full = {
        "m1": "m1_answer",
        "m2": "m2_context",
        "m3": "m3_negatives",
    }
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "_" not in tok:
            raise SystemExit(f"잘못된 config 형식: {tok}")
        s, m_short = tok.split("_", 1)
        if s not in SOURCES:
            raise SystemExit(f"잘못된 source: {s}")
        if m_short not in short_to_full:
            raise SystemExit(f"잘못된 mode: {m_short}")
        out.append((s, short_to_full[m_short]))
    return out


def cmd_run(args):
    if getattr(args, "model", None):
        global MODEL
        if args.model in BACKBONES:
            MODEL = BACKBONES[args.model]
        else:
            MODEL = args.model  # 직접 model_id 도 허용
    out_root = make_out_root(args.stamp)
    if args.configs:
        configs = _parse_configs(args.configs)
    else:
        configs = list(CONFIGS)
    run_sweep(configs, out_root, skip_existing=args.skip_existing)


def cmd_analyze(args):
    p = ROOT / "paper1" / "data" / "runs" / args.stamp / "summary.json"
    if not p.exists():
        print(f"없음: {p}")
        sys.exit(1)
    s = json.loads(p.read_text(encoding="utf-8"))
    ok = [r for r in s if r.get("status") in ("ok", "skipped")]
    print(f"\n=== paper 1 / {args.stamp} ===")
    for r in ok:
        cfg = r.get("cfg_id", "?")
        acc = r.get("accuracy_overall", 0)
        pc = r.get("per_class", {})
        reg = pc.get("format-regular", {}).get("mean_acc", 0)
        sem = pc.get("format-semi-regular", {}).get("mean_acc", 0)
        free = pc.get("format-free", {}).get("mean_acc", 0)
        print(f"  {cfg:<22} overall={acc:.4f}  reg={reg:.4f}  semi={sem:.4f}  free={free:.4f}")


def cmd_all(args):
    print("=== build ===")
    build_all(source="both", force=False)
    print("\n=== run ===")
    out_root = make_out_root(args.stamp)
    run_sweep(list(CONFIGS), out_root, skip_existing=args.skip_existing)


def main():
    if len(sys.argv) == 1:
        sys.argv.extend(["all", "--skip-existing"])

    ap = argparse.ArgumentParser(description="§24-22 논문 1 — 노이즈/학습방식 비교")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="silver 6 디렉터리 빌드")
    p.add_argument("--source", choices=["rule", "llm", "both"], default="both")
    p.add_argument("--force", action="store_true", help="기존 빌드 덮어쓰기")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("run", help="6 configs sweep")
    p.add_argument("--configs", default=None,
                   help="쉼표구분 (예: rule_m1,rule_m2,llm_m1)")
    p.add_argument("--model", default=None,
                   help=f"backbone (default klue/bert-base, 단축: {list(BACKBONES.keys())})")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stamp", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="결과 집계")
    p.add_argument("stamp")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("all", help="build + run")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--stamp", default=None)
    p.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
