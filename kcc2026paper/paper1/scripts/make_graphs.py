"""paper 1 (§24-22) sweep 종합 그래프 생성기 — 3 학습 입력 모드 비교 (M1 / M2 / M3).

═══════════════════════════════════════════════════════════════════════
■ 무엇을 하는 스크립트?
═══════════════════════════════════════════════════════════════════════

paper1.py 가 학습한 3 개 config (rule_m1_answer / rule_m2_context / rule_m3_negatives)
의 결과를 읽어서 학술대회 논문에 들어갈 11 개 figure (G1 ~ G11) 를 자동 생성.

비교 축은 학습 입력 모드 1 개 (mode):
  - M1 답만:        BIO 정답 토큰만, 앞뒤 텍스트 0
  - M2 문장:        BIO + 자연 문맥 (silver 원본 형태)
  - M3 문장+neg:    M2 에 25% all-O negative 샘플 혼합

각 그래프에서 mode 별로 색상이 구분됨:
  - M1 답만        → 빨간색 (#d62728)  실선
  - M2 문장        → 파란색 (#1f77b4)  파선
  - M3 문장+neg    → 초록색 (#2ca02c)  점선

═══════════════════════════════════════════════════════════════════════
■ 입력 (Input)
═══════════════════════════════════════════════════════════════════════

paper/paper1/data/runs/<stamp>/  디렉터리 — paper1.py sweep 산출물.
각 stamp 디렉터리는 다음 구조:

    <stamp>/
    ├── summary.json                          # 3 config 결과 통합 (mode 별 acc, per-class, per-label)
    └── <config>/log/                         # FullLogger (paper_module/log.py) 출력
        ├── scalars.jsonl                     # step 별 loss, learning_rate, grad_norm + epoch 별 eval_*
        ├── gpu.jsonl                         # step 10 마다 GPU memory snapshot
        ├── params.jsonl                      # step 50 마다 layer 별 weight L2 등
        ├── events.jsonl                      # HF Trainer callback 이벤트 (on_train_begin, on_log, ...)
        └── log_history.json                  # 학습 종료 시 state.log_history 전체 dump

stamp 미지정 시 가장 최신 sweep 자동 탐지 (find_latest_sweep).

═══════════════════════════════════════════════════════════════════════
■ 출력 (Output)
═══════════════════════════════════════════════════════════════════════

paper/paper1/figures/run_<stamp>/G*.png  - 11 개 figure:

    G1  training_loss          step 별 loss (3 mode 겹쳐서 비교)
    G2  lr_schedule            warmup → linear decay 검증
    G3  grad_norm              gradient L2 norm (clipped) 추이
    G4  eval_metrics           silver val 의 epoch 별 accuracy/F1/precision/recall
    G5  eval_loss              silver val 의 epoch 별 loss
    G6  overall_accuracy       Gold OOD overall accuracy (mode 별 bar)
    G7  per_class_accuracy     format-regular / semi-regular / free 별 정확도 (grouped bar)
    G8  per_label_accuracy     26 라벨 × 3 mode dot plot (M2 기준 정렬)
    G9  gpu_memory             allocated + peak (RTX 5070 12GB)
    G10 wall_clock             train + eval 소요 시간 (mode 별 bar)
    G11 weight_l2_evolution    encoder layer 0/5/11 + classifier head 의 weight L2 진화

히트맵 제외 — 사용자 요청 (히트맵·색 많은 그래프 X).

═══════════════════════════════════════════════════════════════════════
■ 한글 폰트 처리
═══════════════════════════════════════════════════════════════════════

matplotlib 의 기본 DejaVu Sans 는 한글 글리프 부재. 그림 안에 한글이 들어가면
"□□□□" 식으로 깨짐. koreanize_matplotlib 패키지를 import 하면 NanumGothic 폰트가
자동으로 설정되어 한글 정상 출력.

설치 (한 번만):
    .venv/bin/pip install koreanize-matplotlib

═══════════════════════════════════════════════════════════════════════
■ 사용법
═══════════════════════════════════════════════════════════════════════

    # 가장 최신 sweep 자동 사용
    .venv/bin/python paper1/scripts/make_graphs.py

    # 특정 stamp 명시
    .venv/bin/python paper1/scripts/make_graphs.py 20260427_143110

═══════════════════════════════════════════════════════════════════════
■ 함수 구조 (한 줄 요약)
═══════════════════════════════════════════════════════════════════════

    find_latest_sweep()       paper/paper1/data/runs/ 에서 가장 최신 sweep dir 반환
    load_jsonl(p)             jsonl 파일 → list of dict (손상 line skip)
    load_cfg_data(sweep, cfg) 1 개 config 의 모든 jsonl/json 로드
    split_train_eval(scalars) scalars.jsonl 을 train log / eval log 로 분리
    setup_axes(ax, ...)       모든 plot 의 공통 축 스타일 (xlabel/ylabel/title/grid)
    g1_~ g11_*                각 figure 한 개씩 생성하는 함수 (위 출력 표 참조)
    _aggregate_layer_l2()     특정 layer 의 weight/grad L2 시계열 추출 (G11 용)
    main()                    위 함수들을 순차 호출
"""
from __future__ import annotations

import json
import sys
import statistics
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")          # GUI backend 비활성화 — 서버/WSL 환경 안전
import koreanize_matplotlib    # noqa: F401  (한글 폰트 자동 설정 — import 만으로 효과)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ["rule_m1_answer", "rule_m2_context", "rule_m3_negatives"]
CFG_LABELS = {
    "rule_m1_answer": "M1",
    "rule_m2_context": "M2",
    "rule_m3_negatives": "M3",
}
CFG_COLORS = {
    "rule_m1_answer": "#d62728",      # red — 답만 (성능 낮음)
    "rule_m2_context": "#1f77b4",     # blue — 문장
    "rule_m3_negatives": "#2ca02c",   # green — 문장+neg
}
CFG_STYLES = {
    "rule_m1_answer": "-",
    "rule_m2_context": "--",
    "rule_m3_negatives": ":",
}


def find_latest_sweep() -> Path:
    runs_dir = ROOT / "paper1" / "data" / "runs"
    cands = sorted(
        (d for d in runs_dir.iterdir()
         if d.is_dir() and (d / "summary.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    for d in cands:
        if all((d / cfg / "log").is_dir() for cfg in CONFIGS):
            return d
    raise SystemExit(f"적합한 sweep 디렉터리를 {runs_dir} 에서 찾지 못했습니다.")


def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 손상된 line skip
    return out


def load_cfg_data(sweep: Path, cfg: str) -> Dict[str, Any]:
    log_dir = sweep / cfg / "log"
    return {
        "cfg": cfg,
        "scalars": load_jsonl(log_dir / "scalars.jsonl"),
        "gpu": load_jsonl(log_dir / "gpu.jsonl"),
        "params": load_jsonl(log_dir / "params.jsonl"),
        "events": load_jsonl(log_dir / "events.jsonl"),
        "log_history": (json.loads((log_dir / "log_history.json").read_text(encoding="utf-8"))
                        if (log_dir / "log_history.json").exists() else {}),
    }


def _dedup_by_step(rows):
    """ts 정렬 후 step 단조 증가 prefix 만 keep. 두 번 이상 학습 누적 케이스 처리."""
    rows_sorted = sorted(rows, key=lambda r: r.get("ts", ""))
    cleaned = []
    last_step = -1
    for r in rows_sorted:
        s = r.get("step", -1)
        if s <= last_step:
            break  # step regress = 새 학습 시작 → drop
        cleaned.append(r)
        last_step = s
    return cleaned


def split_train_eval(scalars):
    train = [r for r in scalars if r.get("kind") != "eval" and "loss" in r]
    evals = [r for r in scalars if r.get("kind") == "eval"]
    # 같은 stamp 디렉터리에 학습이 두 번 이상 일어나 scalars.jsonl 에 두 학습
    # 기록이 누적되면 step 가 역행하는 line 이 그려짐. train + eval 모두 dedup.
    return _dedup_by_step(train), _dedup_by_step(evals)


def setup_axes(ax, *, xlabel="step", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


# ════════════════════════════════════════════════════════════════════════
# 그래프
# ════════════════════════════════════════════════════════════════════════


def g1_training_loss(data: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for cfg, d in data.items():
        train, _ = split_train_eval(d["scalars"])
        if not train:
            continue
        steps = [r["step"] for r in train]
        loss = [r["loss"] for r in train]
        ax.plot(steps, loss, label=CFG_LABELS[cfg],
                color=CFG_COLORS[cfg], linestyle="-",     # 모든 mode 실선
                linewidth=1.6, alpha=0.85)
    setup_axes(ax, ylabel="train loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "G1_training_loss.png", dpi=140)
    plt.close(fig)


def g2_lr_schedule(data: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for cfg, d in data.items():
        train, _ = split_train_eval(d["scalars"])
        if not train:
            continue
        steps = [r["step"] for r in train]
        lr = [r.get("learning_rate") for r in train]
        ax.plot(steps, lr, label=CFG_LABELS[cfg],
                color=CFG_COLORS[cfg], linestyle=CFG_STYLES[cfg], linewidth=1.3)
    setup_axes(ax, ylabel="learning rate", title="G2. LR schedule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "G2_lr_schedule.png", dpi=140)
    plt.close(fig)


def g3_grad_norm(data: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for cfg, d in data.items():
        train, _ = split_train_eval(d["scalars"])
        if not train:
            continue
        steps = [r["step"] for r in train]
        gn = [r.get("grad_norm") for r in train]
        ax.plot(steps, gn, label=CFG_LABELS[cfg],
                color=CFG_COLORS[cfg], linestyle=CFG_STYLES[cfg],
                linewidth=1.0, alpha=0.85)
    setup_axes(ax, ylabel="gradient L2 norm (clipped)", title="G3. Gradient norm vs step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "G3_grad_norm.png", dpi=140)
    plt.close(fig)


def g4_eval_metrics(data: Dict[str, Dict], out: Path) -> None:
    metrics = [
        ("eval_accuracy", "accuracy"),
        ("eval_f1", "F1"),
        ("eval_precision", "precision"),
        ("eval_recall", "recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (key, lbl) in zip(axes.flat, metrics):
        for cfg, d in data.items():
            _, evals = split_train_eval(d["scalars"])
            if not evals:
                continue
            ep = [r["epoch"] for r in evals]
            v = [r.get(key) for r in evals]
            ax.plot(ep, v, marker="o", label=CFG_LABELS[cfg],
                    color=CFG_COLORS[cfg], linestyle=CFG_STYLES[cfg], linewidth=1.6)
        setup_axes(ax, xlabel="epoch", ylabel=lbl, title=lbl)
        ax.legend(fontsize=8)
    fig.suptitle("G4. Validation metrics per epoch (mode 비교, silver 내부 holdout)")
    fig.tight_layout()
    fig.savefig(out / "G4_eval_metrics.png", dpi=140)
    plt.close(fig)


def g5_eval_loss(data: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg, d in data.items():
        _, evals = split_train_eval(d["scalars"])
        if not evals:
            continue
        ep = [r["epoch"] for r in evals]
        loss = [r.get("eval_loss") for r in evals]
        ax.plot(ep, loss, marker="s", label=CFG_LABELS[cfg],
                color=CFG_COLORS[cfg], linestyle="-", linewidth=1.6)
    setup_axes(ax, xlabel="epoch", ylabel="eval loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "G5_eval_loss.png", dpi=140)
    plt.close(fig)


def g6_overall_accuracy(summary: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cfgs = [r["cfg_id"] for r in summary]
    accs = [r["accuracy_overall"] for r in summary]
    bars = ax.bar([CFG_LABELS[c] for c in cfgs], accs,
                  color=[CFG_COLORS[c] for c in cfgs], width=0.55, edgecolor="black")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                f"{acc:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    setup_axes(ax, xlabel="mode", ylabel="Gold accuracy")
    fig.tight_layout()
    fig.savefig(out / "G6_overall_accuracy.png", dpi=140)
    plt.close(fig)


def g7_per_class_accuracy(summary: List[Dict], out: Path) -> None:
    classes = ["format-regular", "format-semi-regular", "format-free"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.25
    x_base = list(range(len(classes)))
    for i, r in enumerate(summary):
        cfg = r["cfg_id"]
        vals = [r["per_class"][c]["mean_acc"] for c in classes]
        xs = [x + (i - 1) * width for x in x_base]
        ax.bar(xs, vals, width=width, label=CFG_LABELS[cfg],
               color=CFG_COLORS[cfg], edgecolor="black")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x_base)
    ax.set_xticklabels([c.replace("format-", "") for c in classes])
    setup_axes(ax, xlabel="format-regularity class",
               ylabel="mean per-label accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "G7_per_class_accuracy.png", dpi=140)
    plt.close(fig)


def g8_per_label_accuracy(summary: List[Dict], out: Path) -> None:
    """라벨별 정확도 — 3 mode dot, M2 기준 정렬."""
    labels = sorted(summary[0]["per_label"].keys())
    m2_accs: Dict[str, float] = {}
    m2 = next(r for r in summary if r["cfg_id"] == "rule_m2_context")
    for lb in labels:
        m2_accs[lb] = m2["per_label"][lb]["accuracy"]
    labels_sorted = sorted(labels, key=lambda lb: m2_accs[lb])

    fig, ax = plt.subplots(figsize=(11, 9))
    y = list(range(len(labels_sorted)))
    for r in summary:
        cfg = r["cfg_id"]
        xs = [r["per_label"][lb]["accuracy"] for lb in labels_sorted]
        ax.scatter(xs, y, color=CFG_COLORS[cfg], s=42, alpha=0.85,
                   edgecolor="black", linewidth=0.5, label=CFG_LABELS[cfg])
    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=8)
    setup_axes(ax, xlabel="accuracy", ylabel="",
               title="G8. Per-label accuracy (mode 비교; M2 기준 정렬)")
    ax.set_xlim(-0.02, 1.05)
    ax.axvline(1.0, color="grey", linewidth=0.7, alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "G8_per_label_accuracy.png", dpi=140)
    plt.close(fig)


def g9_gpu_memory(data: Dict[str, Dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for cfg, d in data.items():
        if not d["gpu"]:
            continue
        steps = [r["step"] for r in d["gpu"]]
        alloc = [r["gpus"][0]["memory_allocated_gb"] for r in d["gpu"]]
        peak = [r["gpus"][0]["max_memory_allocated_gb"] for r in d["gpu"]]
        axes[0].plot(steps, alloc, label=CFG_LABELS[cfg],
                     color=CFG_COLORS[cfg], linestyle=CFG_STYLES[cfg],
                     linewidth=1.0, alpha=0.85)
        axes[1].plot(steps, peak, label=CFG_LABELS[cfg],
                     color=CFG_COLORS[cfg], linestyle=CFG_STYLES[cfg],
                     linewidth=1.0, alpha=0.85)
    setup_axes(axes[0], ylabel="allocated (GB)", title="GPU memory (allocated)")
    setup_axes(axes[1], ylabel="peak allocated (GB)", title="GPU memory (peak)")
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("G9. GPU memory vs step (RTX 5070, mode 비교)")
    fig.tight_layout()
    fig.savefig(out / "G9_gpu_memory.png", dpi=140)
    plt.close(fig)


def g10_wall_clock(data: Dict[str, Dict], summary: List[Dict], out: Path) -> None:
    from datetime import datetime
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cfgs = [r["cfg_id"] for r in summary]
    train_secs = []
    for c in cfgs:
        evs = data[c]["events"]
        end = next((e for e in evs if e.get("event") == "on_train_end"), None)
        begin = next((e for e in evs if e.get("event") == "on_train_begin"), None)
        if begin and end:
            t0 = datetime.fromisoformat(begin["ts"])
            t1 = datetime.fromisoformat(end["ts"])
            train_secs.append((t1 - t0).total_seconds())
        else:
            train_secs.append(0)
    eval_secs = []
    for c in cfgs:
        _, evals = split_train_eval(data[c]["scalars"])
        eval_secs.append(evals[-1].get("eval_runtime") if evals else 0)

    axes[0].bar([CFG_LABELS[c] for c in cfgs], train_secs,
                color=[CFG_COLORS[c] for c in cfgs], edgecolor="black", width=0.55)
    for i, t in enumerate(train_secs):
        if t:
            axes[0].text(i, t + max(train_secs) * 0.01, f"{t/60:.1f}m",
                         ha="center", fontsize=9)
    setup_axes(axes[0], xlabel="mode", ylabel="seconds",
               title="train wall-clock per mode")

    axes[1].bar([CFG_LABELS[c] for c in cfgs], eval_secs,
                color=[CFG_COLORS[c] for c in cfgs], edgecolor="black", width=0.55)
    for i, t in enumerate(eval_secs):
        if t:
            axes[1].text(i, t + max(eval_secs) * 0.01, f"{t:.0f}s",
                         ha="center", fontsize=9)
    setup_axes(axes[1], xlabel="mode", ylabel="seconds",
               title="eval runtime (silver holdout)")

    fig.suptitle("G10. Wall-clock comparison")
    fig.tight_layout()
    fig.savefig(out / "G10_wall_clock.png", dpi=140)
    plt.close(fig)


def _aggregate_layer_l2(rows, layer_substr, kind="weight"):
    steps, vals = [], []
    for row in rows:
        step = row.get("step")
        groups = row.get("groups", {})
        l2s = []
        for top, items in groups.items():
            for it in items:
                if layer_substr in it.get("name", ""):
                    key = f"{kind}_l2"
                    if key in it:
                        l2s.append(it[key])
        if l2s:
            steps.append(step)
            vals.append(sum(l2s) / len(l2s))
    return steps, vals


def g12_fit_diagnostic(data: Dict[str, Dict], out: Path) -> None:
    """Overfitting/Underfitting 진단 — train + val loss 를 같은 축에 표시.

    각 mode 마다 한 panel:
      - 회색 가는 선: train loss (step 단위, smoothed via rolling mean)
      - 색상 굵은 선 + 마커: val loss (epoch 단위)

    판독:
      - train/val 모두 ↓ + 격차 작음 → 정상 fit (no over/under)
      - train ↓ but val ↑ → overfitting
      - train, val 모두 plateau 가 높음 → underfitting
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    cfg_order = ["rule_m1_answer", "rule_m2_context", "rule_m3_negatives"]

    for ax, cfg in zip(axes, cfg_order):
        d = data.get(cfg, {})
        train, evals = split_train_eval(d.get("scalars", []))
        if not train:
            ax.set_visible(False)
            continue

        # train loss (step) → epoch 환산해서 같은 X 축
        # epoch 정보는 train row 에 들어 있음
        train_step = [r["step"] for r in train]
        train_epoch = [r.get("epoch", 0) for r in train]
        train_loss = [r["loss"] for r in train]

        # rolling mean 으로 smoothing (window=10)
        import statistics
        win = 10
        sm = []
        for i in range(len(train_loss)):
            lo = max(0, i - win // 2)
            hi = min(len(train_loss), i + win // 2 + 1)
            sm.append(statistics.mean(train_loss[lo:hi]))

        ax.plot(train_epoch, sm, color="grey", linewidth=1.0, alpha=0.7,
                label="train loss")

        # val loss (epoch)
        if evals:
            ev_epoch = [r["epoch"] for r in evals]
            ev_loss = [r.get("eval_loss") for r in evals]
            ax.plot(ev_epoch, ev_loss, color=CFG_COLORS[cfg],
                    marker="s", markersize=8, linewidth=2.0,
                    label="val loss")

        setup_axes(ax, xlabel="epoch", ylabel="loss",
                   title=f"{CFG_LABELS[cfg]}")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(bottom=0)

    fig.suptitle("G12. Train vs Val loss (overfitting/underfitting 진단)")
    fig.tight_layout()
    fig.savefig(out / "G12_fit_diagnostic.png", dpi=140)
    plt.close(fig)


def g11_param_l2(data: Dict[str, Dict], out: Path) -> None:
    layer_targets = [
        ("encoder.layer.0", "early (layer 0)"),
        ("encoder.layer.5", "mid (layer 5)"),
        ("encoder.layer.11", "late (layer 11)"),
        ("classifier", "classifier head"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (sub, title) in zip(axes.flat, layer_targets):
        for cfg, d in data.items():
            steps, l2s = _aggregate_layer_l2(d["params"], sub, kind="weight")
            if not steps:
                continue
            ax.plot(steps, l2s, label=CFG_LABELS[cfg],
                    color=CFG_COLORS[cfg], linestyle=CFG_STYLES[cfg], linewidth=1.3)
        setup_axes(ax, xlabel="step", ylabel="weight L2 (mean)", title=title)
        ax.legend(fontsize=8)
    fig.suptitle("G11. Weight L2 norm evolution by layer (mode 비교)")
    fig.tight_layout()
    fig.savefig(out / "G11_weight_l2_evolution.png", dpi=140)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════


def main() -> None:
    if len(sys.argv) > 1:
        sweep = ROOT / "paper1" / "data" / "runs" / sys.argv[1]
    else:
        sweep = find_latest_sweep()
    print(f"sweep dir: {sweep}")

    summary_path = sweep / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = [r for r in summary if r.get("status") == "ok"]
    if len(summary) < len(CONFIGS):
        print(f"경고: {len(summary)}/{len(CONFIGS)} configs 만 ok 상태")

    data = {cfg: load_cfg_data(sweep, cfg) for cfg in CONFIGS}

    out = ROOT / "paper1" / "figures" / f"run_{sweep.name}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out}")

    g1_training_loss(data, out);          print("  G1 training_loss")
    g2_lr_schedule(data, out);            print("  G2 lr_schedule")
    g3_grad_norm(data, out);              print("  G3 grad_norm")
    g4_eval_metrics(data, out);           print("  G4 eval_metrics")
    g5_eval_loss(data, out);              print("  G5 eval_loss")
    g6_overall_accuracy(summary, out);    print("  G6 overall_accuracy")
    g7_per_class_accuracy(summary, out);  print("  G7 per_class_accuracy")
    g8_per_label_accuracy(summary, out);  print("  G8 per_label_accuracy")
    g9_gpu_memory(data, out);             print("  G9 gpu_memory")
    g10_wall_clock(data, summary, out);   print("  G10 wall_clock")
    g11_param_l2(data, out);              print("  G11 weight_l2_evolution")
    g12_fit_diagnostic(data, out);        print("  G12 fit_diagnostic")
    print(f"\n완료 — 12 figures → {out}")


if __name__ == "__main__":
    main()
