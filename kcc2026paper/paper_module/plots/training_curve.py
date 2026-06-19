"""학습 단계별 곡선 그래프 (loss + F1).

사용 예:
    from paper_module.plots.training_curve import plot_training_curve

    history = [
        {"step": 0,   "train_loss": 0.80, "val_loss": 0.75, "val_f1": 0.30},
        {"step": 100, "train_loss": 0.40, "val_loss": 0.45, "val_f1": 0.60},
        {"step": 200, "train_loss": 0.20, "val_loss": 0.30, "val_f1": 0.78},
    ]
    plot_training_curve(history, save_path="out/curve.png")

디자인:
- 왼쪽 y축: loss (실선=train, 점선=val)
- 오른쪽 y축: F1 (일점쇄선 + 점 마커)
- 선 모양만 다르게 해서 색상은 회색 계열로 통일
"""
from pathlib import Path
import matplotlib.pyplot as plt


def _collect_xy(history, x_key, y_key):
    """history에서 y_key 값이 None이 아닌 (x, y) 쌍만 모은다.

    반환: ([x들], [y들])  — 두 리스트 길이 동일.
    """
    xs, ys = [], []
    for h in history:
        y = h.get(y_key)
        if y is not None:
            xs.append(h[x_key])
            ys.append(y)
    return xs, ys


def plot_training_curve(
    history,
    save_path,
    title="Training Curve",
    x_key="step",
):
    """학습 곡선 그래프를 그린다.

    Args:
        history: 학습 기록 리스트. 각 항목은 dict.
                 키: x_key(보통 "step" 또는 "epoch"),
                     train_loss, val_loss, val_f1
                 일부 키가 없는 항목이 섞여 있어도 괜찮다 (None 으로 처리).
        save_path: 저장 경로 문자열.
        title: 그래프 제목.
        x_key: x축이 될 키 이름 ("step" 또는 "epoch").

    Returns:
        None. 파일 저장만 수행.

    참고:
        - 왼쪽 y축(loss)과 오른쪽 y축(F1)을 동시에 쓰는 dual-axis 구성.
        - twinx() 는 matplotlib이 제공하는 표준 방법이라 복잡하지 않다.
    """
    # --- 1. 지표별로 "None이 아닌 값만" 쌍(x, y)으로 모은다 ---
    # history 는 step마다 어떤 값은 있고 어떤 값은 없을 수 있음.
    # matplotlib의 plot은 None이 섞이면 선이 끊어지므로,
    # 각 지표별로 유효한 (x, y) 쌍만 따로 모아서 그린다.
    train_xy = _collect_xy(history, x_key, "train_loss")
    vloss_xy = _collect_xy(history, x_key, "val_loss")
    vf1_xy = _collect_xy(history, x_key, "val_f1")

    # --- 2. 왼쪽 y축: loss ---
    fig, ax_loss = plt.subplots(figsize=(8, 5))

    if train_xy[0]:
        # 실선 + 검정: train loss
        ax_loss.plot(
            train_xy[0], train_xy[1],
            color="black", linestyle="-", linewidth=1.5,
            label="Train Loss",
        )
    if vloss_xy[0]:
        # 점선 + 검정 + 점 마커: val loss (eval step에서만 찍힘)
        ax_loss.plot(
            vloss_xy[0], vloss_xy[1],
            color="black", linestyle="--", linewidth=1.5,
            marker="s", markersize=3,
            label="Val Loss",
        )

    ax_loss.set_xlabel(x_key.capitalize())  # "Step" or "Epoch"
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # --- 3. 오른쪽 y축: F1 (있을 때만) ---
    # twinx()는 같은 x축을 공유하는 두 번째 y축을 만드는 matplotlib 기능.
    # loss와 F1이 서로 스케일이 달라서 분리된 축이 필요함.
    ax_f1 = None
    if vf1_xy[0]:
        ax_f1 = ax_loss.twinx()
        # 일점쇄선 + 회색 + 점 마커: val F1
        ax_f1.plot(
            vf1_xy[0], vf1_xy[1],
            color="gray", linestyle="-.", linewidth=1.5,
            marker="o", markersize=3,
            label="Val F1",
        )
        ax_f1.set_ylabel("F1-Score")
        ax_f1.set_ylim(0, 1.0)

    # --- 5. 범례: 두 축의 선을 합쳐서 한 번에 표시 ---
    handles_1, labels_1 = ax_loss.get_legend_handles_labels()
    if ax_f1 is not None:
        handles_2, labels_2 = ax_f1.get_legend_handles_labels()
        ax_loss.legend(
            handles_1 + handles_2, labels_1 + labels_2, loc="best",
        )
    else:
        ax_loss.legend(loc="best")

    ax_loss.set_title(title)
    plt.tight_layout()

    # --- 6. 파일 저장 ---
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
