"""라벨별 F1-Score 막대 그래프.

사용 예:
    from paper_module.paper1.plots.f1_bar import plot_f1_per_label

    scores = {"name": 0.92, "address": 0.85, "phone": 0.78}
    plot_f1_per_label(scores, save_path="out/f1.png")

디자인:
- 회색 단색 (색상 최소화)
- 막대 위에 점수를 숫자로 표기
- y-축 범위 0~1 고정 (F1 범위)
"""
from pathlib import Path
import matplotlib.pyplot as plt


def plot_f1_per_label(
    f1_scores,
    save_path,
    title="Label-wise F1 Score",
    sort_desc=True,
):
    """라벨별 F1-Score를 막대 그래프로 그린다.

    Args:
        f1_scores: {라벨명: F1-점수} 형태의 딕셔너리.
                   예: {"name": 0.92, "address": 0.85}
        save_path: 저장 경로 문자열 (예: "out/f1.png").
                   상위 디렉토리는 자동 생성됨.
        title: 그래프 제목.
        sort_desc: True면 F1 내림차순 정렬 (시각적 가독성).
                   False면 딕셔너리 순서 그대로.

    Returns:
        None. 파일 저장만 수행.
    """
    # --- 1. 데이터 추출 + 정렬 ---
    if sort_desc:
        # 점수 내림차순: F1이 높은 라벨이 왼쪽에 오게
        items = sorted(f1_scores.items(), key=lambda kv: kv[1], reverse=True)
    else:
        items = list(f1_scores.items())
    labels = [k for k, _ in items]
    scores = [v for _, v in items]

    # --- 2. 그래프 크기: 라벨 수에 비례 ---
    # 라벨이 많아지면 그래프도 넓어져야 읽을 수 있음
    width = max(8, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(width, 5))

    # --- 3. 막대 그리기: 회색 단색 + 검정 테두리 ---
    ax.bar(labels, scores, color="gray", edgecolor="black", linewidth=0.5)

    # --- 4. 축 꾸미기 ---
    ax.set_xlabel("Label")
    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)

    # x축 라벨 회전: 라벨이 길거나 많으면 겹치므로 45도 기울임
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- 5. 각 막대 위에 점수 숫자 표기 ---
    for x_idx, score in enumerate(scores):
        ax.text(
            x_idx, score + 0.01, f"{score:.2f}",
            ha="center", va="bottom", fontsize=8,
        )

    # --- 6. y축 그리드 (옅게) ---
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)  # 그리드가 막대 뒤에 오도록

    # --- 7. 파일 저장 ---
    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
