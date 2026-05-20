#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_PATH = Path("1.00_kfold_summary.json")
OUT_PATH = "robustness_boxplot_1.00.png"

def get_entry(data, tag: str):
    for k, v in data.items():
        if k.endswith(tag):
            return v
    return None

def get_fold_scores(entry):
    scores = entry.get("fold_scores") or []
    return [float(x) for x in scores]

def main():
    if not SUMMARY_PATH.exists():
        print(f"[Error] Summary file not found: {SUMMARY_PATH}")
        return

    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    pairs = [
        ("mBERT",        "pure_bert",    "crf_bert"),
        ("KLUE-RoBERTa", "pure_roberta", "crf_roberta"),
        ("XLM-R",        "pure_xlm",     "crf_xlm"),
    ]

    # 데이터 수집
    plot_data = [] # list of (label, pure_scores, crf_scores)
    all_scores = []

    for name, pure_tag, crf_tag in pairs:
        pure = get_entry(data, pure_tag)
        crf  = get_entry(data, crf_tag)
        
        if pure and crf:
            p_scores = get_fold_scores(pure)
            c_scores = get_fold_scores(crf)
            plot_data.append((name, p_scores, c_scores))
            all_scores.extend(p_scores)
            all_scores.extend(c_scores)

    if not plot_data:
        print("No data found.")
        return

    # Y축 범위 계산
    min_val = min(all_scores) if all_scores else 0.0
    max_val = max(all_scores) if all_scores else 1.0
    YMIN = max(0.0, min_val - 0.02)
    YMAX = min(1.005, max_val + 0.02)

    # Plotting
    # 가로:세로 = 2:1 비율 (12:6)
    fig, ax = plt.subplots(figsize=(12, 6))

    # 위치 설정
    indices = np.arange(len(plot_data))
    width = 0.3
    
    # 가로형이므로 positions는 Y축 좌표가 됨
    pure_positions = indices - width/1.5
    crf_positions = indices + width/1.5

    # Box Plot 그리기
    pure_scores_list = [d[1] for d in plot_data]
    crf_scores_list = [d[2] for d in plot_data]

    # Pure Models Boxplot (vert=False로 가로 방향)
    bp1 = ax.boxplot(pure_scores_list, positions=pure_positions, widths=width, vert=False,
                     patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.6),
                     medianprops=dict(color="blue"), showfliers=False)

    # CRF Models Boxplot (vert=False)
    bp2 = ax.boxplot(crf_scores_list, positions=crf_positions, widths=width, vert=False,
                     patch_artist=True, boxprops=dict(facecolor="orange", alpha=0.6),
                     medianprops=dict(color="red"), showfliers=False)

    # 개별 데이터 포인트 찍기 (Jitter)
    for i, (name, p_scores, c_scores) in enumerate(plot_data):
        # Pure points (x=score, y=position+jitter)
        x = p_scores
        y = np.random.normal(pure_positions[i], 0.02, size=len(x))
        ax.plot(x, y, 'o', color='blue', alpha=0.6, markersize=4)
        
        # CRF points
        x = c_scores
        y = np.random.normal(crf_positions[i], 0.02, size=len(x))
        ax.plot(x, y, 'o', color='darkred', alpha=0.6, markersize=4)

    # 꾸미기
    ax.set_title("Stability Analysis: F1 Score Distribution (5-Fold CV)")
    ax.set_xlabel("F1 Score")
    ax.set_yticks(indices)
    ax.set_yticklabels([d[0] for d in plot_data])
    ax.set_xlim(YMIN, YMAX)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Y축 반전 (위에서부터 순서대로)
    ax.invert_yaxis()

    # 범례 (Legend) - 박스플롯은 범례 추가가 까다로워서 대리 객체 생성
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Pure (Token Cls)'),
        Patch(facecolor='orange', edgecolor='black', label='BiLSTM+CRF'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()
    print(f"[saved] {OUT_PATH}")

if __name__ == "__main__":
    main()
