"""NER 학습/검증 결과 시각화"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.font_manager as fm
    matplotlib.use('Agg')  # GUI 없이 사용
    HAS_MATPLOTLIB = True
    
    # 한글 폰트 설정
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
        
        font_found = False
        for font_name in korean_fonts:
            if font_name in available_fonts:
                plt.rcParams['font.family'] = font_name
                font_found = True
                break
        
        if not font_found:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MATPLOTLIB = False


def plot_training_history(
    history: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = False,
    model_name: Optional[str] = None
) -> Optional[str]:
    """
    학습 곡선 시각화 (고급 버전)
    
    Args:
        history: 학습 히스토리 (loss, f1, precision, recall 등)
        output_path: 저장 경로 (None이면 자동 생성)
        show: 화면에 표시 여부
        model_name: 모델 이름 (제목에 사용)
    
    Returns:
        저장된 파일 경로
    """
    if not HAS_MATPLOTLIB:
        print("[시각화] matplotlib가 설치되지 않았습니다.")
        return None
    
    if not history:
        print("[시각화] 히스토리 데이터가 없습니다.")
        return None
    
    # 히스토리 데이터 추출
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_f1 = history.get("train_f1", [])
    val_f1 = history.get("val_f1", [])
    val_precision = history.get("val_precision", [])
    val_recall = history.get("val_recall", [])
    
    if not epochs:
        print("[시각화] 에포크 데이터가 없습니다.")
        return None
    
    # 그래프 생성 (2x3 그리드)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'NER Training Results - {model_name or "Model"}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Loss curve
    ax1 = axes[0, 0]
    if train_loss and len(train_loss) > 0:
        # train_loss는 step별이므로 epoch 스케일에 맞춤
        if len(train_loss) > len(epochs):
            # train_loss를 epoch에 맞게 리샘플링
            train_steps = np.linspace(1, len(epochs), len(train_loss))
            ax1.plot(train_steps, train_loss, 'b-', alpha=0.6, 
                    label='Train Loss', linewidth=2.0)
        else:
            # train_loss가 epoch보다 적으면 그대로 사용
            train_epochs = epochs[:len(train_loss)]
            ax1.plot(train_epochs, train_loss, 'b-', alpha=0.6, 
                    label='Train Loss', linewidth=2.0)
    if val_loss and len(val_loss) > 0:
        val_epochs = epochs[:len(val_loss)]
        ax1.plot(val_epochs, val_loss, 'r-o', linewidth=2.5, 
                markersize=6, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training/Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. F1 Score
    ax2 = axes[0, 1]
    if val_f1 and len(val_f1) > 0:
        val_f1_epochs = epochs[:len(val_f1)]
        best_f1_idx = np.argmax(val_f1)
        best_f1 = val_f1[best_f1_idx]
        ax2.plot(val_f1_epochs, val_f1, 'g-o', linewidth=2.5, 
                markersize=6, label='F1 Score', alpha=0.8)
        ax2.axhline(y=best_f1, color='r', linestyle='--', alpha=0.5, 
                   linewidth=2, label=f'Best: {best_f1:.4f}')
    if train_f1 and len(train_f1) > 0:
        if len(train_f1) > len(epochs):
            train_f1_steps = np.linspace(1, len(epochs), len(train_f1))
            ax2.plot(train_f1_steps, train_f1, 'b--', alpha=0.5, 
                    label='Train F1', linewidth=1.5)
        else:
            train_f1_epochs = epochs[:len(train_f1)]
            ax2.plot(train_f1_epochs, train_f1, 'b--', alpha=0.5, 
                    label='Train F1', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    if val_f1:
        f1_min, f1_max = min(val_f1), max(val_f1)
        ax2.set_ylim([max(0, f1_min - 0.1), min(1.05, f1_max + 0.1)])
    
    # 3. Precision
    ax3 = axes[0, 2]
    if val_precision and len(val_precision) > 0:
        val_precision_epochs = epochs[:len(val_precision)]
        ax3.plot(val_precision_epochs, val_precision, 'b-o', 
                linewidth=2.0, markersize=6, label='Precision', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    if val_precision:
        p_min, p_max = min(val_precision), max(val_precision)
        ax3.set_ylim([max(0, p_min - 0.1), min(1.05, p_max + 0.1)])
    
    # 4. Recall
    ax4 = axes[1, 0]
    if val_recall and len(val_recall) > 0:
        val_recall_epochs = epochs[:len(val_recall)]
        ax4.plot(val_recall_epochs, val_recall, 'g-o', 
                linewidth=2.0, markersize=6, label='Recall', alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax4.set_title('Recall', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    if val_recall:
        r_min, r_max = min(val_recall), max(val_recall)
        ax4.set_ylim([max(0, r_min - 0.1), min(1.05, r_max + 0.1)])
    
    # 5. Precision & Recall & F1 비교
    ax5 = axes[1, 1]
    if val_precision and len(val_precision) > 0:
        val_precision_epochs = epochs[:len(val_precision)]
        ax5.plot(val_precision_epochs, val_precision, 'b-', 
                linewidth=2.0, label='Precision', alpha=0.7)
    if val_recall and len(val_recall) > 0:
        val_recall_epochs = epochs[:len(val_recall)]
        ax5.plot(val_recall_epochs, val_recall, 'r-', 
                linewidth=2.0, label='Recall', alpha=0.7)
    if val_f1 and len(val_f1) > 0:
        val_f1_epochs = epochs[:len(val_f1)]
        ax5.plot(val_f1_epochs, val_f1, 'g-', 
                linewidth=2.5, label='F1', alpha=0.8)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax5.set_title('Precision & Recall & F1', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, loc='best', framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_ylim([0, 1.05])
    
    # 6. Precision-Recall Curve
    ax6 = axes[1, 2]
    if val_precision and val_recall and len(val_precision) > 1:
        precision = val_precision
        recall = val_recall
        sorted_pairs = sorted(zip(recall, precision))
        recall_sorted = [x[0] for x in sorted_pairs]
        precision_sorted = [x[1] for x in sorted_pairs]
        ap_score = np.trapz(precision_sorted, recall_sorted)
        ax6.plot(recall_sorted, precision_sorted, 'g-o', linewidth=2.5, 
                markersize=4, label=f'Model (AP={ap_score:.4f})', color='#2E86AB')
        
        # F1 iso-lines
        for f1 in [0.3, 0.5, 0.7, 0.9]:
            x = np.linspace(0.01, 1, 100)
            y = f1 * x / (2 * x - f1)
            y = np.clip(y, 0, 1)
            ax6.plot(x, y, '--', color='gray', alpha=0.2, linewidth=1)
        
        ax6.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax6.set_title(f'Precision-Recall Curve (AP={ap_score:.4f})', 
                     fontsize=13, fontweight='bold')
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        ax6.legend(fontsize=10, loc='lower left', framealpha=0.9)
        ax6.grid(True, alpha=0.3, linestyle='--')
    else:
        ax6.axis('off')
        ax6.text(0.5, 0.5, 'Not enough data\nfor PR curve', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # 저장
    if output_path is None:
        output_path = "data/out/ner_training_history.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"[시각화] 학습 곡선 저장: {output_path}")
    return output_path


def plot_validation_metrics(
    metrics: Dict[str, float],
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    검증 메트릭 시각화
    
    Args:
        metrics: 검증 메트릭 (precision, recall, f1 등)
        output_path: 저장 경로
        show: 화면에 표시 여부
    
    Returns:
        저장된 파일 경로
    """
    if not HAS_MATPLOTLIB:
        print("[시각화] matplotlib가 설치되지 않았습니다.")
        return None
    
    if not metrics:
        print("[시각화] 메트릭 데이터가 없습니다.")
        return None
    
    # 메트릭 추출
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    f1 = metrics.get("f1", 0.0)
    
    # 바 차트 생성
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics_names = ["Precision", "Recall", "F1 Score"]
    metrics_values = [precision, recall, f1]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    
    # 값 표시
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('NER Model Validation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 저장
    if output_path is None:
        output_path = "data/out/ner_validation_metrics.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"[시각화] 검증 메트릭 저장: {output_path}")
    return output_path


def plot_label_performance(
    label_metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[str]:
    """
    라벨별 성능 시각화
    
    Args:
        label_metrics: {label: {precision, recall, f1}}
        output_path: 저장 경로
        show: 화면에 표시 여부
    
    Returns:
        저장된 파일 경로
    """
    if not HAS_MATPLOTLIB:
        print("[시각화] matplotlib가 설치되지 않았습니다.")
        return None
    
    if not label_metrics:
        print("[시각화] 라벨 메트릭 데이터가 없습니다.")
        return None
    
    # 데이터 준비
    labels = list(label_metrics.keys())
    precision_values = [label_metrics[label].get("precision", 0.0) for label in labels]
    recall_values = [label_metrics[label].get("recall", 0.0) for label in labels]
    f1_values = [label_metrics[label].get("f1", 0.0) for label in labels]
    
    # 그래프 생성
    x = range(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))
    
    ax.bar([i - width for i in x], precision_values, width, label='Precision', alpha=0.7)
    ax.bar(x, recall_values, width, label='Recall', alpha=0.7)
    ax.bar([i + width for i in x], f1_values, width, label='F1 Score', alpha=0.7)
    
    ax.set_xlabel('Labels', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('NER Model Performance by Label', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 저장
    if output_path is None:
        output_path = "data/out/ner_label_performance.png"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"[시각화] 라벨별 성능 저장: {output_path}")
    return output_path
