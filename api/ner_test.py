#!/usr/bin/env python3
"""
Simple 6-model NER benchmark using the new BERT+BiLSTM+CRF pipeline.

Models:
  1) google-bert/bert-base-multilingual-cased         (pure token classification head)
  2) klue/roberta-large                               (pure)
  3) FacebookAI/xlm-roberta-large                     (pure)
  4) google-bert/bert-base-multilingual-cased         (BiLSTM+CRF)
  5) klue/roberta-large                               (BiLSTM+CRF)
  6) FacebookAI/xlm-roberta-large                     (BiLSTM+CRF)

Assumes:
  - module/ner/ner_data.py    provides read_conll, build_label_map, NERDataset
  - module/ner/ner_model.py   provides BertBiLstmCrf, NERConfig
"""

from __future__ import annotations

import argparse
import json
import random
import math
import gc
import sys
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import KFold, GroupKFold
import copy

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Add current directory to sys.path to resolve 'module' package issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

# Import data generation logic from ner_train.py
try:
    from src.ner.ner_train import generate_training_samples
    HAS_DATA_GEN = True
except ImportError as e:
    HAS_DATA_GEN = False
    print(f"[FATAL] Could not import data generation module from src.ner.ner_train: {e}")
    print("Please ensure src/ner/ner_train.py exists and has no syntax errors.")
    # Fail hard to avoid generating garbage graphs with dummy data
    sys.exit(1)

# Visualization libraries (optional)
try:
    import matplotlib
    # Set backend to 'Agg' for server/headless environments before importing pyplot
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.font_manager as fm
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, auc
    HAS_VISUALIZATION = True
    
    # Korean Font Setup
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic', 'Gulim']
        font_found = False
        for font_name in korean_fonts:
            if font_name in available_fonts:
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                print(f"[Viz] Set Korean font: {font_name}")
                font_found = True
                break
        if not font_found:
            print("[Viz] No Korean font found. Text may appear broken.")
    except Exception as e:
        print(f"[Viz] Font setup failed: {e}")

except ImportError:
    HAS_VISUALIZATION = False
    plt = None
    sns = None
    confusion_matrix = None
    gridspec = None
    precision_recall_curve = None
    average_precision_score = None
    auc = None

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# our modules
from src.ner.ner_data import read_conll, build_label_map, NERDataset
from src.ner.ner_model import BertBiLstmCrf, NERConfig


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and tensors to standard python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    model_name: str,
    train_file: Path,
    dev_file: Path | None,
    test_file: Path | None,
    max_length: int,
    batch_size: int,
) -> Tuple[
    Dict[str, int],
    Dict[int, str],
    AutoTokenizer,
    DataLoader,
    DataLoader,
    DataLoader | None,
]:
    """Load data files, split if necessary, and build DataLoaders."""
    print(f"[data] Loading train data from {train_file}")
    train_sentences = read_conll(str(train_file))

    if dev_file is not None:
        print(f"[data] Loading dev data from {dev_file}")
        dev_sentences = read_conll(str(dev_file))
    else:
        # Fallback split if dev_file not provided (e.g. user supplied only train_file)
        print("[data] No dev_file provided. Splitting train data 8:2...")
        random.seed(42) 
        random.shuffle(train_sentences)
        split_idx = int(len(train_sentences) * 0.8)
        dev_sentences = train_sentences[split_idx:]
        train_sentences = train_sentences[:split_idx]
        print(f"[data] Split result: Train={len(train_sentences)}, Dev={len(dev_sentences)}")

    test_sentences = None
    if test_file is not None:
        print(f"[data] Loading test data from {test_file}")
        test_sentences = read_conll(str(test_file))

    # label mapping from TRAIN only (to avoid leakage)
    _, label2id, id2label = build_label_map(train_sentences)
    print(f"[data] Labels ({len(label2id)}): {sorted(label2id.keys())}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    train_dataset = NERDataset(train_sentences, tokenizer, label2id, max_length)
    dev_dataset = NERDataset(dev_sentences, tokenizer, label2id, max_length)
    test_dataset = (
        NERDataset(test_sentences, tokenizer, label2id, max_length)
        if test_sentences is not None
        else None
    )

    # Optimize DataLoader for GPU utilization
    # num_workers: parallel data loading (CPU -> RAM)
    # pin_memory: faster transfer (RAM -> VRAM)
    num_workers = min(4, os.cpu_count() or 1)
    print(f"[data] DataLoader config: batch_size={batch_size}, num_workers={num_workers}, pin_memory=True")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    test_loader = (
        DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        if test_dataset is not None
        else None
    )

    return label2id, id2label, tokenizer, train_loader, dev_loader, test_loader


def collect_seqeval_inputs_from_crf_batch(
    pred_paths: List[List[int]],
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert CRF decode outputs + labels into seqeval input format."""
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    batch_size = labels.size(0)

    for i in range(batch_size):
        true_labels = []
        pred_labels = []
        
        # pred_paths[i]는 attention_mask가 1인 토큰에 대한 예측
        pred_idx = 0
        
        for j in range(len(labels[i])):
            # Skip padded tokens
            if attention_mask[i][j] == 0:
                continue
                
            # Skip ignored tokens (-100) but consume prediction
            if labels[i][j] != -100:
                true_labels.append(id2label[labels[i][j].item()])
                if pred_idx < len(pred_paths[i]):
                    pred_labels.append(id2label[pred_paths[i][pred_idx]])
                else:
                    pred_labels.append("O")  # fallback
            
            pred_idx += 1

        all_true.append(true_labels)
        all_pred.append(pred_labels)

    return all_true, all_pred


def collect_seqeval_inputs_from_logits_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert softmax logits + labels into seqeval input format."""
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    # (B, L, num_labels) -> (B, L)
    preds = logits.argmax(dim=-1)

    batch_size = labels.size(0)
    for i in range(batch_size):
        true_labels = []
        pred_labels = []
        
        for j, label_id in enumerate(labels[i]):
            # Skip padded tokens
            if attention_mask[i][j] == 0:
                continue
                
            # Skip ignored tokens
            if label_id == -100:
                continue
            
            true_labels.append(id2label[label_id.item()])
            pred_labels.append(id2label[preds[i][j].item()])

        all_true.append(true_labels)
        all_pred.append(pred_labels)

    return all_true, all_pred


def apply_token_noise(token: str) -> str:
    """Apply simple OCR-like noise to a token."""
    # Increased noise probability from 10% to 30% to make training harder/more realistic
    if len(token) < 2 or random.random() > 0.3: 
        return token
    
    noise_type = random.choice(['jamo', 'typo', 'space'])
    if noise_type == 'space':
        # Insert random space
        split_idx = random.randint(1, len(token)-1)
        return token[:split_idx] + " " + token[split_idx:]
    elif noise_type == 'typo':
        # Replace a char with similar looking one (dummy)
        idx = random.randint(0, len(token)-1)
        char = token[idx]
        # Simple mutation
        return token[:idx] + "?" + token[idx+1:]
    return token

def write_bio_word_level(samples: List[Dict], filepath: Path, apply_noise: bool = False) -> None:
    """
    Save samples to BIO format (Word/Token Level).
    Compatible with NERDataset which expects space-separated tokens.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = sample.get('text', '')
            entities = sample.get('entities', []) or []
            
            # Simple Word Tokenization (Space split)
            tokens = text.split()
            if not tokens:
                continue
                
            # Initialize all labels as 'O'
            labels = ['O'] * len(tokens)
            
            # Create character-level label array
            char_labels = ['O'] * len(text)
            
            # First, mark all entity positions at character level
            # Use a list of (start, end, type) to handle duplicates correctly
            entity_spans = []
            search_start_pos = 0
            
            current_search_pos = 0
            for entity_text, entity_type in entities:
                entity_start = text.find(entity_text, current_search_pos)
                if entity_start == -1:
                    # Try searching from beginning if not found
                    entity_start = text.find(entity_text)
                    
                if entity_start != -1:
                    entity_end = entity_start + len(entity_text)
                    entity_spans.append((entity_start, entity_end, entity_type))
                    if entity_start >= current_search_pos:
                        current_search_pos = entity_end
            
            for start, end, etype in entity_spans:
                char_labels[start] = f"B-{etype}"
                for i in range(start + 1, end):
                    char_labels[i] = f"I-{etype}"
            
            # Now map character labels to token labels
            char_pos = 0
            for token_idx, token in enumerate(tokens):
                # Find token position in text
                token_start = text.find(token, char_pos)
                if token_start == -1:
                    char_pos += len(token) + 1
                    continue
                    
                token_end = token_start + len(token)
                
                # Check if this token contains any entity label
                for i in range(token_start, token_end):
                    if i < len(char_labels) and char_labels[i] != 'O':
                        labels[token_idx] = char_labels[i]
                        break
                
                char_pos = token_end
            
            # Write to file
            for token, label in zip(tokens, labels):
                final_token = token
                # Apply noise to ALL tokens (including entities) if requested
                # This makes the task much harder and more realistic (OCR errors affect names too!)
                if apply_noise:
                    final_token = apply_token_noise(token)
                f.write(f"{final_token}\t{label}\n")
            f.write("\n")


def generate_dynamic_dataset(output_root: Path, num_samples: int = 10000) -> Tuple[Path, Path]:
    """Generate separate Train (8000) and Dev (2000) datasets."""
    train_file = output_root / "dynamic_train.txt"
    dev_file = output_root / "dynamic_dev.txt"
    
    if not HAS_DATA_GEN:
        print("[Error] Data generation module missing.")
        # Dummy
        train_file.write_text("Hello O\n", encoding="utf-8")
        dev_file.write_text("Hello O\n", encoding="utf-8")
        return train_file, dev_file

    # Fixed split as requested: 8000 Train, 2000 Dev
    num_train = 8000
    num_dev = 2000
    
    print(f"[DataGen] Generating {num_train:,} Train samples (with OCR noise)...")
    train_samples = generate_training_samples(num_train, balanced=True, dataset_type='train')
    write_bio_word_level(train_samples, train_file, apply_noise=True)
    
    print(f"[DataGen] Generating {num_dev:,} Dev samples (Clean)...")
    dev_samples = generate_training_samples(num_dev, balanced=True, dataset_type='dev')
    write_bio_word_level(dev_samples, dev_file, apply_noise=False)
            
    return train_file, dev_file


def compute_seqeval_metrics(
    all_true: List[List[str]],
    all_pred: List[List[str]],
) -> Dict[str, float | str]:
    """Return precision, recall, f1 and report string."""
    p = precision_score(all_true, all_pred, zero_division=0)
    r = recall_score(all_true, all_pred, zero_division=0)
    f = f1_score(all_true, all_pred, zero_division=0)
    rep = classification_report(all_true, all_pred, digits=4, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "report": rep}


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    id2label: Dict[int, str],
    device: torch.device,
    model_type: Literal["pure", "crf"],
    return_probs: bool = False,
) -> Tuple[Dict[str, float | str], List[List[str]], List[List[str]], Dict[str, List[float]] | None]:
    """
    Evaluate a model.
    Returns: (metrics, all_true_labels, all_pred_labels, prob_data)
    prob_data is None unless return_probs=True.
    """
    model.eval()
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []
    
    # For PR curve (binary classification: Entity vs O)
    # We collect "probability of being an entity" and "is actually entity"
    y_scores_entity = []  # Probability of NOT being 'O'
    y_true_entity = []    # 1 if not 'O', else 0

    # Find 'O' label ID
    label2id = {v: k for k, v in id2label.items()}
    o_label_id = label2id.get("O")
    
    if return_probs:
        if o_label_id is None:
            print("[Debug] 'O' label not found in label2id keys:", list(label2id.keys())[:10])
            # Fallback
            for k, v in id2label.items():
                if v == 'O':
                    o_label_id = k
                    break
            if o_label_id is None:
                o_label_id = 0
                print("[Warning] Still could not find 'O'. Using 0.")
        # print(f"[Debug] Using O label ID: {o_label_id}")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Calculate Logits & Preds
            if model_type == "crf":
                # For CRF, we can't easily get token-level probabilities for PR curve
                # So we just use decode output.
                # If we really needed probs, we'd need marginal probabilities from CRF, 
                # which torchcrf doesn't provide easily.
                # We skip prob collection for CRF for now or use dummy.
                pred_paths = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                bt_true, bt_pred = collect_seqeval_inputs_from_crf_batch(
                    pred_paths, labels, attention_mask, id2label
                )
                
                # Dummy probs for CRF (since hard preds only)
                # This will make PR curve look like steps, which is expected for hard predictions.
                if return_probs:
                     # Reconstruct flattened binary arrays from bt_true/bt_pred
                    for t_seq, p_seq in zip(bt_true, bt_pred):
                        for t, p in zip(t_seq, p_seq):
                            # 1 if Entity, 0 if O
                            y_true_entity.append(0.0 if t == 'O' else 1.0)
                            y_scores_entity.append(0.0 if p == 'O' else 1.0)

            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs.logits
                bt_true, bt_pred = collect_seqeval_inputs_from_logits_batch(
                    logits, labels, attention_mask, id2label
                )
                
                if return_probs:
                    # Softmax for probabilities
                    probs = torch.softmax(logits, dim=-1) # (B, L, C)
                    
                    # Probability of O
                    # Ensure o_label_id is valid
                    if o_label_id is None:
                        # Fallback: Assume 0 is O
                        o_idx = 0
                    else:
                        o_idx = o_label_id
                        
                    probs_o = probs[:, :, o_idx] # (B, L)
                    # Probability of Entity = 1 - Prob(O)
                    probs_entity = 1.0 - probs_o
                    
                    # Collect valid tokens (matching bt_true logic)
                    mask = attention_mask.bool()
                    
                    for i in range(labels.size(0)):
                        seq_len = int(mask[i].sum())
                        # Get valid labels from original tensor (exclude padding)
                        valid_labels_tensor = labels[i][:seq_len]
                        valid_probs_tensor = probs_entity[i][:seq_len]
                        
                        # Filter out -100 (ignored tokens)
                        # This must match exactly what collect_seqeval_inputs does
                        active_indices = (valid_labels_tensor != -100)
                        
                        # Convert to list
                        final_probs = valid_probs_tensor[active_indices].cpu().tolist()
                        final_labels_cpu = valid_labels_tensor[active_indices].cpu().tolist()
                        
                        # Collect data directly from tensors
                        if len(final_probs) > 0:
                            # 1 if Entity (not O), 0 if O
                            is_entity = [(1.0 if lid != o_label_id else 0.0) for lid in final_labels_cpu]
                            y_true_entity.extend(is_entity)
                            y_scores_entity.extend(final_probs)

            all_true.extend(bt_true)
            all_pred.extend(bt_pred)

    metrics = compute_seqeval_metrics(all_true, all_pred)
    
    prob_data = None
    if return_probs:
        if not y_true_entity:
            print("[Warning] No entity probability data collected! (Lists are empty)")
        else:
            # Check if we have enough data for PR curve (at least one positive and one negative ideally, but sklearn handles it)
            pass
            
        prob_data = {
            "y_true": y_true_entity,
            "y_scores": y_scores_entity
        }
        
    return metrics, all_true, all_pred, prob_data


def save_comprehensive_dashboard(
    history: Dict,
    test_metrics: Dict | None,
    prob_data: Dict | None,
    output_dir: Path,
    model_name: str,
    model_type: str,
    best_epoch: int
):
    """Generate the 8-panel dashboard as requested."""
    if not HAS_VISUALIZATION:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use evaluation index for x-axis (simpler and consistent)
    x_label = "Epoch"
    steps = history.get("steps", range(1, len(history["eval_f1"]) + 1))
    
    # Setup Figure with GridSpec
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.3)
    
    plt.suptitle(f"NER Training Results - {model_name} ({model_type})", fontsize=20, y=0.95)

    # 1. Training/Validation Loss
    ax1 = plt.subplot(gs[0, 0])
    if history["train_loss"]:
        # Ensure steps length matches data length
        curr_steps = steps[:len(history["train_loss"])]
        ax1.plot(curr_steps, history["train_loss"], 'b-', label='Train Loss', alpha=0.7)
    ax1.set_title("Training Loss", fontsize=14)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # 2. F1 Score (Span-level)
    ax2 = plt.subplot(gs[0, 1])
    if "eval_f1" in history and history["eval_f1"]:
        curr_steps = steps[:len(history["eval_f1"])]
        ax2.plot(curr_steps, history["eval_f1"], 'g-o', label='F1 Score', markersize=3)
        ax2.axhline(y=max(history["eval_f1"]), color='r', linestyle='--', alpha=0.5, label='Max F1')
    ax2.set_title("F1 Score (Span-level)", fontsize=14)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    # 3. Precision-Recall Curve (No change needed for x-axis)
    ax3 = plt.subplot(gs[0, 2])
    ap_score = 0.0
    if prob_data and prob_data["y_true"] and prob_data["y_scores"]:
        y_true = prob_data["y_true"]
        y_scores = prob_data["y_scores"]
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        ax3.plot(recall, precision, 'b-', label=f'Model (AP={ap_score:.4f})')
        ax3.plot(1, 1, 'bo') # End point
    else:
        ax3.text(0.5, 0.5, "No Probability Data\n(N/A for CRF or Missing)", ha='center', va='center', fontsize=12)
    
    ax3.set_title(f"Precision-Recall Curve (AP={ap_score:.4f})", fontsize=14)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_xlim(0, 1.05)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower left')

    # 4. Precision & Recall History
    ax4 = plt.subplot(gs[1, 0])
    if "eval_precision" in history and history["eval_precision"]:
        curr_steps = steps[:len(history["eval_precision"])]
        ax4.plot(curr_steps, history["eval_precision"], 's-', label='Precision', color='blue', markersize=3)
    if "eval_recall" in history and history["eval_recall"]:
        curr_steps = steps[:len(history["eval_recall"])]
        ax4.plot(curr_steps, history["eval_recall"], '^-', label='Recall', color='red', markersize=3)
    ax4.set_title("Precision & Recall", fontsize=14)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend()

    # 5. F1 Progress (Best-so-far)
    ax5 = plt.subplot(gs[1, 1])
    if "eval_f1" in history and history["eval_f1"]:
        f1s = history["eval_f1"]
        curr_steps = steps[:len(f1s)]
        best_so_far = [max(f1s[:i+1]) for i in range(len(f1s))]
        ax5.plot(curr_steps, f1s, 'g-o', label='F1 Score', alpha=0.5, markersize=3)
        ax5.plot(curr_steps, best_so_far, 'r-', label='Best-so-far', linewidth=2)
    ax5.set_title("F1 Progress (Best-so-far)", fontsize=14)
    ax5.set_xlabel(x_label)
    ax5.set_ylabel("F1 Score")
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.legend()

    # 6. Convergence Analysis (F1 Derivative)
    ax6 = plt.subplot(gs[1, 2])
    if "eval_f1" in history and len(history["eval_f1"]) > 1:
        f1s = np.array(history["eval_f1"])
        curr_steps = steps[:len(f1s)]
        # Calculate derivative (change per step)
        derivative = np.gradient(f1s)
        ax6.plot(curr_steps, derivative, 'b-')
        ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax6.set_title("Convergence Analysis (F1 Derivative)", fontsize=14)
    ax6.set_xlabel(x_label)
    ax6.set_ylabel("F1 Improvement Rate")
    ax6.grid(True, linestyle='--', alpha=0.5)

    # 7. Final Evaluation Results (Text Box)
    ax7 = plt.subplot(gs[2, 0])
    ax7.axis('off')
    
    final_f1 = test_metrics['f1'] if test_metrics and 'f1' in test_metrics else (history["eval_f1"][-1] if history["eval_f1"] else 0.0)
    
    # Use history as fallback if test_metrics is zero or missing
    if test_metrics and test_metrics.get('precision', 0.0) > 0:
        final_prec = test_metrics['precision']
    else:
        final_prec = history["eval_precision"][-1] if history["eval_precision"] else 0.0
        
    if test_metrics and test_metrics.get('recall', 0.0) > 0:
        final_rec = test_metrics['recall']
    else:
        final_rec = history["eval_recall"][-1] if history["eval_recall"] else 0.0
        
    best_f1 = max(history["eval_f1"]) if history["eval_f1"] else 0.0
    
    result_text = (
        "============================================\n"
        "          Final Evaluation Results\n"
        "============================================\n\n"
        f"F1 Score...................: {final_f1:.4f}\n"
        f"Precision..................: {final_prec:.4f}\n"
        f"Recall.....................: {final_rec:.4f}\n"
        f"Best F1 (Dev)..............: {best_f1:.4f}\n"
        f"Average Precision (AP).....: {ap_score:.4f}\n\n"
        f"Total Evaluations: {len(history['eval_f1']) if history['eval_f1'] else 0}\n"
        f"Model: {model_name}\n"
        "============================================"
    )
    
    # Create a fancy box
    props = dict(boxstyle='round', facecolor='aliceblue', alpha=0.5)
    ax7.text(0.05, 0.5, result_text, transform=ax7.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace', bbox=props)

    # 8. Training Information (Text Box)
    ax8 = plt.subplot(gs[2, 1:]) # Span 2 columns
    ax8.axis('off')
    
    info_text = (
        "===============================================================\n"
        "                   Training Information\n"
        "===============================================================\n\n"
        f"Model: {model_name}\n"
        f"Type: {model_type}\n\n"
        f"Evaluation with Best F1: {best_epoch}\n"
        f"Best F1 Score: {best_f1:.4f}\n\n"
        "Note:\n"
        "- PR Curve shows Precision vs Recall (Entity vs O)\n"
        "- AP (Average Precision) is the area under PR curve\n"
        "- Convergence shows F1 improvement rate\n"
        "==============================================================="
    )
    
    props2 = dict(boxstyle='round', facecolor='floralwhite', alpha=0.5)
    ax8.text(0.05, 0.5, info_text, transform=ax8.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace', bbox=props2)

    plt.savefig(output_dir / f"{model_name.replace('/', '-')}_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(
    all_true: List[List[str]], 
    all_pred: List[List[str]], 
    output_dir: Path, 
    model_name: str
):
    """Generate and save confusion matrix heatmap."""
    if not HAS_VISUALIZATION:
        return

    # Flatten lists
    y_true = [tag for sentence in all_true for tag in sentence]
    y_pred = [tag for sentence in all_pred for tag in sentence]
    
    # Get unique labels sorted
    labels = sorted(list(set(y_true + y_pred)))
    # Remove 'O' to focus on entities if too many labels
    if 'O' in labels and len(labels) > 10:
        labels_no_o = [l for l in labels if l != 'O']
        # If we still have labels, use them. Otherwise keep O.
        if labels_no_o:
             # This filters out 'O' from the confusion matrix visualization
             # We need to filter y_true/y_pred as well or just let sklearn handle it (it will count 'O' as misclassification if not in labels)
             pass 

    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"[Visualization] Failed to plot confusion matrix: {e}")


def save_comparison_plots(summary: Dict, output_root: Path):
    """Save comparison bar charts for all models."""
    if not HAS_VISUALIZATION:
        return
        
    output_root.mkdir(parents=True, exist_ok=True)
    
    model_names = list(summary.keys())
    dev_f1s = [info['dev_f1'] for info in summary.values()]
    test_f1s = [info['test_f1'] if info['test_f1'] is not None and not math.isnan(info['test_f1']) else 0.0 for info in summary.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, dev_f1s, width, label='Dev F1')
    plt.bar(x + width/2, test_f1s, width, label='Test F1')
    
    plt.ylabel('F1 Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_root / "model_comparison.png")
    plt.close()


def resolve_model_path(model_name: str) -> str:
    """
    Check if a local version of the model exists in 'model_downloaded'.
    If so, return the local path. Otherwise, return the HF Hub name.
    """
    # Map HF names to directory names (usually hyphens instead of slashes)
    dir_name = model_name.replace("/", "-")
    local_path = Path("model_downloaded") / dir_name
    
    if local_path.exists() and local_path.is_dir():
        print(f"[setup] Found local model for {model_name} at {local_path}")
        return str(local_path)
    return model_name


def train_one_model(
    *,
    model_name: str,
    model_type: Literal["pure", "crf"],
    train_file: Path,
    dev_file: Path | None,
    test_file: Path | None,
    output_dir: Path,
    model_dir: Path,
    max_length: int = 128,
    batch_size: int = 16,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    num_evaluations: int = 20,
    seed: int = 42,
    save_weights: bool = True,
) -> Dict:
    """Train a single model (pure or CRF) and evaluate on dev/test."""

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    if device.type == 'cuda':
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[train] Initial VRAM: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
    print(f"[train] Output Dir (Plots): {output_dir}")
    if save_weights:
        print(f"[train] Model Dir (Weights): {model_dir}")
    else:
        print(f"[train] Model weights will NOT be saved (save_weights=False)")

    label2id, id2label, tokenizer, train_loader, dev_loader, test_loader = (
        build_dataloaders(
            model_name=model_name,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            max_length=max_length,
            batch_size=batch_size,
        )
    )

    # Total steps for 1 epoch
    total_steps = num_epochs * len(train_loader)
    print(f"[train] Total steps: {total_steps}, Epochs: {num_epochs}")

    num_labels = len(label2id)
    print(f"[train] num_labels determined from data: {num_labels}")
    if num_labels < 3:
        print(f"[WARNING] num_labels is very small ({num_labels}). Check if data is loaded correctly.")
        print(f"[Debug] label2id keys: {list(label2id.keys())}")

    # build model
    cfg = None  # Initialize cfg variable
    if model_type == "pure":
        print(f"[train] Building pure token classification model: {model_name}")
        # Load config explicitly to ensure num_labels is applied
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels,
            id2label={i: l for l, i in label2id.items()},
            label2id=label2id,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True  # Re-added to fix size mismatch error
        )
    else:
        print(f"[train] Building BiLSTM+CRF model on top of: {model_name}")
        cfg = NERConfig(
            model_name_or_path=model_name,
            lstm_hidden_size=256,
            lstm_num_layers=1,
            dropout=0.1,
            max_length=max_length,
        )
        model = BertBiLstmCrf(cfg, num_labels=num_labels)

    model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {
        "train_loss": [],
        "steps": [],  # Track global steps for x-axis
        "eval_f1": [],
        "eval_precision": [],
        "eval_recall": [],
    }

    dev_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "report": ""}
    global_step = 0

    # Enable TF32 for faster math on Ampere+ GPUs (like H200)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize GradScaler for AMP
    # Handle deprecation warning for newer PyTorch versions
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler()

    # training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        print(f"\n===== [{model_type.upper()}] Epoch {epoch}/{num_epochs} =====")
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Use AMP for mixed precision training
            with torch.cuda.amp.autocast():
                if model_type == "crf":
                    # CRF model forward handles label preprocessing (converting -100 to 0) internally
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Unscale and step
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            loss_val = loss.item()
            if math.isnan(loss_val):
                print(f"[FATAL] NaN loss detected at epoch {epoch}, step {step}. Stopping training for this model.")
                return {
                    "error": "NaN loss detected",
                    "model_name": model_name,
                    "last_epoch": epoch,
                    "last_step": step,
                }

            total_loss += loss_val

            if step % 100 == 0 or step == len(train_loader):
                avg_loss = total_loss / step
                print(
                    f"[epoch {epoch}] step {step:4d}/{len(train_loader):4d} "
                    f"loss = {avg_loss:.4f}"
                )

        avg_epoch_loss = total_loss / max(len(train_loader), 1)
        print(f"[train] Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
        
        # Epoch-wise Evaluation
        print(f"\n[Epoch {epoch}] Evaluating...")
        dev_metrics, _, _, _ = evaluate_model(
            model=model,
            dataloader=dev_loader,
            id2label=id2label,
            device=device,
            model_type=model_type,
        )
        
        # Record metrics
        history["steps"].append(epoch)
        history["eval_f1"].append(dev_metrics["f1"])
        history["eval_precision"].append(dev_metrics["precision"])
        history["eval_recall"].append(dev_metrics["recall"])
        history["train_loss"].append(avg_epoch_loss)
        
        print(
            f"[Epoch {epoch}] loss={avg_epoch_loss:.4f} "
            f"f1={dev_metrics['f1']:.4f} "
            f"prec={dev_metrics['precision']:.4f} "
            f"rec={dev_metrics['recall']:.4f}"
        )

    # final evaluation on test set (if provided) or dev set (fallback for PR curve)
    test_metrics = None
    prob_data = None
    
    if test_loader is not None:
        print("\n[test] Evaluating on test set...")
        test_metrics, test_true, test_pred, prob_data = evaluate_model(
            model=model,
            dataloader=test_loader,
            id2label=id2label,
            device=device,
            model_type=model_type,
            return_probs=True,
        )
        print(
            f"[test] precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )
        
        # Save confusion matrix for test set
        save_confusion_matrix(test_true, test_pred, output_dir, model_name=model_name)
    else:
        # No test set, use dev set for prob_data collection (PR Curve)
        print("\n[Fallback] No test set. Using dev set for PR Curve...")
        test_metrics, test_true, test_pred, prob_data = evaluate_model(
            model=model,
            dataloader=dev_loader,
            id2label=id2label,
            device=device,
            model_type=model_type,
            return_probs=True,
        )
        save_confusion_matrix(test_true, test_pred, output_dir, model_name=model_name)

    # save comprehensive dashboard
    save_comprehensive_dashboard(
        history=history,
        test_metrics=test_metrics,
        prob_data=prob_data,
        output_dir=output_dir,
        model_name=model_name,
        model_type=model_type,
        best_epoch=history["eval_f1"].index(max(history["eval_f1"])) + 1 if history["eval_f1"] else 0
    )

    # save model (weights to model_dir)
    if save_weights:
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[save] Saving model to {model_dir}")
        if model_type == "pure":
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            with (model_dir / "label_map.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "label2id": label2id,
                        "id2label": {int(v): k for k, v in label2id.items()},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            # simple saving: state dict + config + label map
            torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
            # Ensure cfg is available here (it is defined in else block above)
            if cfg is None:
                 raise ValueError("CRF configuration missing")

            with (model_dir / "ner_config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ner_config": asdict(cfg),
                        "label2id": label2id,
                        "id2label": {int(v): k for k, v in label2id.items()},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            tokenizer.save_pretrained(model_dir)
    else:
        print("[save] Skipping model weight save (save_weights=False)")

    result = {
        "model_name": model_name,
        "model_type": model_type,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "prob_data": prob_data, # Added prob_data for K-Fold aggregation
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
    }
    return result


def run_six_model_benchmark(
    train_file: Path,
    dev_file: Path | None,
    test_file: Path | None,
    output_root: Path,
    model_root: Path,
    num_epochs: int = 5,
    batch_size_pure: int = 16,
    batch_size_crf: int = 12,
    max_length: int = 128,
    learning_rate: float = 5e-5,
) -> Dict:
    """Train + evaluate the 6 models described above."""
    output_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    # (model_type, hf_name, short_tag)
    model_specs = [
        ("pure", "google-bert/bert-base-multilingual-cased", "pure_bert"),
        ("pure", "klue/roberta-large", "pure_roberta"),
        ("pure", "FacebookAI/xlm-roberta-large", "pure_xlm"),
        ("crf", "google-bert/bert-base-multilingual-cased", "crf_bert"),
        ("crf", "klue/roberta-large", "crf_roberta"),
        ("crf", "FacebookAI/xlm-roberta-large", "crf_xlm"),
    ]

    all_results: Dict[str, Dict] = {}
    for model_type, hf_name, tag in model_specs:
        print("\n" + "=" * 80)
        # Check for local model
        real_model_path = resolve_model_path(hf_name)
        print(f"[RUN] Training model: {tag} ({real_model_path}, type={model_type})")
        print("=" * 80)

        batch_size = 20 # Fixed batch size as requested to align with 400-sample evaluation intervals
        
        # Separate paths
        out_dir = output_root / tag  # For plots/logs
        model_dir = model_root / tag # For model weights

        res = train_one_model(
            model_name=real_model_path,
            model_type=model_type,  # type: ignore[arg-type]
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            output_dir=out_dir,
            model_dir=model_dir,
            max_length=max_length,
            batch_size=batch_size,
            num_epochs=1, # Fixed to 1 epoch as requested
            learning_rate=learning_rate,
            num_evaluations=20
        )

        all_results[tag] = res
        
        # Clear memory
        del res
        torch.cuda.empty_cache()
        gc.collect()

    # summary
    summary = {
        tag: {
            "model_name": res["model_name"],
            "model_type": res["model_type"],
            "dev_f1": res["dev_metrics"]["f1"],
            "test_f1": (res["test_metrics"] or {}).get("f1")
            if res["test_metrics"] is not None
            else None,
        }
        for tag, res in all_results.items()
    }

    print("\n" + "=" * 80)
    print("Summary (dev F1 / test F1)")
    print("=" * 80)
    for tag, info in summary.items():
        print(
            f"{tag:12s} "
            f"type={info['model_type']:4s} "
            f"dev_f1={info['dev_f1']:.4f} "
            f"test_f1={(info['test_f1'] if info['test_f1'] is not None else float('nan')):.4f}"
        )

    results = {"results": all_results, "summary": summary}
    # save as JSON
    # Ensure all data is JSON serializable (numpy floats, etc.)
    results = make_serializable(results)
    
    json_path = output_root / "six_model_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[save] Full results saved to {json_path}")
    
    # Save comparison plots
    save_comparison_plots(summary, output_root)

    return results


def run_kfold_benchmark(
    output_root: Path,
    model_root: Path,
    num_epochs: int = 1,
    batch_size_pure: int = 20,
    batch_size_crf: int = 20,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    k_folds: int = 5,
    num_samples: int = 10000,
    tag_prefix: str = "",
) -> Dict:
    """Run 5-Fold Cross Validation for all 6 models."""
    output_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    # 1. Generate Full Dataset
    # We combine 'train' and 'dev' templates to create a diverse pool for CV
    print(f"[K-Fold] Generating {num_samples} samples for Cross Validation...")
    
    # Generate 50/50 'train' type and 'dev' type to mix patterns
    half = num_samples // 2
    samples_a = generate_training_samples(half, balanced=True, dataset_type='train')
    samples_b = generate_training_samples(num_samples - half, balanced=True, dataset_type='dev')
    all_samples = samples_a + samples_b
    random.shuffle(all_samples)
    
    print(f"[K-Fold] Total samples: {len(all_samples)}")

    # (model_type, hf_name, short_tag)
    model_specs = [
        ("pure", "google-bert/bert-base-multilingual-cased", "pure_bert"),
        ("pure", "klue/roberta-large", "pure_roberta"),
        ("pure", "FacebookAI/xlm-roberta-large", "pure_xlm"),
        ("crf", "google-bert/bert-base-multilingual-cased", "crf_bert"),
        ("crf", "klue/roberta-large", "crf_roberta"),
        ("crf", "FacebookAI/xlm-roberta-large", "crf_xlm"),
    ]

    # Use GroupKFold to prevent template leakage
    # Groups are the templates used to generate the samples
    groups = [s.get('template', 'UNKNOWN') for s in all_samples]
    
    # Verify groups are sufficient
    unique_groups = set(groups)
    print(f"[K-Fold] Unique templates (groups): {len(unique_groups)}")
    if len(unique_groups) < k_folds:
        print(f"[Warning] Number of unique templates ({len(unique_groups)}) is less than k_folds ({k_folds}).")
        print("Falling back to standard KFold (Leakage possible!)")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        split_generator = kf.split(all_samples)
    else:
        print(f"[K-Fold] Using GroupKFold to ensure NO TEMPLATE LEAKAGE between train and val.")
        gkf = GroupKFold(n_splits=k_folds)
        split_generator = gkf.split(all_samples, groups=groups)
    
    # We need to materialize the split generator because we iterate it multiple times (once per model? No, loop is outside)
    # Wait, the loop is: For each model -> For each fold.
    # So we need to regenerate the splits or save them.
    # Let's save the indices.
    splits = list(split_generator)

    final_results = {}

    for model_type, hf_name, base_tag in model_specs:
        tag = f"{tag_prefix}_{base_tag}" if tag_prefix else base_tag
        print("\n" + "=" * 80)
        real_model_path = resolve_model_path(hf_name)
        print(f"[K-Fold] Benchmarking model: {tag} ({k_folds} folds)")
        print("=" * 80)

        batch_size = batch_size_pure if model_type == "pure" else batch_size_crf
        
        fold_metrics = []
        fold_histories = []
        all_fold_y_true = []
        all_fold_y_scores = []
        
        # Create directory for this model's folds
        model_out_dir = output_root / tag
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1}/{k_folds} ---")
            
            # Split data
            train_fold = [all_samples[i] for i in train_idx]
            val_fold = [all_samples[i] for i in val_idx]
            
            # LEAKAGE CHECK
            train_templates = set(s.get('template', 'A') for s in train_fold)
            val_templates = set(s.get('template', 'B') for s in val_fold)
            intersection = train_templates.intersection(val_templates)
            if intersection:
                print(f"[WARNING] Template Leakage Detected! {len(intersection)} templates shared.")
                # print(f"Shared: {list(intersection)[:3]}...")
            else:
                print(f"[Check] No Template Leakage. Train/Val templates are disjoint.")
            
            # Write to temp files
            fold_train_file = model_out_dir / f"fold_{fold_idx+1}_train.txt"
            fold_val_file = model_out_dir / f"fold_{fold_idx+1}_val.txt"
            
            write_bio_word_level(train_fold, fold_train_file, apply_noise=True)
            write_bio_word_level(val_fold, fold_val_file, apply_noise=False)
            
            # Train
            # Only save weights for the first fold to save space/time, or last? 
            # User didn't specify, but saving 30 models is heavy. Let's save none or just best?
            # Let's save weights for Fold 1 only as a representative artifact.
            save_weights = (fold_idx == 0)
            
            res = train_one_model(
                model_name=real_model_path,
                model_type=model_type, # type: ignore
                train_file=fold_train_file,
                dev_file=fold_val_file,
                test_file=None, # No separate test set in CV usually, or use val as test
                output_dir=model_out_dir / f"fold_{fold_idx+1}",
                model_dir=model_root / tag / f"fold_{fold_idx+1}",
                max_length=max_length,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                num_evaluations=20,
                save_weights=save_weights
            )
            
            # Collect results
            # We use the best dev f1 from history or the final one?
            # Usually CV reports the score on the validation set.
            # train_one_model returns 'dev_metrics' (last step) and 'test_metrics' (if test_file provided).
            # Since we didn't provide test_file, we use the best F1 from history or the last one.
            # Let's use the BEST F1 achieved during the fold.
            best_f1 = max(res["history"]["eval_f1"])
            fold_metrics.append(best_f1)
            
            # Calculate per-fold AP and other stats
            fold_ap = 0.0
            if res.get("prob_data"):
                try:
                    fold_ap = average_precision_score(res["prob_data"]["y_true"], res["prob_data"]["y_scores"])
                except:
                    fold_ap = 0.0
            
            # Inject summary stats into history
            res["history"]["summary"] = {
                "best_f1": best_f1,
                "final_f1": res["history"]["eval_f1"][-1] if res["history"]["eval_f1"] else 0.0,
                "final_precision": res["history"]["eval_precision"][-1] if res["history"]["eval_precision"] else 0.0,
                "final_recall": res["history"]["eval_recall"][-1] if res["history"]["eval_recall"] else 0.0,
                "ap_score": fold_ap
            }
            
            fold_histories.append(res["history"])
            
            # Collect probability data for aggregated PR curve
            if res.get("prob_data"):
                all_fold_y_true.extend(res["prob_data"]["y_true"])
                all_fold_y_scores.extend(res["prob_data"]["y_scores"])
            
            # Cleanup
            del res
            torch.cuda.empty_cache()
            gc.collect()
            
            # Remove temp files
            if fold_train_file.exists(): fold_train_file.unlink()
            if fold_val_file.exists(): fold_val_file.unlink()

        # --- Aggregation ---
        mean_f1 = np.mean(fold_metrics)
        std_f1 = np.std(fold_metrics)
        
        print(f"\n[Result] {tag} {k_folds}-Fold CV: Mean F1 = {mean_f1:.4f} (+/- {std_f1:.4f})")
        
        # Average History for Graph
        # Assuming all folds have same number of steps (they should, as data size is constant)
        # history['eval_f1'] is a list of floats.
        avg_history = copy.deepcopy(fold_histories[0])
        
        # Average the lists
        for key in ["train_loss", "eval_f1", "eval_precision", "eval_recall"]:
            # Stack lists: (K, num_steps)
            values = [h[key] for h in fold_histories]
            # Compute mean along axis 0
            # Note: lengths must match. If not (due to slight batch rounding?), truncate to min length.
            min_len = min(len(v) for v in values)
            truncated_values = [v[:min_len] for v in values]
            avg_values = np.mean(truncated_values, axis=0).tolist()
            avg_history[key] = avg_values

        # Calculate Convergence (F1 Derivative) for JSON
        if "eval_f1" in avg_history and len(avg_history["eval_f1"]) > 1:
            avg_history["f1_convergence"] = np.gradient(avg_history["eval_f1"]).tolist()

        # Calculate Convergence for each individual fold as well
        for fh in fold_histories:
            if "eval_f1" in fh and len(fh["eval_f1"]) > 1:
                fh["f1_convergence"] = np.gradient(fh["eval_f1"]).tolist()
            
        # Generate Averaged Dashboard
        print(f"[Graph] Generating Averaged Dashboard for {tag}...")
        
        # Construct aggregated prob_data
        agg_prob_data = None
        ap_score = 0.0
        if all_fold_y_true and all_fold_y_scores:
            agg_prob_data = {
                "y_true": all_fold_y_true,
                "y_scores": all_fold_y_scores
            }
            try:
                ap_score = average_precision_score(all_fold_y_true, all_fold_y_scores)
            except:
                ap_score = 0.0
            
        save_comprehensive_dashboard(
            history=avg_history,
            test_metrics={"f1": mean_f1, "precision": 0.0, "recall": 0.0}, # Dummy prec/rec for text box
            prob_data=agg_prob_data, 
            output_dir=model_out_dir, # Save in model folder root
            model_name=f"{tag} (Average of {k_folds} Folds)",
            model_type=model_type,
            best_epoch=1
        )
        
        # Extract scalar metrics for easier reading
        final_f1 = avg_history["eval_f1"][-1] if avg_history.get("eval_f1") else 0.0
        best_f1 = max(avg_history["eval_f1"]) if avg_history.get("eval_f1") else 0.0
        final_prec = avg_history["eval_precision"][-1] if avg_history.get("eval_precision") else 0.0
        final_rec = avg_history["eval_recall"][-1] if avg_history.get("eval_recall") else 0.0

        final_results[tag] = {
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "fold_scores": fold_metrics,
            "ap_score": ap_score, # Added AP Score
            "best_f1": best_f1,   # Added Best F1
            "final_f1": final_f1, # Added Final F1
            "final_precision": final_prec, # Added Final Precision
            "final_recall": final_rec,     # Added Final Recall
            "avg_history": avg_history,  # Save detailed epoch-by-epoch stats (Average)
            "fold_histories": fold_histories # Save detailed stats for EACH fold
        }

    # Save Summary
    summary_filename = f"{tag_prefix}_kfold_summary.json" if tag_prefix else "kfold_summary.json"
    summary_path = output_root / summary_filename
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(final_results), f, indent=2)
        
    print(f"\n[Done] K-Fold Benchmark Complete. Summary saved to {summary_path}")
    return final_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="6-model NER benchmark")
    p.add_argument("--train_file", type=str, default=None, help="Train BIO file. If not provided, generated dynamically.")
    p.add_argument("--dev_file", type=str, default=None, help="Dev BIO file (Optional. If not provided, splits train_file 8:2)")
    p.add_argument("--test_file", type=str, default=None, help="Test BIO file (optional)")
    p.add_argument(
        "--output_root",
        type=str,
        default="data/out/ner_benchmark",
        help="Directory to store results (logs, plots)",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="models/ner_benchmark",
        help="Directory to store trained models (weights)",
    )
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size_pure", type=int, default=160)
    p.add_argument("--batch_size_crf", type=int, default=160)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps (dynamic if not set)")
    p.add_argument("--single_run", action="store_true", help="Run single benchmark instead of 5-Fold CV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    model_dir = Path(args.model_dir)
    
    if not args.single_run:
        print(">>> Starting Data Scaling Benchmark (25%, 50%, 75%, 100%) <<<")
        
        ratios = [0.25, 0.50, 0.75, 1.00]
        base_samples = 10000
        
        for ratio in ratios:
            num_samples = int(base_samples * ratio)
            prefix = f"{ratio:.2f}" # e.g. "0.25"
            
            print(f"\n\n>>> Running Benchmark with {num_samples} samples ({int(ratio*100)}%) <<<")
            
            run_kfold_benchmark(
                output_root=output_root,
                model_root=model_dir,
                num_epochs=args.num_epochs,
                batch_size_pure=args.batch_size_pure,
                batch_size_crf=args.batch_size_crf,
                max_length=args.max_length,
                learning_rate=args.learning_rate,
                k_folds=5,
                num_samples=num_samples,
                tag_prefix=prefix
            )
        return

    if args.train_file:
        train_file = Path(args.train_file)
        dev_file = Path(args.dev_file) if args.dev_file else None
    else:
        # Dynamic Generation (Separate Train/Dev)
        # Generate data in output_root (results dir) so it can be downloaded/checked
        print("[setup] Generating independent Train/Dev datasets...")
        train_file, dev_file = generate_dynamic_dataset(output_root, num_samples=10000)
        
    test_file = Path(args.test_file) if args.test_file is not None else None

    run_six_model_benchmark(
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file,
        output_root=output_root,
        model_root=model_dir,
        num_epochs=args.num_epochs,
        batch_size_pure=args.batch_size_pure,
        batch_size_crf=args.batch_size_crf,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )

    # Uncomment to run K-Fold Benchmark
    # run_kfold_benchmark(
    #     output_root=output_root,
    #     model_root=model_dir,
    #     num_epochs=args.num_epochs,
    #     batch_size_pure=args.batch_size_pure,
    #     batch_size_crf=args.batch_size_crf,
    #     max_length=args.max_length,
    #     learning_rate=args.learning_rate,
    #     k_folds=5,
    # )


if __name__ == "__main__":
    main()
