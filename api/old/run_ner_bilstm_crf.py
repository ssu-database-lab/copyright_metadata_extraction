import os
import sys
import json
import random
import math
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter

# Transformers & Torch
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
import seqeval.metrics

# Import data generation from ner_train
sys.path.append(str(Path(__file__).parent))
try:
    from src.ner.ner_train import generate_training_samples
    HAS_DATA_GEN = True
except ImportError as e:
    print(f"[Warning] Could not import generate_training_samples: {e}")
    HAS_DATA_GEN = False

# ==========================================
# Configuration
# ==========================================
NUM_SAMPLES = 30000
MODEL_NAME = "google-bert/bert-base-multilingual-cased"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MAX_LEN = 128
OUTPUT_DIR = Path("api/new/data")
MODEL_SAVE_DIR = Path("models/ner_bilstm_crf")
OCR_DIR = Path("data/out/ocr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure Output Directory Exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Check GPU
if not torch.cuda.is_available():
    print("[WARNING] CUDA is not available! Training will be very slow on CPU.")
    print("[WARNING] Please ensure CUDA 12.8 is properly installed.")
else:
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")

# Entity Types
ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
    "CONTRACT_TYPE", "CONSENT_TYPE", "RIGHT_INFO", "MONEY", "PERIOD",
    "PROJECT_NAME", "LAW_REFERENCE", "ID_NUM", "TITLE", "URL",
    "DESCRIPTION", "TYPE", "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY"
]

# BIO Labels
def create_bio_labels():
    labels = ["O"]
    for entity in ENTITY_TYPES:
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")
    return labels

BIO_LABELS = create_bio_labels()
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# ==========================================
# Model Definition (BERT + BiLSTM + CRF)
# ==========================================

@dataclass
class NERConfig:
    model_name_or_path: str = MODEL_NAME
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 1
    dropout: float = 0.1
    max_length: int = MAX_LEN

class BertBiLstmCrf(nn.Module):
    def __init__(self, config: NERConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        # BERT backbone
        self.bert = AutoModel.from_pretrained(config.model_name_or_path)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=config.lstm_hidden_size // 2,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Linear layer to num_labels
        self.classifier = nn.Linear(config.lstm_hidden_size, num_labels)

        # CRF layer - Use same approach as src/ner/ner_model.py
        try:
            self.crf = CRF(num_labels, batch_first=True)
            self.batch_first = True
        except TypeError:
            self.crf = CRF(num_labels)
            self.batch_first = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        emissions = self.classifier(lstm_output)
        mask = attention_mask.bool()

        # CLS/SEP 등의 위치는 labels가 -100으로 표시되어 있어 CRF 계산에서 제외
        if labels is not None:
            mask = mask & (labels != -100)

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if labels is not None:
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0
            
            if not self.batch_first:
                labels_for_crf = labels_for_crf.transpose(0, 1)
            
            try:
                loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction="mean")
            except TypeError:
                log_likelihood = self.crf(emissions, labels_for_crf, mask=mask)
                if log_likelihood.dim() > 0:
                    loss = -torch.mean(log_likelihood)
                else:
                    loss = -log_likelihood / input_ids.size(0)
            return loss

        # Decode: Use same approach as src/ner/ner_model.py
        if hasattr(self.crf, 'decode'):
            predictions = self.crf.decode(emissions, mask=mask)
        elif hasattr(self.crf, 'viterbi_decode'):
            # torchcrf package exposes viterbi_decode instead of decode
            predictions = self.crf.viterbi_decode(emissions, mask=mask)
        elif hasattr(self.crf, 'viterbi_tags'):
            decoded = self.crf.viterbi_tags(emissions, mask=mask)
            predictions = [tags for tags, score in decoded]
        else:
            raise AttributeError("CRF object has no decode/viterbi_decode method")
                
        return predictions

# ==========================================
# Data Generation (Using ner_train.py)
# ==========================================

def convert_samples_to_bio_format(samples: List[Dict]) -> List[Dict]:
    """Convert samples from ner_train format to BIO format for training."""
    bio_samples = []
    seen_texts = set()
    
    for sample in samples:
        text = sample.get('text', '')
        entities = sample.get('entities', [])
        
        if text in seen_texts:
            continue
        seen_texts.add(text)
        
        tokens = text.split()
        if not tokens:
            continue
            
        labels = ['O'] * len(tokens)
        char_labels = ['O'] * len(text)
        
        entity_spans = []
        current_search_pos = 0
        
        for entity_text, entity_type in entities:
            entity_start = text.find(entity_text, current_search_pos)
            if entity_start != -1:
                entity_end = entity_start + len(entity_text)
                entity_spans.append((entity_start, entity_end, entity_type))
                if entity_start >= current_search_pos:
                    current_search_pos = entity_end
        
        for start, end, etype in entity_spans:
            if start < len(char_labels):
                char_labels[start] = f"B-{etype}"
                for i in range(start + 1, min(end, len(char_labels))):
                    char_labels[i] = f"I-{etype}"
        
        char_pos = 0
        for token_idx, token in enumerate(tokens):
            token_start = text.find(token, char_pos)
            if token_start == -1:
                char_pos += len(token) + 1
                continue
                
            token_end = token_start + len(token)
            
            for i in range(token_start, min(token_end, len(char_labels))):
                if char_labels[i] != 'O':
                    labels[token_idx] = char_labels[i]
                    break
            
            char_pos = token_end
        
        ner_tags = [LABEL_TO_ID.get(label, LABEL_TO_ID["O"]) for label in labels]
        
        bio_samples.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })
    
    return bio_samples

def generate_training_data(num_samples: int):
    """Generate training data using ner_train.py's generate_training_samples."""
    if not HAS_DATA_GEN:
        print("[ERROR] Cannot import generate_training_samples. Using fallback.")
        return []
    
    print(f"[DataGen] Generating {num_samples} diverse samples with noise...")
    
    samples = generate_training_samples(
        num_samples=num_samples,
        balanced=True,
        noise_level=0.1,
        dataset_type='train'
    )
    
    print(f"[DataGen] Generated {len(samples)} unique samples.")
    
    bio_samples = convert_samples_to_bio_format(samples)
    
    print(f"[DataGen] Converted to {len(bio_samples)} BIO format samples.")
    
    entity_counts = Counter()
    for sample in bio_samples:
        tags = [ID_TO_LABEL[tag] for tag in sample['ner_tags']]
        for tag in tags:
            if tag != 'O':
                entity_type = tag.split('-')[1] if '-' in tag else tag
                entity_counts[entity_type] += 1
    
    print(f"[DataGen] Entity type distribution:")
    for etype, count in entity_counts.most_common(10):
        print(f"  {etype}: {count}")
    
    return bio_samples

# ==========================================
# Dataset & Training
# ==========================================

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]

        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        word_ids = tokenized_inputs.word_ids(batch_index=0)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(ner_tags[word_idx])
            else:
                label_ids.append(ner_tags[word_idx])
            previous_word_idx = word_idx

        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

def compute_label_wise_accuracy(all_true: List[List[str]], all_pred: List[List[str]]) -> Dict[str, Dict[str, float]]:
    """Compute accuracy per label: (predicted entities) / (total entities) for each label."""
    from seqeval.metrics import classification_report
    
    label_accuracy = {}
    
    try:
        report = classification_report(all_true, all_pred, output_dict=True, zero_division="0")
        
        for label in ENTITY_TYPES:
            b_label = f"B-{label}"
            
            if isinstance(report, dict) and b_label in report:
                label_data = report[b_label]
                if isinstance(label_data, dict):
                    support = int(label_data.get("support", 0))
                    precision = float(label_data.get("precision", 0.0))
                    recall = float(label_data.get("recall", 0.0))
                    f1_score = float(label_data.get("f1-score", 0.0))
                    
                    predicted_entities = int(support / recall) if recall > 0 else support
                    correct_entities = int(support * recall)
                    
                    label_accuracy[label] = {
                        "total_entities": support,
                        "predicted_entities": predicted_entities,
                        "correct_entities": correct_entities,
                        "accuracy": recall,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score
                    }
                else:
                    label_accuracy[label] = {
                        "total_entities": 0, "predicted_entities": 0, "correct_entities": 0,
                        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0
                    }
            else:
                label_accuracy[label] = {
                    "total_entities": 0, "predicted_entities": 0, "correct_entities": 0,
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0
                }
    except Exception as e:
        print(f"[WARNING] Error in compute_label_wise_accuracy: {e}")
        for label in ENTITY_TYPES:
            label_accuracy[label] = {
                "total_entities": 0, "predicted_entities": 0, "correct_entities": 0,
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0
            }
    
    return label_accuracy

def train_model():
    print(f"Using Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"\n{'='*60}")
    print(f"Generating {NUM_SAMPLES} training samples...")
    print(f"{'='*60}")
    raw_data = generate_training_data(NUM_SAMPLES)
    
    if len(raw_data) == 0:
        print("[ERROR] Failed to generate training data!")
        return None, None, None
    
    print("\n=== Data Generation Check ===")
    for i in range(min(3, len(raw_data))):
        sample = raw_data[i]
        print(f"\nSample {i+1}:")
        print(f"Tokens: {sample['tokens'][:10]}...")
        tags_readable = [ID_TO_LABEL[tag] for tag in sample['ner_tags'][:10]]
        print(f"Tags:   {tags_readable}...")
        entity_count = sum(1 for tag in tags_readable if tag != 'O')
        print(f"Entity tokens in first 10: {entity_count}")
    print("=" * 50)
    
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_data = raw_data[:split_idx]
    val_data = raw_data[split_idx:]
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NERDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = NERDataset(val_data, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    config = NERConfig()
    model = BertBiLstmCrf(config, num_labels=len(BIO_LABELS))
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    training_history = {
        "epoch": [],
        "train_loss": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting Training for {EPOCHS} epochs...")
    print(f"{'='*60}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loss_values = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] Invalid loss detected: {loss.item()}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            loss_values.append(loss_val)
            total_loss += loss_val
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        min_loss = min(loss_values) if loss_values else 0
        max_loss = max(loss_values) if loss_values else 0
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} (min: {min_loss:.4f}, max: {max_loss:.4f})")
        
        val_metrics = evaluate_model(model, val_loader, compute_label_wise=False)
        
        training_history["epoch"].append(epoch + 1)
        training_history["train_loss"].append(float(avg_loss))
        training_history["val_f1"].append(float(val_metrics.get("f1", 0.0)))
        training_history["val_precision"].append(float(val_metrics.get("precision", 0.0)))
        training_history["val_recall"].append(float(val_metrics.get("recall", 0.0)))
        
        print(f"Validation - F1: {val_metrics.get('f1', 0.0):.4f}, "
              f"Precision: {val_metrics.get('precision', 0.0):.4f}, "
              f"Recall: {val_metrics.get('recall', 0.0):.4f}")

    save_model_complete(model, tokenizer, OUTPUT_DIR / "saved_model")
    
    return model, tokenizer, training_history

def save_model_complete(model, tokenizer, save_dir):
    """Save the complete model with all necessary components."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_dir / "model.pth")
    tokenizer.save_pretrained(save_dir / "tokenizer")
    
    config_dict = {
        "model_config": asdict(model.config),
        "num_labels": model.num_labels,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "entity_types": ENTITY_TYPES,
        "model_name": MODEL_NAME,
        "max_length": MAX_LEN
    }
    
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
    
    print(f"\nModel saved to {save_dir}")

def load_saved_model(save_dir):
    """Load model/tokenizer if previously saved."""
    save_dir = Path(save_dir)
    model_path = save_dir / "model.pth"
    config_path = save_dir / "config.json"
    tok_path = save_dir / "tokenizer"
    
    if not (model_path.exists() and config_path.exists() and tok_path.exists()):
        return None, None
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    model_cfg_dict = cfg.get("model_config", {})
    num_labels = cfg.get("num_labels", len(BIO_LABELS))
    model_cfg = NERConfig(**model_cfg_dict) if model_cfg_dict else NERConfig()
    
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = BertBiLstmCrf(model_cfg, num_labels=num_labels)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded saved model from {save_dir}")
    return model, tokenizer

def evaluate_model(model, dataloader, compute_label_wise=False):
    """Evaluate NER model using seqeval metrics and optionally label-wise accuracy."""
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            pred_paths = model(input_ids, attention_mask)
            
            for i in range(min(len(labels), len(pred_paths))):
                t_seq = []
                p_seq = []

                seq_len = min(len(labels[i]), len(pred_paths[i]))
                for j in range(seq_len):
                    if attention_mask[i][j] == 0:
                        continue
                    if labels[i][j] == -100:
                        continue

                    t_seq.append(ID_TO_LABEL[labels[i][j].item()])
                    p_seq.append(ID_TO_LABEL[pred_paths[i][j]])
                
                if len(t_seq) > 0:
                    all_true.append(t_seq)
                    all_pred.append(p_seq)
    
    if len(all_true) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    try:
        report_dict = seqeval.metrics.classification_report(all_true, all_pred, output_dict=True, zero_division="0")
        
        if isinstance(report_dict, dict) and 'macro avg' in report_dict:
            macro_avg = report_dict['macro avg']
            if isinstance(macro_avg, dict):
                f1 = float(macro_avg.get('f1-score', 0.0))
                precision = float(macro_avg.get('precision', 0.0))
                recall = float(macro_avg.get('recall', 0.0))
            else:
                f1 = precision = recall = 0.0
        else:
            f1 = precision = recall = 0.0
    except Exception as e:
        print(f"[WARNING] Error computing seqeval metrics: {e}")
        f1 = precision = recall = 0.0
        report_dict = {}
    
    results = {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "full_report": report_dict
    }
    
    if compute_label_wise:
        try:
            label_accuracy = compute_label_wise_accuracy(all_true, all_pred)
            results["label_wise_accuracy"] = label_accuracy
        except Exception as e:
            print(f"[WARNING] Error computing label-wise accuracy: {e}")
            results["label_wise_accuracy"] = {}
    
    return results

# ==========================================
# Inference & Output Generation
# ==========================================

def select_test_files():
    """Select appropriate test files from OCR directory."""
    all_files = list(OCR_DIR.glob("*.txt"))
    
    jincheon_files = [f for f in all_files if "진천동의서" in f.name]
    
    for subdir in OCR_DIR.iterdir():
        if subdir.is_dir():
            jincheon_files.extend(subdir.glob("**/*진천동의서*.txt"))
    
    selected = []
    for f in jincheon_files:
        try:
            size = f.stat().st_size
            if 1000 < size < 100000:
                selected.append(f)
        except:
            pass
    
    if len(selected) < 3:
        contract_files = [f for f in all_files if "저작물양도계약서" in f.name]
        selected.extend(contract_files[:3])
    
    selected = list(set(selected))[:5]
    
    print(f"\nSelected {len(selected)} test files:")
    for f in selected:
        print(f"  - {f.name}")
    
    return selected

def run_inference_on_files(model, tokenizer):
    """Run inference on selected OCR files."""
    print("\n" + "="*60)
    print("Running Inference on OCR files...")
    print("="*60)
    
    model.eval()
    
    test_files = select_test_files()
    if not test_files:
        print("[WARNING] No test files found!")
        return {}
    
    extracted_entities_data = {}
    
    for file_path in test_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            continue
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        file_entities = []
        
        for line_idx, line in enumerate(lines):
            if not line:
                continue
            
            tokens = line.split()
            if not tokens:
                continue
            
            try:
                inputs = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LEN,
                    padding="max_length"
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    predictions = model(inputs["input_ids"], inputs["attention_mask"])
                
                word_ids = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", 
                                    truncation=True, max_length=MAX_LEN).word_ids(batch_index=0)
                
                pred_tags = []
                prev_word_id = None
                pred_idx = 0
                
                for word_id in word_ids:
                    if word_id is None:
                        continue
                    if word_id != prev_word_id:
                        if pred_idx < len(predictions[0]):
                            pred_tags.append(ID_TO_LABEL.get(predictions[0][pred_idx], "O"))
                        else:
                            pred_tags.append("O")
                    prev_word_id = word_id
                    pred_idx += 1
                
                while len(pred_tags) < len(tokens):
                    pred_tags.append("O")
                pred_tags = pred_tags[:len(tokens)]
                
                current_entity = []
                current_type = None
                
                for token, tag in zip(tokens[:len(pred_tags)], pred_tags):
                    if tag.startswith("B-"):
                        if current_entity:
                            file_entities.append({
                                "text": " ".join(current_entity),
                                "type": current_type
                            })
                        current_type = tag[2:]
                        current_entity = [token]
                    elif tag.startswith("I-") and current_type == tag[2:]:
                        current_entity.append(token)
                    else:
                        if current_entity:
                            file_entities.append({
                                "text": " ".join(current_entity),
                                "type": current_type
                            })
                        current_entity = []
                        current_type = None
                
                if current_entity:
                    file_entities.append({
                        "text": " ".join(current_entity),
                        "type": current_type
                    })
                    
            except Exception as e:
                print(f"[WARNING] Error processing line {line_idx}: {e}")
                continue
        
        extracted_entities_data[file_path.name] = file_entities
        print(f"  Extracted {len(file_entities)} entities")
    
    output_file = OUTPUT_DIR / "extracted_entities.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_entities_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nExtracted entities saved to {output_file}")
    return extracted_entities_data

def save_training_results(metrics, extracted_data, training_history):
    """Save training results with proper JSON serialization."""
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    output = {
        "training_config": {
            "num_samples": NUM_SAMPLES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model_name": MODEL_NAME,
            "max_length": MAX_LEN
        },
        "training_history": convert_to_json_serializable(training_history),
        "final_metrics": {
            "f1_score": metrics.get("f1", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "label_wise_accuracy": convert_to_json_serializable(metrics.get("label_wise_accuracy", {}))
        },
        "entity_extraction_summary": {
            "files_processed": list(extracted_data.keys()),
            "total_entities_extracted": sum(len(v) for v in extracted_data.values()),
            "entities_per_file": {k: len(v) for k, v in extracted_data.items()}
        }
    }
    
    output_file = OUTPUT_DIR / "training_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"\nTraining results saved to {output_file}")
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Final F1 Score: {metrics.get('f1', 0.0):.4f}")
    print(f"Final Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"Final Recall: {metrics.get('recall', 0.0):.4f}")
    print(f"\nLabel-wise Accuracy:")
    label_acc = metrics.get("label_wise_accuracy", {})
    for label, stats in sorted(label_acc.items()):
        if stats.get("total_entities", 0) > 0:
            print(f"  {label:20s}: Accuracy={stats.get('accuracy', 0.0):.4f}, "
                  f"F1={stats.get('f1_score', 0.0):.4f}, "
                  f"Total={stats.get('total_entities', 0)}")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("NER Training with BERT + BiLSTM + CRF")
    print("="*60)
    
    saved_model_dir = MODEL_SAVE_DIR
    model, tokenizer = load_saved_model(saved_model_dir)
    training_history = {}
    
    if model is None or tokenizer is None:
        print("[INFO] Saved model not found. Starting training...")
        result = train_model()
        if result is None or len(result) < 3:
            print("[ERROR] Training failed!")
            return
        model, tokenizer, training_history = result
        
        print("\n" + "="*60)
        print("Final Evaluation...")
        print("="*60)
        
        print("\nComputing final metrics with label-wise accuracy...")
        
        val_samples = generate_training_data(1000)
        if val_samples:
            from torch.utils.data import DataLoader
            val_dataset = NERDataset(val_samples, tokenizer, MAX_LEN)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            metrics = evaluate_model(model, val_loader, compute_label_wise=True)
        else:
            print("[WARNING] Could not generate validation data for final evaluation")
            metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "label_wise_accuracy": {}}
        
        extracted_data = run_inference_on_files(model, tokenizer)
        
        save_training_results(metrics, extracted_data, training_history)
    else:
        print("[INFO] Saved model detected. Skipping training and running inference.")
        extracted_data = run_inference_on_files(model, tokenizer)
        print("[INFO] Inference completed using saved model.")
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)

if __name__ == "__main__":
    main()

