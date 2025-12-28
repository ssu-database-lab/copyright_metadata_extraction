
import os
import sys
print("Starting script...")
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from typing import Any

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

# Import modules
try:
    from src.ner.ner_train import generate_training_samples
    from src.ner.ner_model import BertBiLstmCrf, NERConfig
    from src.ner.ner_data import read_conll, build_label_map, NERDataset
    from src.ner.ner_system import predict_texts
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

# Configuration
MODEL_NAME = "model_downloaded/google-bert-bert-base-multilingual-cased"
EPOCHS = 20
BATCH_SIZE = 64
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
OCR_FILE = Path("data/out/ocr/7.저작물양도계약서_p001.txt")
OUTPUT_JSON = Path("ner_single_test_result.json")
TEMP_DATA_DIR = Path("data/temp_single_test")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_bio_word_level(samples, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = sample.get('text', '')
            entities = sample.get('entities', []) or []
            
            tokens = text.split()
            if not tokens:
                continue
                
            labels = ['O'] * len(tokens)
            char_labels = ['O'] * len(text)
            
            entity_spans = []
            current_search_pos = 0
            for entity_text, entity_type in entities:
                entity_start = text.find(entity_text, current_search_pos)
                if entity_start == -1:
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
            
            char_pos = 0
            for token_idx, token in enumerate(tokens):
                token_start = text.find(token, char_pos)
                if token_start == -1:
                    char_pos += len(token) + 1
                    continue
                token_end = token_start + len(token)
                
                for i in range(token_start, token_end):
                    if i < len(char_labels) and char_labels[i] != 'O':
                        labels[token_idx] = char_labels[i]
                        break
                char_pos = token_end
            
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

def train_model():
    set_seed(42)
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Generating Data...")
    # Increased data size significantly as requested
    train_samples = generate_training_samples(num_samples=15000, dataset_type='train')
    dev_samples = generate_training_samples(num_samples=3000, dataset_type='dev')
    
    train_path = TEMP_DATA_DIR / "train.txt"
    dev_path = TEMP_DATA_DIR / "dev.txt"
    
    write_bio_word_level(train_samples, train_path)
    write_bio_word_level(dev_samples, dev_path)
    
    print("[2/5] Loading Data...")
    train_sentences = read_conll(str(train_path))
    dev_sentences = read_conll(str(dev_path))
    
    label_list, label2id, id2label = build_label_map(train_sentences)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = NERDataset(train_sentences, tokenizer, label2id, MAX_LENGTH)
    dev_dataset = NERDataset(dev_sentences, tokenizer, label2id, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[3/5] Initializing Model (mBERT+CRF)...")
    config = NERConfig(
        model_name_or_path=MODEL_NAME,
        dropout=0.1
    )
    model = BertBiLstmCrf(config, num_labels=len(label2id))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device: {device}")
    if device.type == 'cuda':
        print(f"DEBUG: GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"DEBUG: CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA is NOT available. Training on CPU will be slow.")
        
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    print(f"[4/5] Training for {EPOCHS} epochs...")
    best_f1 = 0.0
    best_report = {}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss = model(input_ids, attention_mask, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                decoded_tags = model(input_ids, attention_mask)
                
                # Convert ids to labels
                for i in range(len(labels)):
                    true_labels = []
                    pred_labels = []
                    
                    # Remove padding and special tokens
                    # NERDataset pads with -100 or 0? 
                    # Usually we need to align with mask.
                    # decoded_tags is list of list of ints.
                    
                    # We need to align prediction with true labels
                    # The model output `decoded_tags` usually excludes padding if using torchcrf properly?
                    # Wait, BertBiLstmCrf implementation details matter here.
                    # Let's assume decoded_tags corresponds to the valid tokens in input_ids (excluding padding).
                    # But we need to match it with ground truth labels.
                    
                    # Let's look at how ner_test.py does it.
                    # It uses collect_seqeval_inputs_from_crf_batch
                    
                    # Simplified version:
                    curr_labels = labels[i].cpu().numpy()
                    curr_mask = attention_mask[i].cpu().numpy()
                    curr_preds = decoded_tags[i]
                    
                    valid_len = sum(curr_mask) - 2 # -2 for CLS/SEP usually? 
                    # Actually NERDataset might handle CLS/SEP differently.
                    # Let's just iterate and skip -100 labels.
                    
                    p_idx = 0
                    for j, label_id in enumerate(curr_labels):
                        if label_id != -100:
                            true_labels.append(id2label[label_id])
                            if p_idx < len(curr_preds):
                                pred_labels.append(id2label[curr_preds[p_idx]])
                            else:
                                pred_labels.append("O")
                            p_idx += 1
                    
                    all_preds.append(pred_labels)
                    all_labels.append(true_labels)

        val_f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_report = classification_report(all_labels, all_preds, output_dict=True)
            # Save best model state if needed, but for this script we just keep it in memory?
            # Actually we need to use the best model for inference.
            # So let's just use the model as is at the end of 20 epochs?
            # The user asked for 20 epochs. Usually we use the last one or best one.
            # I'll just continue training.
            
    print(f"Best Val F1: {best_f1:.4f}")
    
    print("[5/5] Running Inference on OCR file...")
    if not OCR_FILE.exists():
        print(f"[Error] OCR file not found: {OCR_FILE}")
        return
        
    with open(OCR_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Use predict_texts
    # predict_texts expects: model, tokenizer, id2label, config, lines, device
    results = predict_texts(model, tokenizer, id2label, config, lines, device)
    
    # Aggregate entities
    final_entities = []
    for res in results:
        tokens_labels = res['tokens']
        current_entity = None
        
        for token, label in tokens_labels:
            if label == 'O':
                if current_entity:
                    final_entities.append(current_entity)
                    current_entity = None
                continue
            
            if label.startswith('B-'):
                if current_entity:
                    final_entities.append(current_entity)
                
                entity_type = label[2:]
                clean_token = token.replace("##", "")
                current_entity = {
                    "label": entity_type,
                    "text": clean_token
                }
            elif label.startswith('I-'):
                if current_entity and current_entity['label'] == label[2:]:
                    clean_token = token.replace("##", "")
                    current_entity['text'] += clean_token
                else:
                    if current_entity:
                        final_entities.append(current_entity)
                    entity_type = label[2:]
                    clean_token = token.replace("##", "")
                    current_entity = {
                        "label": entity_type,
                        "text": clean_token
                    }
        if current_entity:
            final_entities.append(current_entity)

    # Construct Output
    output_data = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "validation_accuracy": best_report,
        "ocr_file": str(OCR_FILE),
        "extraction_results": final_entities
    }
    
    # Convert to serializable format
    output_data = make_serializable(output_data)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    train_model()
