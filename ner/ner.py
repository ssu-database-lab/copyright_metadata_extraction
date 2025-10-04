import os
import sys
import subprocess
import importlib.util

# 필요한 패키지 확인 및 설치
required_packages = [
    'pandas', 'torch', 'transformers', 'datasets', 'tqdm', 'numpy', 'scikit-learn', 'glob', 're'
]

for package in required_packages:
    if package not in ['os', 'sys', 'subprocess', 'importlib', 're', 'glob']:  # 표준 라이브러리는 건너뜀
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"{package} 패키지를 설치합니다...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

import glob
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import re
from datasets import Dataset
import logging
import warnings
from tqdm import tqdm
import argparse
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Constants
OCR_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OCR", "ocr_document")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MODEL_NAME = "klue/roberta-large"  # Using Korean language model
MAX_LENGTH = 512

# Define entity labels for contract documents
CONTRACT_LABELS = [
    "저작물명", "대상저작물상세정보", "양수자기관명", "양수자주소", 
    "양도자기관명", "양도자소속", "양도자대표주소", "양도자연락처", 
    "계약체결일"
]

# Define entity labels for consent documents
CONSENT_LABELS = [
    "양도인성명", "양도인전화번호", "양도인주소", "양수인기관명", 
    "양수인대표자명", "양수인대표자주소", "양수인대표자연락처", 
    "동의여부", "동의날짜"
]

# Output filenames
CONSENT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_consent_results.csv")
CONTRACT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_contract_results.csv")
CONSENT_PRED_FILE = os.path.join(OUTPUT_DIR, "consent_prediction_results.csv")
CONTRACT_PRED_FILE = os.path.join(OUTPUT_DIR, "contract_prediction_results.csv")

# Model directories
CONSENT_MODEL_DIR = os.path.join(OUTPUT_DIR, "consent_model")
CONTRACT_MODEL_DIR = os.path.join(OUTPUT_DIR, "contract_model")
UNIFIED_MODEL_DIR = os.path.join(OUTPUT_DIR, "unified_model")

# Function to check if directories and files exist
def check_paths():
    # Check if OCR document directory exists
    if not os.path.exists(OCR_DOCUMENT_PATH):
        logger.warning(f"OCR document directory not found: {OCR_DOCUMENT_PATH}")
        logger.warning("Creating empty OCR document directory...")
        os.makedirs(OCR_DOCUMENT_PATH, exist_ok=True)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Created output directory: {OUTPUT_DIR}")

# Function to load training data from CSV files
def load_training_data(consent_file=CONSENT_TRAIN_FILE, contract_file=CONTRACT_TRAIN_FILE):
    consent_data = None
    contract_data = None
    combined_data = None
    
    if os.path.exists(consent_file):
        consent_data = pd.read_csv(consent_file)
        logger.info(f"Loaded consent training data: {len(consent_data)} records")
    else:
        logger.warning(f"Consent training data file not found: {consent_file}")
    
    if os.path.exists(contract_file):
        contract_data = pd.read_csv(contract_file)
        logger.info(f"Loaded contract training data: {len(contract_data)} records")
    else:
        logger.warning(f"Contract training data file not found: {contract_file}")
    
    # Combine both datasets if available
    if consent_data is not None and contract_data is not None:
        combined_data = pd.concat([consent_data, contract_data], ignore_index=True)
        logger.info(f"Combined training data: {len(combined_data)} records")
    elif consent_data is not None:
        combined_data = consent_data
    elif contract_data is not None:
        combined_data = contract_data
    
    return consent_data, contract_data, combined_data

# Function to process OCR text files in a directory
def process_ocr_files(directory):
    text_files = glob.glob(os.path.join(directory, "*.txt"))
    
    if not text_files:
        logger.warning(f"No text files found in directory: {directory}")
        # Try to find text files in subdirectories (one level deep)
        for subdir in glob.glob(os.path.join(directory, "*")):
            if os.path.isdir(subdir):
                sub_text_files = glob.glob(os.path.join(subdir, "*.txt"))
                if sub_text_files:
                    text_files.extend(sub_text_files)
                    logger.info(f"Found {len(sub_text_files)} text files in subdirectory: {subdir}")
    
    if not text_files:
        logger.warning(f"No text files found in directory or subdirectories: {directory}")
        return None
    
    # Combine all text files from the directory into a single text
    combined_text = ""
    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                combined_text += file_content + "\n"
        except UnicodeDecodeError:
            # Try different encodings if utf-8 fails
            try:
                with open(text_file, 'r', encoding='cp949') as f:
                    file_content = f.read()
                    combined_text += file_content + "\n"
            except Exception as e:
                logger.error(f"Error reading file {text_file} with cp949 encoding: {e}")
        except Exception as e:
            logger.error(f"Error reading file {text_file}: {e}")
    
    return combined_text

# Function to prepare dataset for NER training
def prepare_ner_dataset(data, tokenizer, label_list):
    if data is None or len(data) == 0:
        return None, None, None
    
    try:
        # Create label to id mapping
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}
        
        # Add special tokens for NER tagging (B-label, I-label format)
        ner_labels = ['O']  # Outside any entity
        for label in label_list:
            ner_labels.extend([f'B-{label}', f'I-{label}'])
        
        # Update label mappings to include NER tags
        ner_label2id = {label: i for i, label in enumerate(ner_labels)}
        ner_id2label = {i: label for i, label in enumerate(ner_labels)}
        
        # Convert dataset to format suitable for HuggingFace training
        tokenized_inputs = []
        labels = []
        
        for _, row in data.iterrows():
            # Create text from all column values
            text_parts = []
            entity_positions = []
            current_pos = 0
            
            for col_name, value in row.items():
                if pd.notna(value) and str(value).strip():
                    text_part = str(value).strip()
                    start_pos = current_pos
                    end_pos = current_pos + len(text_part)
                    
                    # Store entity information
                    entity_positions.append({
                        'text': text_part,
                        'label': col_name,
                        'start': start_pos,
                        'end': end_pos
                    })
                    
                    text_parts.append(text_part)
                    current_pos = end_pos + 1  # +1 for space
            
            # Create full text
            text = ' '.join(text_parts)
            
            # Tokenize text
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Truncate if too long
            if len(token_ids) > MAX_LENGTH:
                token_ids = token_ids[:MAX_LENGTH]
                tokens = tokens[:MAX_LENGTH]
            
            # Create NER tags
            ner_tags = ['O'] * len(token_ids)
            
            # Assign tags based on entities
            for entity in entity_positions:
                entity_text = entity['text']
                entity_type = entity['label']
                
                # Find entity in the concatenated text
                if entity_text in text:
                    start_idx = text.find(entity_text)
                    end_idx = start_idx + len(entity_text)
                    
                    # Map character positions to token positions
                    token_start = None
                    token_end = None
                    char_pos = 0
                    
                    for i, token in enumerate(tokens):
                        token_text = token.replace('##', '')
                        
                        if char_pos <= start_idx < char_pos + len(token_text):
                            token_start = i
                        
                        if char_pos < end_idx <= char_pos + len(token_text):
                            token_end = i
                            break
                        
                        char_pos += len(token_text)
                        
                        # Account for spaces between tokens
                        if i < len(tokens) - 1 and not tokens[i+1].startswith('##'):
                            char_pos += 1
                    
                    # Assign BIO tags
                    if token_start is not None and token_end is not None:
                        if token_start < len(ner_tags):
                            ner_tags[token_start] = f'B-{entity_type}'
                        for i in range(token_start + 1, min(token_end + 1, len(ner_tags))):
                            ner_tags[i] = f'I-{entity_type}'
            
            tokenized_inputs.append(token_ids)
            labels.append([ner_label2id.get(tag, 0) for tag in ner_tags])
        
        # Convert to Dataset
        dataset_dict = {
            'input_ids': tokenized_inputs,
            'labels': labels
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Split dataset into train and eval
        dataset = dataset.train_test_split(test_size=0.1)
        return dataset, ner_id2label, ner_label2id
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return None, None, None

# Function to create and train NER model
def train_ner_model(train_dataset, id2label, label2id, model_name=MODEL_NAME, output_dir=None):
    if train_dataset is None:
        logger.error("Cannot train model: no training data provided")
        return None
    
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir or "./model_output",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=False,
    )
    
    # Create data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    if output_dir:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# Function to predict entities from text
def predict_entities(text, model, tokenizer, id2label):
    if not text:
        return []
    
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process predictions
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    current_entity = None
    
    for i, (token, prediction) in enumerate(zip(tokens, predictions[0])):
        tag = id2label[prediction.item()]
        
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            
            current_entity = {
                'label': tag[2:],
                'text': token.replace('##', '')
            }
        elif tag.startswith('I-') and current_entity and current_entity['label'] == tag[2:]:
            current_entity['text'] += token.replace('##', '')
        elif tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# Function to predict entities from consent documents
def predict_consent_documents(model, tokenizer, id2label):
    """Predict entities from consent documents"""
    consent_predictions = []
    
    # Find consent document directories
    consent_dirs = []
    
    # First check if there's a directory named '동의서' directly in OCR_DOCUMENT_PATH
    direct_consent_dir = os.path.join(OCR_DOCUMENT_PATH, "동의서")
    if os.path.exists(direct_consent_dir) and os.path.isdir(direct_consent_dir):
        consent_dirs.append(direct_consent_dir)
        logger.info(f"Found consent directory: {direct_consent_dir}")
    
    # Also look for any directory with '동의서' in its name
    for root, dirs, _ in os.walk(OCR_DOCUMENT_PATH):
        for d in dirs:
            if "동의서" in d and os.path.join(root, d) not in consent_dirs:
                consent_dirs.append(os.path.join(root, d))
                logger.info(f"Found consent directory: {os.path.join(root, d)}")
    
    if not consent_dirs:
        logger.warning(f"No consent document directories found in: {OCR_DOCUMENT_PATH}")
        return
    
    for consent_dir in consent_dirs:
        # Process files in the consent directory itself
        combined_text = process_ocr_files(consent_dir)
        if combined_text:
            entities = predict_entities(combined_text, model, tokenizer, id2label)
            
            # Extract directory name as document ID
            doc_id = os.path.basename(consent_dir)
            
            # Create result row
            result = {"document_id": doc_id}
            
            # Extract entity values - only include CONSENT_LABELS
            for label in CONSENT_LABELS:
                value = next((e["text"] for e in entities if e["label"] == label), "")
                result[label] = value
            
            consent_predictions.append(result)
        
        # Also check subdirectories
        for dir_path, _, _ in os.walk(consent_dir):
            if dir_path != consent_dir:  # Skip the parent directory we just processed
                logger.info(f"Processing subdirectory: {dir_path}")
                combined_text = process_ocr_files(dir_path)
                
                if combined_text:
                    entities = predict_entities(combined_text, model, tokenizer, id2label)
                    
                    # Extract directory name as document ID
                    doc_id = os.path.basename(dir_path)
                    
                    # Create result row
                    result = {"document_id": doc_id}
                    
                    # Extract entity values - only include CONSENT_LABELS
                    for label in CONSENT_LABELS:
                        value = next((e["text"] for e in entities if e["label"] == label), "")
                        result[label] = value
                    
                    consent_predictions.append(result)
    
    # Save consent predictions to CSV
    if consent_predictions:
        consent_df = pd.DataFrame(consent_predictions)
        consent_df.to_csv(CONSENT_PRED_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"Saved consent predictions to: {CONSENT_PRED_FILE}")
    else:
        logger.warning("No consent predictions generated")

# Function to predict entities from contract documents
def predict_contract_documents(model, tokenizer, id2label):
    """Predict entities from contract documents"""
    contract_predictions = []
    
    # Find contract document directories
    contract_dirs = []
    
    # First check if there's a directory named '계약서' directly in OCR_DOCUMENT_PATH
    direct_contract_dir = os.path.join(OCR_DOCUMENT_PATH, "계약서")
    if os.path.exists(direct_contract_dir) and os.path.isdir(direct_contract_dir):
        contract_dirs.append(direct_contract_dir)
        logger.info(f"Found contract directory: {direct_contract_dir}")
    
    # Also look for any directory with '계약서' in its name
    for root, dirs, _ in os.walk(OCR_DOCUMENT_PATH):
        for d in dirs:
            if "계약서" in d and os.path.join(root, d) not in contract_dirs:
                contract_dirs.append(os.path.join(root, d))
                logger.info(f"Found contract directory: {os.path.join(root, d)}")
    
    if not contract_dirs:
        logger.warning(f"No contract document directories found in: {OCR_DOCUMENT_PATH}")
        return
    
    for contract_dir in contract_dirs:
        # Process files in the contract directory itself
        combined_text = process_ocr_files(contract_dir)
        if combined_text:
            entities = predict_entities(combined_text, model, tokenizer, id2label)
            
            # Extract directory name as document ID
            doc_id = os.path.basename(contract_dir)
            
            # Create result row
            result = {"document_id": doc_id}
            
            # Extract entity values - only include CONTRACT_LABELS
            for label in CONTRACT_LABELS:
                value = next((e["text"] for e in entities if e["label"] == label), "")
                result[label] = value
            
            contract_predictions.append(result)
        
        # Also check subdirectories
        for dir_path, _, _ in os.walk(contract_dir):
            if dir_path != contract_dir:  # Skip the parent directory we just processed
                logger.info(f"Processing subdirectory: {dir_path}")
                combined_text = process_ocr_files(dir_path)
                
                if combined_text:
                    entities = predict_entities(combined_text, model, tokenizer, id2label)
                    
                    # Extract directory name as document ID
                    doc_id = os.path.basename(dir_path)
                    
                    # Create result row
                    result = {"document_id": doc_id}
                    
                    # Extract entity values - only include CONTRACT_LABELS
                    for label in CONTRACT_LABELS:
                        value = next((e["text"] for e in entities if e["label"] == label), "")
                        result[label] = value
                    
                    contract_predictions.append(result)
    
    # Save contract predictions to CSV
    if contract_predictions:
        contract_df = pd.DataFrame(contract_predictions)
        contract_df.to_csv(CONTRACT_PRED_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"Saved contract predictions to: {CONTRACT_PRED_FILE}")
    else:
        logger.warning("No contract predictions generated")

# Main function to run the NER system
def main():
    parser = argparse.ArgumentParser(description='NER System for Contract and Consent Documents')
    parser.add_argument('--predict-consent', action='store_true', help='Predict entities from consent documents')
    parser.add_argument('--predict-contract', action='store_true', help='Predict entities from contract documents')
    parser.add_argument('--predict-all', action='store_true', help='Predict entities from all documents', default=True)
    args = parser.parse_args()
    
    try:
        # Check paths
        check_paths()
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load training data
        consent_data, contract_data, combined_data = load_training_data()
        
        # Load or train model
        model = None
        id2label = None
        
        # Train unified model if there's combined data
        if combined_data is not None and len(combined_data) > 0:
            logger.info("Preparing combined dataset for NER training...")
            # Combine all labels
            all_labels = list(set(CONSENT_LABELS + CONTRACT_LABELS))
            
            # Prepare combined dataset
            combined_dataset, combined_id2label, combined_label2id = prepare_ner_dataset(
                combined_data, tokenizer, all_labels
            )
            
            if combined_dataset is not None:
                # Check if we already have a model
                unified_model_dir = os.path.join(OUTPUT_DIR, "unified_model")
                if os.path.exists(unified_model_dir):
                    logger.info("Loading existing unified model...")
                    model = AutoModelForTokenClassification.from_pretrained(unified_model_dir)
                    id2label = model.config.id2label
                else:
                    logger.info("Training new unified model...")
                    model, _ = train_ner_model(
                        combined_dataset,
                        combined_id2label,
                        combined_label2id,
                        output_dir=unified_model_dir
                    )
                    id2label = combined_id2label
        else:
            logger.warning("No training data available. Please create training data first.")
            return
        
        if model is None:
            logger.error("Failed to load or train a model.")
            return
            
        # Always predict both document types
        logger.info("Processing consent documents...")
        predict_consent_documents(model, tokenizer, id2label)
        
        logger.info("Processing contract documents...")
        predict_contract_documents(model, tokenizer, id2label)
        
        logger.info("NER processing completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
