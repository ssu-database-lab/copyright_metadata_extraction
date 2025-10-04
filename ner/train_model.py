import os
import argparse
import sys
import subprocess
import importlib.util

# 필요한 패키지 확인 및 설치
required_packages = [
    'pandas', 'torch', 'transformers', 'datasets', 'tqdm', 'numpy', 'scikit-learn'
]

for package in required_packages:
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(f"{package} 패키지를 설치합니다...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
import logging
import warnings
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Constants
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

def load_training_data(consent_file=CONSENT_TRAIN_FILE, contract_file=CONTRACT_TRAIN_FILE):
    consent_data = None
    contract_data = None
    
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
    
    return consent_data, contract_data

def prepare_ner_dataset(data, tokenizer, label_list):
    if data is None or len(data) == 0:
        logger.error("No training data provided")
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

def train_ner_model(train_dataset, id2label, label2id, model_name=MODEL_NAME, output_dir=None):
    if train_dataset is None:
        logger.error("Cannot train model: no training data provided")
        return None, None
    
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
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
        logger.info(f"Model saved to: {output_dir}")
    
    return model, tokenizer

def train_consent_model():
    """Train NER model for consent documents"""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load training data
    consent_data, _ = load_training_data()
    
    if consent_data is None:
        logger.error("동의서 학습 데이터가 없습니다.")
        return
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare dataset
    logger.info("동의서 학습 데이터셋 준비 중...")
    consent_dataset, consent_id2label, consent_label2id = prepare_ner_dataset(
        consent_data, tokenizer, CONSENT_LABELS
    )
    
    if consent_dataset is None:
        logger.error("동의서 데이터셋 준비 실패")
        return
    
    # Train model
    logger.info("동의서 NER 모델 학습 중...")
    consent_model_dir = os.path.join(OUTPUT_DIR, "consent_model")
    train_ner_model(
        consent_dataset, 
        consent_id2label, 
        consent_label2id,
        output_dir=consent_model_dir
    )
    
    logger.info("동의서 NER 모델 학습 완료")

def train_contract_model():
    """Train NER model for contract documents"""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load training data
    _, contract_data = load_training_data()
    
    if contract_data is None:
        logger.error("계약서 학습 데이터가 없습니다.")
        return
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare dataset
    logger.info("계약서 학습 데이터셋 준비 중...")
    contract_dataset, contract_id2label, contract_label2id = prepare_ner_dataset(
        contract_data, tokenizer, CONTRACT_LABELS
    )
    
    if contract_dataset is None:
        logger.error("계약서 데이터셋 준비 실패")
        return
    
    # Train model
    logger.info("계약서 NER 모델 학습 중...")
    contract_model_dir = os.path.join(OUTPUT_DIR, "contract_model")
    train_ner_model(
        contract_dataset, 
        contract_id2label, 
        contract_label2id,
        output_dir=contract_model_dir
    )
    
    logger.info("계약서 NER 모델 학습 완료")

def train_all_models():
    """Train both consent and contract NER models"""
    train_consent_model()
    train_contract_model()

def main():
    parser = argparse.ArgumentParser(description='NER 모델 학습')
    parser.add_argument('--doc-type', choices=['동의서', '계약서', 'all'], 
                        help='처리할 문서 유형 (동의서, 계약서, 또는 all)', default='all')
    args = parser.parse_args()
    
    try:
        # Always train all models regardless of argument
        train_all_models()
    
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()
