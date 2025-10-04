import os
import argparse
import pandas as pd
import glob
import logging
import re
import json
from tqdm import tqdm
from entity_extraction import extract_consent_entities, extract_contract_entities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OCR_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OCR", "ocr_document")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Define entity labels for contract documents
CONTRACT_LABELS = [
    "저작물명", "대상 저작물 상세정보", "양수자 기관명", "양수자 주소", 
    "양도자 기관(개인)명", "양도자 소속", "양도자 대표주소", "양도자 연락처", 
    "동의여부", "날짜"
]

# Define entity labels for consent documents
CONSENT_LABELS = [
    "양수인 성명", "양도인 주소", "양도인 전화번호", "양수인 기관명", 
    "양수인 대표자명", "양수인 대표자 주소", "양수인 대표자 연락처", 
    "동의여부", "동의날짜"
]

# Output filenames
CONSENT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_consent_results.csv")
CONTRACT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_contract_results.csv")

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

def create_training_data_interactive():
    """Create training data through interactive mode"""
    doc_type = input("문서 유형을 선택하세요 (1: 동의서, 2: 계약서): ")
    
    if doc_type == "1":
        create_consent_training_data_interactive()
    elif doc_type == "2":
        create_contract_training_data_interactive()
    else:
        logger.error("잘못된 문서 유형입니다. 1 또는 2를 입력하세요.")

def create_consent_training_data_interactive():
    """Create consent document training data interactively"""
    training_data = []
    
    while True:
        logger.info("동의서 학습 데이터를 입력합니다.")
        
        text = input("텍스트를 입력하세요 (끝내려면 빈 줄을 입력하세요): ")
        if not text:
            break
        
        entities = []
        for label in CONSENT_LABELS:
            value = input(f"{label}에 해당하는 텍스트를 입력하세요: ")
            if value:
                entities.append({
                    'text': value,
                    'label': label
                })
        
        training_data.append({
            'text': text,
            'entities': entities
        })
    
    if training_data:
        df = pd.DataFrame(training_data)
        df.to_csv(CONSENT_TRAIN_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"동의서 학습 데이터가 {CONSENT_TRAIN_FILE}에 저장되었습니다.")
    else:
        logger.warning("입력된 데이터가 없습니다.")

def create_contract_training_data_interactive():
    """Create contract document training data interactively"""
    training_data = []
    
    while True:
        logger.info("계약서 학습 데이터를 입력합니다.")
        
        text = input("텍스트를 입력하세요 (끝내려면 빈 줄을 입력하세요): ")
        if not text:
            break
        
        entities = []
        for label in CONTRACT_LABELS:
            value = input(f"{label}에 해당하는 텍스트를 입력하세요: ")
            if value:
                entities.append({
                    'text': value,
                    'label': label
                })
        
        training_data.append({
            'text': text,
            'entities': entities
        })
    
    if training_data:
        df = pd.DataFrame(training_data)
        df.to_csv(CONTRACT_TRAIN_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"계약서 학습 데이터가 {CONTRACT_TRAIN_FILE}에 저장되었습니다.")
    else:
        logger.warning("입력된 데이터가 없습니다.")

def create_consent_training_data_auto():
    """Create consent document training data automatically"""
    check_paths()
    
    training_data = []
    
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
        logger.warning(f"동의서 디렉토리를 찾을 수 없습니다: {OCR_DOCUMENT_PATH}")
        return
    
    for consent_dir in consent_dirs:
        # Process the main directory
        combined_text = process_ocr_files(consent_dir)
        if combined_text:
            # Simple pattern-based extraction for demonstration
            entities = extract_consent_entities(combined_text)
            
            if entities:
                training_data.append({
                    'text': combined_text,
                    'entities': entities
                })
        
        # Process subdirectories
        for dir_path, _, _ in os.walk(consent_dir):
            if dir_path != consent_dir:  # Skip the parent directory we just processed
                logger.info(f"디렉토리 처리 중: {dir_path}")
                combined_text = process_ocr_files(dir_path)
                
                if combined_text:
                    # Extract entities using simple patterns
                    entities = extract_consent_entities(combined_text)
                    
                    if entities:
                        training_data.append({
                            'text': combined_text,
                            'entities': entities
                        })
    
    if training_data:
        # Append to existing training data if it exists
        existing_data = []
        if os.path.exists(CONSENT_TRAIN_FILE):
            existing_df = pd.read_csv(CONSENT_TRAIN_FILE)
            existing_data = existing_df.to_dict('records')
        
        # Combine existing and new data
        combined_data = existing_data + training_data
        
        df = pd.DataFrame(combined_data)
        df.to_csv(CONSENT_TRAIN_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"{len(training_data)}개의 동의서 학습 데이터가 {CONSENT_TRAIN_FILE}에 저장되었습니다.")
    else:
        logger.warning("추출된 데이터가 없습니다.")

def create_contract_training_data_auto():
    """Create contract document training data automatically"""
    check_paths()
    
    training_data = []
    
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
        logger.warning(f"계약서 디렉토리를 찾을 수 없습니다: {OCR_DOCUMENT_PATH}")
        return
    
    for contract_dir in contract_dirs:
        # Process the main directory
        combined_text = process_ocr_files(contract_dir)
        if combined_text:
            # Extract entities using patterns
            entities = extract_contract_entities(combined_text)
            
            if entities:
                training_data.append({
                    'text': combined_text,
                    'entities': entities
                })
        
        # Process subdirectories
        for dir_path, _, _ in os.walk(contract_dir):
            if dir_path != contract_dir:  # Skip the parent directory we just processed
                logger.info(f"디렉토리 처리 중: {dir_path}")
                combined_text = process_ocr_files(dir_path)
                
                if combined_text:
                    # Extract entities using patterns
                    entities = extract_contract_entities(combined_text)
                    
                    if entities:
                        training_data.append({
                            'text': combined_text,
                            'entities': entities
                        })
    
    if training_data:
        # Append to existing training data if it exists
        existing_data = []
        if os.path.exists(CONTRACT_TRAIN_FILE):
            existing_df = pd.read_csv(CONTRACT_TRAIN_FILE)
            existing_data = existing_df.to_dict('records')
        
        # Combine existing and new data
        combined_data = existing_data + training_data
        
        df = pd.DataFrame(combined_data)
        df.to_csv(CONTRACT_TRAIN_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"{len(training_data)}개의 계약서 학습 데이터가 {CONTRACT_TRAIN_FILE}에 저장되었습니다.")
    else:
        logger.warning("추출된 데이터가 없습니다.")
    
    for contract_dir in contract_dirs:
        for dir_path, _, _ in os.walk(contract_dir):
            if os.path.basename(dir_path) != os.path.basename(contract_dir):  # Process subdirectories
                logger.info(f"디렉토리 처리 중: {dir_path}")
                combined_text = process_ocr_files(dir_path)
                
                if combined_text:
                    # Simple pattern-based extraction for demonstration
                    # In a real system, you might want to use more sophisticated extraction methods
                    
                    entities = []
                    
                    # Extract entities using simple patterns
                    patterns = [
                        (r'저작물\s*명\s*[:：]?\s*([^\n,]+)', '저작물명'),
                        (r'저작물\s*상세\s*정보\s*[:：]?\s*([^\n]+)', '대상 저작물 상세정보'),
                        (r'양수자\s*기관\s*명\s*[:：]?\s*([^\n,]+)', '양수자 기관명'),
                        (r'양수자\s*주소\s*[:：]?\s*([^\n]+)', '양수자 주소'),
                        (r'양도자\s*기관\s*명\s*[:：]?\s*([^\n,]+)', '양도자 기관(개인)명'),
                        (r'양도자\s*소속\s*[:：]?\s*([^\n,]+)', '양도자 소속'),
                        (r'양도자\s*주소\s*[:：]?\s*([^\n]+)', '양도자 대표주소'),
                        (r'양도자\s*연락처\s*[:：]?\s*([\d\-]+)', '양도자 연락처'),
                        (r'동의\s*여부\s*[:：]?\s*([^\n,]+)', '동의여부'),
                        (r'날짜\s*[:：]?\s*([^\n]+)', '날짜')
                    ]
                    
                    for pattern, label in patterns:
                        matches = re.findall(pattern, combined_text)
                        if matches:
                            entities.append({
                                'text': matches[0].strip(),
                                'label': label
                            })
                    
                    if entities:
                        training_data.append({
                            'text': combined_text,
                            'entities': entities
                        })
    
    if training_data:
        # Append to existing training data if it exists
        existing_data = []
        if os.path.exists(CONTRACT_TRAIN_FILE):
            existing_df = pd.read_csv(CONTRACT_TRAIN_FILE)
            existing_data = existing_df.to_dict('records')
        
        # Combine existing and new data
        combined_data = existing_data + training_data
        
        df = pd.DataFrame(combined_data)
        df.to_csv(CONTRACT_TRAIN_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"{len(training_data)}개의 계약서 학습 데이터가 {CONTRACT_TRAIN_FILE}에 저장되었습니다.")
    else:
        logger.warning("추출된 데이터가 없습니다.")

def main():
    parser = argparse.ArgumentParser(description='NER 학습 데이터 생성')
    parser.add_argument('--auto', action='store_true', help='자동 추출 모드 사용')
    parser.add_argument('--doc-type', choices=['동의서', '계약서'], help='처리할 문서 유형')
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        if args.auto:
            if args.doc_type == '동의서':
                create_consent_training_data_auto()
            elif args.doc_type == '계약서':
                create_contract_training_data_auto()
            else:
                logger.error("잘못된 문서 유형입니다. '동의서' 또는 '계약서'를 입력하세요.")
        else:
            create_training_data_interactive()
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()
