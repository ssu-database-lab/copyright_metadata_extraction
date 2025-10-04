import os
import glob
import pandas as pd
import re
import json
from pathlib import Path

# OCR 문서 경로
OCR_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OCR", "ocr_document")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 라벨 정의
CONTRACT_LABELS = [
    "저작물명", "대상 저작물 상세정보", "양수자 기관명", "양수자 주소", 
    "양도자 기관(개인)명", "양도자 소속", "양도자 대표주소", "양도자 연락처", 
    "동의여부", "날짜"
]

CONSENT_LABELS = [
    "양수인 성명", "양도인 주소", "양도인 전화번호", "양수인 기관명", 
    "양수인 대표자명", "양수인 대표자 주소", "양수인 대표자 연락처", 
    "동의여부", "동의날짜"
]

def read_ocr_file(file_path):
    """OCR 텍스트 파일을 읽어서 내용을 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"파일 읽기 오류: {file_path} - {e}")
            return None
    except Exception as e:
        print(f"파일 읽기 오류: {file_path} - {e}")
        return None

def extract_contract_entities(text):
    """계약서에서 엔티티를 추출합니다."""
    entities = []
    
    # 저작물명 추출
    patterns = {
        "저작물명": [
            r"저작물명\s*[:：]\s*([^\n]+)",
            r"○\s*저작물명\s*[:：]\s*([^\n]+)",
            r"저작물\s*[:：]\s*([^\n]+)",
        ],
        "대상 저작물 상세정보": [
            r"대상\s*저작물\s*상세정보\s*[:：]\s*([^\n]+)",
            r"○\s*대상\s*저작물\s*상세정보\s*[:：]\s*([^\n]+)",
        ],
        "양수자 기관명": [
            r"양수자.*?기관명\s*[:：]\s*([^\n]+)",
            r"○\s*기관명\s*[:：]\s*([^\n]+)",
            r"기관명\s*[:：]\s*([^\n]+)",
        ],
        "양수자 주소": [
            r"양수자.*?주소\s*[:：]\s*([^\n]+)",
            r"○\s*주소\s*[:：]\s*([^\n]+)",
            r"주소\s*[:：]\s*([^\n]+)",
        ],
        "양도자 기관(개인)명": [
            r"양도자.*?기관.*?명\s*[:：]\s*([^\n]+)",
            r"양도자.*?기관\s*[:：]\s*([^\n]+)",
        ],
        "양도자 소속": [
            r"양도자.*?소속\s*[:：]\s*([^\n]+)",
            r"소속\s*[:：]\s*([^\n]+)",
        ],
        "양도자 대표주소": [
            r"양도자.*?주소\s*[:：]\s*([^\n]+)",
            r"양도자.*?대표.*?주소\s*[:：]\s*([^\n]+)",
        ],
        "양도자 연락처": [
            r"양도자.*?연락처\s*[:：]\s*([^\n]+)",
            r"양도자.*?전화\s*[:：]\s*([^\n]+)",
        ],
        "동의여부": [
            r"동의함|동의|합의|승인",
        ],
        "날짜": [
            r"(\d{4}[\.\-\/년]\s*\d{1,2}[\.\-\/월]\s*\d{1,2}[일]?)",
            r"(\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2})",
        ]
    }
    
    for label, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                entity_text = match.group(1).strip() if match.groups() else match.group(0).strip()
                if entity_text and len(entity_text) > 1:
                    entities.append({
                        'text': entity_text,
                        'label': label
                    })
                    break  # 첫 번째 매치만 사용
        
        if label == "동의여부" and not any(e['label'] == label for e in entities):
            # 동의여부가 없으면 기본값 추가
            if "동의" in text or "합의" in text or "승인" in text:
                entities.append({
                    'text': "동의",
                    'label': label
                })
    
    return entities

def extract_consent_entities(text):
    """동의서에서 엔티티를 추출합니다."""
    entities = []
    
    patterns = {
        "양수인 성명": [
            r"양도인\s*성명\s*[:：]\s*([^\s\n]+)",
            r"성명\s*[:：]\s*([^\s\n]+)",
            r"양도인.*?성명.*?[:：]\s*([^\s\n]+)",
        ],
        "양도인 주소": [
            r"양도인.*?주소\s*[:：]\s*([^\n]+)",
            r"주소\s*[:：]\s*([^\n]+)",
        ],
        "양도인 전화번호": [
            r"양도인.*?전화번호\s*[:：]\s*([^\s\n]+)",
            r"전화번호\s*[:：]\s*([^\s\n]+)",
            r"전화\s*[:：]\s*([^\s\n]+)",
            r"(\d{3}-\d{4}-\d{4})",
            r"(\d{2,3}-\d{3,4}-\d{4})",
        ],
        "양수인 기관명": [
            r"양수인\s*기관명\s*[:：]\s*([^\n]+)",
            r"기관명\s*[:：]\s*([^\n]+)",
        ],
        "양수인 대표자명": [
            r"양수인.*?대표자명\s*[:：]\s*([^\s\n]+)",
            r"대표자명\s*[:：]\s*([^\s\n]+)",
            r"대표자\s*[:：]\s*([^\s\n]+)",
        ],
        "양수인 대표자 주소": [
            r"양수인.*?대표자.*?주소\s*[:：]\s*([^\n]+)",
            r"대표자.*?주소\s*[:：]\s*([^\n]+)",
        ],
        "양수인 대표자 연락처": [
            r"양수인.*?대표자.*?연락처\s*[:：]\s*([^\s\n]+)",
            r"대표자.*?연락처\s*[:：]\s*([^\s\n]+)",
        ],
        "동의여부": [
            r"동의함|동의|합의|승인",
        ],
        "동의날짜": [
            r"(\d{4}[\.\-\/년]\s*\d{1,2}[\.\-\/월]\s*\d{1,2}[일]?)",
            r"(\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2})",
        ]
    }
    
    for label, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                entity_text = match.group(1).strip() if match.groups() else match.group(0).strip()
                if entity_text and len(entity_text) > 1:
                    entities.append({
                        'text': entity_text,
                        'label': label
                    })
                    break  # 첫 번째 매치만 사용
        
        if label == "동의여부" and not any(e['label'] == label for e in entities):
            # 동의여부가 없으면 기본값 추가
            if "동의" in text or "합의" in text or "승인" in text:
                entities.append({
                    'text': "동의",
                    'label': label
                })
    
    return entities

def process_contract_files():
    """계약서 파일들을 처리합니다."""
    contract_data = []
    
    # 계약서 디렉토리 찾기
    contract_dirs = [
        os.path.join(OCR_DOCUMENT_PATH, "7.저작물양도계약서")
    ]
    
    for contract_dir in contract_dirs:
        if os.path.exists(contract_dir):
            txt_files = glob.glob(os.path.join(contract_dir, "*.txt"))
            
            for txt_file in txt_files:
                content = read_ocr_file(txt_file)
                if content:
                    # 텍스트 정리
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    # 엔티티 추출
                    entities = extract_contract_entities(content)
                    
                    if entities:  # 엔티티가 있는 경우만 추가
                        contract_data.append({
                            'text': content,
                            'entities': str(entities)
                        })
                        print(f"계약서 처리 완료: {txt_file} - {len(entities)}개 엔티티")
    
    return contract_data

def process_consent_files():
    """동의서 파일들을 처리합니다."""
    consent_data = []
    
    # 동의서 디렉토리들 찾기
    consent_dirs = [
        os.path.join(OCR_DOCUMENT_PATH, "동의서"),
        os.path.join(OCR_DOCUMENT_PATH, "진천동의서 11명")
    ]
    
    for consent_dir in consent_dirs:
        if os.path.exists(consent_dir):
            # 직접 txt 파일이 있는 경우
            txt_files = glob.glob(os.path.join(consent_dir, "*.txt"))
            for txt_file in txt_files:
                content = read_ocr_file(txt_file)
                if content:
                    # 텍스트 정리
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    # 엔티티 추출
                    entities = extract_consent_entities(content)
                    
                    if entities:  # 엔티티가 있는 경우만 추가
                        consent_data.append({
                            'text': content,
                            'entities': str(entities)
                        })
                        print(f"동의서 처리 완료: {txt_file} - {len(entities)}개 엔티티")
            
            # 하위 디렉토리에 txt 파일이 있는 경우
            subdirs = [d for d in os.listdir(consent_dir) if os.path.isdir(os.path.join(consent_dir, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(consent_dir, subdir)
                txt_files = glob.glob(os.path.join(subdir_path, "*.txt"))
                
                # 하위 디렉토리의 모든 txt 파일을 합쳐서 하나의 문서로 처리
                combined_content = ""
                for txt_file in txt_files:
                    content = read_ocr_file(txt_file)
                    if content:
                        combined_content += content + " "
                
                if combined_content:
                    # 텍스트 정리
                    combined_content = re.sub(r'\s+', ' ', combined_content).strip()
                    
                    # 엔티티 추출
                    entities = extract_consent_entities(combined_content)
                    
                    if entities:  # 엔티티가 있는 경우만 추가
                        consent_data.append({
                            'text': combined_content,
                            'entities': str(entities)
                        })
                        print(f"동의서 처리 완료: {subdir} - {len(entities)}개 엔티티")
    
    return consent_data

def main():
    """메인 함수"""
    print("OCR 데이터에서 학습용 CSV 파일을 생성합니다...")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 계약서 데이터 처리
    print("\n계약서 파일들을 처리중...")
    contract_data = process_contract_files()
    
    # 동의서 데이터 처리
    print("\n동의서 파일들을 처리중...")
    consent_data = process_consent_files()
    
    # CSV 파일로 저장
    if contract_data:
        contract_df = pd.DataFrame(contract_data)
        contract_file = os.path.join(OUTPUT_DIR, "train_contract_results.csv")
        contract_df.to_csv(contract_file, index=False, encoding='utf-8-sig')
        print(f"\n계약서 학습 데이터 저장: {contract_file} ({len(contract_data)}개 샘플)")
    
    if consent_data:
        consent_df = pd.DataFrame(consent_data)
        consent_file = os.path.join(OUTPUT_DIR, "train_consent_results.csv")
        consent_df.to_csv(consent_file, index=False, encoding='utf-8-sig')
        print(f"동의서 학습 데이터 저장: {consent_file} ({len(consent_data)}개 샘플)")
    
    print(f"\n총 {len(contract_data) + len(consent_data)}개의 학습 데이터가 생성되었습니다.")

if __name__ == "__main__":
    main()
