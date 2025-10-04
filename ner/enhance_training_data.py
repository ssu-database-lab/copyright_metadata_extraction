import os
import glob
import pandas as pd
import re
import random
from pathlib import Path

# OCR 문서 경로
OCR_DOCUMENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OCR", "ocr_document")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

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

def extract_enhanced_contract_entities(text):
    """향상된 계약서 엔티티 추출"""
    entities = []
    
    # 더 정교한 패턴 매칭
    patterns = {
        "저작물명": [
            r"저작물명\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)",
            r"○\s*저작물명\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)",
            r"저작물\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)",
            r"○\s*저작물\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)",
        ],
        "대상 저작물 상세정보": [
            r"대상\s*저작물\s*상세정보\s*[:：]\s*([^\n\r○●]+?)(?:\s*[□○●]\s*|\s*$)",
            r"○\s*대상\s*저작물\s*상세정보\s*[:：]\s*([^\n\r○●]+?)(?:\s*[□○●]\s*|\s*$)",
        ],
        "양수자 기관명": [
            r"양수자.*?기관명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
            r"○\s*기관명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
            r"기관명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
        ],
        "양수자 주소": [
            r"양수자.*?주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)",
            r"○\s*주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)",
            r"주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)",
        ],
        "양도자 기관(개인)명": [
            r"양도자.*?기관.*?명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
            r"양도자.*?기관\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
            r"양도자\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
        ],
        "양도자 소속": [
            r"양도자.*?소속\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
            r"소속\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
        ],
        "양도자 대표주소": [
            r"양도자.*?주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)",
            r"양도자.*?대표.*?주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)",
        ],
        "양도자 연락처": [
            r"양도자.*?연락처\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
            r"양도자.*?전화\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)",
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
        found = False
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                entity_text = match.group(1).strip() if match.groups() else match.group(0).strip()
                entity_text = re.sub(r'\s+', ' ', entity_text)  # 공백 정리
                if entity_text and len(entity_text) > 1 and len(entity_text) < 200:
                    entities.append({
                        'text': entity_text,
                        'label': label
                    })
                    found = True
                    break
            if found:
                break
        
        # 동의여부 기본값
        if label == "동의여부" and not found:
            if any(word in text for word in ["동의", "합의", "승인", "체결"]):
                entities.append({
                    'text': "동의",
                    'label': label
                })
    
    return entities

def extract_enhanced_consent_entities(text):
    """향상된 동의서 엔티티 추출"""
    entities = []
    
    patterns = {
        "양수인 성명": [
            r"양도인\s*성명\s*[:：]\s*([^\s\n\r(]+)",
            r"성명\s*[:：]\s*([^\s\n\r(]+)",
            r"양도인.*?성명.*?[:：]\s*([^\s\n\r(]+)",
            r"양도자\s*본인은.*?([가-힣]{2,4})\s*\(",
            r"본인\s*([가-힣]{2,4})\s*\(",
        ],
        "양도인 주소": [
            r"양도인.*?주소\s*[:：]\s*([^\n\r]+?)(?:\s*양수인|\s*전화|\s*$)",
            r"주소\s*[:：]\s*([^\n\r]+?)(?:\s*양수인|\s*전화|\s*$)",
            r"주소.*?[:：]\s*([^\n\r]+?)(?:\s*양수인|\s*전화|\s*$)",
        ],
        "양도인 전화번호": [
            r"양도인.*?전화번호\s*[:：]\s*([0-9\-]+)",
            r"전화번호\s*[:：]\s*([0-9\-]+)",
            r"전화\s*[:：]\s*([0-9\-]+)",
            r"(\d{3}-\d{4}-\d{4})",
            r"(\d{2,3}-\d{3,4}-\d{4})",
        ],
        "양수인 기관명": [
            r"양수인\s*기관명\s*[:：]\s*([^\n\r]+?)(?:\s*대표자|\s*$)",
            r"기관명\s*[:：]\s*([^\n\r]+?)(?:\s*대표자|\s*$)",
        ],
        "양수인 대표자명": [
            r"양수인.*?대표자명\s*[:：]\s*([^\s\n\r]+)",
            r"대표자명\s*[:：]\s*([^\s\n\r]+)",
            r"대표자\s*[:：]\s*([^\s\n\r]+)",
        ],
        "양수인 대표자 주소": [
            r"양수인.*?대표자.*?주소\s*[:：]\s*([^\n\r]+?)(?:\s*대표자.*?연락처|\s*$)",
            r"대표자.*?주소\s*[:：]\s*([^\n\r]+?)(?:\s*대표자.*?연락처|\s*$)",
        ],
        "양수인 대표자 연락처": [
            r"양수인.*?대표자.*?연락처\s*[:：]\s*([0-9\-]+)",
            r"대표자.*?연락처\s*[:：]\s*([0-9\-]+)",
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
        found = False
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                entity_text = match.group(1).strip() if match.groups() else match.group(0).strip()
                entity_text = re.sub(r'\s+', ' ', entity_text)  # 공백 정리
                if entity_text and len(entity_text) > 1 and len(entity_text) < 200:
                    entities.append({
                        'text': entity_text,
                        'label': label
                    })
                    found = True
                    break
            if found:
                break
        
        # 동의여부 기본값
        if label == "동의여부" and not found:
            if any(word in text for word in ["동의", "합의", "승인"]):
                entities.append({
                    'text': "동의",
                    'label': label
                })
    
    return entities

def generate_synthetic_data(existing_data, num_synthetic=50):
    """기존 데이터를 바탕으로 합성 데이터를 생성합니다."""
    synthetic_data = []
    
    # 샘플 데이터 템플릿
    contract_templates = [
        "저작물명: {저작물명}, 대상 저작물 상세정보: {대상 저작물 상세정보}, 양수자 기관명: {양수자 기관명}, 양수자 주소: {양수자 주소}, 양도자 기관명: {양도자 기관명}, 양도자 소속: {양도자 소속}, 양도자 대표주소: {양도자 대표주소}, 양도자 연락처: {양도자 연락처}, 동의여부: {동의여부}, 날짜: {날짜}",
        "본 계약서는 {저작물명}에 관한 저작재산권 양도에 관한 것입니다. 대상 저작물: {대상 저작물 상세정보}. 양수자는 {양수자 기관명} (주소: {양수자 주소})이며, 양도자는 {양도자 기관명} ({양도자 소속} 소속, 주소: {양도자 대표주소}, 연락처: {양도자 연락처})입니다. 양도자는 본 계약에 {동의여부}하며, 계약 체결일은 {날짜}입니다.",
    ]
    
    consent_templates = [
        "본인 {양수인 성명} (주소: {양도인 주소}, 전화번호: {양도인 전화번호})은 {양수인 기관명}의 {양수인 대표자명} 대표자 (주소: {양수인 대표자 주소}, 연락처: {양수인 대표자 연락처})에게 저작권을 양도하는 것에 {동의여부}합니다. 동의날짜: {동의날짜}",
        "저작인접권 양도 동의서. 양도인: {양수인 성명}, 주소: {양도인 주소}, 전화: {양도인 전화번호}. 양수인 기관: {양수인 기관명}, 대표자: {양수인 대표자명}, 대표자 주소: {양수인 대표자 주소}, 대표자 연락처: {양수인 대표자 연락처}. 동의여부: {동의여부}, 동의날짜: {동의날짜}",
    ]
    
    # 샘플 값들
    sample_values = {
        "저작물명": ["디지털 문화유산 사진집", "한국 전통음악 컬렉션", "현대미술 작품집", "한국사 교육 자료", "지역문화 영상자료"],
        "대상 저작물 상세정보": ["문화유산 사진 500장", "전통 국악 녹음 50곡", "현대미술 작품 100점", "한국사 교육영상 20편", "지역문화 다큐멘터리"],
        "양수자 기관명": ["국립박물관", "한국문화정보원", "문화재청", "한국예술종합학교", "지역문화재단"],
        "양수자 주소": ["서울시 용산구", "서울시 마포구", "대전시 서구", "서울시 서초구", "부산시 해운대구"],
        "양도자 기관명": ["디지털문화재단", "전통음악보존회", "현대미술협회", "교육콘텐츠제작소", "지역문화연구소"],
        "양도자 소속": ["문화재청", "문화체육관광부", "교육부", "지방자치단체", "민간단체"],
        "양도자 대표주소": ["대전시 서구", "서울시 강남구", "부산시 중구", "광주시 동구", "대구시 수성구"],
        "양도자 연락처": ["042-123-4567", "02-987-6543", "051-555-1234", "062-777-8888", "053-999-0000"],
        "동의여부": ["동의", "동의함", "합의", "승인"],
        "날짜": ["2025년 8월 5일", "2025년 7월 15일", "2025년 9월 1일", "2025년 6월 20일", "2025년 10월 10일"],
        "양수인 성명": ["김철수", "박영희", "이민수", "정수연", "최영진"],
        "양도인 주소": ["서울시 종로구", "부산시 해운대구", "대구시 중구", "광주시 서구", "대전시 유성구"],
        "양도인 전화번호": ["010-1234-5678", "010-9876-5432", "010-5555-1234", "010-7777-8888", "010-9999-0000"],
        "양수인 대표자명": ["이영희", "홍길동", "김민정", "박준혁", "조은아"],
        "양수인 대표자 주소": ["서울시 용산구", "서울시 강남구", "부산시 중구", "대전시 서구", "광주시 동구"],
        "양수인 대표자 연락처": ["02-123-4567", "02-987-6543", "051-555-1234", "042-777-8888", "062-999-0000"],
        "동의날짜": ["2025년 8월 12일", "2025년 7월 15일", "2025년 9월 5일", "2025년 6월 25일", "2025년 10월 15일"],
    }
    
    # 계약서 합성 데이터 생성
    for i in range(num_synthetic // 2):
        template = random.choice(contract_templates)
        values = {key: random.choice(value_list) for key, value_list in sample_values.items()}
        
        try:
            synthetic_text = template.format(**values)
            entities = extract_enhanced_contract_entities(synthetic_text)
            
            if entities:
                synthetic_data.append({
                    'text': synthetic_text,
                    'entities': str(entities)
                })
        except KeyError:
            continue
    
    # 동의서 합성 데이터 생성
    for i in range(num_synthetic // 2):
        template = random.choice(consent_templates)
        values = {key: random.choice(value_list) for key, value_list in sample_values.items()}
        
        try:
            synthetic_text = template.format(**values)
            entities = extract_enhanced_consent_entities(synthetic_text)
            
            if entities:
                synthetic_data.append({
                    'text': synthetic_text,
                    'entities': str(entities)
                })
        except KeyError:
            continue
    
    return synthetic_data

def main():
    """메인 함수"""
    print("향상된 OCR 데이터 처리 및 합성 데이터 생성...")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 기존 데이터 로드
    try:
        existing_contract = pd.read_csv(os.path.join(OUTPUT_DIR, "train_contract_results.csv"))
        existing_consent = pd.read_csv(os.path.join(OUTPUT_DIR, "train_consent_results.csv"))
        print(f"기존 데이터 로드: 계약서 {len(existing_contract)}개, 동의서 {len(existing_consent)}개")
    except:
        existing_contract = pd.DataFrame()
        existing_consent = pd.DataFrame()
    
    # 합성 데이터 생성
    print("합성 데이터 생성중...")
    synthetic_data = generate_synthetic_data([], 50)
    
    # 계약서와 동의서로 분류
    synthetic_contract = []
    synthetic_consent = []
    
    for item in synthetic_data:
        if any(label in item['entities'] for label in ['저작물명', '양수자 기관명']):
            synthetic_contract.append(item)
        else:
            synthetic_consent.append(item)
    
    # 기존 데이터와 합성 데이터 결합
    all_contract_data = existing_contract.to_dict('records') + synthetic_contract
    all_consent_data = existing_consent.to_dict('records') + synthetic_consent
    
    # CSV 파일로 저장
    if all_contract_data:
        contract_df = pd.DataFrame(all_contract_data)
        contract_file = os.path.join(OUTPUT_DIR, "enhanced_train_contract_results.csv")
        contract_df.to_csv(contract_file, index=False, encoding='utf-8-sig')
        print(f"계약서 학습 데이터 저장: {contract_file} ({len(all_contract_data)}개 샘플)")
    
    if all_consent_data:
        consent_df = pd.DataFrame(all_consent_data)
        consent_file = os.path.join(OUTPUT_DIR, "enhanced_train_consent_results.csv")
        consent_df.to_csv(consent_file, index=False, encoding='utf-8-sig')
        print(f"동의서 학습 데이터 저장: {consent_file} ({len(all_consent_data)}개 샘플)")
    
    print(f"\n총 {len(all_contract_data) + len(all_consent_data)}개의 학습 데이터가 준비되었습니다.")

if __name__ == "__main__":
    main()
