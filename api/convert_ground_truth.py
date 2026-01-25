#!/usr/bin/env python3
"""
Ground Truth JSON을 NER 훈련 데이터로 변환
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def load_ground_truth(json_path: Path) -> Dict:
    """Ground Truth JSON 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_ocr_text(txt_path: Path) -> str:
    """OCR 텍스트 로드"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

def find_entity_in_text(text: str, entity: str) -> List[Tuple[int, int]]:
    """텍스트에서 엔티티의 모든 위치 찾기"""
    positions = []
    # 정확한 매칭 시도
    start = 0
    while True:
        pos = text.find(entity, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(entity)))
        start = pos + 1
    
    # 못 찾으면 공백 제거하고 재시도
    if not positions:
        entity_no_space = entity.replace(' ', '')
        text_no_space = text.replace(' ', '')
        start = 0
        while True:
            pos = text_no_space.find(entity_no_space, start)
            if pos == -1:
                break
            # 원본 텍스트에서 위치 계산 (근사값)
            positions.append((pos, pos + len(entity_no_space)))
            start = pos + 1
    
    return positions

def create_bio_labels(text: str, entities: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """텍스트와 엔티티로부터 BIO 라벨 생성"""
    # 문자 단위로 분리 (공백 제외)
    chars = [ch for ch in text if ch != ' ' and ch != '\n' and ch != '\t']
    labels = ['O'] * len(chars)
    
    # 각 엔티티 타입별로 처리
    for entity_type, entity_list in entities.items():
        if entity_type in ['N/A']:
            continue
        
        for entity in entity_list:
            if not entity or entity == 'N/A':
                continue
            
            # 엔티티 문자 (공백 제거)
            entity_chars = [ch for ch in entity if ch != ' ' and ch != '\n' and ch != '\t']
            if len(entity_chars) == 0:
                continue
            
            # 텍스트에서 엔티티 찾기
            text_no_space = ''.join(chars)
            entity_no_space = ''.join(entity_chars)
            
            start = 0
            while True:
                pos = text_no_space.find(entity_no_space, start)
                if pos == -1:
                    break
                
                # BIO 태그 적용
                if pos < len(labels):
                    # 이미 태깅된 위치는 건너뛰기
                    if labels[pos] == 'O':
                        labels[pos] = f'B-{entity_type}'
                        for i in range(pos + 1, min(pos + len(entity_chars), len(labels))):
                            if labels[i] == 'O':
                                labels[i] = f'I-{entity_type}'
                
                start = pos + 1
    
    return chars, labels

def convert_ground_truth_to_train(gt_json_path: Path, ocr_txt_path: Path, output_path: Path):
    """Ground Truth를 훈련 데이터로 변환"""
    gt_data = load_ground_truth(gt_json_path)
    ocr_text = load_ocr_text(ocr_txt_path)
    
    # OCR 텍스트를 청크로 분리 (페이지별 또는 문단별)
    # "--- XXX.png ---" 같은 페이지 구분자가 있으면 분리
    pages = re.split(r'\n---\s+\d+\.png\s+---\n', ocr_text)
    
    train_samples = []
    
    for page_idx, page_text in enumerate(pages):
        if not page_text.strip():
            continue
        
        # 각 페이지를 문장 단위로 분리 (최대 200자)
        lines = page_text.split('\n')
        current_chunk = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if len(current_chunk) + len(line) > 200:
                if current_chunk:
                    # BIO 라벨 생성
                    chars, labels = create_bio_labels(current_chunk, gt_data)
                    if len(chars) > 0:
                        train_samples.append({
                            "tokens": chars,
                            "labels": labels
                        })
                current_chunk = line
            else:
                current_chunk += " " + line if current_chunk else line
        
        # 마지막 청크 처리
        if current_chunk:
            chars, labels = create_bio_labels(current_chunk, gt_data)
            if len(chars) > 0:
                train_samples.append({
                    "tokens": chars,
                    "labels": labels
                })
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 생성된 샘플: {len(train_samples)}개")
    print(f"   저장 위치: {output_path}")
    
    # 통계 출력
    total_entities = 0
    for sample in train_samples:
        total_entities += sum(1 for l in sample['labels'] if l.startswith('B-'))
    print(f"   총 엔티티: {total_entities}개")

if __name__ == "__main__":
    # Ground Truth 파일 목록
    gt_dir = Path("data/ground_truth")
    ocr_dir = Path("data/out/ocr/google")
    output_dir = Path("data/in/real_document_train")
    
    # 7.저작물양도계약서
    convert_ground_truth_to_train(
        gt_dir / "7.저작물양도계약서.json",
        ocr_dir / "7.저작물양도계약서" / "7.저작물양도계약서.txt",
        output_dir / "7.저작물양도계약서.json"
    )
    
    print("\n✅ 변환 완료!")
