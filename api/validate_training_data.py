#!/usr/bin/env python3
"""
훈련 데이터 검증 스크립트
- BIO 태그 형식 오류 검출
- 엔티티 타입과 값의 일치성 검증
- 토큰-라벨 불일치 검출
- 비정상적인 패턴 발견
"""
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# 엔티티 타입별 검증 패턴
ENTITY_VALIDATION_PATTERNS = {
    'PHONE': r'^\d{2,3}-\d{3,4}-\d{4}$',  # 전화번호 형식
    'DATE': r'^\d{4}\.\d{1,2}\.\d{1,2}',  # 날짜 형식
    'EMAIL': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # 이메일
    'ID_NUM': r'^\d{6}-\d{7}$',  # 주민등록번호
    'URL': r'^https?://',  # URL
}

# 엔티티 타입별 예상 키워드
ENTITY_KEYWORDS = {
    'RIGHT_INFO': ['권', '권리', '저작', '재산권', '인격권', '인접권', '복제', '배포', '공연', '전시', '송신'],
    'LAW_REFERENCE': ['법', '조', '항', '저작권법', '공공데이터', '법률'],
    'CONTRACT_TYPE': ['계약', '동의서', '협약', '합의서'],
    'POSITION': ['이사', '장', '원', '팀장', '과장', '부장', '실장'],
    'COMPANY': ['원', '청', '단', '관', '회', '사', '주식회사', '재단', '위원회'],
    'PROJECT_NAME': ['사업', '프로젝트', '과제', '구축', '개발', '제작'],
}

class DataValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def load_bio_file(self, filepath: Path) -> List[List[Tuple[str, str]]]:
        """BIO 형식 파일 로드"""
        samples = []
        current_sample = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line:
                    if current_sample:
                        samples.append(current_sample)
                        current_sample = []
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    self.errors.append(f"Line {line_num}: 잘못된 형식 - '{line}'")
                    continue
                
                token, label = parts
                current_sample.append((token, label, line_num))
        
        if current_sample:
            samples.append(current_sample)
        
        return samples
    
    def validate_bio_sequence(self, sample: List[Tuple[str, str, int]]) -> List[str]:
        """BIO 태그 시퀀스 검증"""
        errors = []
        prev_label = 'O'
        
        for i, (token, label, line_num) in enumerate(sample):
            # 라벨 형식 검증
            if label != 'O':
                if not (label.startswith('B-') or label.startswith('I-')):
                    errors.append(f"Line {line_num}: 잘못된 라벨 형식 '{label}'")
                    continue
                
                # I- 태그가 B- 없이 시작하는지 검증
                if label.startswith('I-'):
                    entity_type = label[2:]
                    expected_b = f'B-{entity_type}'
                    expected_i = f'I-{entity_type}'
                    
                    if prev_label not in [expected_b, expected_i]:
                        errors.append(f"Line {line_num}: I-{entity_type}가 B-{entity_type} 없이 시작됨")
            
            prev_label = label
        
        return errors
    
    def extract_entities(self, sample: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
        """샘플에서 엔티티 추출"""
        entities = []
        current_entity = []
        current_type = None
        start_line = None
        
        for token, label, line_num in sample:
            if label.startswith('B-'):
                if current_entity:
                    entity_value = ''.join(current_entity)
                    entities.append((current_type, entity_value, start_line))
                
                current_type = label[2:]
                current_entity = [token]
                start_line = line_num
                
            elif label.startswith('I-'):
                if current_entity:
                    current_entity.append(token)
                    
            else:  # O
                if current_entity:
                    entity_value = ''.join(current_entity)
                    entities.append((current_type, entity_value, start_line))
                    current_entity = []
                    current_type = None
        
        if current_entity:
            entity_value = ''.join(current_entity)
            entities.append((current_type, entity_value, start_line))
        
        return entities
    
    def validate_entity_value(self, entity_type: str, entity_value: str, line_num: int) -> List[str]:
        """엔티티 값 검증"""
        errors = []
        
        # 패턴 기반 검증
        if entity_type in ENTITY_VALIDATION_PATTERNS:
            pattern = ENTITY_VALIDATION_PATTERNS[entity_type]
            if not re.match(pattern, entity_value):
                errors.append(f"Line {line_num}: {entity_type} 형식 오류 - '{entity_value}'")
        
        # 키워드 기반 검증
        if entity_type in ENTITY_KEYWORDS:
            keywords = ENTITY_KEYWORDS[entity_type]
            if not any(kw in entity_value for kw in keywords):
                self.warnings.append(f"Line {line_num}: {entity_type}에 예상 키워드 없음 - '{entity_value}'")
        
        # 길이 검증
        if len(entity_value) < 2:
            self.warnings.append(f"Line {line_num}: {entity_type} 값이 너무 짧음 - '{entity_value}'")
        
        if len(entity_value) > 200:
            errors.append(f"Line {line_num}: {entity_type} 값이 너무 김 - '{entity_value[:50]}...'")
        
        # 특수 문자만으로 구성된 경우
        if re.match(r'^[^\w\s가-힣]+$', entity_value):
            errors.append(f"Line {line_num}: {entity_type}이 특수문자만으로 구성됨 - '{entity_value}'")
        
        return errors
    
    def validate_file(self, filepath: Path) -> Dict:
        """파일 전체 검증"""
        print(f"\n검증 중: {filepath.name}")
        print("=" * 80)
        
        samples = self.load_bio_file(filepath)
        print(f"총 샘플 수: {len(samples):,}")
        
        total_tokens = 0
        entity_counts = defaultdict(int)
        entity_examples = defaultdict(list)
        
        for sample_idx, sample in enumerate(samples, 1):
            total_tokens += len(sample)
            
            # BIO 시퀀스 검증
            seq_errors = self.validate_bio_sequence(sample)
            self.errors.extend(seq_errors)
            
            # 엔티티 추출 및 검증
            entities = self.extract_entities(sample)
            
            for entity_type, entity_value, line_num in entities:
                entity_counts[entity_type] += 1
                
                # 예제 수집 (각 타입별 최대 5개)
                if len(entity_examples[entity_type]) < 5:
                    entity_examples[entity_type].append(entity_value)
                
                # 엔티티 값 검증
                val_errors = self.validate_entity_value(entity_type, entity_value, line_num)
                self.errors.extend(val_errors)
            
            if (sample_idx % 1000 == 0):
                print(f"  진행: {sample_idx:,} / {len(samples):,} 샘플")
        
        print(f"\n총 토큰 수: {total_tokens:,}")
        print(f"\n엔티티 타입별 통계:")
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count:,}개")
            print(f"    예시: {', '.join(entity_examples[entity_type][:3])}")
        
        return {
            'samples': len(samples),
            'tokens': total_tokens,
            'entities': dict(entity_counts),
            'examples': dict(entity_examples)
        }
    
    def check_duplicates(self, samples: List[List[Tuple[str, str, int]]]) -> List[str]:
        """중복 샘플 검출"""
        duplicates = []
        seen = set()
        
        for sample in samples:
            sample_str = ''.join([token for token, _, _ in sample])
            if sample_str in seen:
                duplicates.append(sample_str[:100])
            else:
                seen.add(sample_str)
        
        return duplicates
    
    def print_summary(self):
        """검증 결과 요약"""
        print("\n" + "=" * 80)
        print("검증 결과 요약")
        print("=" * 80)
        
        print(f"\n❌ 오류: {len(self.errors)}개")
        if self.errors:
            print("\n주요 오류 (최대 20개):")
            for error in self.errors[:20]:
                print(f"  - {error}")
            
            if len(self.errors) > 20:
                print(f"  ... 외 {len(self.errors) - 20}개")
        
        print(f"\n⚠️  경고: {len(self.warnings)}개")
        if self.warnings:
            print("\n주요 경고 (최대 10개):")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            
            if len(self.warnings) > 10:
                print(f"  ... 외 {len(self.warnings) - 10}개")
        
        if not self.errors and not self.warnings:
            print("\n✅ 오류 및 경고 없음! 데이터가 깨끗합니다.")

def main():
    print("=" * 80)
    print("훈련 데이터 검증 시작")
    print("=" * 80)
    
    validator = DataValidator()
    
    files_to_validate = [
        Path("data/in/realistic_train.txt"),
        Path("data/in/real_document_train.txt"),
    ]
    
    all_stats = {}
    
    for filepath in files_to_validate:
        if filepath.exists():
            stats = validator.validate_file(filepath)
            all_stats[filepath.name] = stats
        else:
            print(f"\n⚠️  파일을 찾을 수 없음: {filepath}")
    
    # 최종 요약
    validator.print_summary()
    
    # 통계 비교
    print("\n" + "=" * 80)
    print("파일별 통계 비교")
    print("=" * 80)
    
    for filename, stats in all_stats.items():
        print(f"\n{filename}:")
        print(f"  샘플: {stats['samples']:,}개")
        print(f"  토큰: {stats['tokens']:,}개")
        print(f"  엔티티: {sum(stats['entities'].values()):,}개")
    
    print("\n✅ 검증 완료!")

if __name__ == "__main__":
    main()
