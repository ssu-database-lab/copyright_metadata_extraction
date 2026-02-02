"""
Weak Supervision을 사용하여 실제 문서에서 학습 데이터 자동 생성

사용 방법:
1. 문서 디렉토리에서 텍스트 추출 (OCR 또는 텍스트 파일)
2. 정규식 패턴으로 자동 라벨링
3. 각 라벨별 jsonl 파일에 추가

예시:
    python scripts/generate_training_data_from_documents.py \
        --input_dir data/in/document \
        --output_dir configs/training/ner_labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import re

from module.extractor import text as text_module
try:
    from module.extractor.ner.weak_supervision import WeakSupervisionLabeler  # type: ignore
except Exception:
    WeakSupervisionLabeler = None  # type: ignore
TRAINING_DATA_DIR = Path("configs/training")
from module.extractor import ocr as ocr_module


def load_regex_patterns() -> Dict[str, str]:
    """labels.yaml에서 정규식 패턴 로드"""
    import yaml
    config_path = Path("configs/labels.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    regex_labels = config.get("regular", {}).get("regex_labels", {})
    
    # NER 라벨과 매핑
    ner_label_map = {
        "phone": "phone_number",
        "email": "email",
        "url": "url",
        "date": "date",
    }
    
    patterns = {}
    for regex_label, pattern in regex_labels.items():
        ner_label = ner_label_map.get(regex_label, regex_label)
        patterns[ner_label] = pattern
    
    return patterns


def label_with_keywords(
    tokens: List[str],
    keyword_map: Dict[str, List[str]],
) -> List[str]:
    """
    키워드 기반 라벨링 (person_name, company_name, address 등)
    
    Args:
        tokens: 토큰 리스트
        keyword_map: {"person_name": ["대표", "사장", "이름"], ...}
    
    Returns:
        BIO 라벨 리스트
    """
    labels = ["O"] * len(tokens)
    text = " ".join(tokens).lower()
    
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in text:
                # 키워드 다음에 오는 단어들을 엔티티로 간주
                keyword_tokens = keyword.split()
                for i in range(len(tokens) - len(keyword_tokens) + 1):
                    if " ".join(tokens[i:i+len(keyword_tokens)]).lower() == keyword_lower:
                        # 키워드 다음 1-3개 토큰을 엔티티로 라벨링
                        start = i + len(keyword_tokens)
                        end = min(start + 3, len(tokens))
                        if start < len(tokens):
                            labels[start] = f"B-{label}"
                            for j in range(start + 1, end):
                                if j < len(labels):
                                    labels[j] = f"I-{label}"
                        break
    
    return labels


def extract_text_from_file(file_path: Path) -> Optional[str]:
    """파일에서 텍스트 추출 (OCR 또는 텍스트 파일)"""
    if file_path.suffix.lower() in ['.txt', '.md']:
        # 텍스트 파일
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception:
            return None
    else:
        # OCR 필요 (이미지, PDF 등)
        try:
            pipeline = ocr_module.get_pipeline()
            raw_text, _ = ocr_module.process_file_for_metadata(file_path, use_temp_dir=True)
            return raw_text
        except Exception as e:
            print(f"[경고] {file_path} OCR 실패: {e}")
            return None


def generate_training_data_from_documents(
    input_dir: str,
    output_dir: Optional[str] = None,
    min_tokens: int = 3,
    max_samples_per_file: int = 100,
) -> Dict[str, int]:
    """
    문서 디렉토리에서 학습 데이터 자동 생성
    
    Args:
        input_dir: 입력 문서 디렉토리
        output_dir: 출력 디렉토리 (기본: configs/training/ner_labels)
        min_tokens: 최소 토큰 수 (너무 짧은 샘플 제외)
        max_samples_per_file: 파일당 최대 샘플 수
    
    Returns:
        라벨별 생성된 샘플 수
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"입력 디렉토리가 없습니다: {input_dir}")
    
    output_path = Path(output_dir) if output_dir else TRAINING_DATA_DIR / "ner_labels"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 정규식 패턴 로드
    regex_patterns = load_regex_patterns()
    print(f"로드된 정규식 패턴: {list(regex_patterns.keys())}")
    
    # WeakSupervisionLabeler 초기화
    if WeakSupervisionLabeler is None:
        raise RuntimeError("WeakSupervisionLabeler 모듈이 없습니다. zero-shot 모드에서는 사용하지 않습니다.")
    labeler = WeakSupervisionLabeler(output_dir=str(output_path))
    
    # 통계
    stats: Dict[str, int] = {}
    file_count = 0
    total_samples = 0
    
    # 지원하는 파일 확장자
    supported_extensions = ['.txt', '.md', '.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 모든 파일 처리
    for file_path in input_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() not in supported_extensions:
            continue
        
        file_count += 1
        print(f"\n[{file_count}] 처리 중: {file_path.name}")
        
        # 텍스트 추출
        raw_text = extract_text_from_file(file_path)
        if not raw_text or len(raw_text.strip()) < 10:
            print(f"  → 텍스트 없음 또는 너무 짧음")
            continue
        
        # 토큰화
        struct = text_module.read_text(raw_text)
        sentences = struct.get("sentences", [])
        tokens_list = struct.get("tokens", [])
        
        if not sentences:
            print(f"  → 문장 없음")
            continue
        
        # 문장별로 처리
        samples_per_file = 0
        for sent_idx, sentence in enumerate(sentences):
            if samples_per_file >= max_samples_per_file:
                break
            
            sent_text = sentence.get("text", "")
            if not sent_text:
                continue
            
            # 문장의 토큰 추출
            sent_id = sentence.get("sent_id")
            if sent_id is None:
                continue
            
            sent_tokens = [t.get("text", "") for t in tokens_list 
                          if t.get("sent_id") == sent_id]
            
            if len(sent_tokens) < min_tokens:
                continue
            
            # 정규식으로 라벨링
            regex_labels = labeler.label_with_regex(sent_tokens, regex_patterns)
            
            # 키워드 기반 라벨링 (person_name, company_name, address)
            keyword_patterns = {
                "person_name": ["대표", "사장", "이름", "대표이사", "대표자", "담당자", "작성자"],
                "company_name": ["회사", "기관", "법인", "주식회사", "(주)", "㈜", "기업", "단체"],
                "address": ["주소", "소재지", "위치", "본사", "사무소", "도로명", "지번"],
            }
            keyword_labels = label_with_keywords(sent_tokens, keyword_patterns)
            
            # 두 라벨링 결과 병합 (정규식 우선)
            labels = regex_labels.copy()
            for i, (regex_l, keyword_l) in enumerate(zip(regex_labels, keyword_labels)):
                if regex_l == "O" and keyword_l != "O":
                    labels[i] = keyword_l
            
            # 라벨이 있는지 확인
            has_label = any(l != "O" for l in labels)
            if not has_label:
                continue
            
            # 각 라벨별로 파일에 추가
            label_counts: Dict[str, int] = {}
            for label in labels:
                if label.startswith("B-"):
                    base_label = label[2:]
                    label_counts[base_label] = label_counts.get(base_label, 0) + 1
            
            # 각 라벨별로 별도 샘플로 저장
            for base_label in label_counts.keys():
                sample_id = f"{file_path.stem}_sent{sent_idx}"
                output_file = output_path / f"{base_label}.jsonl"
                
                # 해당 라벨만 포함하는 라벨 리스트 생성 (다른 라벨은 O로)
                label_specific_labels = []
                for i, lbl in enumerate(labels):
                    if lbl.startswith(f"B-{base_label}") or lbl.startswith(f"I-{base_label}"):
                        label_specific_labels.append(lbl)
                    else:
                        label_specific_labels.append("O")
                
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "id": sample_id,
                        "tokens": sent_tokens,
                        "labels": label_specific_labels
                    }, ensure_ascii=False) + "\n")
                
                stats[base_label] = stats.get(base_label, 0) + 1
                total_samples += 1
                samples_per_file += 1
        
        print(f"  → {samples_per_file}개 샘플 생성")
    
    print(f"\n{'='*60}")
    print(f"[완료] 학습 데이터 생성")
    print(f"{'='*60}")
    print(f"  처리된 파일: {file_count}개")
    print(f"  총 생성된 샘플: {total_samples}개")
    print(f"  라벨별 샘플 수:")
    for label, count in sorted(stats.items()):
        print(f"    - {label}: {count}개")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Weak Supervision을 사용하여 문서에서 학습 데이터 자동 생성"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/in/document",
        help="입력 문서 디렉토리 (기본: data/in/document)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 (기본: configs/training/ner_labels)"
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=3,
        help="최소 토큰 수 (기본: 3)"
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=100,
        help="파일당 최대 샘플 수 (기본: 100)"
    )
    
    args = parser.parse_args()
    
    generate_training_data_from_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_tokens=args.min_tokens,
        max_samples_per_file=args.max_samples_per_file,
    )


if __name__ == "__main__":
    main()
