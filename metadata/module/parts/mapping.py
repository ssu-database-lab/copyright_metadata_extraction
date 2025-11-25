"""CSV 헤더 → 영문 라벨 매핑 유틸리티"""
import yaml
from pathlib import Path


def load_header_mapping(config_path='configs/labels.yaml'):
    """헤더 별칭 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('header_aliases', {}), config.get('extra_columns', {})


def korean_to_english(korean_header, header_aliases=None, extra_columns=None):
    """
    한국어 컬럼명 → 영문 라벨 변환
    
    Args:
        korean_header: CSV 헤더명 (예: "순번", "사이트명")
        header_aliases: schema_labels에 포함된 별칭 매핑
        extra_columns: 추가 컬럼 별칭 매핑
    
    Returns:
        영문 라벨 또는 None
    """
    if header_aliases is None or extra_columns is None:
        header_aliases, extra_columns = load_header_mapping()
    
    # schema_labels 우선 검색
    for en_label, ko_list in header_aliases.items():
        if korean_header in ko_list:
            return en_label
    
    # extra_columns 검색
    for en_label, ko_list in extra_columns.items():
        if korean_header in ko_list:
            return en_label
    
    return None


def english_to_korean(english_label, header_aliases=None):
    """
    영문 라벨 → 한국어 컬럼명 (첫 번째 별칭 반환)
    
    Args:
        english_label: 영문 라벨 (예: "seq_number")
        header_aliases: 별칭 매핑
    
    Returns:
        한국어 헤더명 또는 None
    """
    if header_aliases is None:
        header_aliases, _ = load_header_mapping()
    
    ko_list = header_aliases.get(english_label, [])
    return ko_list[0] if ko_list else None


def map_csv_columns(csv_columns, mode='schema_only'):
    """
    CSV 컬럼 리스트를 영문 라벨로 일괄 매핑
    
    Args:
        csv_columns: CSV 헤더 리스트
        mode: 'schema_only' (schema_labels만), 'all' (extra 포함)
    
    Returns:
        dict: {korean_header: english_label}
    """
    header_aliases, extra_columns = load_header_mapping()
    
    mapping = {}
    for col in csv_columns:
        # schema_labels 우선
        en_label = korean_to_english(col, header_aliases, {})
        if en_label:
            mapping[col] = en_label
        elif mode == 'all':
            # extra_columns 포함
            en_label = korean_to_english(col, {}, extra_columns)
            if en_label:
                mapping[col] = en_label
    
    return mapping


def get_label_category(english_label, config=None):
    """
    영문 라벨이 어느 카테고리에 속하는지 반환
    
    Args:
        english_label: 영문 라벨
        config: labels.yaml 설정 (없으면 자동 로드)
    
    Returns:
        'regex', 'datetime', 'numeric', 'ner', 'text' 중 하나 또는 None
    """
    if config is None:
        with open('configs/labels.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # regex는 dict 형태
    if english_label in config.get('regex_labels', {}):
        return 'regex'
    
    # 나머지는 list 형태
    for category in ['datetime', 'numeric', 'ner', 'text']:
        key = f'{category}_labels'
        if english_label in config.get(key, []):
            return category
    
    return None


if __name__ == "__main__":
    # 테스트
    print("한→영 변환 테스트:")
    test_headers = ["순번", "사이트명", "URL", "작성일", "영상"]
    for h in test_headers:
        en = korean_to_english(h)
        cat = get_label_category(en) if en else None
        print(f"  {h:15s} → {en or 'N/A':20s} [{cat or '-'}]")
    
    print("\n영→한 변환 테스트:")
    test_labels = ["seq_number", "site_name", "url", "created_date", "video_count"]
    for label in test_labels:
        ko = english_to_korean(label)
        print(f"  {label:20s} → {ko or 'N/A'}")
