"""CSV 데이터 검증 및 매핑 도구"""
# import
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from library.mapping import map_csv_columns, get_label_category
from library.csv import load_config


# -----------------------------------------------------------------------------
# 변수 선언 (없음)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# function 선언 (export)
# -----------------------------------------------------------------------------

def validate_csv_file(csv_path, verbose=True):
    """
    CSV 파일의 헤더를 labels.yaml과 비교 검증
    
    Args:
        csv_path: CSV 파일 경로
        verbose: 상세 출력 여부
    
    Returns:
        dict: 매핑 결과 통계
    """
    # CSV 읽기
    df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=0)
    headers = df.columns.tolist()
    
    # 매핑
    mapping = map_csv_columns(headers, mode='all')
    config = load_config()
    
    # 통계
    stats = {
        'total_columns': len(headers),
        'mapped_schema': 0,
        'mapped_extra': 0,
        'unmapped': [],
        'by_category': {'regex': 0, 'datetime': 0, 'numeric': 0, 'ner': 0, 'text': 0}
    }
    
    for header in headers:
        en_label = mapping.get(header)
        if en_label:
            category = get_label_category(en_label, config)
            if category:
                stats['mapped_schema'] += 1
                stats['by_category'][category] += 1
            else:
                stats['mapped_extra'] += 1
        else:
            stats['unmapped'].append(header)
    
    if verbose:
        print(f"\n📄 {Path(csv_path).name}")
        print(f"   총 컬럼: {stats['total_columns']}")
        print(f"   Schema 매핑: {stats['mapped_schema']}")
        print(f"   Extra 매핑: {stats['mapped_extra']}")
        print(f"   미매핑: {len(stats['unmapped'])}")
        
        if stats['by_category']:
            print(f"\n   카테고리별:")
            for cat, cnt in stats['by_category'].items():
                if cnt > 0:
                    print(f"     - {cat:10s}: {cnt}")
        
        if stats['unmapped']:
            print(f"\n   미매핑 컬럼:")
            for h in stats['unmapped']:
                print(f"     - {h}")
    
    return stats


def validate_dataset(data_dir='data/in/training_csv'):
    """
    전체 데이터셋 검증
    
    Args:
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        dict: 전체 통계
    """
    root = Path(data_dir)
    csv_files = list(root.rglob('*.csv'))
    
    total_stats = {
        'total_files': len(csv_files),
        'total_columns': 0,
        'mapped_schema': 0,
        'mapped_extra': 0,
        'unmapped_set': set(),
        'by_category': {'regex': 0, 'datetime': 0, 'numeric': 0, 'ner': 0, 'text': 0}
    }
    
    print(f"🔍 데이터셋 검증 시작 ({len(csv_files)} 파일)")
    print("=" * 60)
    
    for csv_file in csv_files:
        stats = validate_csv_file(csv_file, verbose=False)
        total_stats['total_columns'] += stats['total_columns']
        total_stats['mapped_schema'] += stats['mapped_schema']
        total_stats['mapped_extra'] += stats['mapped_extra']
        total_stats['unmapped_set'].update(stats['unmapped'])
        
        for cat in total_stats['by_category']:
            total_stats['by_category'][cat] += stats['by_category'][cat]
    
    print("\n📊 전체 통계:")
    print(f"   총 파일: {total_stats['total_files']}")
    print(f"   총 컬럼 출현: {total_stats['total_columns']}")
    print(f"   Schema 매핑: {total_stats['mapped_schema']}")
    print(f"   Extra 매핑: {total_stats['mapped_extra']}")
    print(f"   고유 미매핑: {len(total_stats['unmapped_set'])}")
    
    print(f"\n   카테고리별 컬럼 수:")
    for cat, cnt in sorted(total_stats['by_category'].items()):
        print(f"     - {cat:10s}: {cnt:4d}")
    
    if total_stats['unmapped_set']:
        print(f"\n   미매핑 컬럼 목록:")
        for h in sorted(total_stats['unmapped_set']):
            print(f"     - {h}")
    
    return total_stats


def export_mapping_table(output_path='configs/column_mapping.csv'):
    """
    한국어↔영문 매핑 테이블을 CSV로 내보내기
    
    Args:
        output_path: 출력 파일 경로
    """
    from library.mapping import load_header_mapping, get_label_category
    
    header_aliases, extra_columns = load_header_mapping()
    config = load_config()
    
    rows = []
    
    # Schema labels
    for en_label, ko_list in header_aliases.items():
        category = get_label_category(en_label, config)
        for ko_header in ko_list:
            rows.append({
                'korean_header': ko_header,
                'english_label': en_label,
                'category': category or 'undefined',
                'type': 'schema'
            })
    
    # Extra columns
    for en_label, ko_list in extra_columns.items():
        for ko_header in ko_list:
            rows.append({
                'korean_header': ko_header,
                'english_label': en_label,
                'category': 'extra',
                'type': 'reference'
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['type', 'category', 'english_label'])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 매핑 테이블 저장: {output_path}")
    print(f"   총 {len(rows)}개 매핑")


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 단일 파일 검증
        validate_csv_file(sys.argv[1])
    else:
        # 전체 데이터셋 검증
        validate_dataset()
        
        # 매핑 테이블 내보내기
        print("\n" + "=" * 60)
        export_mapping_table()
