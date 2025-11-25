import pandas as pd
import os
from pathlib import Path
import re


def clean_csv_file(input_file, output_dir, file_type):
    """
    CSV 파일을 정리하여 학습용 데이터로 변환합니다.
    
    Parameters:
    - input_file: 입력 CSV 파일 경로
    - output_dir: 출력 디렉토리
    - file_type: '권리확인' 또는 '권리처리'
    """
    try:
        # CSV 파일 읽기 (헤더 없이)
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        # 첫 번째 컬럼이 모두 NaN인 경우 제거
        if df.iloc[:, 0].isna().all():
            df = df.iloc[:, 1:]
        
        # 실제 헤더 찾기 (보통 2-3행 사이)
        header_row = None
        for i in range(min(5, len(df))):
            row_values = df.iloc[i].astype(str).tolist()
            # '순번'이 포함된 행을 헤더로 간주
            if '순번' in row_values or 'Unnamed' not in str(row_values):
                header_row = i
                break
        
        if header_row is not None:
            # 헤더 설정
            new_columns = df.iloc[header_row].tolist()
            df = df.iloc[header_row + 1:].reset_index(drop=True)
            df.columns = new_columns
            
            # 첫 번째 컬럼이 빈 문자열이면 제거
            if df.columns[0] == '' or pd.isna(df.columns[0]):
                df = df.iloc[:, 1:]
        
        # 빈 행 제거 (모든 값이 NaN이거나 빈 문자열)
        df = df.dropna(how='all')
        df = df[~(df.astype(str).apply(lambda x: x.str.strip() == '').all(axis=1))]
        
        # '검토 불가', '확인불가' 등이 포함된 행 제거
        if file_type == '권리확인':
            if '개방여부' in df.columns:
                df = df[~df['개방여부'].astype(str).str.contains('검토 불가|검토불가', na=False, case=False)]
            if '최종 공공누리 유형' in df.columns or '공공누리유형' in df.columns:
                col_name = '최종 공공누리 유형' if '최종 공공누리 유형' in df.columns else '공공누리유형'
                df = df[~df[col_name].astype(str).str.contains('검토 불가|검토불가', na=False, case=False)]
        
        # 순번 컬럼이 숫자가 아닌 행 제거 (헤더 중복 등)
        if '순번' in df.columns:
            df = df[pd.to_numeric(df['순번'], errors='coerce').notna()]
        
        # 결측치 표준화 (-, 빈칸 등을 NaN으로)
        df = df.replace(['-', '', ' ', 'nan', 'NaN'], pd.NA)
        
        # 인덱스 리셋
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"오류 발생 - {input_file}: {str(e)}")
        return None


def process_all_csv():
    """
    모든 CSV 파일을 처리하여 training_csv 폴더에 저장합니다.
    """
    input_dir = Path('data/in/csv')
    output_dir = Path('data/in/training_csv')
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 권리확인과 권리처리 폴더 생성
    (output_dir / '01_권리확인').mkdir(exist_ok=True)
    (output_dir / '02_권리처리').mkdir(exist_ok=True)
    
    # CSV 파일 찾기
    csv_files = list(input_dir.rglob('*.csv'))
    
    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.\n")
    
    # 파일 분류 및 처리
    processed_count = {'권리확인': 0, '권리처리': 0}
    skipped_files = []
    
    for csv_file in sorted(csv_files):
        relative_path = csv_file.relative_to(input_dir)
        
        # 파일 유형 판단
        if '권리확인' in str(csv_file):
            file_type = '권리확인'
            output_subdir = output_dir / '01_권리확인'
        elif '권리처리' in str(csv_file):
            file_type = '권리처리'
            output_subdir = output_dir / '02_권리처리'
        else:
            skipped_files.append(str(relative_path))
            continue
        
        # CSV 정리
        cleaned_df = clean_csv_file(csv_file, output_subdir, file_type)
        
        if cleaned_df is not None and len(cleaned_df) > 0:
            # 출력 파일명 생성 (연도_기관명_유형.csv)
            year = '2023' if '2023' in str(csv_file) else '2024'
            filename = csv_file.stem  # 확장자 제외한 파일명
            
            # 파일명 정리 (너무 길면 축약)
            clean_filename = re.sub(r'_권리확인.*?_|_권리처리.*?_', '_', filename)
            output_filename = f"{year}_{clean_filename}.csv"
            
            # 저장
            output_path = output_subdir / output_filename
            cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            processed_count[file_type] += 1
            print(f"✓ {relative_path}")
            print(f"  → {output_path.relative_to(output_dir)}")
            print(f"  행 수: {len(cleaned_df)}, 컬럼 수: {len(cleaned_df.columns)}\n")
        else:
            skipped_files.append(str(relative_path))
    
    # 결과 요약
    print("\n" + "="*80)
    print("전처리 완료!")
    print(f"  권리확인 파일: {processed_count['권리확인']}개")
    print(f"  권리처리 파일: {processed_count['권리처리']}개")
    print(f"  건너뛴 파일: {len(skipped_files)}개")
    print(f"  저장 위치: {output_dir}")
    
    if skipped_files:
        print("\n건너뛴 파일:")
        for f in skipped_files[:10]:  # 처음 10개만 표시
            print(f"  - {f}")
        if len(skipped_files) > 10:
            print(f"  ... 외 {len(skipped_files) - 10}개")
    
    # 최종 데이터셋 통계
    print("\n" + "="*80)
    print("생성된 데이터셋:")
    
    for subdir_name, type_name in [('01_권리확인', '권리확인'), ('02_권리처리', '권리처리')]:
        subdir = output_dir / subdir_name
        files = sorted(list(subdir.glob('*.csv')))
        
        if files:
            print(f"\n[{type_name}] - {len(files)}개 파일")
            total_rows = 0
            
            for idx, file in enumerate(files, 1):
                df = pd.read_csv(file, encoding='utf-8-sig')
                total_rows += len(df)
                print(f"  {idx:3d}. {file.name:60s} ({len(df):5d} rows)")
            
            print(f"\n  총 데이터 행 수: {total_rows:,}")


if __name__ == "__main__":
    process_all_csv()
