"""CSV 및 Excel 처리 유틸리티"""
import pandas as pd
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from module.parts import directory

# ---------- Conversion (Excel -> CSV) ----------

def convert_excel_to_csv(
    input_base_dir: str = 'data/in/excel',
    output_base_dir: str = 'data/in/csv'
) -> None:
    """
    Excel 파일들을 CSV로 변환합니다.
    각 Excel 파일의 시트마다 별도의 CSV 파일로 저장됩니다.
    """
    input_path = Path(input_base_dir)
    output_path = directory.ensure_outdir(output_base_dir)
    
    excel_files = list(directory.iter_excel_files(input_path))
    
    if not excel_files:
        print(f"'{input_base_dir}' 폴더에 Excel 파일이 없습니다.")
        return
    
    print(f"총 {len(excel_files)}개의 Excel 파일을 찾았습니다.\n")
    
    total_files = 0
    total_sheets = 0
    
    for excel_file in excel_files:
        try:
            relative_path = excel_file.relative_to(input_path)
            excel_data = pd.ExcelFile(excel_file)
            sheet_names = excel_data.sheet_names
            
            print(f"처리중: {relative_path}")
            print(f"  시트 개수: {len(sheet_names)}")
            
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # 출력 경로 (폴더 구조 유지)
                output_dir = output_path / relative_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일명 생성
                base_filename = excel_file.stem
                safe_sheet_name = str(sheet_name).replace('/', '_').replace('\\', '_')
                csv_filename = f"{base_filename}_{safe_sheet_name}.csv"
                csv_path = output_dir / csv_filename
                
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                print(f"    → {csv_path.relative_to(output_path)}")
                total_sheets += 1
            
            total_files += 1
            print()
            
        except Exception as e:
            print(f"오류 발생 - {excel_file}: {str(e)}\n")
            continue
    
    print(f"변환 완료!")
    print(f"  파일: {total_files}개")
    print(f"  시트: {total_sheets}개")
    print(f"  저장 위치: {output_base_dir}")


# ---------- Preprocessing (Clean CSV) ----------

def _clean_csv_dataframe(input_file: Path, file_type: str) -> Optional[pd.DataFrame]:
    """단일 CSV 파일을 정리하여 DataFrame으로 반환"""
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        # 첫 번째 컬럼이 모두 NaN인 경우 제거
        if not df.empty and df.iloc[:, 0].isna().all():
            df = df.iloc[:, 1:]
        
        # 실제 헤더 찾기
        header_row = None
        for i in range(min(5, len(df))):
            row_values = df.iloc[i].astype(str).tolist()
            if '순번' in row_values or 'Unnamed' not in str(row_values):
                header_row = i
                break
        
        if header_row is not None:
            new_columns = df.iloc[header_row].tolist()
            df = df.iloc[header_row + 1:].reset_index(drop=True)
            df.columns = new_columns
            
            if len(df.columns) > 0 and (df.columns[0] == '' or pd.isna(df.columns[0])):
                df = df.iloc[:, 1:]
        
        # 빈 행 제거
        df = df.dropna(how='all')
        if not df.empty:
            df = df[~(df.astype(str).apply(lambda x: x.str.strip() == '').all(axis=1))]
        
        # '검토 불가' 등 제거 (권리확인용)
        if file_type == '권리확인':
            for col in ['개방여부', '최종 공공누리 유형', '공공누리유형']:
                if col in df.columns:
                    df = df[~df[col].astype(str).str.contains('검토 불가|검토불가', na=False, case=False)]
        
        # 순번 정리
        if '순번' in df.columns:
            df = df[pd.to_numeric(df['순번'], errors='coerce').notna()]
        
        # 결측치 표준화
        df = df.replace(['-', '', ' ', 'nan', 'NaN'], pd.NA)
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"오류 발생 - {input_file}: {str(e)}")
        return None

def preprocess_csv_dataset(
    input_dir: str = 'data/in/csv',
    output_dir: str = 'data/in/training_csv'
) -> None:
    """
    모든 CSV 파일을 전처리하여 학습용 데이터셋 폴더로 저장합니다.
    """
    input_path = Path(input_dir)
    output_path = directory.ensure_outdir(output_dir)
    
    # 하위 폴더 생성
    (output_path / '01_권리확인').mkdir(exist_ok=True)
    (output_path / '02_권리처리').mkdir(exist_ok=True)
    
    csv_files = list(directory.iter_csv_files(input_path))
    print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.\n")
    
    processed_count = {'권리확인': 0, '권리처리': 0}
    skipped_files = []
    
    for csv_file in sorted(csv_files):
        relative_path = csv_file.relative_to(input_path)
        
        # 파일 유형 판단
        str_path = str(csv_file)
        if '권리확인' in str_path:
            file_type = '권리확인'
            output_subdir = output_path / '01_권리확인'
        elif '권리처리' in str_path:
            file_type = '권리처리'
            output_subdir = output_path / '02_권리처리'
        else:
            skipped_files.append(str(relative_path))
            continue
        
        # 정리 수행
        cleaned_df = _clean_csv_dataframe(csv_file, file_type)
        
        if cleaned_df is not None and len(cleaned_df) > 0:
            year = '2023' if '2023' in str_path else '2024'
            clean_filename = re.sub(r'_권리확인.*?_|_권리처리.*?_', '_', csv_file.stem)
            output_filename = f"{year}_{clean_filename}.csv"
            
            save_path = output_subdir / output_filename
            cleaned_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            processed_count[file_type] += 1
            print(f"✓ {relative_path} -> {save_path.relative_to(output_path)}")
        else:
            skipped_files.append(str(relative_path))
            
    print("\n" + "="*80)
    print(f"전처리 완료! 확인: {processed_count['권리확인']}, 처리: {processed_count['권리처리']}, 건너뜀: {len(skipped_files)}")
    print(f"저장 위치: {output_dir}")
