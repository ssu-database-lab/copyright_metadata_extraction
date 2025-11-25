import pandas as pd
import os
from pathlib import Path


def excel_to_csv(input_base_dir='data/in/excel', output_base_dir='data/in/csv'):
    """
    Excel 파일들을 CSV로 변환합니다.
    각 Excel 파일의 시트마다 별도의 CSV 파일로 저장됩니다.
    
    Parameters:
    - input_base_dir: Excel 파일이 있는 기본 디렉토리
    - output_base_dir: CSV 파일을 저장할 기본 디렉토리
    """
    input_path = Path(input_base_dir)
    output_path = Path(output_base_dir)
    
    # output 디렉토리가 없으면 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 처리된 파일 개수 추적
    total_files = 0
    total_sheets = 0
    
    # Excel 파일 찾기 (.xlsx, .xls)
    excel_files = list(input_path.rglob('*.xlsx')) + list(input_path.rglob('*.xls'))
    
    if not excel_files:
        print(f"'{input_base_dir}' 폴더에 Excel 파일이 없습니다.")
        return
    
    print(f"총 {len(excel_files)}개의 Excel 파일을 찾았습니다.\n")
    
    for excel_file in excel_files:
        try:
            # 상대 경로 구조 유지
            relative_path = excel_file.relative_to(input_path)
            
            # Excel 파일 읽기
            excel_data = pd.ExcelFile(excel_file)
            sheet_names = excel_data.sheet_names
            
            print(f"처리중: {relative_path}")
            print(f"  시트 개수: {len(sheet_names)}")
            
            # 각 시트를 CSV로 변환
            for sheet_name in sheet_names:
                # 시트 데이터 읽기
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # 출력 파일 경로 생성 (폴더 구조 유지)
                # 예: 2023/01. 권리확인 목록/파일명_시트명.csv
                output_dir = output_path / relative_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일명 생성 (원본파일명_시트명.csv)
                base_filename = excel_file.stem  # 확장자 제외한 파일명
                safe_sheet_name = str(sheet_name).replace('/', '_').replace('\\', '_')  # 시트명에서 경로 구분자 제거
                csv_filename = f"{base_filename}_{safe_sheet_name}.csv"
                csv_path = output_dir / csv_filename
                
                # CSV로 저장 (UTF-8 BOM 인코딩으로 한글 호환성 향상)
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


if __name__ == "__main__":
    # Excel 파일을 CSV로 변환
    excel_to_csv()

