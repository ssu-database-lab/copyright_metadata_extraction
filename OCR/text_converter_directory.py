# png, jpg first convert
# Naver Clova OCR API...! Please Confirm that It Costs Enormous Money!!!

import requests
import uuid
import time
import json
import glob
import os
from pathlib import Path
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리

api_url = 'https://bdyivdm6wb.apigw.ntruss.com/custom/v1/45180/02f46cd018c9da67e2487e81d9f22a44f0d016e28ad7b572b909e41484108b84/general'
secret_key = 'bXBudXBFY25xZXpPVVJTd1FmdHJvaG5QZk5sYWJVTHY='

# 디렉토리 경로 설정
src_root_document = Path("./converted_document")
text_ocr_document = Path("./ocr_document")

def call_naver_ocr(image_path: Path) -> dict:
    request_json = {
        'images': [
            {
                'format': image_path.suffix.lstrip('.').lower(),
                'name': image_path.stem
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    headers = {
      'X-OCR-SECRET': secret_key
    }

    # image opening...
    with image_path.open('rb') as image:
        files = [
          ('file', image)
        ]
        # POST로 Request, File Send.
        response = requests.request("POST", api_url, headers=headers, data = payload, files = files)

    # Error Checking...for Debug
    try:
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"OCR Error : {e}\n{response.text[:500]}")

    return data

# dict Data를 받고, list[str]를 뱉습니다.
def extract_texts(ocr_json: dict) -> list[str]:
    img = ocr_json.get('images')[0]
    if not img:
        return []
    fields = img.get('fields', []) or []
    return [f.get('inferText') for f in fields if isinstance(f, dict) and f.get('inferText')]

# output path 설정
def extract_to_path(image_path: Path, src_root: Path, text_ocr_root: Path) -> Path:
    # relative_path
    relative_path = image_path.resolve().relative_to(src_root.resolve())
    out_path = (text_ocr_root / relative_path).with_suffix('.txt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def save_ocr_output(image_path: Path, src_root: Path, text_ocr_root: Path, add_metadata_header: bool = True) -> Path:
    ocr_json = call_naver_ocr(image_path)
    texts = extract_texts(ocr_json)

    out_path = extract_to_path(image_path, src_root, text_ocr_root)
    with out_path.open('w', encoding='utf-8') as out_file:
        if add_metadata_header:
            # Already Sets image_path's output name through parameter
            out_file.write(f'# {image_path}\n')
        out_file.write(' '.join(texts))
    return out_path

def process_directory(input_dir: Path, src_root: Path, text_ocr_root: Path) -> list[Path]:
    """
    디렉토리 내의 모든 이미지 파일을 OCR 처리합니다.
    
    Args:
        input_dir: 처리할 이미지가 있는 디렉토리 경로
        src_root: 소스 루트 디렉토리 (상대 경로 계산용)
        text_ocr_root: OCR 결과를 저장할 루트 디렉토리
        
    Returns:
        처리된 파일 경로 목록
    """
    # 지원하는 이미지 확장자
    supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
    
    processed_files = []
    total_files = 0
    
    # 지원하는 모든 확장자에 대해 파일 개수 계산
    for ext in supported_extensions:
        pattern = os.path.join(input_dir, '**', ext)
        total_files += len(glob.glob(pattern, recursive=True))
    
    print(f"총 {total_files}개의 이미지 파일을 처리합니다...")
    
    # 모든 이미지 파일 처리
    with tqdm(total=total_files, desc="OCR 처리 중") as pbar:
        for ext in supported_extensions:
            pattern = os.path.join(input_dir, '**', ext)
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    image_path = Path(file_path)
                    out_path = save_ocr_output(image_path, src_root, text_ocr_root)
                    processed_files.append(out_path)
                    pbar.update(1)
                    
                    # API 속도 제한을 피하기 위한 짧은 지연
                    time.sleep(0.1)
                except Exception as e:
                    print(f"파일 처리 중 오류 발생: {file_path} - {str(e)}")
    
    return processed_files

if __name__ == '__main__':
    # 명령줄 인자로 처리할 디렉토리를 받거나 기본값 사용
    import sys
    
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = src_root_document
    
    print(f"디렉토리 처리 시작: {target_dir}")
    processed = process_directory(target_dir, src_root_document, text_ocr_document)
    print(f"처리 완료! {len(processed)}개 파일이 다음 위치에 저장되었습니다: {text_ocr_document}")
