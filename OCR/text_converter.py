# png, jpg first convert
# Naver Clova OCR API...! Please Confirm that It Costs Enormous Money!!!

import requests
import uuid
import time
import json
from pathlib import Path

api_url = 'https://bdyivdm6wb.apigw.ntruss.com/custom/v1/45180/02f46cd018c9da67e2487e81d9f22a44f0d016e28ad7b572b909e41484108b84/general'
secret_key = 'bXBudXBFY25xZXpPVVJTd1FmdHJvaG5QZk5sYWJVTHY='
# test
image_file = Path('./converted_document/test/7.저작물양도계약서/002.png')
# all
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

if __name__ == '__main__':
    saved = save_ocr_output(image_file, src_root_document, text_ocr_document)
    print("saved! : ", saved)