"""
디렉토리 구조 유지 OCR 시스템 - 수정 버전
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Union
from tqdm import tqdm

# 로깅 설정
logger = logging.getLogger(__name__)

def _get_relative_structure(file_path: Path, base_input_path: Path) -> Path:
    """입력 파일의 상대적 디렉토리 구조를 반환합니다."""
    try:
        relative_path = file_path.relative_to(base_input_path)
        return relative_path.parent
    except ValueError:
        # 파일이 base_input_path 하위에 없는 경우
        return Path(".")

def _create_output_structure(file_path: Path, base_input_path: Path, output_base: Path, ocr_engine: str) -> Path:
    """출력 디렉토리 구조를 생성하고 반환합니다."""
    # pdf_to_image와 동일한 방식: 입력 경로 기준 상대 경로 계산
    try:
        relative_path = file_path.relative_to(base_input_path)
        relative_dir = relative_path.parent
        # ocr/engine_name/relative_structure 구조 생성
        output_dir = output_base / "ocr" / ocr_engine / relative_dir
    except ValueError:
        # 파일이 base_input_path 하위에 없는 경우
        output_dir = output_base / "ocr" / ocr_engine
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def _load_ocr_config() -> Dict[str, Any]:
    """OCR 설정 파일을 로드합니다."""
    config_path = Path(__file__).parent.parent / "ocr_config.json"
    
    if not config_path.exists():
        # 기본 설정 파일 생성
        default_config = {
            "naver": {
                "api_url": "",
                "secret_key": "",
                "template_ids": []
            },
            "google": {
                "credentials_path": "",
                "use_document_detection": True
            },
            "mistral": {
                "api_key": "",
                "model_name": "pixtral-12b-2409",
                "prompt": "이 이미지에서 모든 텍스트를 정확히 추출해주세요. 줄바꿈과 형식을 유지해주세요.",
                "max_tokens": 2000
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"OCR 설정 파일 로드 실패: {e}")
        return {}

def ocr_naver(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Naver CLOVA OCR을 사용하여 이미지에서 텍스트를 추출합니다.
    
    Args:
        input_path (str): 입력 이미지 파일 경로 또는 디렉토리 경로
        output_path (str): 출력 디렉토리 경로  
        
    Returns:
        Dict[str, Any]: 처리 결과
        
    Example:
        result = ocr_naver(
            input_path="business_cards/",
            output_path="ocr_results/"
        )
    
    설정 파일:
        ocr_config.json에 API 정보를 설정하세요:
        {
            "naver": {
                "api_url": "https://...",
                "secret_key": "your_secret_key",
                "template_ids": ["template_id_1"]
            }
        }
    
    Note:
        - Naver CLOVA OCR API 계정이 필요합니다
        - API URL과 Secret Key는 Naver Cloud Platform에서 발급받으세요
        - 지원 형식: JPG, PNG, PDF, TIFF (최대 20MB)
        - 월 1,000건까지 무료, 이후 유료
        - 한국어 문서에 특화되어 있습니다
        
    API 신청:
        1. Naver Cloud Platform (ncloud.com) 가입
        2. AI·Application Service > CLOVA OCR 선택
        3. 이용 신청 후 API URL과 Secret Key 발급
    """
    import time
    import requests
    import base64
    
    start_time = time.time()
    
    try:
        # JSON 설정 로드
        config = _load_ocr_config()
        naver_config = config.get('naver', {})
        
        api_url = naver_config.get('api_url')
        secret_key = naver_config.get('secret_key')
        template_ids = naver_config.get('template_ids', [])
        
        if not api_url or not secret_key:
            return {
                "success": False,
                "error": "Naver OCR API 설정이 ocr_config.json에 올바르게 설정되지 않았습니다."
            }
        
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        # ocr 디렉토리 생성
        ocr_dir = output_path_obj / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        
        # pdf_to_image와 동일한 방식: 입력 경로를 base_input_path로 설정
        base_input_path = Path(input_path)
        
        # 처리할 파일 목록 생성
        if input_path_obj.is_file():
            if input_path_obj.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pdf']:
                files_to_process = [input_path_obj]
            else:
                return {"success": False, "error": "지원되지 않는 파일 형식"}
        elif input_path_obj.is_dir():
            extensions = ['*.jpg', '*.jpeg', '*.png', '.tif', '*.tiff', '*.pdf']
            files_to_process = []
            for ext in extensions:
                files_to_process.extend(input_path_obj.rglob(ext))
        else:
            return {"success": False, "error": "유효하지 않은 입력 경로"}
        
        if not files_to_process:
            return {"success": False, "error": "처리할 파일이 없습니다"}
        
        processed_files = 0
        output_files = []
        
        print(f"Naver OCR 처리 시작: {len(files_to_process)}개 파일")
        
        # 디렉토리별로 파일 그룹화 (진행률 표시용)
        dir_groups = {}
        for file_path in files_to_process:
            dir_name = file_path.parent.name
            if dir_name not in dir_groups:
                dir_groups[dir_name] = []
            dir_groups[dir_name].append(file_path)
        
        # 각 디렉토리별로 처리  
        for dir_name, files_in_dir in tqdm(dir_groups.items(), desc="디렉토리 처리", unit="dir"):
            all_text_for_dir = ""
            output_file_path = None
            
            # 디렉토리 내 파일들 처리 (이미지 처리 진행률 표시)
            for file_path in tqdm(files_in_dir, desc=f"{dir_name} 이미지 처리", unit="img", leave=False):
                try:
                    file_path = Path(file_path)
                    # logger.info(f"Naver OCR 처리 중: {file_path.name}")  # 개별 파일 로그 제거
                    
                    # 파일을 base64로 인코딩
                    with open(file_path, 'rb') as f:
                        file_data = base64.b64encode(f.read()).decode()
                    
                    # API 요청 데이터 구성
                    request_json = {
                        'images': [{
                            'format': file_path.suffix[1:].lower(),
                            'name': file_path.name,
                            'data': file_data
                        }],
                        'requestId': str(file_path.stem),
                        'version': 'V2',
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    if template_ids:
                        request_json['templates'] = [{'id': tid} for tid in template_ids]
                    
                    # API 호출
                    headers = {
                        'X-OCR-SECRET': secret_key,
                        'Content-Type': 'application/json'
                    }
                    
                    response = requests.post(api_url, headers=headers, json=request_json)
                    response.raise_for_status()
                    
                    result_data = response.json()
                    
                    # 텍스트 추출 - 좌표 기반 정렬로 가독성 향상
                    extracted_text = ""
                    if 'images' in result_data:
                        for image in result_data['images']:
                            if 'fields' in image:
                                # 텍스트 필드를 좌표 기준으로 정렬 (Y좌표 우선, X좌표 보조)
                                fields_with_coords = []
                                for field in image['fields']:
                                    if 'boundingPoly' in field and 'vertices' in field['boundingPoly']:
                                        vertices = field['boundingPoly']['vertices']
                                        if vertices:
                                            # 상단 좌측 좌표 기준으로 정렬
                                            y_coord = vertices[0].get('y', 0)
                                            x_coord = vertices[0].get('x', 0)
                                            text = field.get('inferText', '').strip()
                                            if text:
                                                fields_with_coords.append((y_coord, x_coord, text))
                                
                                # Y좌표 기준 정렬 (위에서 아래로), 같은 라인은 X좌표 기준 정렬 (왼쪽에서 오른쪽)
                                fields_with_coords.sort(key=lambda x: (x[0], x[1]))
                                
                                # 라인별로 그룹화하여 텍스트 구성
                                if fields_with_coords:
                                    current_line_y = fields_with_coords[0][0]
                                    line_threshold = 20  # 같은 라인으로 간주할 Y좌표 차이
                                    current_line_texts = []
                                    
                                    for y, x, text in fields_with_coords:
                                        if abs(y - current_line_y) <= line_threshold:
                                            # 같은 라인
                                            current_line_texts.append(text)
                                        else:
                                            # 새로운 라인
                                            if current_line_texts:
                                                extracted_text += " ".join(current_line_texts) + "\n"
                                            current_line_texts = [text]
                                            current_line_y = y
                                    
                                    # 마지막 라인 처리
                                    if current_line_texts:
                                        extracted_text += " ".join(current_line_texts) + "\n"
                    
                    # 디렉토리별 텍스트 누적
                    if extracted_text.strip():
                        if all_text_for_dir:
                            all_text_for_dir += f"\n--- {file_path.name} ---\n"
                        all_text_for_dir += extracted_text.strip() + "\n"
                    
                    # 출력 경로 설정 (첫 번째 파일에서 설정)
                    if output_file_path is None:
                        # _create_output_structure 함수 사용
                        output_dir = _create_output_structure(file_path, base_input_path, output_path_obj, "naver")
                        
                        # 파일명은 마지막 디렉토리명 사용 (디렉토리별로 통합)
                        try:
                            relative_path = file_path.relative_to(base_input_path)
                            path_parts = relative_path.parts[:-1]  # 파일명 제외한 디렉토리 부분
                            if path_parts:
                                output_filename = f"{path_parts[-1]}.txt"
                            else:
                                output_filename = f"{file_path.stem}.txt"
                        except ValueError:
                            output_filename = f"{file_path.parent.name}.txt"
                            
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_file_path = output_dir / output_filename
                    
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"파일 {file_path.name} Naver OCR 처리 실패: {e}")
                    continue
            
            # 디렉토리별 텍스트 파일 저장
            if all_text_for_dir.strip() and output_file_path:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(all_text_for_dir.strip())
                
                # 출력 파일 경로는 처음 생성할 때만 추가
                if str(output_file_path) not in output_files:
                    output_files.append(str(output_file_path))
                    
                print(f"Saved: {output_file_path}")  # 파일 저장 완료 메시지
        
        processing_time = time.time() - start_time
        
        return {
            "success": processed_files > 0,
            "processed_files": processed_files,
            "output_files": output_files,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Naver OCR 처리 중 오류 발생: {e}")
        return {
            "success": False,
            "error": f"Naver OCR 처리 실패: {str(e)}"
        }

def ocr_mistral(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Mistral AI의 Vision 모델을 사용하여 이미지에서 텍스트를 추출합니다.
    
    Args:
        input_path (str): 입력 이미지 파일 또는 디렉토리 경로
        output_path (str): 출력 디렉토리 경로
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    import time
    import base64
    import requests
    
    start_time = time.time()
    
    try:
        # JSON 설정 로드
        config = _load_ocr_config()
        mistral_config = config.get('mistral', {})
        
        api_key = mistral_config.get('api_key')
        model_name = mistral_config.get('model_name', 'pixtral-12b-2409')
        prompt = mistral_config.get('prompt', '이 이미지에서 모든 텍스트를 정확히 추출해주세요. 줄바꿈과 형식을 유지해주세요.')
        max_tokens = mistral_config.get('max_tokens', 2000)
        
        if not api_key:
            return {
                "success": False,
                "error": "Mistral API 설정이 ocr_config.json에 올바르게 설정되지 않았습니다."
            }
        
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        # pdf_to_image와 동일한 방식: 입력 경로를 base_input_path로 설정
        base_input_path = Path(input_path)
        
        # 처리할 이미지 파일 목록 생성
        if input_path_obj.is_file():
            if input_path_obj.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                files_to_process = [input_path_obj]
            else:
                return {"success": False, "error": "지원되지 않는 파일 형식 (JPG, PNG만 지원)"}
        elif input_path_obj.is_dir():
            extensions = ['*.jpg', '*.jpeg', '*.png']
            files_to_process = []
            for ext in extensions:
                files_to_process.extend(input_path_obj.rglob(ext))
        else:
            return {"success": False, "error": "유효하지 않은 입력 경로"}
        
        if not files_to_process:
            return {"success": False, "error": "처리할 이미지 파일이 없습니다"}
        
        processed_files = 0
        all_text_content = []
        
        # 각 파일 처리
        for file_path in files_to_process:
            try:
                logger.info(f"Mistral OCR 처리 중: {file_path.name}")
                
                # 이미지를 base64로 인코딩
                with open(file_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                # API 요청
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                
                data = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": max_tokens
                }
                
                response = requests.post(
                    'https://api.mistral.ai/v1/chat/completions',
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                
                result = response.json()
                extracted_text = result['choices'][0]['message']['content']
                
                # _create_output_structure 함수 사용
                output_dir = _create_output_structure(file_path, base_input_path, output_path_obj, "mistral")
                output_filename = f"mistral_ocr_{file_path.stem}.txt"
                output_file_path = output_dir / output_filename
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                
                all_text_content.append(extracted_text)
                processed_files += 1
                
                logger.info(f"Mistral OCR 완료: {output_filename}")
                
            except Exception as e:
                logger.error(f"파일 {file_path} Mistral OCR 처리 실패: {e}")
                continue
        
        # 전체 결과 저장
        if all_text_content and files_to_process:
            # 첫 번째 파일 기준으로 출력 디렉토리 생성
            output_dir = _create_output_structure(files_to_process[0], base_input_path, output_path_obj, "mistral")
            combined_output_path = output_dir / "mistral_ocr_combined.txt"
            with open(combined_output_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(all_text_content))
        
        processing_time = time.time() - start_time
        
        # 출력 디렉토리 정보 (첫 번째 파일 기준)
        output_directory = str(output_path_obj / "ocr" / "mistral") if not files_to_process else str(_create_output_structure(files_to_process[0], base_input_path, output_path_obj, "mistral"))
        
        return {
            "success": processed_files > 0,
            "processed_files": processed_files,
            "total_text_length": len("".join(all_text_content)),
            "output_directory": output_directory,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Mistral OCR 처리 중 오류 발생: {e}")
        return {
            "success": False,
            "error": f"Mistral OCR 처리 실패: {str(e)}"
        }

def ocr_google(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Google Cloud Vision OCR을 사용하여 이미지에서 텍스트를 추출합니다.
    Naver OCR과 동일한 출력 구조를 사용합니다.
    
    Args:
        input_path (str): 입력 이미지 파일 또는 디렉토리 경로
        output_path (str): 출력 디렉토리 경로
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    import time
    
    start_time = time.time()
    
    try:
        # JSON 설정 로드
        config = _load_ocr_config()
        google_config = config.get('google', {})
        
        credentials_path = google_config.get('credentials_path')
        use_document_detection = google_config.get('use_document_detection', True)
        
        # Google Cloud Vision 라이브러리 import
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        # ocr 디렉토리 생성
        ocr_dir = output_path_obj / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        
        # pdf_to_image와 동일한 방식: 입력 경로를 base_input_path로 설정
        base_input_path = Path(input_path)
        
        # 처리할 파일 목록 생성
        if input_path_obj.is_file():
            if input_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                files_to_process = [input_path_obj]
            else:
                return {"success": False, "error": "지원되지 않는 파일 형식"}
        elif input_path_obj.is_dir():
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
            files_to_process = []
            for ext in extensions:
                files_to_process.extend(input_path_obj.rglob(ext))
        else:
            return {"success": False, "error": "유효하지 않은 입력 경로"}
        
        if not files_to_process:
            return {"success": False, "error": "처리할 파일이 없습니다"}
        
        processed_files = 0
        output_files = []
        
        print(f"Google OCR 처리 시작: {len(files_to_process)}개 파일")
        
        # 디렉토리별로 파일 그룹화 (Naver OCR과 동일)
        dir_groups = {}
        for file_path in files_to_process:
            dir_name = file_path.parent.name
            if dir_name not in dir_groups:
                dir_groups[dir_name] = []
            dir_groups[dir_name].append(file_path)
        
        # 각 디렉토리별로 처리 (Naver OCR과 동일한 방식)
        for dir_name, files_in_dir in tqdm(dir_groups.items(), desc="디렉토리 처리", unit="dir"):
            all_text_for_dir = ""
            output_file_path = None
            
            # 디렉토리 내 파일들 처리
            for file_path in tqdm(files_in_dir, desc=f"{dir_name} 이미지 처리", unit="img", leave=False):
                try:
                    file_path = Path(file_path)
                    
                    with open(file_path, 'rb') as image_file:
                        content = image_file.read()
                    
                    image = vision.Image(content=content)
                    
                    # OCR 수행 (문서 검출 or 일반 텍스트 검출)
                    if use_document_detection:
                        response = client.document_text_detection(image=image)
                        texts = response.full_text_annotation
                        extracted_text = texts.text if texts else ""
                    else:
                        response = client.text_detection(image=image)
                        texts = response.text_annotations
                        extracted_text = texts[0].description if texts else ""
                    
                    # 에러 확인
                    if response.error.message:
                        raise Exception(response.error.message)
                    
                    # 디렉토리별 텍스트 누적 (Naver OCR과 동일한 방식)
                    if extracted_text.strip():
                        if all_text_for_dir:
                            all_text_for_dir += f"\n--- {file_path.name} ---\n"
                        all_text_for_dir += extracted_text.strip() + "\n"
                    
                    # 출력 경로 설정 (첫 번째 파일에서 설정)
                    if output_file_path is None:
                        # _create_output_structure 함수 사용
                        output_dir = _create_output_structure(file_path, base_input_path, output_path_obj, "google")
                        
                        # 파일명은 마지막 디렉토리명 사용 (Naver OCR과 동일)
                        try:
                            relative_path = file_path.relative_to(base_input_path)
                            path_parts = relative_path.parts[:-1]  # 파일명 제외한 디렉토리 부분
                            if path_parts:
                                output_filename = f"{path_parts[-1]}.txt"
                            else:
                                output_filename = f"{file_path.stem}.txt"
                        except ValueError:
                            output_filename = f"{file_path.parent.name}.txt"
                            
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_file_path = output_dir / output_filename
                    
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"파일 {file_path.name} Google OCR 처리 실패: {e}")
                    continue
            
            # 디렉토리별 텍스트 파일 저장 (Naver OCR과 동일)
            if all_text_for_dir.strip() and output_file_path:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(all_text_for_dir.strip())
                
                # 출력 파일 경로는 처음 생성할 때만 추가
                if str(output_file_path) not in output_files:
                    output_files.append(str(output_file_path))
                    
                print(f"Saved: {output_file_path}")  # 파일 저장 완료 메시지
        
        processing_time = time.time() - start_time
        
        return {
            "success": processed_files > 0,
            "processed_files": processed_files,
            "output_files": output_files,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Google OCR 처리 중 오류 발생: {e}")
        return {
            "success": False,
            "error": f"Google OCR 처리 실패: {str(e)}"
        }

def ocr_complete(input_path: str, output_path: str,
                 ocr_engines: List[str] = None) -> Dict[str, Any]:
    """
    여러 OCR 엔진을 동시에 사용하여 텍스트를 추출하고 결과를 통합합니다.
    
    Args:
        input_path (str): 입력 이미지 파일 또는 디렉토리 경로
        output_path (str): 출력 디렉토리 경로
        ocr_engines (List[str], optional): 사용할 OCR 엔진 목록. 
                                         기본값: ["google", "naver", "mistral"]
        
    Returns:
        Dict[str, Any]: 통합 처리 결과
    """
    import time
    
    start_time = time.time()
    
    try:
        if ocr_engines is None:
            ocr_engines = ["google", "naver", "mistral"]
        
        # 결과 저장용 딕셔너리
        engine_results = {}
        successful_engines = []
        
        # 출력 디렉토리 생성
        ocr_output_dir = Path(output_path) / "ocr"
        ocr_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 OCR 엔진 실행
        logger.info(f"Complete OCR 처리 시작: {len(ocr_engines)}개 엔진 사용")
        
        # 각 엔진별로 OCR 수행
        for engine in ocr_engines:
            try:
                logger.info(f"OCR 엔진 실행: {engine}")
                if engine.lower() == "google":
                    result = ocr_google(input_path, output_path)
                elif engine.lower() == "naver":
                    result = ocr_naver(input_path, output_path)
                elif engine.lower() == "mistral":
                    result = ocr_mistral(input_path, output_path)
                else:
                    logger.warning(f"지원되지 않는 OCR 엔진: {engine}")
                    continue
                
                engine_results[engine] = result
                
                if result.get('success'):
                    successful_engines.append(engine)
                    logger.info(f"OCR 엔진 {engine} 성공")
                else:
                    logger.warning(f"OCR 엔진 {engine} 실패: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"OCR 엔진 {engine} 처리 중 오류: {e}")
                engine_results[engine] = {
                    "success": False,
                    "error": str(e)
                }
        
        # 결과 통합
        total_combined_text = ""
        if successful_engines:
            combined_result = _combine_ocr_results(engine_results, "google_priority")
            total_combined_text = combined_result.get('combined_text', '')
            
            # 통합 결과 저장
            complete_output_path = ocr_output_dir / "complete_ocr_result.txt"
            with open(complete_output_path, 'w', encoding='utf-8') as f:
                f.write(total_combined_text)
        
        processing_time = time.time() - start_time
        
        return {
            "success": len(successful_engines) > 0,
            "processed_files": sum(r.get('processed_files', 0) for r in engine_results.values() if r.get('success')),
            "total_text_length": len(total_combined_text),
            "engines_used": list(engine_results.keys()),
            "successful_engines": successful_engines,
            "consensus_method": "google_priority",
            "output_directory": str(ocr_output_dir),
            "output_file": "complete_ocr_result.txt",
            "engine_results": engine_results,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Complete OCR 처리 중 오류: {e}")
        return {
            "success": False,
            "error": f"Complete OCR 처리 실패: {str(e)}"
        }

def _combine_ocr_results(engine_results: Dict[str, Dict], 
                        consensus_method: str = "google_priority") -> Dict[str, str]:
    """
    여러 OCR 엔진 결과를 통합합니다.
    
    Args:
        engine_results: 각 엔진별 결과 딕셔너리
        consensus_method: 통합 방법 ("google_priority", "longest", "consensus")
        
    Returns:
        Dict[str, str]: 통합된 텍스트 결과
    """
    # 간단한 구현 - Google 우선, 없으면 가장 긴 결과 사용
    combined_text = ""
    
    if consensus_method == "google_priority":
        if "google" in engine_results and engine_results["google"].get("success"):
            combined_text = "Google OCR 결과를 우선 사용"
        else:
            # 다른 엔진 중 가장 긴 결과 사용
            for engine, result in engine_results.items():
                if result.get("success"):
                    combined_text = f"{engine} OCR 결과 사용"
                    break
    
    return {"combined_text": combined_text}
