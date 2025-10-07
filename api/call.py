#!/usr/bin/env python3
"""
REST API 서버: PDF → OCR → NER 파이프라인

기능:
1. PDF 파일 업로드
2. OCR 처리 (Google Vision API)
3. NER 엔티티 추출
4. 결과 JSON 반환
5. 요청 로그 기록 (시간순)

사용법:
    python call.py

엔드포인트:
    POST /process - PDF 파일을 받아서 NER 결과 반환
    GET /health - 서버 상태 확인
    GET /logs - 요청 로그 조회
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# api.py의 기능들 import
from api import pdf_to_image, ocr_google, ner_predict

# Flask 앱 초기화
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 최대 50MB

# 로그 디렉토리 설정
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)

# 임시 파일 디렉토리
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# 모델 설정 로드
def load_model_config():
    """model_config.json에서 모델 설정 로드"""
    config_path = Path(__file__).parent / "model_config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.warning(f"모델 설정 로드 실패: {e}, 기본값 사용")
        return {
            "ner": {
                "default_model": "xlm-roberta-large"
            }
        }

MODEL_CONFIG = load_model_config()
DEFAULT_NER_MODEL = MODEL_CONFIG.get("ner", {}).get("default_model", "xlm-roberta-large")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'server_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RequestLogger:
    """요청 로그를 시간순으로 기록하는 클래스"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
    
    def log_request(self, request_data: Dict[str, Any]) -> str:
        """요청 정보를 로그 파일에 저장"""
        timestamp = datetime.now()
        log_filename = timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".json"
        log_path = self.log_dir / log_filename
        
        request_data['timestamp'] = timestamp.isoformat()
        request_data['log_file'] = log_filename
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Request logged: {log_filename}")
        return log_filename
    
    def get_all_logs(self, limit: int = 100) -> list:
        """모든 로그를 시간 역순으로 조회"""
        log_files = sorted(self.log_dir.glob("*.json"), reverse=True)[:limit]
        logs = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs.append(json.load(f))
            except Exception as e:
                logger.error(f"Error reading log {log_file}: {e}")
        
        return logs


# 로거 인스턴스 생성
request_logger = RequestLogger(LOG_DIR)


def allowed_file(filename: str) -> bool:
    """허용된 파일 형식인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


def process_pdf(pdf_path: Path, output_base: Path, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    PDF 처리 파이프라인: PDF → 이미지 → OCR → NER
    
    Args:
        pdf_path: PDF 파일 경로
        output_base: 출력 기본 디렉토리
        model_name: NER 모델 이름 (None이면 기본 모델 사용)
    
    Returns:
        처리 결과 딕셔너리
    """
    # 모델 이름이 지정되지 않으면 기본 모델 사용
    if model_name is None:
        model_name = DEFAULT_NER_MODEL
    
    logger.info(f"사용 모델: {model_name}")
    
    result = {
        'success': False,
        'error': None,
        'steps': {},
        'entities': None,
        'entity_count': 0,
        'model_name': model_name
    }
    
    try:
        # 1단계: PDF → 이미지
        logger.info(f"[1/3] PDF → 이미지 변환 시작: {pdf_path.name}")
        image_dir = output_base / "images"
        image_dir.mkdir(exist_ok=True, parents=True)
        
        pdf_result = pdf_to_image(str(pdf_path), str(image_dir))
        result['steps']['pdf_to_image'] = {
            'success': pdf_result.get('success', False),
            'images_created': pdf_result.get('total_images', 0)
        }
        
        if not pdf_result.get('success'):
            result['error'] = "PDF to image conversion failed"
            return result
        
        logger.info(f"✓ 이미지 변환 완료: {pdf_result.get('total_images', 0)}개")
        
        # 2단계: OCR 처리
        logger.info("[2/3] OCR 처리 시작")
        ocr_dir = output_base / "ocr"
        ocr_dir.mkdir(exist_ok=True, parents=True)
        
        ocr_result = ocr_google(str(image_dir), str(output_base))
        result['steps']['ocr'] = {
            'success': ocr_result.get('success', False),
            'files_processed': ocr_result.get('processed_files', 0)
        }
        
        if not ocr_result.get('success'):
            result['error'] = "OCR processing failed"
            return result
        
        logger.info(f"✓ OCR 완료: {ocr_result.get('processed_files', 0)}개 파일")
        
        # 3단계: NER 처리
        logger.info(f"[3/3] NER 엔티티 추출 시작 (모델: {model_name})")
        ner_dir = output_base / "ner"
        ner_dir.mkdir(exist_ok=True, parents=True)
        
        ner_result = ner_predict(str(ocr_dir / "google"), str(output_base), model_name=model_name)
        result['steps']['ner'] = {
            'success': ner_result.get('success', False),
            'files_processed': ner_result.get('processed_files', 0),
            'total_entities': ner_result.get('total_entities', 0),
            'model_name': model_name
        }
        
        if not ner_result.get('success'):
            result['error'] = "NER processing failed"
            return result
        
        logger.info(f"✓ NER 완료: {ner_result.get('total_entities', 0)}개 엔티티")
        
        # NER 결과 파일 읽기 (재귀적으로 검색)
        ner_dir_path = output_base / "ner"
        logger.info(f"NER 결과 디렉토리 확인: {ner_dir_path}")
        logger.info(f"디렉토리 존재 여부: {ner_dir_path.exists()}")
        
        if ner_dir_path.exists():
            all_files = list(ner_dir_path.rglob("*"))
            logger.info(f"디렉토리 내 전체 파일/폴더: {len(all_files)}개")
        
        # rglob으로 재귀적 검색
        ner_result_files = list(ner_dir_path.rglob("*_entities.json"))
        logger.info(f"NER 결과 파일 수: {len(ner_result_files)}")
        
        if ner_result_files:
            logger.info(f"발견된 파일: {[f.name for f in ner_result_files]}")
            # 새 형식: 타입별로 그룹화된 엔티티 병합
            all_entities_grouped = {}
            
            for ner_file in ner_result_files:
                logger.info(f"파일 읽기: {ner_file}")
                with open(ner_file, 'r', encoding='utf-8') as f:
                    ner_data = json.load(f)
                    entities = ner_data.get('entities', {})
                    logger.info(f"파일 {ner_file.name}에서 {len(entities)}가지 타입 발견")
                    
                    # 각 타입별로 엔티티 병합
                    for entity_type, entity_list in entities.items():
                        if entity_type not in all_entities_grouped:
                            all_entities_grouped[entity_type] = []
                        all_entities_grouped[entity_type].extend(entity_list)
            
            # 중복 제거 및 정렬
            for entity_type in all_entities_grouped:
                all_entities_grouped[entity_type] = sorted(list(set(all_entities_grouped[entity_type])))
            
            # 전체 엔티티 개수 계산
            total_count = sum(len(v) for v in all_entities_grouped.values())
            
            result['entities'] = all_entities_grouped
            result['entity_count'] = total_count
            result['entity_types'] = sorted(list(all_entities_grouped.keys()))
            logger.info(f"최종 엔티티: {total_count}개, {len(all_entities_grouped)}가지 타입")
        else:
            logger.warning("NER 결과 파일을 찾을 수 없습니다!")
        
        result['success'] = True
        logger.info("✓ 전체 파이프라인 완료")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}", exc_info=True)
        result['error'] = str(e)
    
    return result


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/logs', methods=['GET'])
def get_logs():
    """요청 로그 조회"""
    limit = request.args.get('limit', 100, type=int)
    logs = request_logger.get_all_logs(limit=limit)
    
    return jsonify({
        'total': len(logs),
        'logs': logs
    })


@app.route('/process', methods=['POST'])
def process_document():
    """
    PDF 문서 처리 엔드포인트
    
    Request:
        - file: PDF 파일 (multipart/form-data)
        - model_name: (선택) NER 모델 이름 (기본값: model_config.json의 default_model)
    
    Response:
        - success: 성공 여부
        - entities: 추출된 엔티티 리스트
        - entity_count: 엔티티 개수
        - steps: 각 단계별 결과
        - model_name: 사용된 모델 이름
        - log_file: 로그 파일명
    """
    start_time = datetime.now()
    
    # 요청 로그 초기화
    log_data = {
        'endpoint': '/process',
        'method': 'POST',
        'start_time': start_time.isoformat(),
        'client_ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    }
    
    try:
        # model_name 파라미터 받기 (선택적)
        model_name = request.form.get('model_name', None)
        if model_name:
            log_data['model_name'] = model_name
            logger.info(f"요청된 모델: {model_name}")
        else:
            log_data['model_name'] = DEFAULT_NER_MODEL
            logger.info(f"기본 모델 사용: {DEFAULT_NER_MODEL}")
        
        # 파일 확인
        if 'file' not in request.files:
            log_data['error'] = 'No file provided'
            log_data['success'] = False
            request_logger.log_request(log_data)
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            log_data['error'] = 'Empty filename'
            log_data['success'] = False
            request_logger.log_request(log_data)
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            log_data['error'] = 'Invalid file type (only PDF allowed)'
            log_data['success'] = False
            request_logger.log_request(log_data)
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # 파일명 안전화
        filename = secure_filename(file.filename)
        log_data['filename'] = filename
        
        # 임시 디렉토리 생성
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_request_dir = TEMP_DIR / request_id
        temp_request_dir.mkdir(exist_ok=True, parents=True)
        
        log_data['request_id'] = request_id
        
        # 파일 저장
        pdf_path = temp_request_dir / filename
        file.save(str(pdf_path))
        
        file_size = pdf_path.stat().st_size
        log_data['file_size_bytes'] = file_size
        log_data['file_size_mb'] = round(file_size / (1024 * 1024), 2)
        
        logger.info(f"Processing request {request_id}: {filename} ({log_data['file_size_mb']}MB)")
        
        # PDF 처리 (model_name 전달)
        output_dir = temp_request_dir / "output"
        result = process_pdf(pdf_path, output_dir, model_name=model_name)
        
        # 처리 시간 계산
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 로그 데이터 업데이트
        log_data['success'] = result['success']
        log_data['error'] = result.get('error')
        log_data['steps'] = result['steps']
        log_data['entity_count'] = result.get('entity_count', 0)
        log_data['processing_time_seconds'] = round(processing_time, 2)
        log_data['end_time'] = end_time.isoformat()
        
        # 로그 저장
        log_file = request_logger.log_request(log_data)
        
        # 임시 파일 정리 (디버깅용 주석)
        # try:
        #     shutil.rmtree(temp_request_dir)
        #     logger.info(f"Cleaned up temporary files for request {request_id}")
        # except Exception as e:
        #     logger.warning(f"Failed to clean up temp files: {e}")
        logger.info(f"임시 파일 보존 (디버깅): {temp_request_dir}")
        
        # 응답 생성
        response = {
            'success': result['success'],
            'request_id': request_id,
            'filename': filename,
            'model_name': result.get('model_name', model_name),
            'entities': result.get('entities', []),
            'entity_count': result.get('entity_count', 0),
            'steps': result['steps'],
            'processing_time_seconds': round(processing_time, 2),
            'log_file': log_file
        }
        
        if result.get('error'):
            response['error'] = result['error']
        
        status_code = 200 if result['success'] else 500
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in /process: {str(e)}", exc_info=True)
        
        # 에러 로그 저장
        log_data['success'] = False
        log_data['error'] = str(e)
        log_data['end_time'] = datetime.now().isoformat()
        request_logger.log_request(log_data)
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """파일 크기 초과 에러 핸들러"""
    return jsonify({
        'error': 'File too large (maximum 50MB)'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """404 에러 핸들러"""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error'
    }), 500


def main():
    """서버 시작"""
    print("="*60)
    print("REST API Server - PDF → OCR → NER Pipeline")
    print("="*60)
    print(f"Log directory: {LOG_DIR.absolute()}")
    print(f"Temp directory: {TEMP_DIR.absolute()}")
    print()
    print("Available endpoints:")
    print("  POST /process - Process PDF file")
    print("  GET  /health  - Server health check")
    print("  GET  /logs    - View request logs")
    print()
    print("Starting server on http://0.0.0.0:5000")
    print("="*60)
    print()
    
    # Flask 서버 시작
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
