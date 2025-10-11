#!/usr/bin/env python3
"""
PDF/이미지 NER 처리 웹 애플리케이션 (FastAPI)

기능:
- PDF/이미지 업로드 (드래그 앤 드롭 지원)
- OCR 처리 (Google/Naver/Mistral)
- NER 엔티티 추출 (3개 모델 선택 가능)
- 실시간 처리 진행 상황 표시
- 자동 API 문서화 (Swagger UI)
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 상위 디렉토리(api)를 경로에 추가
current_dir = Path(__file__).parent
api_dir = current_dir.parent
sys.path.insert(0, str(api_dir))

# api 모듈 import
from api import pdf_to_image, ocr_google, ocr_naver, ocr_mistral, ner_predict

# FastAPI 앱 초기화
app = FastAPI(
    title="NER 엔티티 추출 API",
    description="PDF/이미지에서 자동으로 엔티티를 추출하는 AI API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
UPLOAD_DIR = api_dir / "web" / "uploads"
RESULTS_DIR = api_dir / "web" / "results"
TEMP_DIR = api_dir / "web" / "temp"

for directory in [UPLOAD_DIR, RESULTS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 템플릿 설정
templates = Jinja2Templates(directory=str(current_dir / "templates"))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# OCR 설정 확인
def check_ocr_availability() -> Dict[str, bool]:
    """OCR 엔진 사용 가능 여부 확인"""
    config_path = api_dir / "ocr_key" / "ocr_config.json"
    available = {
        'google': False,
        'naver': False,
        'mistral': False
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Google 확인
        ocr_key_dir = api_dir / "ocr_key"
        if ocr_key_dir.exists():
            json_files = list(ocr_key_dir.glob("*.json"))
            available['google'] = len(json_files) > 0
        
        # Naver 확인
        naver_config = config.get('naver', {})
        if naver_config.get('api_url') and naver_config.get('secret_key'):
            available['naver'] = True
        
        # Mistral 확인
        mistral_config = config.get('mistral', {})
        if mistral_config.get('api_key'):
            available['mistral'] = True
            
    except Exception as e:
        logger.warning(f"OCR 설정 확인 실패: {e}")
    
    return available

OCR_AVAILABILITY = check_ocr_availability()

# 사용 가능한 NER 모델
AVAILABLE_MODELS = {
    'klue-roberta-large': {
        'name': 'klue/roberta-large',
        'display_name': 'KLUE RoBERTa Large',
        'description': '한국어 특화 모델 (뛰어난 성능)',
        'accuracy': '92.72%',
        'speed': '보통'
    },
    'xlm-roberta': {
        'name': 'FacebookAI/xlm-roberta-large',
        'display_name': 'XLM-RoBERTa Large',
        'description': '다국어 특화 모델 (최고 정확도)',
        'accuracy': '95.88%',
        'speed': '느림'
    },
    'google-bert': {
        'name': 'google-bert/bert-base-multilingual-cased',
        'display_name': 'Google mBERT',
        'description': '제일 가벼운 모델 (빠른 속도)',
        'accuracy': '87.60%',
        'speed': '빠름'
    }
}

# 사용 가능한 OCR 엔진
AVAILABLE_OCR_ENGINES = {
    'mistral': {
        'name': 'Mistral Vision AI',
        'description': '높은 정확도',
        'quality': '높음',
        'speed': '빠름',
        'available': OCR_AVAILABILITY['mistral'],
        'setup_guide': 'api/ocr_key/ocr_config.json에 Mistral API Key를 설정하세요'
    },
    'naver': {
        'name': 'Naver CLOVA OCR',
        'description': '표 분석에 강함',
        'quality': '높음',
        'speed': '빠름',
        'available': OCR_AVAILABILITY['naver'],
        'setup_guide': 'api/ocr_key/ocr_config.json에 Naver API URL과 Secret Key를 설정하세요'
    },
    'google': {
        'name': 'Google Vision API',
        'description': '텍스트 분석률 좋음',
        'quality': '높음',
        'speed': '빠름',
        'available': OCR_AVAILABILITY['google'],
        'setup_guide': 'api/ocr_key/ 디렉토리에 Google Cloud 인증 JSON 파일을 배치하세요'
    }
}

def allowed_file(filename: str) -> bool:
    """허용된 파일 형식인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename: str) -> str:
    """파일명을 안전하게 변환"""
    import re
    filename = str(filename).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', filename)

async def process_document(file_path: Path, output_dir: Path, model_name: str, ocr_engine: str = 'google') -> Dict[str, Any]:
    """
    문서 처리 파이프라인
    
    1. PDF → 이미지 변환 (필요시)
    2. OCR 처리
    3. NER 엔티티 추출
    """
    result = {
        'success': False,
        'steps': {},
        'entities': [],
        'entity_count': 0,
        'model_name': model_name,
        'ocr_engine': ocr_engine
    }
    
    try:
        # 1단계: PDF를 이미지로 변환
        if file_path.suffix.lower() == '.pdf':
            logger.info("Step 1: PDF → 이미지 변환")
            image_dir = output_dir / "images"
            
            try:
                pdf_result = pdf_to_image(str(file_path), str(image_dir), dpi=300)
                
                result['steps']['pdf_to_image'] = {
                    'success': pdf_result.get('success', False),
                    'image_count': pdf_result.get('total_images', 0),
                    'time': pdf_result.get('processing_time', 0)
                }
                
                if not pdf_result.get('success'):
                    result['error'] = 'PDF 변환 실패: ' + pdf_result.get('error', '알 수 없는 오류')
                    return result
                
                input_for_ocr = image_dir
            except Exception as e:
                result['error'] = f'PDF 변환 오류: {str(e)}'
                result['steps']['pdf_to_image'] = {'success': False, 'error': str(e)}
                return result
        else:
            # 이미지 파일은 바로 OCR
            input_for_ocr = file_path
            result['steps']['pdf_to_image'] = {'success': True, 'skipped': True}
        
        # 2단계: OCR 처리
        logger.info(f"Step 2: OCR 처리 (엔진: {ocr_engine})")
        ocr_dir = output_dir / "ocr"
        
        try:
            # OCR 엔진 선택
            if ocr_engine == 'google':
                ocr_result = ocr_google(str(input_for_ocr), str(ocr_dir))
            elif ocr_engine == 'naver':
                ocr_result = ocr_naver(str(input_for_ocr), str(ocr_dir))
            elif ocr_engine == 'mistral':
                ocr_result = ocr_mistral(str(input_for_ocr), str(ocr_dir))
            else:
                result['error'] = f'지원하지 않는 OCR 엔진: {ocr_engine}'
                return result
            
            result['steps']['ocr'] = {
                'success': ocr_result.get('success', False),
                'files_processed': ocr_result.get('processed_files', 0),
                'time': ocr_result.get('processing_time', 0),
                'engine': ocr_engine
            }
            
            if not ocr_result.get('success'):
                error_msg = ocr_result.get('error', 'OCR 처리 실패')
                
                # Google Cloud 인증 오류 처리
                if 'credentials' in error_msg.lower() or 'authentication' in error_msg.lower():
                    result['error'] = 'Google Cloud Vision API 인증 오류: 환경 변수 GOOGLE_APPLICATION_CREDENTIALS를 설정하거나 api/ocr_key/ 디렉토리에 인증 파일을 배치하세요.'
                else:
                    result['error'] = f'OCR 처리 실패: {error_msg}'
                return result
            
        except Exception as e:
            error_str = str(e)
            if 'credentials' in error_str.lower():
                result['error'] = 'Google Cloud Vision API 인증 설정이 필요합니다. 관리자에게 문의하세요.'
            else:
                result['error'] = f'OCR 처리 오류: {error_str}'
            result['steps']['ocr'] = {'success': False, 'error': str(e)}
            return result
        
        # 3단계: NER 엔티티 추출
        logger.info(f"Step 3: NER 엔티티 추출 (모델: {model_name})")
        ner_dir = output_dir / "ner"
        
        try:
            ner_result = ner_predict(
                str(ocr_dir),
                str(ner_dir),
                model_name=model_name,
                debug=False
            )
            
            result['steps']['ner'] = {
                'success': ner_result.get('success', False),
                'entity_count': ner_result.get('total_entities', 0),
                'time': ner_result.get('processing_time', 0)
            }
            
            if not ner_result.get('success'):
                result['error'] = 'NER 처리 실패: ' + ner_result.get('error', '알 수 없는 오류')
                return result
            
            # 결과 수집
            result['success'] = True
            result['entities'] = ner_result.get('entity_types', {})
            result['entity_count'] = ner_result.get('total_entities', 0)
            result['output_files'] = ner_result.get('output_files', [])
            
        except Exception as e:
            result['error'] = f'NER 처리 오류: {str(e)}'
            result['steps']['ner'] = {'success': False, 'error': str(e)}
            return result
        
        return result
        
    except Exception as e:
        logger.error(f"처리 중 예상치 못한 오류: {e}", exc_info=True)
        result['error'] = f'처리 중 오류 발생: {str(e)}'
        return result

# ============================================================================
# 라우트 정의
# ============================================================================

@app.get("/")
async def index(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": AVAILABLE_MODELS,
            "ocr_engines": AVAILABLE_OCR_ENGINES
        }
    )

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = Form("klue-roberta-large"),
    ocr_engine: str = Form("google")
):
    """파일 업로드 및 처리"""
    try:
        # 파일 확인
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 비어있습니다")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f'지원하지 않는 파일 형식입니다. 허용: {", ".join(ALLOWED_EXTENSIONS)}'
            )
        
        # 모델 선택
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail="잘못된 모델 선택")
        
        model_name = AVAILABLE_MODELS[model]['name']
        
        # OCR 엔진 선택
        if ocr_engine not in AVAILABLE_OCR_ENGINES:
            raise HTTPException(status_code=400, detail="잘못된 OCR 엔진 선택")
        
        # OCR 엔진 사용 가능 여부 확인
        if not AVAILABLE_OCR_ENGINES[ocr_engine]['available']:
            setup_guide = AVAILABLE_OCR_ENGINES[ocr_engine]['setup_guide']
            raise HTTPException(
                status_code=400,
                detail=f'{AVAILABLE_OCR_ENGINES[ocr_engine]["name"]} 설정이 필요합니다. {setup_guide}'
            )
        
        # 파일 저장
        filename = secure_filename(file.filename)
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        upload_path = UPLOAD_DIR / request_id / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 내용 저장
        content = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"업로드 완료: {filename} ({file_size_mb:.2f}MB), 모델: {model_name}, OCR: {ocr_engine}")
        
        # 결과 디렉토리
        result_dir = RESULTS_DIR / request_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 처리 시작
        start_time = datetime.now()
        result = await process_document(upload_path, result_dir, model_name, ocr_engine)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 응답 생성
        response = {
            'success': result['success'],
            'request_id': request_id,
            'filename': filename,
            'file_size_mb': round(file_size_mb, 2),
            'model': AVAILABLE_MODELS[model]['display_name'],
            'model_key': model,
            'ocr_engine': AVAILABLE_OCR_ENGINES[ocr_engine]['name'],
            'ocr_engine_key': ocr_engine,
            'entities': result.get('entities', {}),
            'entity_count': result.get('entity_count', 0),
            'steps': result.get('steps', {}),
            'processing_time': round(processing_time, 2)
        }
        
        if not result['success']:
            response['error'] = result.get('error', '알 수 없는 오류')
        
        # 결과 JSON 저장
        result_json_path = result_dir / 'result.json'
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        status_code = 200 if result['success'] else 500
        return JSONResponse(content=response, status_code=status_code)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"업로드 처리 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{request_id}")
async def download_result(request_id: str, type: str = "entities"):
    """결과 다운로드
    
    Args:
        request_id: 요청 ID
        type: 다운로드 타입 ("entities" 또는 "stats")
    """
    try:
        result_dir = RESULTS_DIR / request_id
        
        if not result_dir.exists():
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")
        
        if type == "entities":
            # 추출된 엔티티 파일 찾기 (*_entities.json)
            entities_files = list(result_dir.rglob("*_entities.json"))
            
            if not entities_files:
                raise HTTPException(status_code=404, detail="엔티티 파일을 찾을 수 없습니다")
            
            # 첫 번째 엔티티 파일 사용 (원본 파일명 유지)
            file_path = entities_files[0]
            filename = file_path.name
        else:
            # 통계 리포트 파일 찾기 (summary.json)
            summary_files = list(result_dir.rglob("summary.json"))
            
            if not summary_files:
                raise HTTPException(status_code=404, detail="통계 파일을 찾을 수 없습니다")
            
            # 첫 번째 summary 파일 사용 (원본 파일명 유지)
            file_path = summary_files[0]
            filename = file_path.name
        
        return FileResponse(
            path=file_path,
            media_type='application/json',
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"다운로드 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_models': list(AVAILABLE_MODELS.keys()),
        'available_ocr_engines': [k for k, v in AVAILABLE_OCR_ENGINES.items() if v['available']]
    }

@app.get("/api/info")
async def api_info():
    """API 정보"""
    return {
        'title': 'NER 엔티티 추출 API',
        'version': '2.0.0',
        'framework': 'FastAPI',
        'models': AVAILABLE_MODELS,
        'ocr_engines': AVAILABLE_OCR_ENGINES
    }

if __name__ == '__main__':
    print("=" * 80)
    print("  NER 엔티티 추출 웹 애플리케이션 (FastAPI)")
    print("=" * 80)
    print(f"\n업로드 디렉토리: {UPLOAD_DIR}")
    print(f"결과 디렉토리: {RESULTS_DIR}")
    
    print(f"\n사용 가능한 OCR 엔진:")
    for key, info in AVAILABLE_OCR_ENGINES.items():
        status = "✓" if info['available'] else "✗"
        print(f"  {status} {info['name']}")
    
    print(f"\n사용 가능한 NER 모델:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  - {info['display_name']}: {info['description']}")
    
    print("\n" + "=" * 80)
    print("서버 시작: http://localhost:5000")
    print("API 문서: http://localhost:5000/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
