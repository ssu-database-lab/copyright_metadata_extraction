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
from dotenv import load_dotenv

# Load environment variables — search project root first, then api/, then api/web/
_project_root_env = Path(__file__).parent.parent.parent / ".env"
_api_env = Path(__file__).parent.parent / ".env"
if _project_root_env.exists():
    load_dotenv(_project_root_env)
elif _api_env.exists():
    load_dotenv(_api_env)
else:
    load_dotenv()  # fallback: cwd

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json

# Directory references (no sys.path hack needed — .pth file handles imports)
current_dir = Path(__file__).parent
api_dir = current_dir.parent

# api 모듈 import
from api import pdf_to_image, ner_predict

# 새로운 OCR 모듈 import
from module.ocr import UniversalOCRProcessor

# LLM extraction 모듈 import
from module.llm_extraction import LLMExtractionProcessor

# Consolidation 모듈 import
from module.consolidator import ConsolidationAgent

# Pipeline orchestrator
from web.pipeline import PipelineOrchestrator

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
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'webp', 'bmp', 'gif'}

# OCR 설정 확인
def check_ocr_availability() -> Dict[str, bool]:
    """OCR 엔진 사용 가능 여부 확인"""
    available = {
        'google': False,
        'naver': False,
        'mistral': False,
        'alibaba': False
    }
    
    try:
        # Google 확인 - Google Cloud credentials JSON 파일
        google_creds_path = api_dir / "google_credentials.json"
        if google_creds_path.exists():
            available['google'] = True
        
        # Mistral 확인 - 환경 변수에서 API 키
        mistral_api_key = os.getenv('MISTRAL_API_KEY')
        if mistral_api_key and mistral_api_key != 'your_mistral_api_key_here':
            available['mistral'] = True
        
        # Naver 확인 - 환경 변수에서 API 키
        naver_api_url = os.getenv('NAVER_OCR_API_URL')
        naver_secret_key = os.getenv('NAVER_OCR_SECRET_KEY')
        if naver_api_url and naver_secret_key and naver_api_url != 'your_naver_api_url_here':
            available['naver'] = True
        
        # Alibaba 확인 - 환경 변수에서 API 키
        alibaba_api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
        if alibaba_api_key and alibaba_api_key != 'your_alibaba_api_key_here':
            available['alibaba'] = True
        
        return available
            
    except Exception as e:
        logger.error(f"OCR 설정 확인 중 오류: {e}")
    return available

OCR_AVAILABILITY = check_ocr_availability()

# OCR 엔진 정보 정의
AVAILABLE_OCR_ENGINES = {
    'google': {
        'name': 'Google Cloud Vision API',
        'available': OCR_AVAILABILITY['google'],
        'setup_guide': 'Google Cloud credentials JSON 파일을 설정하세요'
    },
    'mistral': {
        'name': 'Mistral OCR API',
        'available': OCR_AVAILABILITY['mistral'],
        'setup_guide': 'MISTRAL_API_KEY 환경변수를 설정하세요'
    },
    'naver': {
        'name': 'Naver CLOVA OCR API',
        'available': OCR_AVAILABILITY['naver'],
        'setup_guide': 'NAVER_OCR_API_URL과 NAVER_OCR_SECRET_KEY 환경변수를 설정하세요'
    },
    'alibaba': {
        'name': 'Alibaba Cloud Qwen3-VL',
        'available': OCR_AVAILABILITY['alibaba'],
        'setup_guide': 'DASHSCOPE_API_KEY 또는 ALIBABA_API_KEY 환경변수를 설정하세요'
    }
}

# LLM extraction processor 초기화
llm_processor = LLMExtractionProcessor(output_dir=str(Path(__file__).parent / "results"))

# 사용 가능한 NER 모델
AVAILABLE_MODELS = {
    'klue-roberta-large': {
        'name': 'klue/roberta-large',
        'display_name': 'KLUE RoBERTa Large',
        'description': '다국어 특화 모델 (최고 정확도)',
        'accuracy': '95.88%',
        'speed': '보통'
    },
    'google-bert': {
        'name': 'google-bert/bert-base-multilingual-cased',
        'display_name': 'Google mBERT',
        'description': '제일 가벼운 모델 (빠른 속도)',
        'accuracy': '92.72%',
        'speed': '빠름'
    },
    'xlm-roberta': {
        'name': 'FacebookAI/xlm-roberta-large',
        'display_name': 'XLM-RoBERTa Large',
        'description': '한국어 특화 모델',
        'accuracy': '87.60%',
        'speed': '느림'
    }
}

# Pipeline orchestrator 초기화
pipeline_orchestrator = PipelineOrchestrator(
    llm_processor=llm_processor,
    ner_predict_fn=ner_predict,
    available_ner_models=AVAILABLE_MODELS,
    upload_dir=UPLOAD_DIR,
    results_dir=RESULTS_DIR,
)

# Universal OCR는 별도 엔드포인트(/api/ocr-universal)에서 처리

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
        
        # 2단계: OCR 처리 (Universal OCR 사용)
        logger.info(f"Step 2: OCR 처리 (엔진: {ocr_engine})")
        ocr_dir = output_dir / "ocr"
        
        try:
            # Universal OCR Processor 사용
            processor = UniversalOCRProcessor(output_dir=str(ocr_dir))
            
            # OCR 엔진별 모델 설정
            if ocr_engine == 'google':
                provider = 'google'
                model = None
            elif ocr_engine == 'mistral':
                provider = 'mistral'
                model = 'mistral-ocr-latest'
            elif ocr_engine == 'alibaba':
                provider = 'alibaba'
                model = 'qwen3-vl-235b-a22b-instruct'
            else:
                result['error'] = f'지원하지 않는 OCR 엔진: {ocr_engine}'
                return result
            
            # OCR 처리 실행
            if isinstance(input_for_ocr, Path) and input_for_ocr.is_dir():
                # PDF에서 변환된 이미지 디렉토리 처리
                ocr_result = processor.process_directory(
                    input_dir=str(input_for_ocr),
                    provider=provider,
                    model=model,
                    stream=False
                )
            else:
                # 단일 파일 처리
                ocr_result = processor.process_file(
                    file_path=str(input_for_ocr),
                    provider=provider,
                    model=model,
                    stream=False
                )
            
            result['steps']['ocr'] = {
                'success': ocr_result.get('success', False),
                'files_processed': ocr_result.get('processed_files', 0),
                'time': ocr_result.get('processing_time', 0),
                'engine': ocr_engine
            }
            
            if not ocr_result.get('success'):
                error_msg = ocr_result.get('error', 'OCR 처리 실패')
                result['error'] = f'OCR 처리 실패: {error_msg}'
                return result
            
        except Exception as e:
            error_str = str(e)
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
            # Extract entity_types_count from statistics
            statistics = ner_result.get('statistics', {})
            result['entities'] = statistics.get('entity_types_count', {})
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

async def process_document_with_universal_ocr(file_path: Path, output_dir: Path, model_name: str, ocr_provider: str, ocr_model: str = None) -> Dict[str, Any]:
    """
    Universal OCR을 사용한 문서 처리 파이프라인
    
    1. Universal OCR 처리 (PDF → 이미지 변환 포함)
    2. NER 엔티티 추출
    """
    result = {
        'success': False,
        'steps': {},
        'entities': [],
        'entity_count': 0,
        'model_name': model_name,
        'ocr_provider': ocr_provider,
        'ocr_model': ocr_model
    }
    
    try:
        # 1단계: Universal OCR 처리 (PDF 변환 포함)
        logger.info(f"Step 1: Universal OCR 처리 (제공자: {ocr_provider}, 모델: {ocr_model})")
        ocr_dir = output_dir / "ocr"
        
        try:
            # Universal OCR Processor 사용
            processor = UniversalOCRProcessor(provider=ocr_provider, output_dir=str(ocr_dir), model=ocr_model)
            
            # OCR 처리 실행 (PDF → 이미지 변환 자동 처리)
            ocr_result = processor.process_single_file(str(file_path))
            
            result['steps']['ocr'] = {
                'success': ocr_result.get('status') == 'success',
                'files_processed': ocr_result.get('total_pages', 0),
                'time': ocr_result.get('processing_time', 0),
                'provider': ocr_provider,
                'model': ocr_model
            }
            
            if ocr_result.get('status') != 'success':
                error_msg = ocr_result.get('error', 'OCR 처리 실패')
                result['error'] = f'OCR 처리 실패: {error_msg}'
                return result
            
            # OCR 텍스트 추출
            ocr_text = ocr_result.get('full_text', '')
            if not ocr_text:
                result['error'] = 'OCR에서 텍스트를 추출할 수 없습니다'
                return result
            
            logger.info(f"OCR 텍스트 추출 완료: {len(ocr_text)} 문자")
            
        except Exception as e:
            error_str = str(e)
            result['error'] = f'OCR 처리 오류: {error_str}'
            result['steps']['ocr'] = {'success': False, 'error': str(e)}
            return result
        
        # 2단계: NER 엔티티 추출
        logger.info(f"Step 2: NER 엔티티 추출 (모델: {model_name})")
        ner_dir = output_dir / "ner"
        
        try:
            # OCR 텍스트를 임시 파일로 저장
            temp_text_file = ocr_dir / f"{file_path.stem}.ocr"
            with open(temp_text_file, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            
            ner_result = ner_predict(
                str(ocr_dir),  # OCR 결과 디렉토리
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
            # Extract entity_types_count from statistics
            statistics = ner_result.get('statistics', {})
            result['entities'] = statistics.get('entity_types_count', {})
            result['entity_count'] = ner_result.get('total_entities', 0)
            result['output_files'] = ner_result.get('output_files', [])
            result['ner_processing_time'] = ner_result.get('processing_time', 0)
            
            # Debug logging
            logger.info(f"process_document_with_universal_ocr - NER 결과 - total_entities: {ner_result.get('total_entities', 0)}, processing_time: {ner_result.get('processing_time', 0)}")
            logger.info(f"process_document_with_universal_ocr - statistics: {statistics}")
            logger.info(f"process_document_with_universal_ocr - entity_types_count: {statistics.get('entity_types_count', {})}")
            logger.info(f"process_document_with_universal_ocr - result['entity_count'] 설정: {result['entity_count']}")
            
        except Exception as e:
            result['error'] = f'NER 처리 오류: {str(e)}'
            result['steps']['ner'] = {'success': False, 'error': str(e)}
            return result
        
        return result
        
    except Exception as e:
        logger.error(f"처리 중 예상치 못한 오류: {e}", exc_info=True)
        result['error'] = f'처리 중 오류 발생: {str(e)}'
        return result

def _format_ner_entities(ner_result: Dict[str, Any]) -> Dict[str, int]:
    """NER 결과에서 엔티티 타입별 개수를 추출"""
    # Try to get entity_types_count from statistics first (new format)
    statistics = ner_result.get('statistics', {})
    if statistics and 'entity_types_count' in statistics:
        return statistics.get('entity_types_count', {})
    
    # Fallback to old format (entities dict)
    entities_data = ner_result.get('entities', {})
    entity_types_count = {}
    
    for entity_type, entity_list in entities_data.items():
        if isinstance(entity_list, list):
            entity_types_count[entity_type] = len(entity_list)
    
    return entity_types_count

def _count_ner_entities(ner_result: Dict[str, Any]) -> int:
    """NER 결과에서 총 엔티티 개수를 계산"""
    # First try to get total_entities directly from ner_result (ner_predict format)
    if 'total_entities' in ner_result:
        return ner_result.get('total_entities', 0)
    
    # Try to get from statistics
    statistics = ner_result.get('statistics', {})
    if statistics and 'entity_types_count' in statistics:
        entity_types_count = statistics.get('entity_types_count', {})
        # Sum up all entity counts
        return sum(entity_types_count.values()) if isinstance(entity_types_count, dict) else 0
    
    # Fallback to old format (entities dict)
    entities_data = ner_result.get('entities', {})
    total_entities = 0
    
    for entity_type, entity_list in entities_data.items():
        if isinstance(entity_list, list):
            total_entities += len(entity_list)
    
    return total_entities

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
            "models": AVAILABLE_MODELS
        }
    )

@app.get("/pair")
async def index_pair(request: Request):
    """검사 모드 — 계약서+저작물 통합 검사 (contract→work 상속 데모/테스트 UI)"""
    return templates.TemplateResponse("pair.html", {"request": request})

@app.get("/v2")
async def index_v2(request: Request):
    """메인 페이지 v2 (새 디자인)"""
    return templates.TemplateResponse(
        "index2.html",
        {
            "request": request,
            "models": AVAILABLE_MODELS
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
        type: 다운로드 타입 ("entities", "stats", "llm")
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
        elif type == "llm":
            # LLM 메타데이터 파일 찾기 (llm_metadata.json)
            llm_files = list(result_dir.rglob("llm_metadata.json"))
            
            if not llm_files:
                raise HTTPException(status_code=404, detail="LLM 메타데이터 파일을 찾을 수 없습니다")
            
            # 첫 번째 LLM 파일 사용
            file_path = llm_files[0]
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
        'universal_ocr_providers': [k for k, v in OCR_AVAILABILITY.items() if v]
    }

@app.post("/api/ocr-universal")
async def process_universal_ocr(
    file: UploadFile = File(...),
    provider: str = Form("google"),
    model: str = Form(None),
    stream: bool = Form(False)
):
    """Universal OCR processing endpoint
    
    Args:
        file: Uploaded file (PDF, DOCX, DOC, PPTX, XLS, XLSX, PPT, HWP, images)
        provider: OCR provider (google, mistral, naver, alibaba)
        model: Model name (for Alibaba Cloud)
        stream: Enable streaming output
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 없습니다")
        
        file_ext = Path(file.filename).suffix.lower()
        supported_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.hwp',
                               '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'}
        
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"지원되지 않는 파일 형식: {file_ext}. 지원 형식: {', '.join(supported_extensions)}"
            )
        
        # Validate provider
        if provider not in ['google', 'mistral', 'naver', 'alibaba']:
            raise HTTPException(status_code=400, detail="지원되지 않는 OCR 제공자")
        
        # Check provider availability
        if not OCR_AVAILABILITY.get(provider, False):
            raise HTTPException(status_code=400, detail=f"{provider} OCR 제공자가 사용할 수 없습니다")
        
        # Create timestamped result directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        result_dir = RESULTS_DIR / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = result_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Universal OCR processing: {file.filename} with {provider}")
        
        # Initialize OCR processor
        ocr_output_dir = result_dir / "ocr"
        processor = UniversalOCRProcessor(provider, str(ocr_output_dir), model)
        
        start_time = datetime.now()
        
        if stream:
            # Streaming processing
            from fastapi.responses import StreamingResponse
            
            def generate_stream():
                try:
                    for chunk in processor.process_single_file_streaming(str(file_path)):
                        yield chunk
                except Exception as e:
                    yield f"Error: {str(e)}"
            
            # Encode filename for HTTP headers (RFC 5987)
            import urllib.parse
            encoded_filename = urllib.parse.quote(file.filename.encode('utf-8'))
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}.ocr"
                }
            )
        else:
            # Regular processing
            result = processor.process_single_file(str(file_path))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                'request_id': timestamp,
                'filename': file.filename,
                'provider': provider,
                'model': model or 'default',
                'success': result['status'] == 'success',
                'total_pages': result.get('total_pages', 0),
                'total_text_length': result.get('total_text_length', 0),
                'processing_time': round(processing_time, 2),
                'result_directory': str(result_dir)
            }
            
            if result['status'] == 'success':
                response['extracted_text'] = result.get('full_text', '')
                response['pages'] = result.get('pages', [])
            else:
                response['error'] = result.get('error', '알 수 없는 오류')
            
            # Save result JSON
            result_json_path = result_dir / 'universal_ocr_result.json'
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            
            status_code = 200 if result['status'] == 'success' else 500
            return JSONResponse(content=response, status_code=status_code)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Universal OCR 처리 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/info")
async def api_info():
    """API 정보"""
    return {
        'title': 'NER 엔티티 추출 API',
        'version': '2.0.0',
        'framework': 'FastAPI',
        'models': AVAILABLE_MODELS,
        'universal_ocr_providers': [k for k, v in OCR_AVAILABILITY.items() if v],
        'llm_models': llm_processor.get_available_models()
    }

def _send_progress_update(message: str, step: int, percent: int, data: Dict = None):
    """Helper function to format SSE progress update"""
    update = {
        "message": message,
        "step": step,
        "percent": percent,
        "timestamp": datetime.now().isoformat()
    }
    if data:
        update.update(data)
    sse_message = f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
    logger.debug(f"SSE update: step={step}, percent={percent}, has_result={'result' in update}")
    return sse_message

@app.post("/api/llm-extract")
async def llm_extract_metadata(
    file: UploadFile = File(...),
    model_name: str = Form(default="alibaba-qwen3.5-122b-a10b"),
    document_type: str = Form(default="기타문서"),
    ocr_provider: str = Form(default="alibaba"),
    ocr_model: str = Form(default=None),
    ner_model: str = Form(default="klue-roberta-large"),
    consolidate: bool = Form(default=True),
    consolidation_model: str = Form(default="alibaba-qwen3.5-122b-a10b"),
    vlm_prefer: str = Form(default="gemma"),  # 멀티모달(이미지) VLM 우선: gemma(기본)→qwen 폴백
    contract_metadata: str = Form(default=None),  # 계약서→저작물 상속용: 계약서 분석 결과 JSON (이미지 처리 시)
    stream: bool = Form(default=False)
):
    """LLM을 사용한 메타데이터 추출 (SSE 지원, 통합 기능 포함)"""

    # Read file content before async generator (FastAPI closes handle after)
    try:
        file_content = await file.read()
        filename = file.filename
    except Exception as e:
        logger.error(f"파일 읽기 오류: {e}")
        error_response = {"success": False, "error": f"파일 읽기 오류: {str(e)}"}
        if stream:
            async def error_stream():
                yield _send_progress_update(str(e), 0, 0, {"error": str(e), "result": error_response})
            return StreamingResponse(error_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        return JSONResponse(content=error_response, status_code=400)

    # 계약서 메타데이터(JSON 문자열) 파싱 — 잘못된 JSON 은 400 으로 조기 반환
    contract_meta_dict = None
    if contract_metadata:
        if len(contract_metadata) > 200_000:
            return JSONResponse(content={"success": False,
                                         "error": "contract_metadata too large (>200KB)"}, status_code=400)
        try:
            contract_meta_dict = json.loads(contract_metadata)
            if not isinstance(contract_meta_dict, dict):
                raise ValueError("contract_metadata must be a JSON object")
        except Exception as e:
            return JSONResponse(content={"success": False,
                                         "error": f"contract_metadata JSON 파싱 실패: {e}"}, status_code=400)

    # Validate inputs
    if not filename or not allowed_file(filename):
        err = f'지원하지 않는 파일 형식입니다. 허용: {", ".join(ALLOWED_EXTENSIONS)}'
        if stream:
            async def err_stream():
                yield _send_progress_update(err, 0, 0, {"error": err})
            return StreamingResponse(err_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        return JSONResponse(content={"success": False, "error": err}, status_code=400)

    if ner_model not in AVAILABLE_MODELS:
        err = f"잘못된 NER 모델: {ner_model}"
        return JSONResponse(content={"success": False, "error": err}, status_code=400)
    
    # ── SSE streaming mode ──
    if stream:
        async def process_with_progress():
            try:
                ctx = pipeline_orchestrator.setup(file_content, filename)
                yield _send_progress_update("파일 업로드 완료", 1, 10, {"request_id": ctx["request_id"]})
                await asyncio.sleep(0.01)

                # ── Modality routing: image → VLM(Gemma→Qwen); video/audio → guarded; doc/text → OCR path ──
                modality = pipeline_orchestrator.run_router(ctx)["modality"]
                yield _send_progress_update(f"입력 유형 감지: {modality}", 1, 15, {"modality": modality})
                await asyncio.sleep(0.01)

                if modality == "image":
                    yield _send_progress_update(f"시각 메타데이터 추출 중 (VLM: {vlm_prefer})...", 2, 35)
                    await asyncio.sleep(0.01)
                    vlm_result = await asyncio.to_thread(pipeline_orchestrator.run_vlm, ctx, vlm_prefer)
                    backend = vlm_result.get("model_used")
                    if not vlm_result.get("success"):
                        error_response = {"success": False, "error": vlm_result.get("error", "VLM 추출 실패"),
                                          "request_id": ctx["request_id"], "filename": ctx["filename"],
                                          "modality": "image", "vlm_backend": backend}
                        yield _send_progress_update(f"VLM 추출 실패 ({backend})", 0, 0, {"error": vlm_result.get("error"), "result": error_response})
                        return
                    yield _send_progress_update(f"VLM 추출 완료 ({backend})", 3, 70, {"vlm_backend": backend})
                    await asyncio.sleep(0.01)
                    con_result, con_success, con_error = None, False, None
                    do_con = consolidate and vlm_result.get("success")
                    if do_con:
                        yield _send_progress_update("메타데이터 통합 중...", 4, 90)
                        await asyncio.sleep(0.01)
                        con_result, con_success, con_error = await asyncio.to_thread(
                            pipeline_orchestrator.run_consolidation, vlm_result,
                            {"success": False, "entities": {}, "total_entities": 0, "extracted_entities": []},
                            "", document_type, ctx["result_dir"], consolidation_model)
                    response = pipeline_orchestrator.build_response(
                        ctx, model_name=backend or "VLM", document_type=document_type,
                        ocr_text="", ocr_provider="(image/VLM)", ocr_model=None,
                        llm_result=vlm_result, ner_model=None, ner_result=None,
                        consolidate=do_con, consolidation_model=consolidation_model,
                        consolidation_result=con_result, consolidation_success=con_success,
                        consolidation_error=con_error)
                    response["modality"] = "image"; response["vlm_backend"] = backend
                    response["vlm_raw"] = vlm_result.get("_vlm_raw")
                    if contract_meta_dict:
                        yield _send_progress_update("계약서 권리정보 상속 중...", 4, 95)
                        pipeline_orchestrator.apply_contract_inheritance(response, contract_meta_dict)
                    pipeline_orchestrator.save_results(ctx["result_dir"], response, con_result, con_success)
                    yield _send_progress_update("처리 완료", 5, 100, {"result": response})
                    await asyncio.sleep(0.01)
                    return

                if modality in ("video", "audio"):
                    error_response = {"success": False, "modality": modality, "request_id": ctx["request_id"],
                                      "filename": ctx["filename"],
                                      "error": f"{modality} 처리는 아직 지원되지 않습니다 (멀티모달 P3 예정)"}
                    yield _send_progress_update(f"{modality} 미지원 (P3 예정)", 0, 0, {"error": error_response["error"], "result": error_response})
                    return

                yield _send_progress_update("OCR 텍스트 추출 중...", 2, 20)
                await asyncio.sleep(0.01)
                ocr_text, ocr_result = pipeline_orchestrator.run_ocr(ctx, ocr_provider, ocr_model)
                yield _send_progress_update(f"OCR 완료 ({len(ocr_text)} 문자)", 2, 40)
                await asyncio.sleep(0.01)

                # Guard: stop early if OCR returned no text
                if not ocr_text or not ocr_text.strip():
                    error_response = {
                        "success": False,
                        "error": "OCR에서 텍스트를 추출하지 못했습니다. 다른 OCR 제공자를 선택하거나 파일을 확인해 주세요.",
                        "request_id": ctx["request_id"],
                        "filename": ctx["filename"],
                        "ocr_text": "",
                        "ocr_provider": ocr_provider,
                    }
                    yield _send_progress_update("OCR 텍스트가 비어있어 처리를 중단합니다", 0, 0, {"error": "OCR 텍스트 없음", "result": error_response})
                    return

                yield _send_progress_update("LLM + NER 동시 추출 중...", 3, 50)
                await asyncio.sleep(0.01)

                # Run LLM and NER concurrently using threads
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    llm_future = executor.submit(
                        pipeline_orchestrator.run_llm, ocr_text, document_type, ctx["filename"], model_name
                    )
                    ner_future = executor.submit(
                        pipeline_orchestrator.run_ner, ocr_result, ctx["result_dir"], ner_model, ocr_text
                    )
                    llm_result = llm_future.result()
                    ner_result = ner_future.result()

                entity_count = pipeline_orchestrator._count_ner_entities(ner_result)
                yield _send_progress_update(f"LLM + NER 완료 (엔티티 {entity_count}개)", 4, 90)
                await asyncio.sleep(0.01)

                con_result, con_success, con_error = None, False, None
                if consolidate:
                    yield _send_progress_update("메타데이터 통합 중...", 5, 95)
                    await asyncio.sleep(0.01)
                    con_result, con_success, con_error = pipeline_orchestrator.run_consolidation(
                        llm_result, ner_result, ocr_text, document_type,
                        ctx["result_dir"], consolidation_model,
                    )
                    msg = "메타데이터 통합 완료" if con_success else f"통합 실패: {con_error}"
                    yield _send_progress_update(msg, 5, 98)
                    await asyncio.sleep(0.01)

                response = pipeline_orchestrator.build_response(
                    ctx,
                    model_name=model_name, document_type=document_type,
                    ocr_text=ocr_text, ocr_provider=ocr_provider, ocr_model=ocr_model,
                    llm_result=llm_result, ner_model=ner_model, ner_result=ner_result,
                    consolidate=consolidate, consolidation_model=consolidation_model,
                    consolidation_result=con_result,
                    consolidation_success=con_success, consolidation_error=con_error,
                )
                pipeline_orchestrator.save_results(ctx["result_dir"], response, con_result, con_success)

                yield _send_progress_update("처리 완료", 5, 100, {"result": response})
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                error_response = {"success": False, "error": str(e)}
                yield _send_progress_update(str(e), 0, 0, {"error": str(e), "result": error_response})

        return StreamingResponse(
            process_with_progress(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # ── Non-streaming mode (simple) ──
    try:
        result = pipeline_orchestrator.run(
            file_content, filename,
            model_name=model_name, document_type=document_type,
            ocr_provider=ocr_provider, ocr_model=ocr_model,
            ner_model=ner_model, consolidate=consolidate,
            consolidation_model=consolidation_model,
            vlm_prefer=vlm_prefer,
            contract_metadata=contract_meta_dict,
        )
        status_code = 200 if result.get("success", False) else 500
        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/api/llm-models")
async def get_llm_models():
    """사용 가능한 LLM 모델 목록"""
    return llm_processor.get_available_models()

@app.get("/api/list-files/{request_id}")
async def list_result_files(request_id: str):
    """결과 디렉토리의 파일 목록 반환"""
    try:
        result_dir = RESULTS_DIR / request_id
        
        if not result_dir.exists():
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")
        
        # 모든 파일 찾기 (재귀적으로)
        all_files = []
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                # 상대 경로로 변환
                relative_path = file_path.relative_to(result_dir)
                all_files.append(str(relative_path))
        
        return {"files": all_files}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{request_id}/{file_path:path}")
async def download_specific_file(request_id: str, file_path: str):
    """결과 디렉토리에서 특정 파일 다운로드"""
    try:
        result_dir = RESULTS_DIR / request_id
        
        if not result_dir.exists():
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")
        
        # 파일 경로 보안 검사
        target_file = result_dir / file_path
        
        # 경로 조작 공격 방지
        try:
            target_file.resolve().relative_to(result_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다")
        
        if not target_file.exists() or not target_file.is_file():
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
        
        return FileResponse(
            path=target_file,
            filename=target_file.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 다운로드 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ner-extract")
async def ner_extract_entities(
    file: UploadFile = File(...),
    model: str = Form("klue-roberta-large"),
    ocr_provider: str = Form("google"),
    ocr_model: str = Form(None)
):
    """NER 엔티티 추출 전용 엔드포인트"""
    try:
        # 파일 검증
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일명이 비어있습니다")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f'지원하지 않는 파일 형식입니다. 허용: {", ".join(ALLOWED_EXTENSIONS)}'
            )
        
        # 모델 검증
        logger.info(f"NER 엔티티 추출 요청 - 받은 모델 파라미터: {model}")
        if model not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail="잘못된 모델 선택")
        
        model_name = AVAILABLE_MODELS[model]['name']
        logger.info(f"사용할 NER 모델: {model_name} (키: {model})")
        
        # OCR 제공자 검증
        if ocr_provider not in AVAILABLE_OCR_ENGINES:
            raise HTTPException(status_code=400, detail="잘못된 OCR 제공자 선택")
        
        if not AVAILABLE_OCR_ENGINES[ocr_provider]['available']:
            setup_guide = AVAILABLE_OCR_ENGINES[ocr_provider]['setup_guide']
            raise HTTPException(
                status_code=400,
                detail=f'{AVAILABLE_OCR_ENGINES[ocr_provider]["name"]} 설정이 필요합니다. {setup_guide}'
            )
        
        # 파일 저장
        filename = secure_filename(file.filename)
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        upload_path = UPLOAD_DIR / request_id / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"NER 처리 시작: {filename} ({file_size_mb:.2f}MB), 모델: {model_name}, OCR: {ocr_provider}")
        
        # 결과 디렉토리
        result_dir = RESULTS_DIR / request_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 처리 시작
        start_time = datetime.now()
        result = await process_document_with_universal_ocr(upload_path, result_dir, model_name, ocr_provider, ocr_model)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 응답 생성
        # Use processing_time from result if available, otherwise use calculated time
        # The result should include total processing time from ner_predict
        final_processing_time = result.get('ner_processing_time') or processing_time
        
        response = {
            'success': result['success'],
            'request_id': request_id,
            'filename': filename,
            'file_size_mb': round(file_size_mb, 2),
            'model': AVAILABLE_MODELS[model]['display_name'],
            'model_key': model,
            'ocr_engine': AVAILABLE_OCR_ENGINES[ocr_provider]['name'],
            'ocr_engine_key': ocr_provider,
            'entities': result.get('entities', {}),
            'entity_count': result.get('entity_count', 0),
            'steps': result.get('steps', {}),
            'processing_time': round(final_processing_time, 2)
        }
        
        # Debug logging
        logger.info(f"NER 응답 생성 - entity_count: {response['entity_count']}, processing_time: {response['processing_time']}")
        logger.info(f"result에서 가져온 값들 - entity_count: {result.get('entity_count', 0)}, entities: {result.get('entities', {})}")
        logger.info(f"ner_result에서 가져온 값들 - total_entities: {result.get('steps', {}).get('ner', {}).get('entity_count', 'N/A')}")
        
        if not result['success']:
            response['error'] = result.get('error', '알 수 없는 오류')
        
        # 결과 JSON 저장
        result_json_path = result_dir / 'ner_result.json'
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        status_code = 200 if result['success'] else 500
        return JSONResponse(content=response, status_code=status_code)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NER 처리 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm-test")
async def test_llm_extraction(model_name: str = Form(default="alibaba-qwen3.5-122b-a10b")):
    """LLM 추출 테스트"""
    try:
        result = llm_processor.test_extraction(model_name)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.error(f"LLM 테스트 오류: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"LLM 테스트 오류: {str(e)}"},
            status_code=500
        )

# ====== 무하유 프론트엔드 정적 파일 서빙 ======
# kogl-classifier (npm run export → out/) 빌드 결과물
_frontend_dir = api_dir.parent / "kogl-classifier" / "out"
if _frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logging.getLogger(__name__).info(f"프론트엔드 마운트: /frontend → {_frontend_dir}")

if __name__ == '__main__':
    print("=" * 80)
    print("  NER 엔티티 추출 웹 애플리케이션 (FastAPI)")
    print("=" * 80)
    print(f"\n업로드 디렉토리: {UPLOAD_DIR}")
    print(f"결과 디렉토리: {RESULTS_DIR}")
    
    print(f"\n사용 가능한 Universal OCR 제공자:")
    provider_names = {
        'google': 'Google Vision API',
        'mistral': 'Mistral OCR',
        'naver': 'Naver CLOVA OCR',
        'alibaba': 'Alibaba Cloud Qwen3-VL'
    }
    for key, available in OCR_AVAILABILITY.items():
        status = "✓" if available else "✗"
        name = provider_names.get(key, key)
        print(f"  {status} {name}")
    
    print(f"\n사용 가능한 NER 모델:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  - {info['display_name']}: {info['description']}")
    
    print(f"\n사용 가능한 LLM 모델:")
    llm_models = llm_processor.get_available_models()
    for key, info in llm_models.items():
        model_type = "로컬" if info['type'] == 'local' else "클라우드"
        print(f"  - {info['name']}: {info['description']} ({model_type})")
    
    print("\n" + "=" * 80)
    print("서버 시작: http://localhost:5000")
    print("API 문서: http://localhost:5000/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
