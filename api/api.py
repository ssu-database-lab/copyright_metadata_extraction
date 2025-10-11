#!/usr/bin/env python3
"""
통합 AI API: PDF 처리, OCR, NER 기능을 제공하는 모듈화된 API

이 API는 다음 기능들을 제공합니다:
1. PDF → 이미지 변환 (pdf_to_image)
2. OCR 처리 (ocr_naver, ocr_mistral, ocr_google, ocr_complete)
3. NER 훈련 (ner_train)
4. NER 예측 (ner_predict)

각 함수는 독립적으로 사용 가능하며, input_path와 output_path를 지정하여 사용합니다.

사용법:
    from api import pdf_to_image, ocr_google, ner_predict
    
    # PDF를 이미지로 변환
    result = pdf_to_image("document.pdf", "images/")
    
    # Google OCR로 텍스트 추출
    result = ocr_google("images/", "ocr_results/")
    
    # NER로 엔티티 추출
    result = ner_predict("ocr_results/", "ner_results/", "models/ner_model")
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# 모듈 import
from module import (
    pdf_to_image,           # PDF → 이미지 변환
    ocr_naver,              # Naver OCR
    ocr_mistral,            # Mistral OCR  
    ocr_google,             # Google OCR
    ocr_complete,           # 통합 OCR
    ner_predict,            # NER 예측
    ner_evaluate            # NER 평가
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API 버전 정보
__version__ = "1.0.0"
__author__ = "DBLab"

# API에서 제공하는 함수들
__all__ = [
    'pdf_to_image',
    'ocr_naver', 'ocr_mistral', 'ocr_google', 'ocr_complete',
    'ner_predict',  # NER 예측
    'ner_evaluate',  # NER 평가
    'process_pdf_to_ner',  # 통합 파이프라인
    'get_api_info'  # API 정보
]

def process_pdf_to_ner(
    input_pdf_path: str,
    output_dir: str,
    model_name: str = None,  # None일 경우 roberta_large 기본 사용
    ocr_engine: str = "google",
    ner_confidence_threshold: float = 0.85,
    save_intermediate_files: bool = False
) -> Dict[str, Any]:
    """
    PDF에서 NER까지 전체 파이프라인 처리
    
    Args:
        input_pdf_path (str): 입력 PDF 파일 경로
        output_dir (str): 결과 저장 디렉토리
        model_name (str): NER 모델명 (기본값: "roberta_large")
        ocr_engine (str): 사용할 OCR 엔진 ("google", "naver", "mistral", "complete")
        ner_confidence_threshold (float): NER 신뢰도 임계값 (기본값: 0.85)
        save_intermediate_files (bool): 중간 파일 저장 여부
    
    Returns:
        Dict[str, Any]: 전체 처리 결과
    
    사용법:
        # 단일 PDF 처리
        result = process_pdf_to_ner(
            input_pdf_path="document.pdf",
            output_dir="results/"
        )
        
        # 고급 설정
        result = process_pdf_to_ner(
            input_pdf_path="contract.pdf",
            output_dir="results/",
            model_name="roberta_contract",
            ocr_engine="complete",
            ner_confidence_threshold=0.9,
            save_intermediate_files=True
        )
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"전체 파이프라인 시작: {input_pdf_path}")
        
        # 출력 디렉토리 설정
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 중간 파일 디렉토리
        temp_dir = output_path / "temp" if save_intermediate_files else Path("temp_processing")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "success": True,
            "input_file": input_pdf_path,
            "output_directory": str(output_path),
            "processing_steps": []
        }
        
        # 1단계: PDF → 이미지
        logger.info("1단계: PDF → 이미지 변환")
        image_dir = temp_dir / "images"
        pdf_result = pdf_to_image(
            input_path=input_pdf_path,
            output_path=str(image_dir)
        )
        
        if not pdf_result.get("success", False):
            return {
                "success": False,
                "error": f"PDF 변환 실패: {pdf_result.get('error', 'Unknown error')}",
                "processing_time": time.time() - start_time
            }
        
        results["processing_steps"].append({
            "step": "pdf_to_image",
            "success": True,
            "details": pdf_result
        })
        
        # 2단계: 이미지 → 텍스트 (OCR)
        logger.info(f"2단계: OCR 처리 ({ocr_engine})")
        ocr_dir = temp_dir / "ocr"
        
        if ocr_engine == "google":
            ocr_result = ocr_google(str(image_dir), str(ocr_dir))
        elif ocr_engine == "naver":
            ocr_result = ocr_naver(str(image_dir), str(ocr_dir))
        elif ocr_engine == "mistral":
            ocr_result = ocr_mistral(str(image_dir), str(ocr_dir))
        elif ocr_engine == "complete":
            ocr_result = ocr_complete(str(image_dir), str(ocr_dir))
        else:
            return {
                "success": False,
                "error": f"지원하지 않는 OCR 엔진: {ocr_engine}",
                "processing_time": time.time() - start_time
            }
        
        if not ocr_result.get("success", False):
            return {
                "success": False,
                "error": f"OCR 처리 실패: {ocr_result.get('error', 'Unknown error')}",
                "processing_time": time.time() - start_time
            }
        
        results["processing_steps"].append({
            "step": "ocr_processing",
            "engine": ocr_engine,
            "success": True,
            "details": ocr_result
        })
        
        # 3단계: 텍스트 → 엔티티 (NER)
        logger.info("3단계: NER 처리")
        ner_result = ner_predict(
            input_path=str(ocr_dir),
            output_path=str(output_path),
            model_name=model_name,  # 사용자가 지정한 model_name 사용
            confidence_threshold=ner_confidence_threshold,
            output_format="both"
        )
        
        if not ner_result.get("success", False):
            return {
                "success": False,
                "error": f"NER 처리 실패: {ner_result.get('error', 'Unknown error')}",
                "processing_time": time.time() - start_time
            }
        
        results["processing_steps"].append({
            "step": "ner_processing",
            "success": True,
            "details": ner_result
        })
        
        # 4단계: 임시 파일 정리
        if not save_intermediate_files and temp_dir.name == "temp_processing":
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("임시 파일 정리 완료")
            except Exception as e:
                logger.warning(f"임시 파일 정리 중 오류: {e}")
        
        # 최종 결과
        processing_time = time.time() - start_time
        results.update({
            "total_processing_time": processing_time,
            "final_outputs": ner_result.get("output_files", []),
            "entities_found": ner_result.get("total_entities", 0),
            "entity_types": ner_result.get("entity_types", {}),
        })
        
        logger.info(f"전체 파이프라인 완료: {processing_time:.1f}초")
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def get_api_info() -> Dict[str, Any]:
    """
    API 정보 및 사용 가능한 기능 반환
    
    Returns:
        Dict[str, Any]: API 정보
    """
    return {
        "version": __version__,
        "author": __author__,
        "available_functions": {
            "pdf_processing": {
                "pdf_to_image": "PDF 파일을 이미지로 변환"
            },
            "ocr_processing": {
                "ocr_naver": "Naver CLOVA OCR API 사용",
                "ocr_mistral": "Mistral AI Vision API 사용", 
                "ocr_google": "Google Cloud Vision API 사용",
                "ocr_complete": "여러 OCR 엔진 통합 사용"
            },
            "ner_processing": {
                "ner_predict": "NER 예측 수행 (모델은 별도 훈련 필요)"
            },
            "integrated": {
                "process_pdf_to_ner": "PDF → OCR → NER 전체 파이프라인"
            }
        },
        "supported_formats": {
            "input": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "TXT"],
            "output": ["JSON", "CSV", "TXT"]
        }
    }

def main():
    """
    API 테스트 및 사용 예시
    """
    print("=" * 80)
    print("    통합 AI API - 테스트 및 사용 예시")
    print("=" * 80)
    
    # API 정보 출력
    api_info = get_api_info()
    print(f"\nAPI 버전: {api_info['version']}")
    print(f"개발자: {api_info['author']}")
    
    print("\n사용 가능한 기능:")
    for category, functions in api_info["available_functions"].items():
        print(f"\n  {category.upper()}:")
        for func_name, description in functions.items():
            print(f"    • {func_name}: {description}")
    
    print(f"\n지원 형식:")
    print(f"  입력: {', '.join(api_info['supported_formats']['input'])}")
    print(f"  출력: {', '.join(api_info['supported_formats']['output'])}")
    
    # 테스트 실행
    print("\n" + "=" * 80)
    print("    함수별 테스트 예시")
    print("=" * 80)
    
    # 1. PDF → 이미지 변환 테스트
    print("\n1. PDF → 이미지 변환")
    print("-" * 40)
    print("""
# 사용법 예시:
result = pdf_to_image(
    input_path="document.pdf",      # 변환할 PDF 파일
    output_path="images/",          # 이미지 저장 디렉토리
    dpi=200                         # 이미지 해상도
)

if result["success"]:
    print(f"변환 완료: {result['total_images']}개 이미지 생성")
else:
    print(f"변환 실패: {result['error']}")
    """)
    
    # 2. OCR 처리 테스트
    print("\n2. OCR 처리")
    print("-" * 40)
    print("""
# Google OCR 사용:
result = ocr_google(
    input_path="images/",           # 이미지 디렉토리
    output_path="ocr_results/"      # 텍스트 결과 저장소
)

# Naver OCR 사용 (API 키 필요):
result = ocr_naver(
    input_path="images/",
    output_path="ocr_results/",
    api_url="https://...",          # Naver OCR API URL
    secret_key="your_secret_key"    # Naver OCR Secret Key
)

# 통합 OCR 사용:
result = ocr_complete(
    input_path="images/",
    output_path="ocr_results/",
    ocr_engines=["google", "naver"] # 사용할 엔진들
)
    """)
    
    # 3. NER 처리 테스트
    print("\n3. NER 처리")
    print("-" * 40) 
    print("""
        # NER 모델 훈련: API에서 제공되지 않음 (ner_train.py 직접 실행)
        # 훈련된 모델은 models/ 디렉토리에 있어야 함# NER 예측:
result = ner_predict(
    input_path="ocr_results/",          # 텍스트 파일들
    output_path="ner_results/",         # 결과 저장소  
    model_path="models/my_ner",         # 훈련된 모델
    confidence_threshold=0.8            # 신뢰도 임계값
)
    """)
    
    # 4. 전체 파이프라인 테스트
    print("\n4. 전체 파이프라인")
    print("-" * 40)
    print("""
# PDF → OCR → NER 전체 처리:
result = process_pdf_to_ner(
    input_pdf_path="contract.pdf",      # 입력 PDF
    output_dir="final_results/",        # 최종 결과 저장소
    model_path="models/contract_ner",   # NER 모델
    ocr_engine="google",                # OCR 엔진 선택
    ner_confidence_threshold=0.8        # NER 신뢰도
)

if result["success"]:
    print(f"처리 완료: {result['entities_found']}개 엔티티 발견")
    print(f"출력 파일: {result['final_outputs']}")
else:
    print(f"처리 실패: {result['error']}")
    """)
    
    print("\n" + "=" * 80)
    print("API 사용 준비가 완료되었습니다!")
    print("각 함수의 자세한 사용법은 함수의 docstring을 참고하세요.")

if __name__ == "__main__":
    main()