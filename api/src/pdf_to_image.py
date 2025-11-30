"""
PDF 처리 시스템
PDF 파일을 이미지로 변환하는 기능을 제공합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import pymupdf  # PyMuPDF
from tqdm import tqdm
import shutil

# 로깅 설정
logger = logging.getLogger(__name__)

def _copy_image_files(input_path: Path, output_path: Path) -> List[str]:
    """
    이미지 파일들을 디렉토리 구조를 유지하면서 복사합니다.
    input_path를 기준으로 한 상대 경로만 출력에 반영합니다.
    
    Args:
        input_path: 입력 경로 (기준점)
        output_path: 출력 경로
        
    Returns:
        복사된 이미지 파일 경로 리스트
    """
    copied_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']
    
    try:
        # PDF 변환 출력 디렉토리 생성
        pdf_convert_dir = output_path / "pdf_convert"
        pdf_convert_dir.mkdir(parents=True, exist_ok=True)
        
        if input_path.is_file():
            # 단일 이미지 파일 처리
            if input_path.suffix.lower() in image_extensions:
                # 파일명에서 확장자 제거한 디렉토리 생성
                file_dir = pdf_convert_dir / input_path.stem
                file_dir.mkdir(parents=True, exist_ok=True)
                
                # 이미지 파일 복사 (001.png 형식으로)
                dest_file = file_dir / "001.png"
                shutil.copy2(input_path, dest_file)
                copied_files.append(str(dest_file))
                
        elif input_path.is_dir():
            # 디렉토리 내 이미지 파일들 처리
            for item_path in input_path.rglob("*"):  # 재귀적으로 모든 파일 찾기
                if item_path.is_file() and item_path.suffix.lower() in image_extensions:
                    # input_path를 기준으로 한 상대 경로만 사용
                    try:
                        relative_path = item_path.relative_to(input_path)
                        # input_path 하위의 상대 경로만 출력에 반영
                        if relative_path.parent != Path('.'):
                            # 중간 디렉토리가 있는 경우: a/b/image.jpg -> a/b/ 에 image.png
                            dest_dir = pdf_convert_dir / relative_path.parent
                        else:
                            # input_path 직접 하위: image.jpg -> root에 image.png
                            dest_dir = pdf_convert_dir
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        
                        # PNG 형식으로 변환하여 복사
                        dest_file = dest_dir / f"{item_path.stem}.png"
                        shutil.copy2(item_path, dest_file)
                        copied_files.append(str(dest_file))
                    
                    except ValueError:
                        # 상대 경로 계산 실패시 폴백
                        dest_file = pdf_convert_dir / f"{item_path.stem}.png"
                        shutil.copy2(item_path, dest_file)
                        copied_files.append(str(dest_file))
                    
    except Exception as e:
        logger.error(f"이미지 파일 복사 실패: {e}")
        
    return copied_files

def pdf_to_image(input_path: str, output_path: str, dpi: int = 200, 
                image_format: str = "png") -> Dict[str, Any]:
    """
    PDF 파일을 이미지로 변환
    
    Args:
        input_path (str): PDF 파일 경로 또는 이미지 파일 경로 또는 디렉토리 경로
        output_path (str): 이미지 저장할 디렉토리 경로  
        dpi (int): 이미지 해상도 (기본값: 200, 권장: 150-300)
        image_format (str): 출력 이미지 형식 ("png", "jpg", "jpeg")
    
    Returns:
        Dict[str, Any]: 변환 결과
        {
            "success": bool,              # 성공 여부
            "converted_files": int,       # 변환된 PDF 파일 수
            "total_images": int,          # 생성된 총 이미지 수
            "output_directory": str,      # 출력 디렉토리 경로
            "images": List[str],          # 생성된 이미지 파일 경로들
            "processing_time": float,     # 처리 시간 (초)
            "error": str (if success=False)  # 오류 메시지
        }
    
    사용법:
        # 단일 PDF 파일 변환
        result = pdf_to_image(
            input_path="document.pdf", 
            output_path="output/images/"
        )
        
        # 디렉토리 내 모든 PDF 파일 일괄 변환
        result = pdf_to_image(
            input_path="pdf_folder/", 
            output_path="output/images/"
        )
        
        # 고해상도 변환 (OCR 품질 향상)
        result = pdf_to_image(
            input_path="document.pdf", 
            output_path="output/images/",
            dpi=300,
            image_format="png"
        )
        
        # 용량 절약을 위한 JPG 변환
        result = pdf_to_image(
            input_path="pdf_folder/", 
            output_path="output/images/",
            dpi=150,
            image_format="jpg"
        )
    
    출력 구조:
        output_path/
        ├── document1/
        │   ├── 001.png
        │   ├── 002.png
        │   └── 003.png
        └── document2/
            ├── 001.png
            └── 002.png
    
    Note:
        - PDF가 암호화되어 있으면 변환에 실패할 수 있습니다
        - 대용량 PDF의 경우 메모리 사용량에 주의하세요
        - DPI가 높을수록 이미지 품질은 좋아지지만 파일 크기가 커집니다
        - OCR 용도로는 200-300 DPI가 적당합니다
    """
    import time
    
    start_time = time.time()
    
    try:
        input_path_obj = Path(input_path)
        
        # pdf_convert 디렉토리 생성
        pdf_convert_dir = Path(output_path) / "pdf_convert"
        pdf_convert_dir.mkdir(parents=True, exist_ok=True)
        
        # 지원하는 이미지 형식 확인
        if image_format.lower() not in ['png', 'jpg', 'jpeg']:
            return {
                "success": False,
                "error": f"지원하지 않는 이미지 형식: {image_format}. 'png', 'jpg', 'jpeg' 중 선택하세요."
            }
        
        converted_files = 0
        all_images = []
        
        # 파일 형식별로 분류
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']
        
        if input_path_obj.is_file():
            if input_path_obj.suffix.lower() == '.pdf':
                pdf_files = [input_path_obj]
                image_files = []
            elif input_path_obj.suffix.lower() in image_extensions:
                pdf_files = []
                image_files = [input_path_obj]
            else:
                return {
                    "success": False,
                    "error": f"지원하지 않는 파일 형식: {input_path_obj.suffix}"
                }
        elif input_path_obj.is_dir():
            pdf_files = list(input_path_obj.rglob("*.pdf"))
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path_obj.rglob(f"*{ext}"))
        else:
            return {
                "success": False,
                "error": f"유효하지 않은 입력 경로: {input_path}"
            }
        
        # 이미지 파일 복사 처리
        if image_files:
            print(f"{len(image_files)}개 이미지 파일 복사 시작...")
            copied_images = _copy_image_files(input_path_obj, Path(output_path))
            all_images.extend(copied_images)
            converted_files += len(image_files)
        
        # PDF 파일 변환 처리
        if pdf_files:
            print(f"{len(pdf_files)}개 PDF 파일 변환 시작...")
            
            for pdf_path in tqdm(pdf_files, desc="PDF 파일 처리", unit="파일"):
                try:
                    # PDF 파일의 디렉토리 구조를 유지하여 출력 디렉토리 생성
                    pdf_path = Path(pdf_path)  # Path 객체로 변환
                    
                    # 입력 경로를 기준으로 상대 경로 계산하여 출력 구조 생성
                    if input_path_obj.is_file():
                        # 단일 파일인 경우 - 파일명으로 디렉토리 생성
                        pdf_output_dir = pdf_convert_dir / pdf_path.stem
                    else:
                        # 디렉토리인 경우 - input_path를 기준으로 한 상대 경로 구조 유지
                        try:
                            relative_path = pdf_path.relative_to(input_path_obj)
                            # input_path 하위의 상대 경로 + 파일명(확장자 제외)
                            if relative_path.parent != Path('.'):
                                # 중간 디렉토리가 있는 경우: a/b/c.pdf -> a/b/c/
                                pdf_output_dir = pdf_convert_dir / relative_path.parent / pdf_path.stem
                            else:
                                # input_path 직접 하위 파일: f.pdf -> f/
                                pdf_output_dir = pdf_convert_dir / pdf_path.stem
                        except ValueError:
                            # 상대 경로 계산 실패시 폴백
                            pdf_output_dir = pdf_convert_dir / pdf_path.stem
                    
                    pdf_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # PDF 열기
                    doc = pymupdf.open(str(pdf_path))
                    
                    if len(doc) == 0:
                        doc.close()
                        continue
                    
                    # 각 페이지를 이미지로 변환 (페이지별 프로그레스 바)
                    page_desc = f"{pdf_path.name} 페이지 변환"
                    for page_num in tqdm(range(len(doc)), desc=page_desc, unit="페이지", leave=False):
                        try:
                            page = doc.load_page(page_num)
                            
                            # 해상도 매트릭스 설정
                            zoom = dpi / 72.0  # 72 DPI가 기본값
                            mat = pymupdf.Matrix(zoom, zoom)
                            
                            # 이미지 렌더링
                            pix = page.get_pixmap(matrix=mat)
                            
                            # 이미지 저장
                            if image_format.lower() in ['jpg', 'jpeg']:
                                image_path = pdf_output_dir / f"{page_num + 1:03d}.jpg"
                                pix.save(str(image_path), jpg_quality=95)
                            else:
                                image_path = pdf_output_dir / f"{page_num + 1:03d}.{image_format.lower()}"
                                pix.save(str(image_path))
                            
                            all_images.append(str(image_path))
                            
                        except Exception as e:
                            print(f"ERROR : 페이지 {page_num + 1} 변환 오류 ({pdf_path.name}): {e}")
                            continue
                    
                    doc.close()
                    converted_files += 1
                    
                    # 프로그레스 바 완료 후 저장 완료 메시지 출력
                    print(f"\nSaved >> \"{pdf_output_dir}\"")
                    
                except Exception as e:
                    print(f"ERROR : PDF 변환 오류 ({pdf_path}): {e}")
                    continue
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "converted_files": converted_files,
            "total_images": len(all_images),
            "output_directory": str(pdf_convert_dir),
            "images": all_images,
            "processing_time": processing_time
        }
        
        print("\n" + "="*50)
        print("PDF 변환 완료!")
        print(f"처리된 PDF: {converted_files}개")
        print(f"생성된 이미지: {len(all_images)}개")
        print(f"총 처리 시간: {processing_time:.2f}초")
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"PDF 변환 중 오류 발생: {str(e)}",
            "processing_time": time.time() - start_time
        }