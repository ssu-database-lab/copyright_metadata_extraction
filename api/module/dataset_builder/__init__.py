"""
KOGL 데이터셋 빌더 모듈.

붙임1 원문저작물 메타데이터 Excel(≈144k행)에서 저작권자명이 채워진 행을
분류(이미지/영상/어문)별로 샘플링하고, 공개 썸네일을 내려받아 검증한 뒤,
계약서 생성에 쓸 매니페스트(manifest.xlsx)를 만든다.

3단계 파이프라인:
    SELECT   → selection.csv  (저작권자명 보유 행을 분류별 샘플링)
    DOWNLOAD → {images,documents,videos}/{원문인덱스}.{jpg,png}  (썸네일만)
    MANIFEST → manifest.xlsx  (모든 권리/저작권 필드 + 다운로드 상태)

CLI:
    python -m api.module.dataset_builder.build [옵션]
"""

from .build import main

__all__ = ["main"]
