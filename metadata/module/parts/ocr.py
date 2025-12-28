from typing import Optional
from pathlib import Path
from functools import lru_cache
from paddleocr import PaddleOCRVL

# 모델 로딩 횟수 제한
@lru_cache(maxsize=1)
def get_pipeline() -> PaddleOCRVL:
    pipe=PaddleOCRVL()
    return pipe

# 코드 검사 완료
# AI 코드 더럽게 짜네;;
def extract_text_from_file(file_path: str, save_path: Optional[str] = None) -> str:
    """
    파일에서 OCR을 수행하고 텍스트를 반환합니다.
    save_path가 있으면 해당 경로에 .txt 파일로 저장합니다.
    
    Args:
        file_path: 입력 파일 경로 (이미지/PDF)
        save_path: 저장할 .txt 파일 경로
    """

    pipeline=get_pipeline()
    
    # Predict
    results = pipeline.predict(file_path)

    # 텍스트 추출 (결과 구조에 따라 다를 수 있으나, 일반적으로 results 내에 텍스트가 포함됨)
    # PaddleOCRVL 결과는 리스트 형태. 단순 연결만 수행.
    full_text = []
    for res in results:
        if isinstance(res, dict) and 'text' in res:
             full_text.append(str(res['text']))
        else:
             full_text.append(str(res))
    
    text_content = "\n".join(full_text)
    
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text_content, encoding="utf-8")

    return text_content