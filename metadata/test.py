from paddleocr import PaddleOCRVL
ocr = PaddleOCRVL()

print("model loaded")

# 아주 작은 이미지 (예: 100x100 검정 이미지)를 넣어봄
from PIL import Image
import numpy as np

img = Image.fromarray(np.zeros((128,128,3), dtype=np.uint8))
ocr.predict(img)

print("done")
