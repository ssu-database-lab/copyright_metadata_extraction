# import
from paddleocr import PaddleOCRVL
from PIL import Image
import numpy as np


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------

ocr = PaddleOCRVL()
print("model loaded")

img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
ocr.predict(img)
print("done")
