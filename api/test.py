from api import *

# PDF 변환 테스트 (완료)
# pdf_to_image("data/in/document","data/out", 300)

# 함부로 주석 풀지 말 것. 비용이 엄청나게 들어감.
# ocr_naver("data/out/pdf_convert","data/out",)
# ocr_google("data/out/pdf_convert","data/out",)
# ocr_mistral("data/out/pdf_convert","data/out",)
# ocr_complete("data/out/pdf_convert/7.저작물양도계약서","data/out",)

# 기본 NER 테스트 (자동으로 klue-roberta_large 사용, 자동 다운로드 및 훈련)
ner_predict("data/out/ocr", "data/out", model_name="XLM-RoBERTa-Large")