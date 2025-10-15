from api import *

# pdf를 image로 변환
# input 경로가 디렉토리면 재귀적으로 파일 처리, input이 파일이면 그 파일만 처리
# output 경로 내부에 pdf_convert 디렉토리 생성 후, 이미지 저장
pdf_to_image("data/in", "data/out")

# OCR
# input 경로가 디렉토리면 재귀적으로 파일 처리, input이 파일이면 그 파일만 처리
# output 경로 내부에 ocr 디렉토리 생성 후, OCR 결과 JSON 저장
# ocr_naver("data/out/pdf_convert", "data/out")
# ocr_google("data/out/pdf_convert", "data/out")
# ocr_mistral("data/out/pdf_convert", "data/out")

# NER
# input 경로가 디렉토리면 재귀적으로 파일 처리, input이 파일이면 그 파일만 처리
# output 경로 내부에 ner 디렉토리 생성 후, NER 결과 JSON 저장 (summary.json 포함)
# 기본 모델: "klue-roberta-large"
# 모델이 없으면 자동 다운로드(train=False 파라미터로 학습을 끌 수 있음)
ner_predict("data/out/ocr", "data/out/ner", train=True)
ner_predict("data/out/ocr", "data/out/ner", model_name="FacebookAI/xlm-roberta-large")
ner_predict("data/out/ocr", "data/out/ner", model_name="google-bert/bert-base-multilingual-cased")
