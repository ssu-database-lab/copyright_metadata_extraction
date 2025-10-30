#!/usr/bin/env python3

from api import *

# pdf를 image로 변환
# input 경로가 디렉토리면 재귀적으로 파일 처리, input이 파일이면 그 파일만 처리
# output 경로 내부에 pdf_convert 디렉토리 생성 후, 이미지 저장
# pdf_to_image("data/in", "data/out")

# OCR
# input 경로가 디렉토리면 재귀적으로 파일 처리, input이 파일이면 그 파일만 처리
# output 경로 내부에 ocr 디렉토리 생성 후, OCR 결과 JSON 저장
# ocr_naver("data/out/pdf_convert", "data/out")
# ocr_google("data/out/pdf_convert", "data/out")
# ocr_mistral("data/out/pdf_convert", "data/out")

# NER
# input 경로가 디렉토리면 재귀적으로 파일 처리, input이 파일이면 그 파일만 처리
# output 경로 내부에 ner 디렉토리 생성 후, NER 결과 JSON 저장 (summary.json 포함)
# 기본 모델: "google-bert/bert-base-multilingual-cased"

ner_train(
    iterations=1, 
    epochs=10,
    num_train_samples=300,
    learning_rate=2e-5,
    batch_size=16, 
    enable_visualization=True,
    enable_early_stopping=False,
)

# 예측 수행
ner_predict("data/in/ocr", "data/out")

# 2️⃣ klue/roberta-large
ner_train(
    iterations=1, 
    epochs=10, 
    num_train_samples=300,
    learning_rate=2e-5,
    batch_size=16,
    enable_visualization=True,
    enable_early_stopping=False,
    model_name="klue/roberta-large",
)

# 예측 수행
ner_predict("data/in/ocr", "data/out", model_name="klue/roberta-large")

# 3️⃣ FacebookAI/xlm-roberta-large
ner_train(
    iterations=1, 
    epochs=10, 
    num_train_samples=300,
    learning_rate=2e-5,
    batch_size=16,
    enable_visualization=True,
    enable_early_stopping=False,
    model_name="FacebookAI/xlm-roberta-large",
)

# 예측 수행
ner_predict("data/in/ocr", "data/out", model_name="FacebookAI/xlm-roberta-large")