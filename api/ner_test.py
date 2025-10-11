from api import *

ner_predict("data/out/ocr", "data/out/ner")
ner_predict("data/out/ocr", "data/out/ner", model_name="FacebookAI/xlm-roberta-large")
ner_predict("data/out/ocr", "data/out/ner", model_name="google-bert/bert-base-multilingual-cased")