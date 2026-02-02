from module import api

if __name__ == "__main__":
    # api.ocr_extract("data/in/document", "data/out/ocr", "data/out/ocr/metadata")
    # # 텍스트 파일 메타데이터 일괄 추출
    # api.metadata_extract(file_path="data/out/ocr/result/진천동의서 11명.txt")
    
    api.ner_predict(file_path="data/out/ocr/result/진천동의서 11명.txt")

    # NER 모델 학습 (필요 시 주석 해제 또는 사용)
    # api.ner_train(
    #     model_name="bert-base-multilingual-cased",
    #     model_path=None,  # None이면 model_downloaded/{model_name} 사용
    #     adapter_dir="models/ner/adapters",
    #     epochs=5,
    #     batch_size=16,
    #     learning_rate=2e-5,
    #     train_data_path="configs/training",  # configs/training 디렉토리에서 *.jsonl 또는 training_data.json 읽음
    #     train_ratio=0.8,
    #     random_seed=42
    # )
    