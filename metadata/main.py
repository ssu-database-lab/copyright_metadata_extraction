from module import api

if __name__ == "__main__":
    api.ocr_extract("data/in/document", "data/out/ocr")
    # # 텍스트 파일 메타데이터 일괄 추출
    # api.file_metadata_extract()
    
    # # NER 모델 학습 (필요 시 주석 해제 또는 사용)
    # api.ner_train(
    #     model_type="ner",
    #     model_name="bert-base-multilingual-cased",
    #     model_path="data/models/ner",
    #     epochs=20,
    #     batch_size=32,
    #     learning_rate=2e-5,
    #     train_data_path="data/in/training_csv",
    #     train_ratio=0.8,
    #     random_seed=42,
    #     dataset_size=10000,
    #     plot=True,
    #     plot_output_path="data/out/results/ner_train_history.png"
    # )
