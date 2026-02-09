from module import api


if __name__ == "__main__":
    api.metadata_extract(
        file_path="data/out/ocr/result",
        out_dir="data/out/results"
    )

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
    