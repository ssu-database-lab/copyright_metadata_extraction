# import
from module import api


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 1) OCR 추출
    print("=" * 60)
    print("[TEST] OCR 추출")
    print("=" * 60)
    api.ocr_extract(
        in_path="data/in/text/text.txt",
        out_path="data/out/ocr/result",
    )

    # 2) NER 학습 (자동학습 검사 → 변경 시에만 학습)
    print("=" * 60)
    print("[TEST] NER 학습")
    print("=" * 60)
    train_result = api.ner_train()
    print(f"  → {train_result.get('message')}")

    # 3) NER 예측 (학습 없이 예측만)
    print("=" * 60)
    print("[TEST] NER 예측")
    print("=" * 60)
    predict_result = api.ner_predict(
        file_path="data/out/ocr/result/화순동의서_박광순.txt",
        out_dir="data/out/results",
    )
    print(f"  → labels: {list(predict_result.keys())}")

    # 4) 통합 메타데이터 추출 (OCR + 정규식 + NER 예측)
    print("=" * 60)
    print("[TEST] 통합 메타데이터 추출")
    print("=" * 60)
    meta_result = api.metadata_extract(
        file_path="data/out/ocr/result/화순동의서_박광순.txt",
        out_dir="data/out/results",
    )
    print(f"  → ner={len(meta_result.get('ner_decisions', []))}, "
          f"regular={len(meta_result.get('regular_decisions', []))}")
