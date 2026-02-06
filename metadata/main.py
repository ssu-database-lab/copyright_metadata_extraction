from pathlib import Path
from module import api


def _find_first_file(root: Path, exts: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                return p
        for p in root.rglob(f"*{ext.upper()}"):
            if p.is_file():
                return p
    return None


def _pick_txt_root() -> Path:
    primary = Path("data/out/ocr/result")
    fallback = Path("out/ocr/result")
    return primary if primary.exists() else fallback

if __name__ == "__main__":
    pdf_path = Path("/home/peppermint/copyright_metadata_extraction/metadata/data/in/document/진천동의서 11명.pdf")
    image_dir = Path("data/in/document/동의서/화순")
    image_path = _find_first_file(
        image_dir, (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")
    )
    txt_root = _pick_txt_root()
    txt_path = _find_first_file(txt_root, (".txt",))

    print("=== OCR 테스트: PDF ===")
    if pdf_path.exists():
        api.ocr_extract(str(pdf_path), "data/out/ocr", "data/out/ocr/metadata")
    else:
        print(f"PDF 없음: {pdf_path}")

    print("=== OCR 테스트: 이미지 ===")
    if image_path:
        api.ocr_extract(str(image_path), "data/out/ocr", "data/out/ocr/metadata")
    else:
        print(f"이미지 없음: {image_dir}")

    print("=== NER 테스트: TXT ===")
    if txt_path:
        api.ner_predict(file_path=str(txt_path))
    else:
        print(f"TXT 없음: {txt_root}")

    print("=== OCR+NER 테스트: PDF ===")
    if pdf_path.exists():
        api.ner_metadata_extract(file_path=str(pdf_path))
    else:
        print(f"PDF 없음: {pdf_path}")

    print("=== OCR+NER 테스트: 이미지 ===")
    if image_path:
        api.ner_metadata_extract(file_path=str(image_path))
    else:
        print(f"이미지 없음: {image_dir}")

    print("=== OCR+NER 테스트: TXT ===")
    if txt_path:
        api.ner_metadata_extract(file_path=str(txt_path))
    else:
        print(f"TXT 없음: {txt_root}")

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
    