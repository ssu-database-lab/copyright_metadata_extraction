"""메타데이터 추출 엔트리 포인트."""
from pathlib import Path
from typing import Iterable

from module import api

DEFAULT_INPUT_DIR = Path("data/in/text")  # TODO: 실제 텍스트 입력 경로 확정
DEFAULT_OUTPUT_DIR = Path("data/out/results")


def _iter_text_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 .txt 파일만 순회."""
    if not root.exists():
        return []
    return (path for path in root.rglob("*.txt") if path.is_file())


def file_metadata_extract(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    입력 디렉터리 내 모든 텍스트 파일에 대해 metadata_extract 실행.
    실제 extractor 구현은 module.extractor.* 에서 후속 작업으로 채운다.
    """
    files = list(_iter_text_files(input_dir))
    if not files:
        print(f"처리할 텍스트 파일이 없습니다: {input_dir}")
        return

    for file_path in files:
        print(f"Processing: {file_path}")
        api.metadata_extract(file_path=str(file_path), out_dir=str(output_dir))


if __name__ == "__main__":
    file_metadata_extract()
    api.ner_train(
        model_type="ner",
        model_name="bert-base-multilingual-cased",
        model_path="data/models/ner",
        epochs=20,
        batch_size=32,
        learning_rate=2e-5,
        train_data_path="data/in/training_csv",
        train_ratio=0.8,
        random_seed=42,
        dataset_size=10000,
        plot=True,
        plot_output_path="data/out/results/ner_train_history.png"
    )