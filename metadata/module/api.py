"""API 모듈: main.py에서 사용하는 함수만 노출"""
import json
import traceback

from pathlib import Path
from typing import Optional, Dict, Any, List

from module.parts import ocr
from module.parts import directory
from module.parts import schema
from module.parts import csv
from module.extractor import text as text_module
from module.extractor import regular_extractor
from module.extractor import ner_extractor
from module.extractor import llm_extractor
from module.extractor.ner import base
from module.extractor.ner import visualize as ner_visualize


def metadata_extract(
    *,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    out_dir: str = "data/out/results",
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    메타데이터 추출 (텍스트 또는 파일 입력 가능)
    - text: 문자열 직접 입력
    - file_path: 원본 파일 경로 (없으면 임시 이름 사용)
    - out_dir: 결과 저장 디렉토리 (기본: data/out/results/)
    - use_llm: LLM 사용 여부 (False면 ner > regular 우선순위로 통합)
    """
    if text is None:
        if file_path is None:
            raise ValueError("텍스트 또는 파일 경로 중 하나는 제공되어야 합니다.")
        raw_text = Path(file_path).read_text(encoding="utf-8")
    else:
        raw_text = text

    # preprocessing, tokenization
    struct = text_module.read_text(raw_text)
    sentences, tokens = struct["sentences"], struct["tokens"]

    # stage outputs
    # 1. regular와 ner는 각각 독립적으로 전체 텍스트 처리 (병렬)
    regular_decisions = regular_extractor(sentences=sentences, tokens=tokens)
    ner_decisions = ner_extractor(sentences=sentences, tokens=tokens)
    
    # 2. 통합 (LLM 사용 여부에 따라 분기)
    if use_llm:
        # LLM이 regular + ner 결과를 통합하여 최종 정리
        final_decisions = llm_extractor(
            raw_text=raw_text,
            sentences=sentences,
            tokens=tokens,
            previous_decisions=regular_decisions + ner_decisions,
        )
    else:
        # ner > regular 우선순위로 통합
        from module.extractor.llm import merge_regular_ner
        final_decisions = merge_regular_ner(regular_decisions, ner_decisions)

    # 3. 최종 결과를 JSON으로 변환
    labels_list = directory.load_schema_labels()
    aggregated = {label: [] for label in labels_list}

    for decision in final_decisions:
        label = decision.label
        value = decision.value
        if label in aggregated and value and value not in aggregated[label]:
            aggregated[label].append(value)

    # 빈 리스트를 "N/A"로 변환
    for label in aggregated:
        if not aggregated[label]:
            aggregated[label] = "N/A"

    # prepare output directory
    out_dir_path = directory.ensure_outdir(out_dir)
    out_file = directory.default_outfile(file_path=file_path, out_dir=out_dir_path)

    # save JSON file
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"Metadata saved to: {out_file}")
    return aggregated


def ner_predict(
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> List[Any]:
    """NER 예측"""
    return base.predict(
        sentences=sentences,
        tokens=tokens,
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        **kwargs
    )


def ner_train(
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    train_data_path: Optional[str] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    dataset_size: Optional[int] = None,
    samples_per_file: Optional[int] = None,
    sample_ratio_per_file: Optional[float] = None,
    plot: bool = True,
    plot_output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """NER 모델 학습"""
    result = base.train(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_data_path=train_data_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
        dataset_size=dataset_size,
        samples_per_file=samples_per_file,
        sample_ratio_per_file=sample_ratio_per_file,
        **kwargs
    )
    
    # 시각화
    if plot and result.get("history"):
        ner_visualize.plot_training_history(
            history=result["history"],
            output_path=plot_output_path,
            model_name=model_name
        )
    
    return result


def ner_validate(
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    plot: bool = True,
    plot_output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """NER 모델 검증"""
    metrics = base.validate(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        val_data_path=val_data_path,
        **kwargs
    )
    
    # 시각화
    if plot:
        ner_visualize.plot_validation_metrics(
            metrics=metrics,
            output_path=plot_output_path
        )
    
    return metrics


# 코드 검사 완료
def ocr_extract(
    in_path: str,
    out_path: str = "data/out/results",
) -> None:
    """
    PaddleOCRVL
    
    Args:
        in_path: 입력 파일 또는 디렉토리 경로
        out_path: 출력 파일 또는 디렉토리 경로
        - in_path가 파일일 때: 결과 텍스트가 저장될 파일 경로 (확장자 .txt)
        - in_path가 디렉토리일 때: 결과가 저장될 루트 디렉토리 (구조 유지됨)
    """
    
    input_p = Path(in_path)
    output_p = Path(out_path)
    
    if not input_p.exists():
        print(f"Error: Input path does not exist: {in_path}")
        return

    # 파일 여부 검사
    if input_p.is_file():
        print(f"OCR Processing : {input_p}")
        
        # 출력 경로 결정
        if output_p.suffix:
            save_file = output_p
        else:
            save_file = directory.get_mirror_output_path(input_p, input_p.parent, output_p)
            
        ocr.extract_text_from_file(str(input_p), save_path=str(save_file))

        print(f"Saved to: {save_file}")
    else :
        print(f"OCR Processing(directory): {input_p}")
        
        files = list(directory.iter_document_files(input_p))

        if not files:
            print("No supported document files found.")
            return
            
        print(f"Found {len(files)} files.")
        
        for file in files:
            # 구조 유지 경로 계산
            save_file = directory.get_mirror_output_path(file, input_p, output_p)
            
            print(f"Processing: {file.relative_to(input_p)}")

            try:
                ocr.extract_text_from_file(str(file), save_path=str(save_file))
            except Exception as e:
                traceback.print_exc()
                print(f"Failed : {e}")
                
    print("OCR extraction completed.")


def file_metadata_extract(
    input_dir: str = "data/in/text",
    out_dir: str = "data/out/results",
) -> None:
    """
    입력 디렉터리 내 모든 텍스트 파일에 대해 metadata_extract 실행.
    """
    input_path = Path(input_dir)
    
    # directory 모듈의 iter_text_files 사용
    files = list(directory.iter_text_files(input_path))
    
    if not files:
        print(f"처리할 텍스트 파일이 없습니다: {input_path}")
        return

    for file_path in files:
        print(f"Processing: {file_path}")
        metadata_extract(file_path=str(file_path), out_dir=out_dir)


# ---------- CSV/Excel Utility Functions ----------

def convert_excel_dataset(
    input_dir: str = 'data/in/excel',
    output_dir: str = 'data/in/csv'
) -> None:
    """Excel 데이터셋을 CSV로 일괄 변환"""
    csv.convert_excel_to_csv(input_dir, output_dir)

def preprocess_csv_dataset(
    input_dir: str = 'data/in/csv',
    output_dir: str = 'data/in/training_csv'
) -> None:
    """CSV 데이터셋 전처리 (학습 데이터 생성)"""
    csv.preprocess_csv_dataset(input_dir, output_dir)
