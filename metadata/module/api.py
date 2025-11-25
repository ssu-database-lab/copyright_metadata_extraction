"""API 모듈: main.py에서 사용하는 함수만 노출"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from module.parts import text as text_module
from module.parts import directory
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
    schema = directory.load_schema_labels()
    aggregated = {label: [] for label in schema}

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
    """
    NER 모델 학습
    
    Args:
        model_type: 모델 타입 (ner, bilstm_crf)
        model_name: 모델 이름 (기본: bert-base-multilingual-cased)
        model_path: 모델 저장 경로 (기본: data/models/ner)
        epochs: 학습 에포크 (기본 10)
        batch_size: 배치 크기 (기본 32)
        learning_rate: 학습률 (기본 2e-5)
        train_data_path: 학습 데이터 경로 (기본: data/in/training_csv)
        train_ratio: 학습 데이터 비율 (기본 0.8, 검증 0.2)
        random_seed: 랜덤 시드 (기본 42)
        dataset_size: 전체 데이터셋 크기 제한 (None이면 전체 사용)
        samples_per_file: 각 CSV 파일에서 샘플링할 최대 문장 개수 (None이면 전체)
        sample_ratio_per_file: 각 CSV 파일에서 샘플링할 비율 (0.0 ~ 1.0, None이면 전체)
                              samples_per_file이 지정되면 무시됨
        plot: 학습 곡선 시각화 여부 (기본 True)
        plot_output_path: 시각화 저장 경로 (기본: data/out/results/ner_train_history.png)
    """
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
    """
    NER 모델 검증
    
    Args:
        plot: 검증 메트릭 시각화 여부
        plot_output_path: 시각화 저장 경로 (None이면 자동 생성)
    
    Returns:
        검증 메트릭 딕셔너리
    """
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
