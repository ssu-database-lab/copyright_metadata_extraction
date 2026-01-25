"""API 모듈: main.py에서 사용하는 함수만 노출"""
import json

from pathlib import Path
from typing import Optional, Dict, Any, List

from module.extractor import ocr as ocr_module
from module.parts import directory
from module.extractor import text as text_module
from module.extractor import regular_extractor
from module.extractor import ner_extractor
from module.extractor import llm_extractor
from module.extractor.ner import base
from module.parts.types import Decision


def metadata_extract(
    *,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    out_dir: str = "data/out/results",
    use_llm: bool = False
) -> Dict[str, Any]:
    """
    메타데이터 추출 (텍스트 또는 파일 입력 가능)
    
    플로우:
    1. Regex: word(tokens)만 사용
    2. NER: word + sentence 사용
    3. LLM: NER 결과 + raw_text 사용
    
    Args:
        text: 문자열 직접 입력
        file_path: 원본 파일 경로
            - 이미지/PDF 등 비텍스트 파일: OCR 수행 후 메타데이터 추출
            - txt 파일: OCR 없이 바로 처리
        out_dir: 결과 저장 디렉토리 (기본: data/out/results/)
        use_llm: LLM 사용 여부 (기본값: False)
    
    Returns:
        추출된 메타데이터 딕셔너리
    """
    # 입력 처리
    if text is None:
        if file_path is None:
            raise ValueError("텍스트 또는 파일 경로 중 하나는 제공되어야 합니다.")
        
        file_path_obj = Path(file_path)
        
        # OCR 또는 텍스트 파일 처리 (분리된 함수 사용)
        raw_text, ocr_labeled_metadata = ocr_module.process_file_for_metadata(
            file_path_obj,
            use_temp_dir=True  # 메타데이터 추출 시 임시 디렉토리 사용
        )
    else:
        raw_text = text
        file_path_obj = Path(file_path) if file_path else None
        ocr_labeled_metadata = {}

    # preprocessing, tokenization
    struct = text_module.read_text(raw_text)
    sentences, tokens = struct["sentences"], struct["tokens"]

    # stage outputs
    # 1. Regex: word(tokens)만 사용
    regex_decisions = regular_extractor(sentences=sentences, tokens=tokens)
    
    # 2. NER: word + sentence 사용 (어댑터 기반 Zero-Shot NER)
    # ner_extractor는 기본적으로 adapter_ner를 사용 (base.py에서 기본값 설정)
    ner_decisions = ner_extractor(sentences=sentences, tokens=tokens)
    
    # NER 라벨을 최종 메타데이터 라벨로 매핑
    ner_decisions = _map_ner_labels_to_metadata_labels(ner_decisions)
    
    # # 3. LLM: NER 결과 + raw_text 사용
    # if use_llm:
    #     # LLM이 NER 결과 + raw_text를 받아 최종 정리
    #     final_decisions = llm_extractor(
    #         raw_text=raw_text,
    #         sentences=sentences,
    #         tokens=tokens,
    #         previous_decisions=regex_decisions + ner_decisions,
    #     )
    # else:
    #     # LLM 없이 NER > Regex 우선순위로 통합
    #     from module.extractor.llm import merge_regular_ner
    #     final_decisions = merge_regular_ner(regex_decisions, ner_decisions)
    #
    # # OCR 메타데이터에서 doc_title 추출 (OCR 사용한 경우만)
    # if ocr_labeled_metadata and 'doc_title' in ocr_labeled_metadata:
    #     doc_title_items = ocr_labeled_metadata['doc_title']
    #     for item in doc_title_items:
    #         content = item.get('content', '').strip()
    #         if content:
    #             from module.parts.types import Decision
    #             final_decisions.append(Decision(
    #                 label="doc_title",
    #                 value=content,
    #                 sent_id=None,
    #                 tok_id=None,
    #                 source="ocr"
    #             ))

    # 현재는 regex만 사용
    final_decisions = regex_decisions
    
    # OCR 메타데이터에서 doc_title 추출 (OCR 사용한 경우만)
    if ocr_labeled_metadata and 'doc_title' in ocr_labeled_metadata:
        doc_title_items = ocr_labeled_metadata['doc_title']
        for item in doc_title_items:
            content = item.get('content', '').strip()
            if content:
                from module.parts.types import Decision
                final_decisions.append(Decision(
                    label="doc_title",
                    value=content,
                    sent_id=None,
                    tok_id=None,
                    source="ocr"
                ))

    # 4. 최종 결과를 JSON으로 변환
    # 최종 메타데이터 라벨 목록 (untitled_metadata.json 구조 참고)
    final_metadata_labels = [
        "seq_number", "site_name", "agency_name", "board_name", "board_path",
        "category", "work_title", "url", "description", "created_date",
        "registration_date", "production_date", "valid_period", "attachment",
        "video_count", "photo_count", "document_count", "quantity",
        "kogl_type", "disclosure_type", "copyrightability", "unprotected_work",
        "work_for_hire", "copyright_holder", "co_author", "neighboring_rights_holder",
        "co_author_consent", "third_party_rights", "economic_rights", "commercial_use",
        "portrait_rights", "personal_info", "contract", "review_impossible",
        "work_type", "digital_format", "keyword", "language", "phone", "email", "memo", "doc_title"
    ]
    aggregated = {label: [] for label in final_metadata_labels}

    # final_decisions를 aggregated에 추가
    for decision in final_decisions:
        label = decision.label
        value = decision.value
        if label in aggregated and value and value not in aggregated[label]:
            aggregated[label].append(value)

    # 빈 리스트를 "N/A"로 변환
    for label in aggregated:
        if not aggregated[label]:
            aggregated[label] = ["N/A"]

    # prepare output directory
    out_dir_path = directory.ensure_outdir(out_dir)
    out_file = directory.default_outfile(
        file_path=str(file_path_obj) if file_path_obj else None, 
        out_dir=out_dir_path
    )

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
    """NER 예측 (api에서 호출)"""
    return base.predict(
        sentences=sentences,
        tokens=tokens,
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        **kwargs
    )


def ner_train(
    model_name: str = "bert-base-multilingual-cased",
    model_path: Optional[str] = None,
    adapter_dir: str = "models/ner/adapters",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    train_data_path: Optional[str] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    **kwargs
) -> None:
    """
    NER 모델 학습 (라벨별 어댑터 학습)
    
    Args:
        model_name: 사전 학습된 모델 이름 (기본값: "bert-base-multilingual-cased")
        model_path: 사전 학습된 모델 경로 (None이면 model_downloaded/{model_name} 사용)
        adapter_dir: 어댑터 저장 디렉토리 (기본값: "models/ner/adapters")
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        train_data_path: 학습 데이터 경로 (None이면 configs/training 사용)
        train_ratio: 학습/검증 데이터 비율
        random_seed: 랜덤 시드
    """
    base.train(
        model_name=model_name,
        model_path=model_path,
        adapter_dir=adapter_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_data_path=train_data_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
        **kwargs
    )




def _map_ner_labels_to_metadata_labels(ner_decisions: List[Decision]) -> List[Decision]:
    """
    NER 라벨을 최종 메타데이터 라벨로 매핑
    
    labels.yaml의 NER 라벨 → 최종 메타데이터 라벨 매핑:
    - company_name → agency_name
    - phone_number → phone
    - email → email (이미 동일)
    - url → url (이미 동일)
    - date → created_date, registration_date, production_date, valid_period (모두 가능)
    - address, person_name → 그대로 유지 (최종 라벨에 없으면 무시됨)
    """
    label_mapping = {
        "company_name": "agency_name",
        "phone_number": "phone",
        "email": "email",  # 이미 동일하지만 명시적으로 매핑
        "url": "url",  # 이미 동일하지만 명시적으로 매핑
        # date는 여러 날짜 라벨로 매핑 가능하지만, 일단 그대로 유지
        # (datetime_extractor가 더 정확할 수 있음)
    }
    
    mapped_decisions = []
    for decision in ner_decisions:
        original_label = decision.label
        
        # 매핑이 있으면 변경, 없으면 그대로 유지
        mapped_label = label_mapping.get(original_label, original_label)
        
        # date 라벨의 경우, 여러 날짜 라벨로 복제 가능
        # 하지만 일단 그대로 유지 (datetime_extractor가 더 정확할 수 있음)
        if original_label == "date":
            # date는 그대로 유지하거나, 필요시 여러 라벨로 복제 가능
            # 일단 그대로 유지
            mapped_label = "date"  # 최종 라벨에 없으면 무시됨
        
        mapped_decisions.append(Decision(
            label=mapped_label,
            value=decision.value,
            sent_id=decision.sent_id,
            tok_id=decision.tok_id,
            source=decision.source,
            meta=decision.meta
        ))
    
    return mapped_decisions


def ocr_extract(
    in_path: str,
    out_path: str,
    metadata_path: Optional[str] = None,
) -> None:
    """
    PaddleOCRVL을 사용하여 파일 또는 디렉토리에서 OCR을 수행합니다.
    텍스트는 out_path/result에 저장하고, 메타데이터는 out_path/result/metadata에 저장합니다.
    
    Args:
        in_path: 입력 파일 또는 디렉토리 경로 (필수)
        out_path: 출력 루트 디렉토리 경로 (필수, result와 result/metadata가 생성됨)
        metadata_path: metadata JSON이 저장될 루트 디렉토리 경로 (None이면 저장하지 않음)
    """
    ocr_module.ocr_extract(in_path, out_path, metadata_path)  # type: ignore