from typing import Optional, List, Dict, Any
from pathlib import Path
import re
import json
import yaml
from paddleocr import PaddleOCRVL
from module.parts import directory


def needs_ocr(file_path: Path) -> bool:
    """
    파일이 OCR이 필요한지 확인 (이미지, PDF 등)
    
    Args:
        file_path: 확인할 파일 경로
    
    Returns:
        OCR 필요 여부 (True: OCR 필요, False: 텍스트 파일)
    """
    if not file_path.exists():
        return False
    
    # 텍스트 파일 확장자
    text_extensions = {".txt", ".md", ".csv"}
    
    # OCR 필요한 확장자 (이미지, PDF 등)
    ocr_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    
    ext = file_path.suffix.lower()
    
    if ext in text_extensions:
        return False
    elif ext in ocr_extensions:
        return True
    else:
        # 알 수 없는 확장자는 텍스트로 간주
        return False


# 하위 호환성을 위한 별칭
_needs_ocr = needs_ocr


def _load_ocr_config() -> Dict[str, Any]:
    """OCR 라벨 설정 로드"""
    config_path = Path('configs/labels.yaml')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('ocr', {})
    return {}


def _normalize_label(label: Optional[str], normalize_map: Dict[str, Any]) -> Optional[str]:
    """OCR 라벨을 표준 키로 정규화"""
    if not label:
        return None
    
    # normalize_map에서 직접 매핑 찾기 (키가 입력 라벨, 값이 표준 키)
    if label in normalize_map:
        normalized = normalize_map[label]
        # YAML에서 키워드로 파싱된 경우 문자열로 변환
        return str(normalized) if normalized else label
    
    # 이미 표준 키인 경우 (normalize_map의 값들을 문자열로 변환하여 비교)
    normalized_values = [str(v) for v in normalize_map.values()]
    if label in normalized_values:
        return label
    
    # 매핑되지 않은 경우 원본 반환
    return label


def _extract_ocr_results(results: List[Dict[str, Any]]) -> tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """
    OCR 결과에서 텍스트와 라벨별 메타데이터를 추출합니다.
    
    Args:
        results: pipeline.predict() 결과 리스트
    
    Returns:
        (페이지별 텍스트 리스트, 라벨별 메타데이터 딕셔너리)
    """
    ocr_config = _load_ocr_config()
    normalize_map = ocr_config.get('normalize_map', {})
    
    page_texts = []
    labeled_metadata = {}
    
    for page_idx, page_result_obj in enumerate(results):
        # PaddleOCRVLResult 객체를 딕셔너리로 변환
        try:
            if isinstance(page_result_obj, dict):
                page_result = page_result_obj
            elif hasattr(page_result_obj, 'to_dict'):
                page_result = page_result_obj.to_dict()  # type: ignore
            else:
                # 객체가 아닌 경우 스킵
                continue
        except (AttributeError, TypeError):
            continue
        
        # page_index와 page_count 안전하게 추출
        try:
            page_index = page_result.get('page_index')
            if page_index is None:
                page_index = page_idx
            
            page_count = page_result.get('page_count')
            if page_count is None:
                page_count = len(results)
            
            # 정수 타입 확인
            if not isinstance(page_index, int):
                page_index = page_idx
            if not isinstance(page_count, int):
                page_count = len(results)
            
            parsing_res_list = page_result.get('parsing_res_list', [])
        except (KeyError, TypeError, AttributeError):
            # 기본값 사용
            page_index = page_idx
            page_count = len(results)
            parsing_res_list = []
        
        page_content = []
        page_labeled_items = []
        
        for item_idx, parsing_item_obj in enumerate(parsing_res_list):
            content = None
            label = None
            bbox = None
            
            # PaddleOCRVLBlock 객체를 딕셔너리로 변환 시도
            try:
                if hasattr(parsing_item_obj, 'to_dict'):
                    parsing_item = parsing_item_obj.to_dict()
                elif isinstance(parsing_item_obj, dict):
                    parsing_item = parsing_item_obj
                else:
                    parsing_item = None
            except (AttributeError, TypeError):
                parsing_item = None
            
            if parsing_item and isinstance(parsing_item, dict):
                # 딕셔너리인 경우 직접 접근
                content = parsing_item.get('content')
                label = parsing_item.get('label')
                bbox = parsing_item.get('bbox')
            else:
                # 문자열로 변환하여 파싱
                item_str = str(parsing_item_obj)
                
                # 정규표현식으로 content, label, bbox 추출
                content_match = re.search(r'content:\s*(.+?)(?=\n#################|\Z)', item_str, re.DOTALL)
                label_match = re.search(r'label:\s*(\S+)', item_str)
                bbox_match = re.search(r'bbox:\s*\[([^\]]+)\]', item_str)
                
                if content_match:
                    content = content_match.group(1).strip()
                if label_match:
                    label = label_match.group(1).strip()
                if bbox_match:
                    bbox_str = bbox_match.group(1).strip()
                    try:
                        bbox = [float(x.strip()) for x in bbox_str.split(',')]
                    except:
                        bbox = None
            
            if content:
                # 라벨 정규화
                normalized_label = _normalize_label(label, normalize_map)
                if not normalized_label:
                    normalized_label = 'text'  # 기본값
                
                # 텍스트 추가
                page_content.append(content)
                
                # 메타데이터 추가
                item_data = {
                    'page_index': page_index,
                    'item_index': item_idx,
                    'label': normalized_label,
                    'content': content,
                    'bbox': bbox
                }
                page_labeled_items.append(item_data)
                
                # 라벨별로 그룹화
                if normalized_label not in labeled_metadata:
                    labeled_metadata[normalized_label] = []
                labeled_metadata[normalized_label].append(item_data)
        
        # 페이지 구분자와 함께 텍스트 저장
        if page_content:
            page_text = "\n".join(page_content)
            # page_index와 page_count가 None이 아닌지 확인
            safe_page_index = page_index if isinstance(page_index, int) else page_idx
            safe_page_count = page_count if isinstance(page_count, int) else len(results)
            page_texts.append(f"--- Page {safe_page_index + 1}/{safe_page_count} ---\n{page_text}")
    
    return page_texts, labeled_metadata


def extract_text_from_file(
    pipeline: PaddleOCRVL, 
    file_path: str, 
    save_path: Optional[str] = None
) -> tuple[str, Dict[str, Any]]:
    """
    파일 또는 디렉토리에서 OCR을 수행합니다.
    plain(텍스트)는 save_path/result에 저장하고, metadata는 반환합니다.
    
    Args:
        pipeline: PaddleOCRVL 모델 인스턴스 (외부에서 생성하여 전달)
        file_path: 입력 파일 또는 디렉토리 경로
        save_path: 출력 루트 디렉토리 경로 (result에 저장)
    
    Returns:
        (추출된 텍스트 내용, metadata 딕셔너리)
        - 단일 파일인 경우: (텍스트, 단일 파일 metadata)
        - 디렉토리인 경우: ("", 전체 metadata 딕셔너리)
    """
    input_p = Path(file_path)
    
    if not input_p.exists():
        raise ValueError(f"Input path does not exist: {file_path}")
    
    if not save_path:
        raise ValueError("save_path must be provided")
    
    output_root = Path(save_path)
    # result 디렉토리에 직접 저장 (plain 디렉토리 없음)
    result_dir = output_root / "result"
    
    # 파일인 경우
    if input_p.is_file():
        results = pipeline.predict(str(input_p))
        page_texts, labeled_metadata = _extract_ocr_results(results)
        
        # 텍스트 파일 저장 (result 디렉토리에 직접)
        plain_file = result_dir / f"{input_p.stem}.txt"
        plain_file.parent.mkdir(parents=True, exist_ok=True)
        plain_file.write_text("\n\n".join(page_texts), encoding="utf-8")
        
        # metadata 반환 (저장하지 않음)
        metadata = {
            'source_file': str(input_p),
            'total_pages': len(results),
            'labels': labeled_metadata
        }
        
        return "\n\n".join(page_texts), metadata
    
    # 디렉토리인 경우 재귀적으로 처리
    elif input_p.is_dir():
        files = list(directory.iter_document_files(input_p))
        
        if not files:
            print("No supported document files found.")
            return "", {}
        
        print(f"Found {len(files)} files.")
        
        all_metadata = {}
        
        for file in files:
            print(f"Processing: {file.relative_to(input_p)}")
            
            try:
                results = pipeline.predict(str(file))
                page_texts, labeled_metadata = _extract_ocr_results(results)
                
                # 상대 경로 계산
                rel_path = file.relative_to(input_p)
                
                # result 디렉토리에 텍스트 저장 (상대 경로 유지)
                plain_file = result_dir / rel_path.with_suffix('.txt')
                plain_file.parent.mkdir(parents=True, exist_ok=True)
                plain_file.write_text("\n\n".join(page_texts), encoding="utf-8")
                
                # metadata 수집 (저장하지 않음)
                rel_path_str = str(rel_path)
                all_metadata[rel_path_str] = {
                    'source_file': str(file),
                    'total_pages': len(results),
                    'labels': labeled_metadata
                }
                    
            except Exception as e:
                print(f"Failed: {e}")
                # 계속 진행 (다음 파일 처리)
        
        return "", all_metadata
    
    else:
        raise ValueError(f"Input path does not exist: {file_path}")


def process_file_for_metadata(
    file_path: Path,
    use_temp_dir: bool = True
) -> tuple[str, Dict[str, Any]]:
    """
    메타데이터 추출을 위한 파일 처리 (OCR 또는 텍스트 읽기)
    
    Args:
        file_path: 처리할 파일 경로
        use_temp_dir: OCR 결과를 임시 디렉토리에 저장할지 여부 (True면 임시 디렉토리 사용)
    
    Returns:
        (raw_text, ocr_labeled_metadata) 튜플
        - raw_text: 추출된 텍스트 내용
        - ocr_labeled_metadata: OCR 메타데이터 딕셔너리 (OCR 사용 시 labels 포함, 텍스트 파일이면 빈 딕셔너리)
    
    Raises:
        RuntimeError: OCR 처리 중 오류 발생 시 (에러 메시지 출력 후 빈 텍스트 반환)
    """
    # 텍스트 파일인 경우 바로 읽기
    if not needs_ocr(file_path):
        raw_text = file_path.read_text(encoding="utf-8")
        return raw_text, {}
    
    # OCR 필요한 파일인 경우
    print(f"OCR required for: {file_path}")
    try:
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL()
        
        if use_temp_dir:
            # 임시 디렉토리에 OCR 결과 저장
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                text_result, ocr_metadata = extract_text_from_file(
                    pipeline, 
                    str(file_path), 
                    save_path=str(temp_path)
                )
                raw_text = text_result
                ocr_labeled_metadata = ocr_metadata.get('labels', {}) if ocr_metadata else {}
        else:
            # 영구 디렉토리에 저장 (기본 OCR 출력 경로)
            text_result, ocr_metadata = extract_text_from_file(
                pipeline,
                str(file_path),
                save_path="data/out/ocr"
            )
            raw_text = text_result
            ocr_labeled_metadata = ocr_metadata.get('labels', {}) if ocr_metadata else {}
        
        return raw_text, ocr_labeled_metadata
        
    except Exception as e:
        # OCR 실패 시 에러 메시지 출력하고 빈 텍스트 반환
        print(f"⚠️ OCR 실패 (정규식 처리만 진행): {str(e)}")
        print("⚠️ OCR 없이 정규식 처리를 계속합니다. (빈 텍스트로 진행)")
        return "", {}


def ocr_extract(
    in_path: str,
    out_path: str,
    metadata_path: Optional[str] = None,
) -> None:
    """
    PaddleOCRVL을 사용하여 파일 또는 디렉토리에서 OCR을 수행합니다.
    텍스트는 out_path/result에 저장하고, 메타데이터는 out_path/result/metadata에 저장합니다.
    모델은 1회만 생성되고 모든 파일에 대해 재사용됩니다.
    
    Args:
        in_path: 입력 파일 또는 디렉토리 경로 (필수)
        out_path: 출력 루트 디렉토리 경로 (필수, result와 result/metadata가 생성됨)
        metadata_path: metadata JSON이 저장될 루트 디렉토리 경로 (None이면 저장하지 않음음)
    """
    import traceback
    
    input_p = Path(in_path)
    output_p = Path(out_path)
    
    if not input_p.exists():
        print(f"Error: Input path does not exist: {in_path}")
        return
    
    # metadata_path 설정: 지정되지 않으면 out_path/result/metadata 사용
    if metadata_path is not None:
        metadata_dir = Path(metadata_path)
    else:
        # out_path/result/metadata에 저장
        metadata_dir = None
    
    # 모델은 1회만 생성
    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL()
    
    # 파일/디렉토리 처리
    try:
        if input_p.is_file():
            print(f"OCR Processing: {input_p}")
            # save_path 설정: 파일인 경우 상위 디렉토리, 디렉토리인 경우 그대로
            if output_p.suffix:
                # 출력 경로가 파일인 경우 상위 디렉토리를 루트로 사용
                save_root = output_p.parent
            else:
                # 출력 경로가 디렉토리인 경우 그대로 사용
                save_root = output_p
            
            # plain 저장 및 metadata 반환
            text, metadata = extract_text_from_file(pipeline, str(in_path), save_path=str(save_root))
            
            # plain 저장 경로 출력
            plain_file = save_root / 'result' / f'{input_p.stem}.txt'
            print(f"Plain saved to: {plain_file}")
            
            # 메타데이터 저장
            if metadata_dir is not None:
                metadata_file = metadata_dir / f"{input_p.stem}.json"
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                print(f"Metadata saved to: {metadata_file}")
        else:
            print(f"OCR Processing(directory): {input_p}")
            # plain 저장 및 metadata 반환
            text, all_metadata = extract_text_from_file(pipeline, str(in_path), save_path=str(out_path))
            
            # plain 저장 경로 출력
            print(f"Plain saved to: {output_p / 'result'}")
            
            # 메타데이터 저장
            if metadata_dir is not None:
                saved_count = 0
                for rel_path_str, metadata in all_metadata.items():
                    rel_path = Path(rel_path_str)
                    metadata_file = metadata_dir / rel_path.with_suffix('.json')
                    metadata_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    saved_count += 1
                print(f"Metadata saved to: {metadata_dir} ({saved_count} files)")
        
        print("OCR extraction completed.")
    except Exception as e:
        traceback.print_exc()
        print(f"OCR extraction failed: {e}")
