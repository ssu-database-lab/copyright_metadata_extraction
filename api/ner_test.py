#!/usr/bin/env python3
"""
NER 모델 성능 비교 테스트 환경

단일 모델(metadata)과 새로 설계한 모델(api)의 성능을 비교할 수 있는 테스트 환경을 제공합니다.

사용 예시:
    # 단일 모델 훈련
    single_model_train(
        model_name="google-bert/bert-base-multilingual-cased",
        epochs=20,
        batch_size=32
    )
    
    # 새로 설계한 모델 훈련
    system_model_train(
        model_name="google-bert/bert-base-multilingual-cased",
        num_epochs=20,
        batch_size=12
    )
    
    # 두 모델 비교
    compare_models(
        model_name="google-bert/bert-base-multilingual-cased",
        epochs=20
    )
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== Import 모델들 (api/module/ner) ==========
# api 디렉토리를 경로에 추가 (현재 디렉토리)
api_path = Path(__file__).parent.resolve()
if str(api_path) not in sys.path:
    sys.path.insert(0, str(api_path))

# ========== Import 순수 모델 (api/module/ner) ==========
# 순수 모델: ner_system.py에서 가져오기 (AutoModelForTokenClassification 직접 사용, CRF/BiLSTM 없음)
try:
    from module.ner.ner_system import ner_predict as pure_ner_predict
    from module.ner.ner_evaluate import evaluate_model as pure_ner_evaluate
    # 순수 모델 훈련: ner_train.py의 ner_train() 사용 (wrapper 함수, enable_visualization 지원)
    # 주의: 실제로는 시스템 모델을 훈련하지만, 순수 모델 훈련 함수가 없으므로 임시로 사용
    from module.ner.ner_train import ner_train as pure_ner_train
    SINGLE_MODEL_AVAILABLE = True
    print(f"✓ 순수 모델(api/module/ner) import 성공: {api_path}")
    print(f"   ⚠️ 순수 모델 훈련은 현재 시스템 모델 훈련 함수를 사용합니다.")
    print(f"   TODO: 순수 모델 훈련 함수 구현 필요 (AutoModelForTokenClassification 직접 사용)")
except ImportError as e:
    print(f"⚠️ 순수 모델(api/module/ner) import 실패: {e}")
    print(f"   경로 확인: {api_path}")
    print(f"   ner_system.py 존재 여부: {(api_path / 'module' / 'ner' / 'ner_system.py').exists()}")
    SINGLE_MODEL_AVAILABLE = False

# ========== Import 시스템 모델 (api/module/ner) ==========
try:
    # torchcrf가 필요한지 먼저 확인 (시스템 모델에 필요)
    try:
        import torchcrf
    except ImportError:
        print("⚠️ torchcrf 패키지가 설치되지 않았습니다.")
        print("   설치 명령: pip install torchcrf")
        print("   또는 requirements.txt에 torchcrf>=1.0.0 추가 후: pip install -r requirements.txt")
        raise ImportError("torchcrf 패키지가 필요합니다. pip install torchcrf")
    
    # 시스템 모델: ner_train.py에서 가져오기 (CRF + BiLSTM 추가)
    from module.ner.ner_train import train_ner_model as system_ner_train
    from module.ner.ner_system import ner_predict as system_ner_predict
    from module.ner.ner_evaluate import evaluate_model as system_ner_evaluate
    SYSTEM_MODEL_AVAILABLE = True
    print(f"✓ 시스템 모델(api/module/ner) import 성공: {api_path}")
except ImportError as e:
    print(f"⚠️ 시스템 모델(api/module/ner) import 실패: {e}")
    print(f"   경로 확인: {api_path}")
    print(f"   module/ner 디렉토리 존재 여부: {(api_path / 'module' / 'ner').exists()}")
    if (api_path / 'module' / 'ner').exists():
        print(f"   ner_train.py 존재 여부: {(api_path / 'module' / 'ner' / 'ner_train.py').exists()}")
    SYSTEM_MODEL_AVAILABLE = False

# 한글 폰트 설정
try:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
    
    font_found = False
    for font_name in korean_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            font_found = True
            break
    
    if not font_found:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


# ========== 유틸리티 함수 ==========

def ensure_dir(path: Path) -> Path:
    """디렉토리 생성"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_text_report(data: Dict[str, Any], filepath: Path):
    """텍스트 리포트 저장"""
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"NER 모델 성능 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        for key, value in data.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            elif isinstance(value, list):
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")
    
    print(f"텍스트 리포트 저장: {filepath}")




# ========== 순수 모델 함수들 (metadata 기반, 레이어 추가 없음) ==========

def pure_model_train(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    model_path: Optional[str] = None,
    epochs: int = 20,
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
    force_regenerate_data: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    순수 모델(metadata) 훈련 - 레이어 추가 없이 AutoModelForTokenClassification 직접 사용
    
    Args:
        model_name: 모델 이름
        model_path: 모델 저장 경로
        epochs: 학습 에포크
        batch_size: 배치 크기
        learning_rate: 학습률
        train_data_path: 학습 데이터 경로
        train_ratio: 학습 데이터 비율
        random_seed: 랜덤 시드
        dataset_size: 전체 데이터셋 크기 제한
        samples_per_file: 각 파일에서 샘플링할 최대 문장 개수
        sample_ratio_per_file: 각 파일에서 샘플링할 비율
        plot: 학습 곡선 시각화 여부
        plot_output_path: 시각화 저장 경로
    
    Returns:
        Dict[str, Any]: 훈련 결과
    """
    if not SINGLE_MODEL_AVAILABLE:
        return {"success": False, "error": "순수 모델(metadata)을 사용할 수 없습니다"}
    
    print("=" * 80)
    print("순수 모델(metadata) 훈련 시작 - 레이어 추가 없음")
    print("=" * 80)
    
    try:
        # 순수 모델 훈련: ner_train.py의 train_pure_ner_model 사용
        # 순수 모델은 AutoModelForTokenClassification을 직접 사용 (CRF/BiLSTM 없음)
        from module.ner.ner_train import train_pure_ner_model
        
        num_samples = dataset_size or 30000
        
        result = train_pure_ner_model(
            model_name=model_name,
            num_samples=num_samples,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=model_path,
            use_gpu=True,
            use_realistic_data=kwargs.get("use_realistic_data", True),
            enable_early_stopping=kwargs.get("enable_early_stopping", False),
            force_regenerate_data=force_regenerate_data,
            enable_balanced_sampling=kwargs.get("enable_balanced_sampling", True)
        )
        
        # train_pure_ner_model은 (model, tokenizer, history_metrics) tuple을 반환
        # dict로 변환
        if isinstance(result, tuple):
            model_obj, tokenizer_obj, history_metrics = result
            result_dict = {
                "model": model_obj,
                "tokenizer": tokenizer_obj,
                "history": history_metrics.get("history", {}) if isinstance(history_metrics, dict) else {},
                "epochs": epochs,
                "train_samples": num_samples,
                "val_samples": None  # validation 샘플 수는 history에서 가져올 수 있음
            }
        else:
            result_dict = result if isinstance(result, dict) else {}
        
        # 텍스트 리포트 저장
        model_safe_name = model_name.replace('/', '_')
        report_path = Path(f"data/out/ner_visualization/pure_{model_safe_name}_train_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 히스토리 로드 시도
        history = result_dict.get("history", {})
        if model_path:
            history_path = Path(model_path) / "training_history.json"
        else:
            history_path = Path(f"models/ner_pure/{model_safe_name}/training_history.json")
        
        if not history and history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                history = history_data.get("history", {})
                result_dict["history"] = history
        
        save_text_report({
            "모델 타입": "순수 모델 (레이어 추가 없음)",
            "모델명": model_name,
            "훈련 완료 시간": datetime.now().isoformat(),
            "에포크": epochs,
            "학습 샘플 수": num_samples,
            "배치 크기": batch_size,
            "학습률": learning_rate,
            "모델 경로": model_path or f"models/ner_pure/{model_safe_name}",
            "히스토리": history
        }, report_path)
        
        print(f"\n✓ 순수 모델 훈련 완료")
        return {"success": True, "result": result_dict, "report_path": str(report_path), "history": history}
    
    except ImportError:
        # train_pure_ner_model이 없으면 시스템 모델 훈련 함수를 사용 (임시)
        print("⚠️ 경고: 순수 모델 훈련 함수가 없어 시스템 모델 훈련 함수를 사용합니다.")
        print("   실제로는 CRF+BiLSTM이 포함된 시스템 모델이 훈련됩니다.")
        
        result = pure_ner_train(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_samples=dataset_size or 30000,
            enable_visualization=plot,
            enable_early_stopping=kwargs.get("enable_early_stopping", False),
            enable_balanced_sampling=kwargs.get("enable_balanced_sampling", True),
            force_regenerate_data=kwargs.get("force_regenerate_data", False),
            debug=kwargs.get("debug", False)
        )
        
        # 텍스트 리포트 저장
        report_path = Path(f"data/out/ner_visualization/pure_{model_name.replace('/', '_')}_train_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_text_report({
            "모델 타입": "순수 모델 (레이어 추가 없음) - 임시로 시스템 모델 사용",
            "모델명": model_name,
            "훈련 완료 시간": datetime.now().isoformat(),
            "에포크": result.get("epochs"),
            "학습 샘플 수": result.get("train_samples"),
            "검증 샘플 수": result.get("val_samples"),
            "모델 경로": result.get("model_path"),
            "히스토리": result.get("history", {})
        }, report_path)
        
        print(f"\n✓ 순수 모델 훈련 완료 (임시)")
        return {"success": True, "result": result, "report_path": str(report_path)}
    
    except Exception as e:
        print(f"\n✗ 순수 모델 훈련 실패: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def pure_model_predict(
    text: str,
    model_name: str = "google-bert/bert-base-multilingual-cased",
    model_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    순수 모델(metadata) 예측
    
    Args:
        text: 예측할 텍스트
        model_name: 모델 이름
        model_path: 모델 경로
    
    Returns:
        Dict[str, Any]: 예측 결과 (extracted_entities 포함)
    """
    if not SINGLE_MODEL_AVAILABLE:
        return {"success": False, "error": "단일 모델(metadata)을 사용할 수 없습니다"}
    
    try:
        # 순수 모델 예측: ner_system.py의 extract_entities_from_text 직접 사용
        from module.ner.ner_system import extract_entities_from_text
        
        entities = extract_entities_from_text(
            text=text,
            model_name=model_name,
            model_path=Path(model_path) if model_path else None,
            debug=False
        )
        
        return {
            "success": True,
            "result": {
                "extracted_entities": entities,
                "entity_count": len(entities)
            }
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def pure_model_validate(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    model_path: Optional[str] = None,
    plot: bool = True,
    plot_output_path: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    순수 모델(metadata) 검증
    
    Args:
        model_name: 모델 이름
        model_path: 모델 경로
        plot: 시각화 여부
        plot_output_path: 시각화 저장 경로
        verbose: 상세 출력 여부
    
    Returns:
        Dict[str, Any]: 검증 결과
    """
    if not SINGLE_MODEL_AVAILABLE:
        return {"success": False, "error": "순수 모델(metadata)을 사용할 수 없습니다"}
    
    print("=" * 80)
    print("순수 모델(metadata) 검증 시작 - 레이어 추가 없음")
    print("=" * 80)
    
    try:
        # 순수 모델 검증: ner_evaluate.py의 evaluate_model 사용
        # verbose 중복 방지: kwargs에서 verbose 제거 후 명시적으로 전달
        eval_kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
        metrics = pure_ner_evaluate(
            model_name=model_name,
            use_validation_data=True,
            verbose=verbose,
            **eval_kwargs
        )
        
        # 텍스트 리포트 저장
        report_path = Path(f"data/out/ner_visualization/pure_{model_name.replace('/', '_')}_validate_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_text_report({
            "모델 타입": "순수 모델 (레이어 추가 없음)",
            "모델명": model_name,
            "검증 완료 시간": datetime.now().isoformat(),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1 Score": metrics.get("f1", 0.0),
            "전체 메트릭": metrics
        }, report_path)
        
        print(f"\n✓ 순수 모델 검증 완료")
        return {"success": True, "metrics": metrics, "report_path": str(report_path)}
    
    except Exception as e:
        print(f"\n✗ 순수 모델 검증 실패: {e}")
        return {"success": False, "error": str(e)}


# ========== 시스템 모델 함수들 (api 기반, CRF + BiLSTM 추가) ==========

def system_model_train(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    num_samples: int = 30000,
    num_epochs: int = 20,
    batch_size: int = 12,
    learning_rate: float = 2e-5,
    output_dir: Optional[Path] = None,
    use_gpu: bool = True,
    use_realistic_data: bool = True,
    enable_early_stopping: bool = False,
    force_regenerate_data: bool = True,
    enable_balanced_sampling: bool = True,
    plot: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    시스템 모델(api) 훈련 - CRF + BiLSTM 등 레이어 추가된 원래 시스템
    
    Args:
        model_name: HuggingFace 모델명
        num_samples: 학습 샘플 수
        num_epochs: 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        output_dir: 모델 저장 경로
        use_gpu: GPU 사용 여부
        use_realistic_data: 실전 기반 데이터 사용
        enable_early_stopping: Early stopping 활성화
        force_regenerate_data: 데이터 재생성 여부
        enable_balanced_sampling: 균형 샘플링 활성화
        plot: 시각화 여부
    
    Returns:
        Dict[str, Any]: 훈련 결과
    """
    if not SYSTEM_MODEL_AVAILABLE:
        return {"success": False, "error": "시스템 모델(api)을 사용할 수 없습니다"}
    
    print("=" * 80)
    print("시스템 모델(api) 훈련 시작 - CRF + BiLSTM 추가")
    print("=" * 80)
    
    try:
        # train_ner_model의 올바른 인자만 전달 (enable_visualization 제외)
        result = system_ner_train(
            model_name=model_name,
            num_samples=num_samples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=str(output_dir) if output_dir else None,
            use_gpu=use_gpu,
            use_realistic_data=use_realistic_data,
            enable_early_stopping=enable_early_stopping,
            force_regenerate_data=force_regenerate_data,
            enable_balanced_sampling=enable_balanced_sampling,
            # enable_visualization은 train_ner_model에서 지원하지 않음
            # 시각화는 train_ner_model 내부에서 자동으로 처리됨
            **{k: v for k, v in kwargs.items() if k not in ['enable_visualization', 'plot']}
        )
        
        # 텍스트 리포트 저장
        model_safe_name = model_name.replace('/', '_')
        report_path = Path(f"data/out/ner_visualization/mixed_{model_safe_name}_train_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 히스토리 로드 시도
        history = {}
        if output_dir:
            history_path = Path(output_dir) / "training_history.json"
        else:
            history_path = Path(f"models/ner/{model_safe_name}/training_history.json")
        
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                history = history_data.get("history", {})
        
        save_text_report({
            "모델 타입": "시스템 모델 (CRF + BiLSTM 추가)",
            "모델명": model_name,
            "훈련 완료 시간": datetime.now().isoformat(),
            "에포크": num_epochs,
            "학습 샘플 수": num_samples,
            "배치 크기": batch_size,
            "학습률": learning_rate,
            "모델 경로": output_dir or f"models/ner/{model_safe_name}",
            "히스토리": history
        }, report_path)
        
        print(f"\n✓ 시스템 모델 훈련 완료")
        return {"success": True, "result": result, "report_path": str(report_path), "history": history}
    
    except Exception as e:
        print(f"\n✗ 시스템 모델 훈련 실패: {e}")
        return {"success": False, "error": str(e)}


def system_model_predict(
    text: str,
    model_name: str = "google-bert/bert-base-multilingual-cased",
    use_regex: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    시스템 모델(api) 예측
    
    Args:
        text: 예측할 텍스트
        model_name: 모델 이름
        use_regex: 정규식 백업 사용 여부
    
    Returns:
        Dict[str, Any]: 예측 결과 (entities 포함)
    """
    if not SYSTEM_MODEL_AVAILABLE:
        return {"success": False, "error": "새로 설계한 모델(api)을 사용할 수 없습니다"}
    
    try:
        # 시스템 모델 예측: ner_system.py의 extract_entities_from_text 사용
        from module.ner.ner_system import extract_entities_from_text, extract_entities_by_regex
        
        # BERT-CRF 모델 예측
        entities = extract_entities_from_text(
            text=text,
            model_name=model_name,
            debug=False
        )
        
        # 정규식 백업 (use_regex=True인 경우)
        if use_regex:
            regex_entities = extract_entities_by_regex(text)
            # 중복 제거하면서 합치기
            entity_set = set(entities)
            entity_set.update(regex_entities)
            entities = list(entity_set)
        
        return {
            "success": True,
            "result": {
                "entities": entities,
                "entity_count": len(entities)
            }
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def system_model_validate(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    use_validation_data: bool = True,
    test_data_path: Optional[str] = None,
    use_regex: bool = True,
    auto_download: bool = True,
    verbose: bool = True,
    plot: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    시스템 모델(api) 검증
    
    Args:
        model_name: 모델 이름
        use_validation_data: validation 데이터 사용 여부
        test_data_path: 테스트 데이터 경로
        use_regex: 정규식 백업 사용 여부
        auto_download: 모델 자동 다운로드 여부
        verbose: 상세 출력 여부
        plot: 시각화 여부
    
    Returns:
        Dict[str, Any]: 검증 결과
    """
    if not SYSTEM_MODEL_AVAILABLE:
        return {"success": False, "error": "시스템 모델(api)을 사용할 수 없습니다"}
    
    print("=" * 80)
    print("시스템 모델(api) 검증 시작 - CRF + BiLSTM 추가")
    print("=" * 80)
    
    try:
        test_path = Path(test_data_path) if test_data_path else None
        result = system_ner_evaluate(
            model_name=model_name,
            use_validation_data=use_validation_data,
            test_data_path=test_path,
            use_regex=use_regex,
            auto_download=auto_download,
            verbose=verbose,
            **kwargs
        )
        
        # 텍스트 리포트 저장
        model_safe_name = model_name.replace('/', '_')
        report_path = Path(f"data/out/ner_visualization/mixed_{model_safe_name}_validate_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        overall_metrics = result.get("overall_metrics", {}) if result.get("success") else {}
        save_text_report({
            "모델 타입": "시스템 모델 (CRF + BiLSTM 추가)",
            "모델명": model_name,
            "검증 완료 시간": datetime.now().isoformat(),
            "Precision": overall_metrics.get("precision", 0.0),
            "Recall": overall_metrics.get("recall", 0.0),
            "F1 Score": overall_metrics.get("f1", 0.0),
            "TP": overall_metrics.get("tp", 0),
            "FP": overall_metrics.get("fp", 0),
            "FN": overall_metrics.get("fn", 0),
            "전체 메트릭": overall_metrics,
            "전체 결과": result
        }, report_path)
        
        print(f"\n✓ 시스템 모델 검증 완료")
        return {"success": True, "result": result, "report_path": str(report_path)}
    
    except Exception as e:
        print(f"\n✗ 시스템 모델 검증 실패: {e}")
        return {"success": False, "error": str(e)}


# ========== 각 모델별 테스트, 성능, 예측 결과 모두 보여주기 ==========

def test_model_comprehensive(
    model_name: str,
    model_type: str = "system",  # "system" or "pure"
    test_samples: Optional[List[str]] = None,
    use_regex: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    각 모델에 대해 테스트, 성능, 예측 결과를 모두 보여주는 함수
    
    Args:
        model_name: 모델 이름
        model_type: 모델 타입 ("system" 또는 "pure")
        test_samples: 테스트할 샘플 텍스트 리스트 (None이면 기본 샘플 사용)
        use_regex: 정규식 백업 사용 여부 (시스템 모델만)
        verbose: 상세 출력 여부
    
    Returns:
        Dict[str, Any]: 테스트, 성능, 예측 결과
    """
    print("=" * 80)
    print(f"{'시스템' if model_type == 'system' else '순수'} 모델 종합 테스트: {model_name}")
    print("=" * 80)
    
    results = {
        "model_name": model_name,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "test_results": [],
        "performance_metrics": {},
        "prediction_results": []
    }
    
    # 기본 테스트 샘플
    if test_samples is None:
        test_samples = [
            "저작물 저작재산권 양도 계약서\n\n계약자: 김민수\n전화번호: 010-1234-5678\n이메일: minsu.kim@gmail.com\n주소: 서울시 강남구 테헤란로 123\n\n수탁기관: 한국콘텐츠진흥원\n담당자: 박영희 부장\n계약금: 5,000,000원",
            "작성일: 2024년 1월 15일\n작성자: 이영희\n소속: 주식회사 테크노\n직책: 개발팀장\n연락처: 02-1234-5678\n이메일: yh.lee@techno.co.kr",
            "프로젝트명: AI 기반 저작권 관리 시스템\n계약기간: 2024년 1월 1일부터 2024년 12월 31일까지\n계약금액: 100,000,000원\n법령 근거: 저작권법 제31조"
        ]
    
    # 1. 테스트 (예측 결과 샘플)
    print("\n[1/3] 테스트 - 예측 결과 샘플")
    print("-" * 80)
    
    for idx, sample_text in enumerate(test_samples, 1):
        print(f"\n샘플 {idx}:")
        print(f"입력 텍스트: {sample_text[:100]}..." if len(sample_text) > 100 else f"입력 텍스트: {sample_text}")
        
        try:
            if model_type == "system":
                pred_result = system_model_predict(
                    text=sample_text,
                    model_name=model_name,
                    use_regex=use_regex
                )
            else:
                pred_result = pure_model_predict(
                    text=sample_text,
                    model_name=model_name
                )
            
            if pred_result.get("success"):
                if model_type == "system":
                    entities = pred_result.get("result", {}).get("entities", [])
                else:
                    # 순수 모델: extracted_entities 사용
                    entities = pred_result.get("result", {}).get("extracted_entities", [])
                
                print(f"예측된 엔티티 ({len(entities)}개):")
                for entity, entity_type in entities[:10]:  # 최대 10개만 표시
                    print(f"  - {entity} ({entity_type})")
                if len(entities) > 10:
                    print(f"  ... 외 {len(entities) - 10}개")
                
                results["test_results"].append({
                    "sample_idx": idx,
                    "input_text": sample_text[:200],  # 처음 200자만 저장
                    "predicted_entities": entities,
                    "entity_count": len(entities)
                })
            else:
                error = pred_result.get("error", "알 수 없는 오류")
                print(f"예측 실패: {error}")
                results["test_results"].append({
                    "sample_idx": idx,
                    "input_text": sample_text[:200],
                    "error": error
                })
        
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            results["test_results"].append({
                "sample_idx": idx,
                "input_text": sample_text[:200],
                "error": str(e)
            })
    
    # 2. 성능 (메트릭)
    print("\n[2/3] 성능 - 메트릭")
    print("-" * 80)
    
    try:
        if model_type == "system":
            perf_result = system_model_validate(
                model_name=model_name,
                use_validation_data=True,
                use_regex=use_regex,
                verbose=verbose
            )
        else:
            perf_result = pure_model_validate(
                model_name=model_name,
                verbose=verbose
            )
        
        if perf_result.get("success"):
            if model_type == "system":
                overall_metrics = perf_result.get("result", {}).get("overall_metrics", {})
            else:
                overall_metrics = perf_result.get("metrics", {})
            
            precision = overall_metrics.get("precision", 0.0)
            recall = overall_metrics.get("recall", 0.0)
            f1 = overall_metrics.get("f1", 0.0)
            tp = overall_metrics.get("tp", 0)
            fp = overall_metrics.get("fp", 0)
            fn = overall_metrics.get("fn", 0)
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}")
            
            results["performance_metrics"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "total_true": overall_metrics.get("total_true", 0),
                "total_pred": overall_metrics.get("total_pred", 0)
            }
            
            # 엔티티 타입별 메트릭
            if model_type == "system":
                type_metrics = perf_result.get("result", {}).get("entity_type_metrics", {})
            else:
                type_metrics = perf_result.get("metrics", {}).get("entity_type_metrics", {})
            
            if type_metrics:
                print(f"\n엔티티 타입별 성능:")
                for entity_type in sorted(type_metrics.keys())[:10]:  # 최대 10개만 표시
                    metrics = type_metrics[entity_type]
                    print(f"  {entity_type:20s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
                
                results["performance_metrics"]["entity_type_metrics"] = type_metrics
        else:
            error = perf_result.get("error", "알 수 없는 오류")
            print(f"성능 평가 실패: {error}")
            results["performance_metrics"] = {"error": error}
    
    except Exception as e:
        print(f"성능 평가 중 오류 발생: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        results["performance_metrics"] = {"error": str(e)}
    
    # 3. 예측 결과 (전체 예측 결과)
    print("\n[3/3] 예측 결과 - 전체 예측 결과")
    print("-" * 80)
    
    # 모든 테스트 샘플에 대한 예측 결과 수집
    all_predictions = []
    for idx, sample_text in enumerate(test_samples, 1):
        try:
            if model_type == "system":
                pred_result = system_model_predict(
                    text=sample_text,
                    model_name=model_name,
                    use_regex=use_regex
                )
            else:
                pred_result = pure_model_predict(
                    text=sample_text,
                    model_name=model_name
                )
            
            if pred_result.get("success"):
                if model_type == "system":
                    entities = pred_result.get("result", {}).get("entities", [])
                else:
                    # 순수 모델: extracted_entities 사용
                    entities = pred_result.get("result", {}).get("extracted_entities", [])
                
                all_predictions.append({
                    "sample_idx": idx,
                    "input_text": sample_text,
                    "predicted_entities": entities,
                    "entity_count": len(entities)
                })
        
        except Exception as e:
            all_predictions.append({
                "sample_idx": idx,
                "input_text": sample_text,
                "error": str(e)
            })
    
    results["prediction_results"] = all_predictions
    
    # 요약 출력
    print(f"\n요약:")
    print(f"  - 테스트 샘플 수: {len(test_samples)}")
    print(f"  - 성공한 예측: {sum(1 for r in results['test_results'] if 'error' not in r)}/{len(test_samples)}")
    if results["performance_metrics"] and "error" not in results["performance_metrics"]:
        print(f"  - F1 Score: {results['performance_metrics']['f1']:.4f}")
        print(f"  - Precision: {results['performance_metrics']['precision']:.4f}")
        print(f"  - Recall: {results['performance_metrics']['recall']:.4f}")
    
    print("=" * 80)
    
    # 결과 파일 저장
    model_safe_name = model_name.replace('/', '_')
    model_prefix = "pure" if model_type == "pure" else "mixed"
    
    # JSON 저장
    json_path = Path(f"data/out/ner_visualization/{model_prefix}_{model_safe_name}_comprehensive.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n종합 테스트 결과 JSON 저장: {json_path}")
    
    # 텍스트 리포트 저장
    report_path = Path(f"data/out/ner_visualization/{model_prefix}_{model_safe_name}_comprehensive.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_text_report(results, report_path)
    
    return results


# ========== 비교 함수 ==========

def compare_models(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    # 훈련 파라미터
    epochs: int = 20,
    batch_size_single: int = 32,
    batch_size_system: int = 12,
    learning_rate: float = 2e-5,
    # 단일 모델 파라미터
    dataset_size: Optional[int] = None,
    samples_per_file: Optional[int] = None,
    sample_ratio_per_file: Optional[float] = None,
    # 새로 설계한 모델 파라미터
    num_samples: int = 30000,
    use_realistic_data: bool = True,
    enable_early_stopping: bool = False,
    # 검증 파라미터
    use_regex: bool = True,
    use_validation_data: bool = True,
    # 출력 설정
    plot: bool = True,
    save_comparison: bool = True,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    단일 모델과 새로 설계한 모델의 성능 비교
    
    Args:
        model_name: 모델 이름
        epochs: 학습 에포크
        batch_size_single: 단일 모델 배치 크기
        batch_size_system: 새로 설계한 모델 배치 크기
        learning_rate: 학습률
        dataset_size: 단일 모델 전체 데이터셋 크기 제한
        samples_per_file: 단일 모델 각 파일 샘플링 개수
        sample_ratio_per_file: 단일 모델 각 파일 샘플링 비율
        num_samples: 새로 설계한 모델 학습 샘플 수
        use_realistic_data: 새로 설계한 모델 실전 데이터 사용
        enable_early_stopping: 새로 설계한 모델 Early stopping
        use_regex: 정규식 백업 사용 여부
        use_validation_data: validation 데이터 사용 여부
        plot: 시각화 여부
        save_comparison: 비교 결과 저장 여부
        output_dir: 출력 디렉토리
    
    Returns:
        Dict[str, Any]: 비교 결과
    """
    print("=" * 80)
    print("모델 성능 비교 시작")
    print("=" * 80)
    print(f"모델: {model_name}")
    print(f"에포크: {epochs}")
    print("=" * 80)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = Path("data/out/ner_visualization")
    else:
        output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    model_safe_name = model_name.replace('/', '_')
    
    comparison_results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "single_model": {},
        "system_model": {},
        "comparison": {}
    }
    
    # 1. 순수 모델 훈련
    print("\n[1/8] 순수 모델 훈련 중...")
    pure_train_result = pure_model_train(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size_single,
        learning_rate=learning_rate,
        dataset_size=dataset_size,
        samples_per_file=samples_per_file,
        sample_ratio_per_file=sample_ratio_per_file,
        plot=plot,
        plot_output_path=str(output_dir / f"pure_{model_safe_name}_train.png"),
        **kwargs
    )
    comparison_results["pure_model"] = {"train": pure_train_result}
    
    # 2. 시스템 모델 훈련
    print("\n[2/8] 시스템 모델 훈련 중...")
    system_train_result = system_model_train(
        model_name=model_name,
        num_samples=num_samples,
        num_epochs=epochs,
        batch_size=batch_size_system,
        learning_rate=learning_rate,
        use_realistic_data=use_realistic_data,
        enable_early_stopping=enable_early_stopping,
        plot=plot,
        **kwargs
    )
    comparison_results["system_model"] = {"train": system_train_result}
    
    # 3. 순수 모델 테스트, 성능, 예측 결과 모두 보여주기
    print("\n[3/8] 순수 모델 종합 테스트 중 (테스트, 성능, 예측 결과)...")
    pure_comprehensive_result = test_model_comprehensive(
        model_name=model_name,
        model_type="pure",
        use_regex=False,  # 순수 모델은 정규식 사용 안 함
        verbose=True
    )
    comparison_results["pure_model"]["comprehensive"] = pure_comprehensive_result
    
    # 4. 시스템 모델 테스트, 성능, 예측 결과 모두 보여주기
    print("\n[4/8] 시스템 모델 종합 테스트 중 (테스트, 성능, 예측 결과)...")
    system_comprehensive_result = test_model_comprehensive(
        model_name=model_name,
        model_type="system",
        use_regex=use_regex,
        verbose=True
    )
    comparison_results["system_model"]["comprehensive"] = system_comprehensive_result
    
    # 5. 순수 모델 검증
    print("\n[5/8] 순수 모델 검증 중...")
    pure_validate_result = pure_model_validate(
        model_name=model_name,
        plot=plot,
        plot_output_path=str(output_dir / f"pure_{model_safe_name}_validate.png"),
        **kwargs
    )
    comparison_results["pure_model"]["validate"] = pure_validate_result
    
    # 6. 시스템 모델 검증
    print("\n[6/8] 시스템 모델 검증 중...")
    system_validate_result = system_model_validate(
        model_name=model_name,
        use_validation_data=use_validation_data,
        use_regex=use_regex,
        verbose=True,
        plot=plot,
        **kwargs
    )
    comparison_results["system_model"]["validate"] = system_validate_result
    
    # 7. 결과 비교
    print("\n[7/8] 결과 비교 중...")
    comparison = {}
    
    # 훈련 결과 비교
    if pure_train_result.get("success") and system_train_result.get("success"):
        pure_train = pure_train_result.get("result", {})
        system_train = system_train_result.get("result", {})
        
        # pure_train이 tuple인 경우 처리 (train_pure_ner_model이 tuple 반환)
        if isinstance(pure_train, tuple):
            pure_train = {}
        
        # pure_train이 dict가 아닌 경우 빈 dict로 처리
        if not isinstance(pure_train, dict):
            pure_train = {}
        
        comparison["train"] = {
            "pure": {
                "epochs": pure_train.get("epochs") if isinstance(pure_train, dict) else epochs,
                "train_samples": pure_train.get("train_samples") if isinstance(pure_train, dict) else None,
                "val_samples": pure_train.get("val_samples") if isinstance(pure_train, dict) else None,
            },
            "system": {
                "epochs": system_train.get("num_epochs") if isinstance(system_train, dict) and "num_epochs" in system_train else epochs,
                "samples": system_train.get("num_samples") if isinstance(system_train, dict) and "num_samples" in system_train else num_samples,
            }
        }
    
    # 검증 결과 비교
    pure_metrics = {}
    system_metrics = {}
    
    if pure_validate_result.get("success"):
        pure_metrics = pure_validate_result.get("metrics", {})
    
    if system_validate_result.get("success"):
        system_metrics = system_validate_result.get("result", {}).get("overall_metrics", {})
    
    if pure_metrics and system_metrics:
        comparison["validate"] = {
            "pure": {
                "precision": pure_metrics.get("precision", 0.0),
                "recall": pure_metrics.get("recall", 0.0),
                "f1": pure_metrics.get("f1", 0.0),
            },
            "system": {
                "precision": system_metrics.get("precision", 0.0),
                "recall": system_metrics.get("recall", 0.0),
                "f1": system_metrics.get("f1", 0.0),
            }
        }
        
        # 성능 차이 계산
        pure_f1 = comparison["validate"]["pure"]["f1"]
        system_f1 = comparison["validate"]["system"]["f1"]
        comparison["validate"]["f1_diff"] = system_f1 - pure_f1
        comparison["validate"]["f1_improvement"] = ((system_f1 - pure_f1) / pure_f1 * 100) if pure_f1 > 0 else 0.0
    
    comparison_results["comparison"] = comparison
    
    # 8. 결과 출력
    print("\n[8/8] 비교 결과 요약")
    print("=" * 80)
    
    if "validate" in comparison:
        print(f"\n{'메트릭':<20} {'순수 모델':<15} {'시스템 모델':<15} {'차이':<15}")
        print("-" * 80)
        
        pure_precision = comparison["validate"]["pure"]["precision"]
        system_precision = comparison["validate"]["system"]["precision"]
        print(f"{'Precision':<20} {pure_precision:<15.4f} {system_precision:<15.4f} {system_precision - pure_precision:<15.4f}")
        
        pure_recall = comparison["validate"]["pure"]["recall"]
        system_recall = comparison["validate"]["system"]["recall"]
        print(f"{'Recall':<20} {pure_recall:<15.4f} {system_recall:<15.4f} {system_recall - pure_recall:<15.4f}")
        
        pure_f1 = comparison["validate"]["pure"]["f1"]
        system_f1 = comparison["validate"]["system"]["f1"]
        print(f"{'F1 Score':<20} {pure_f1:<15.4f} {system_f1:<15.4f} {system_f1 - pure_f1:<15.4f}")
        
        if comparison["validate"].get("f1_improvement"):
            improvement = comparison["validate"]["f1_improvement"]
            print(f"\nF1 Score 개선율: {improvement:.2f}%")
    
    print("=" * 80)
    
    # 결과 저장
    if save_comparison:
        # JSON 저장
        json_output_file = output_dir / f"comparison_{model_safe_name}_results.json"
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        print(f"\n비교 결과 JSON 저장: {json_output_file}")
        
        # 텍스트 리포트 저장
        text_output_file = output_dir / f"comparison_{model_safe_name}_report.txt"
        save_text_report(comparison_results, text_output_file)
    
    return comparison_results


# ========== 기존 함수 (하위 호환성) ==========

def evaluate_all_models(
    model_names: list = None,
    use_validation_data: bool = True,
    use_regex: bool = True,
    auto_download: bool = True,
    verbose: bool = True
) -> dict:
    """
    3개 모델을 모두 평가하고 결과를 요약 (기존 함수 유지)
    """
    if not SYSTEM_MODEL_AVAILABLE:
        return {"error": "새로 설계한 모델(api)을 사용할 수 없습니다"}
    
    if model_names is None:
        model_names = [
            "google-bert/bert-base-multilingual-cased",
            "klue/roberta-large",
            "FacebookAI/xlm-roberta-large"
        ]
    
    results = {}
    
    print("=" * 80)
    print("모델 성능 평가 시작")
    print("=" * 80)
    
    for idx, model_name in enumerate(model_names, 1):
        print(f"\n{'=' * 80}")
        print(f"{idx}️⃣ {model_name} 모델 평가")
        print(f"{'=' * 80}")
        
        try:
            result = system_model_validate(
                model_name=model_name,
                use_validation_data=use_validation_data,
                use_regex=use_regex,
                auto_download=auto_download,
                verbose=verbose
            )
            
            if result.get("success"):
                overall = result.get("result", {}).get("overall_metrics", {})
                precision = overall.get("precision", 0.0)
                recall = overall.get("recall", 0.0)
                f1 = overall.get("f1", 0.0)
                
                results[model_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "tp": overall.get("tp", 0),
                    "fp": overall.get("fp", 0),
                    "fn": overall.get("fn", 0),
                    "total_true": overall.get("total_true", 0),
                    "total_pred": overall.get("total_pred", 0),
                }
                
                if verbose:
                    print(f"\n✓ {model_name} 평가 완료")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall:    {recall:.4f}")
                    print(f"  F1 Score:  {f1:.4f}")
            else:
                error = result.get("error", "알 수 없는 오류")
                results[model_name] = {
                    "success": False,
                    "error": error
                }
                
                if verbose:
                    print(f"\n✗ {model_name} 평가 실패: {error}")
        
        except Exception as e:
            results[model_name] = {
                "success": False,
                "error": str(e)
            }
            
            if verbose:
                print(f"\n✗ {model_name} 평가 중 오류 발생: {e}")
    
    # 결과 요약
    summary = {
        "models": results,
    }
    
    # 결과 출력
    print(f"\n{'=' * 80}")
    print("평가 결과 요약")
    print(f"{'=' * 80}")
    print(f"\n{'모델명':<50} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        if "f1_score" in metrics:
            print(f"{model_name:<50} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
        else:
            error = metrics.get("error", "알 수 없는 오류")
            print(f"{model_name:<50} {'실패':<12} {'실패':<12} {'실패':<12} ({error})")
    
    print("-" * 80)
    print(f"{'=' * 80}\n")
    
    # 결과를 JSON 파일로 저장
    output_file = Path("data/out/ner_evaluation_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"평가 결과 저장: {output_file}")
    
    return summary


# ========== 6개 모델 비교 함수 ==========

def compare_all_6_models(
    model_names: Optional[List[str]] = None,
    # 훈련 파라미터
    epochs: int = 20,
    batch_size_pure: int = 32,
    batch_size_system: int = 12,
    learning_rate: float = 2e-5,
    # 순수 모델 파라미터
    dataset_size: Optional[int] = None,
    samples_per_file: Optional[int] = None,
    sample_ratio_per_file: Optional[float] = None,
    # 시스템 모델 파라미터
    num_samples: int = 30000,
    use_realistic_data: bool = True,
    enable_early_stopping: bool = False,
    # 검증 파라미터
    use_regex: bool = True,
    use_validation_data: bool = True,
    # 출력 설정
    plot: bool = True,
    save_comparison: bool = True,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    6개 모델 비교: 시스템 모델 3개 + 순수 모델 3개
    
    Args:
        model_names: 비교할 모델 리스트 (기본: 3개 모델)
        epochs: 학습 에포크
        batch_size_pure: 순수 모델 배치 크기
        batch_size_system: 시스템 모델 배치 크기
        learning_rate: 학습률
        dataset_size: 순수 모델 전체 데이터셋 크기 제한
        samples_per_file: 순수 모델 각 파일 샘플링 개수
        sample_ratio_per_file: 순수 모델 각 파일 샘플링 비율
        num_samples: 시스템 모델 학습 샘플 수
        use_realistic_data: 시스템 모델 실전 데이터 사용
        enable_early_stopping: 시스템 모델 Early stopping
        use_regex: 정규식 백업 사용 여부
        use_validation_data: validation 데이터 사용 여부
        plot: 시각화 여부
        save_comparison: 비교 결과 저장 여부
        output_dir: 출력 디렉토리
    
    Returns:
        Dict[str, Any]: 6개 모델 비교 결과
    """
    if model_names is None:
        model_names = [
            "google-bert/bert-base-multilingual-cased",
            "klue/roberta-large",
            "FacebookAI/xlm-roberta-large"
        ]
    
    print("=" * 80)
    print("6개 모델 성능 비교 시작")
    print("=" * 80)
    print(f"시스템 모델 3개: {model_names}")
    print(f"순수 모델 3개: {model_names}")
    print(f"총 6개 모델 비교")
    print("=" * 80)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = Path("data/out/ner_visualization")
    ensure_dir(output_dir)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "summary": {}
    }
    
    # 각 모델별로 비교 수행
    for model_name in model_names:
        print(f"\n{'=' * 80}")
        print(f"모델 비교: {model_name}")
        print(f"{'=' * 80}")
        
        result = compare_models(
            model_name=model_name,
            epochs=epochs,
            batch_size_single=batch_size_pure,
            batch_size_system=batch_size_system,
            learning_rate=learning_rate,
            dataset_size=dataset_size,
            samples_per_file=samples_per_file,
            sample_ratio_per_file=sample_ratio_per_file,
            num_samples=num_samples,
            use_realistic_data=use_realistic_data,
            enable_early_stopping=enable_early_stopping,
            use_regex=use_regex,
            use_validation_data=use_validation_data,
            plot=plot,
            save_comparison=False,  # 개별 저장은 하지 않고 전체 요약만 저장
            output_dir=output_dir,
            **kwargs
        )
        
        all_results["models"][model_name] = result
    
    # 전체 요약 생성
    print("\n" + "=" * 80)
    print("6개 모델 전체 비교 요약")
    print("=" * 80)
    
    summary_data = []
    for model_name, result in all_results["models"].items():
        comparison = result.get("comparison", {})
        if "validate" in comparison:
            pure_metrics = comparison["validate"].get("pure", {})
            system_metrics = comparison["validate"].get("system", {})
            
            summary_data.append({
                "model": model_name,
                "pure": {
                    "precision": pure_metrics.get("precision", 0.0),
                    "recall": pure_metrics.get("recall", 0.0),
                    "f1": pure_metrics.get("f1", 0.0),
                },
                "system": {
                    "precision": system_metrics.get("precision", 0.0),
                    "recall": system_metrics.get("recall", 0.0),
                    "f1": system_metrics.get("f1", 0.0),
                }
            })
    
    # 요약 출력
    if summary_data:
        print(f"\n{'모델명':<50} {'순수 F1':<12} {'시스템 F1':<12} {'차이':<12}")
        print("-" * 80)
        for data in summary_data:
            pure_f1 = data["pure"]["f1"]
            system_f1 = data["system"]["f1"]
            diff = system_f1 - pure_f1
            print(f"{data['model']:<50} {pure_f1:<12.4f} {system_f1:<12.4f} {diff:<12.4f}")
        print("-" * 80)
    
    all_results["summary"] = summary_data
    
    # 결과 저장
    if save_comparison:
        # JSON 저장
        json_output_file = output_dir / "all_6_models_comparison.json"
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n전체 비교 결과 JSON 저장: {json_output_file}")
        
        # 텍스트 리포트 저장
        text_output_file = output_dir / "all_6_models_comparison_report.txt"
        save_text_report(all_results, text_output_file)
    
    print("\n" + "=" * 80)
    print("6개 모델 비교 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  - 각 모델별 6개 이미지 파일 (총 18개)")
    print("  - 각 모델별 텍스트 리포트 파일")
    print("  - 전체 비교 결과 JSON 및 텍스트 리포트")
    print("=" * 80)
    
    return all_results


# ========== 메인 실행 ==========

if __name__ == "__main__":
    # ner_test.py만 실행해도 모든 것이 동작하도록 구성
    print("=" * 80)
    print("NER 모델 성능 비교 테스트 시작")
    print("=" * 80)
    print("\n6개 모델 비교를 시작합니다...")
    print("(시스템 모델 3개 + 순수 모델 3개)\n")
    
    # 6개 모델 비교 실행
    compare_all_6_models(
        epochs=20,
        batch_size_pure=32,
        batch_size_system=12,
        learning_rate=2e-5,
        plot=True,
        save_comparison=True
    )
    
    print("\n" + "=" * 80)
    print("모든 테스트 완료!")
    print("=" * 80)
