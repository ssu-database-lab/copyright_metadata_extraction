"""NER 학습 — Hugging Face Token Classification (BERT 등) + PEFT LoRA."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from module.extractor.ner._runtime import (
    DEFAULT_MODEL,
    NER_DEBUG_SESSION_DIR,
    attach_ner_debug_file_logging,
    configure_ner_debug,
    ensure_model_ready,
    get_train_dir,
    get_training_data_signature,
    invalidate_model_cache,
    make_ner_debug_session_dir,
    model_display_name,
    ner_debug_print,
    project_root,
    read_train_state,
    resolve_bio_train_dir,
    write_train_state,
)

log = logging.getLogger(__name__)


def ner_train(
    *,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
    input_path: Optional[Union[str, Path]] = None,
    model_path: Optional[str] = None,
    fine_tuning_method: str = "lora",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    train_ratio: float = 8 / 12,
    val_ratio: float = 2 / 12,
    test_ratio: float = 2 / 12,
    split_seed: int = 42,
    debug: bool = False,
    debug_path: Optional[str] = None,
    save_plots: bool = False,
    early_stopping_patience: int = 0,
    extra_input_paths: Optional[Sequence[Union[str, Path]]] = None,
    negative_input_paths: Optional[Sequence[Union[str, Path]]] = None,
    max_per_label: Optional[int] = None,
) -> Dict[str, Any]:
    """NER 학습. model_name = HuggingFace ID 또는 로컬 이름.

    input_path: BIO 학습용 .jsonl이 있는 디렉터리. None이면 ``configs/train`` (하위 ``raw/`` 자동).
    model_path: 모델 디렉터리 루트. None이면 프로젝트의 ``models``. 절대/상대 경로 모두 지원.
    fine_tuning_method: "lora" (기본값) | "full"
    debug_path: debug=True일 때 세션 디렉터리 루트. None이면 프로젝트 ``debug/`` 아래.
    """
    ctx_token = None
    session_dir: Optional[Path] = None
    if debug:
        session_dir = make_ner_debug_session_dir(
            debug_path, model_name, debug_kind="train", threshold_dir="na",
        )
        ctx_token = NER_DEBUG_SESSION_DIR.set(session_dir)
        configure_ner_debug(True)
        attach_ner_debug_file_logging(session_dir)
        meta = {
            "kind": "ner_train",
            "model_name": model_name,
            "input_path": str(input_path) if input_path else None,
            "model_path": model_path,
            "fine_tuning_method": fine_tuning_method,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": split_seed,
            "force": force,
            "debug_path": debug_path,
            "threshold_dir": "na",
            "session_dir": str(session_dir),
        }
        (session_dir / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ner_debug_print(
            f"[NER debug][train] 시작 model={model_name!r} force={force} "
            f"input_path={input_path!r} model_path={model_path!r} session_dir={session_dir}"
        )

    try:
        return _ner_train_impl(
            model_name=model_name,
            force=force,
            input_path=input_path,
            model_path=model_path,
            fine_tuning_method=fine_tuning_method,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            debug=debug,
            save_plots=save_plots,
            early_stopping_patience=early_stopping_patience,
            extra_input_paths=extra_input_paths,
            negative_input_paths=negative_input_paths,
            max_per_label=max_per_label,
            debug_session_dir=str(session_dir) if session_dir else None,
        )
    finally:
        if debug and ctx_token is not None:
            configure_ner_debug(False)
            NER_DEBUG_SESSION_DIR.reset(ctx_token)


def _ner_train_impl(
    *,
    model_name: str,
    force: bool,
    input_path: Optional[Union[str, Path]],
    model_path: Optional[str],
    fine_tuning_method: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    warmup_ratio: float,
    weight_decay: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    debug: bool,
    save_plots: bool,
    early_stopping_patience: int,
    extra_input_paths: Optional[Sequence[Union[str, Path]]],
    negative_input_paths: Optional[Sequence[Union[str, Path]]],
    max_per_label: Optional[int],
    debug_session_dir: Optional[str],
) -> Dict[str, Any]:
    """실제 학습 구현 (디버그 세션 스캐폴딩 제외)."""
    _ = debug_session_dir  # 현재 구현은 ContextVar(NER_DEBUG_SESSION_DIR)만 사용

    display = model_display_name(model_name)
    model_dir = ensure_model_ready(model_name, model_path=model_path)

    if input_path is not None:
        train_dir = Path(input_path)
        if not train_dir.is_absolute():
            train_dir = project_root() / train_dir
        train_dir = train_dir.resolve()
    else:
        train_dir = get_train_dir()

    train_dir = resolve_bio_train_dir(train_dir)
    current_sig = get_training_data_signature(train_dir)
    state = read_train_state(model_name) or {}

    result: Dict[str, Any] = {
        "model": display,
        "model_type": "token_cls",
        "fine_tuning_method": fine_tuning_method,
        "training_needed": False,
        "training_executed": False,
        "debug": debug,
    }

    if debug:
        jsonl_files = sorted(train_dir.glob("*.jsonl"))
        ner_debug_print(
            f"[NER debug][train] model_dir={model_dir} train_dir={train_dir} "
            f"jsonl_files={len(jsonl_files)} "
            f"{[p.name for p in jsonl_files[:12]]}{'...' if len(jsonl_files) > 12 else ''}"
        )
        ner_debug_print(f"[NER debug][train] data_signature={current_sig[:16]}... full={current_sig}")
        ner_debug_print(f"[NER debug][train] train_state={state}")

    if not current_sig:
        result["message"] = "학습 데이터가 없습니다."
        print(f"[NER Train] [{display}] {result['message']}")
        if debug:
            ner_debug_print("[NER debug][train] 종료: 서명 없음(데이터 없음)")
        return result

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-5:
        result["message"] = (
            f"train_ratio + val_ratio + test_ratio 합이 1이어야 합니다 (현재 합={ratio_sum})."
        )
        print(f"[NER Train] [{display}] {result['message']}")
        return result

    adapter_dir = model_dir / "adapter"
    already_trained = adapter_dir.exists() and any(adapter_dir.iterdir())

    if debug:
        ner_debug_print(
            f"[NER debug][train] adapter_dir={adapter_dir} "
            f"already_trained={already_trained} force={force}"
        )

    if not force and state.get("signature") == current_sig and already_trained:
        result["message"] = "학습 데이터 변경 없음. 학습 불필요."
        print(f"[NER Train] [{display}] {result['message']}")
        if debug:
            ner_debug_print("[NER debug][train] 종료: 스킵(시그니처 동일·어댑터 존재)")
        return result

    result["training_needed"] = True
    print(f"[NER Train] [{display}] 학습 시작...")

    extra_dirs = None
    if extra_input_paths:
        extra_dirs = []
        for p in extra_input_paths:
            ep = Path(p)
            if not ep.is_absolute():
                ep = project_root() / ep
            extra_dirs.append(resolve_bio_train_dir(ep.resolve()))

    neg_dirs = None
    if negative_input_paths:
        neg_dirs = []
        for p in negative_input_paths:
            ep = Path(p)
            if not ep.is_absolute():
                ep = project_root() / ep
            neg_dirs.append(ep.resolve())

    success, eval_info = _train_token_cls(
        model_dir,
        train_dir,
        debug=debug,
        fine_tuning_method=fine_tuning_method,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        save_plots=save_plots,
        early_stopping_patience=early_stopping_patience,
        extra_input_dirs=extra_dirs,
        negative_input_dirs=neg_dirs,
        max_per_label=max_per_label,
    )

    if eval_info:
        result["evaluation"] = eval_info

    if success:
        write_train_state(current_sig, str(adapter_dir), model_name)
        invalidate_model_cache(model_name)
        result["training_executed"] = True
        result["message"] = "학습 완료."
    else:
        result["message"] = "학습 실패."

    print(f"[NER Train] [{display}] {result['message']}")
    if debug:
        ner_debug_print(f"[NER debug][train] 종료 result_keys={list(result.keys())} success={success}")
    return result


def _train_token_cls(
    model_dir: Path,
    train_dir: Path,
    *,
    debug: bool = False,
    fine_tuning_method: str = "lora",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    train_ratio: float = 8 / 12,
    val_ratio: float = 2 / 12,
    test_ratio: float = 2 / 12,
    split_seed: int = 42,
    save_plots: bool = False,
    early_stopping_patience: int = 0,
    extra_input_dirs: Optional[List[Path]] = None,
    negative_input_dirs: Optional[List[Path]] = None,
    max_per_label: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    from module.extractor.ner.token_cls import TokenClassNER

    if debug:
        ner_debug_print(
            f"[NER debug][train][token_cls] model_dir={model_dir} train_dir={train_dir} "
            f"method={fine_tuning_method}"
        )
    tc = TokenClassNER(model_dir)
    ok, metrics = tc.train(
        train_dir,
        debug=debug,
        fine_tuning_method=fine_tuning_method,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        save_plots=save_plots,
        early_stopping_patience=early_stopping_patience,
        extra_input_dirs=extra_input_dirs,
        negative_input_dirs=negative_input_dirs,
        max_per_label=max_per_label,
    )
    return ok, metrics
