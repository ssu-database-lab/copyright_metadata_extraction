"""NER 예측 — 텍스트/파일/디렉터리 입력, 임계값 스윕, 디버그 세션 로깅."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from module.parts import text as text_module
from module.extractor.ner._runtime import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    NER_DEBUG_SESSION_DIR,
    attach_ner_debug_file_logging,
    configure_ner_debug,
    disk_adapter_report,
    load_labels_from_yaml,
    load_ner_defaults,
    make_ner_debug_session_dir,
    model_display_name,
    ner_debug_print,
    _predict_decisions as _base_ner_predict,
    ner_predict_at_thresholds as _base_ner_predict_at_thresholds,
    project_root,
    runtime_adapter_report,
)
from module.parts import directory
from module.parts.paths import resolve_out_dir, resolve_user_path
from module.parts.types import Decision


# ═══════════════════════════════════════════════════════════════════════
# private: 어댑터 상태 로그
# ═══════════════════════════════════════════════════════════════════════


def _log_ner_adapter_disk(model_name: str) -> None:
    """예측 전: 디스크에 어댑터 파일이 있는지."""
    d = disk_adapter_report(model_name)
    has = "있음" if d.get("ready_for_adapter_load") else "없음"
    cfg = d.get("adapter_config_path") or "(없음)"
    ner_debug_print(
        f"[NER 어댑터·디스크] {model_name} | 유형={d['model_type']} | "
        f"어댑터파일={has} | config={cfg}"
    )


def _log_ner_adapter_runtime(model_name: str) -> None:
    """예측 직후: 메모리에 올라간 추론기가 어댑터를 실제 적용했는지."""
    r = runtime_adapter_report(model_name)
    hit = "hit" if r.get("inference_cache_hit") else "miss"
    applied = r.get("adapter_applied_path") or "(없음 — 베이스 가중치만)"
    ner_debug_print(
        f"[NER 어댑터·런타임] {model_name} | 캐시={hit} | 적용경로={applied}"
    )


# ═══════════════════════════════════════════════════════════════════════
# private: 임계값 처리
# ═══════════════════════════════════════════════════════════════════════


def _threshold_folder_name(th: Optional[float]) -> str:
    """출력 폴더명: None → ``thr_default``, 0.55 → ``thr_0_55``."""
    if th is None:
        return "thr_default"
    s = f"{float(th):g}".replace(".", "_").replace("-", "neg")
    return f"thr_{s}"


def _normalize_predict_thresholds(
    threshold: Optional[float],
    thresholds: Optional[Sequence[float]],
) -> Tuple[List[Optional[float]], bool]:
    """(임계값 목록, 스윕 여부). ``thresholds``가 주어지면 스윕 모드."""
    if thresholds is not None:
        lst = sorted({float(t) for t in thresholds})
        if not lst:
            raise ValueError("thresholds가 비어 있습니다.")
        return cast(List[Optional[float]], lst), True
    return [threshold], False


def _predict_threshold_key(th: Optional[float]) -> str:
    """반환 dict 키용."""
    return "default" if th is None else str(th)


# ═══════════════════════════════════════════════════════════════════════
# private: decision 집계
# ═══════════════════════════════════════════════════════════════════════


def _aggregate_decisions(
    decisions: List[Decision],
    labels: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Decision 목록을 라벨별 값 리스트로 집계."""
    if labels is None:
        labels = sorted({d.label for d in decisions})
    aggregated: Dict[str, List[str]] = {label: [] for label in labels}
    for decision in decisions:
        label = decision.label
        value = str(decision.value) if not isinstance(decision.value, str) else decision.value
        if label in aggregated and value and value.strip() and value not in aggregated[label]:
            aggregated[label].append(value)
    for label in aggregated:
        if not aggregated[label]:
            aggregated[label] = ["N/A"]
    return aggregated


# ═══════════════════════════════════════════════════════════════════════
# private: 단일 텍스트 / 파일 예측
# ═══════════════════════════════════════════════════════════════════════


def _build_full_metadata(text: str, ner_aggregated: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """NER aggregation + regex(9) + post-process + LLM placeholder(9) → 35-라벨 dict."""
    from module.extractor.regex import regex_extract
    from module.extractor.ner.postprocess import postprocess_metadata
    from module.parts.labels import LLM_DELEGATED_LABEL_SET, NER_LABEL_SET

    out: Dict[str, List[str]] = dict(regex_extract(text))
    for label in NER_LABEL_SET:
        out[label] = ner_aggregated.get(label, ["N/A"])
    out = postprocess_metadata(out, text)
    for label in LLM_DELEGATED_LABEL_SET:
        out.setdefault(label, ["N/A"])
    return out


def _predict_one_text(
    text: str,
    model_name: str,
    threshold: Optional[float],
    *,
    model_path: Optional[str] = None,
    debug: bool = False,
) -> Tuple[Dict[str, List[str]], int, int, int]:
    """텍스트 입력 → (35-라벨 메타데이터, 문장 수, 토큰 수, decision 수)."""
    struct = text_module.read_text(text)
    sentences = struct.get("sentences", [])
    tokens = struct.get("tokens", [])
    decisions = _base_ner_predict(
        sentences=sentences,
        tokens=tokens,
        threshold=threshold,
        model=model_name,
        model_path=model_path,
        debug=debug,
    )
    from module.parts.labels import NER_LABEL_SET
    ner_aggregated = _aggregate_decisions(decisions, sorted(NER_LABEL_SET))
    aggregated = _build_full_metadata(text, ner_aggregated)
    return aggregated, len(sentences), len(tokens), len(decisions)


def _predict_one_file(
    file_path: Path,
    model_name: str,
    threshold: Optional[float],
    *,
    model_path: Optional[str] = None,
    debug: bool = False,
) -> Tuple[Dict[str, List[str]], int, int, int]:
    """단일 파일 → (35-라벨 메타데이터, 문장 수, 토큰 수, decision 수)."""
    if debug:
        ner_debug_print(f"[NER debug][predict] 입력 파일={file_path} (exists={file_path.exists()})")

    if file_path.suffix.lower() in (".txt", ".md"):
        text = file_path.read_text(encoding="utf-8")
    else:
        from module.extractor import ocr as ocr_module
        text, _ = ocr_module.process_file_for_metadata(file_path)

    struct = text_module.read_text(text)
    sentences = struct.get("sentences", [])
    tokens = struct.get("tokens", [])
    if debug:
        ner_debug_print(
            f"[NER debug][predict] 전처리 후 sentences={len(sentences)} tokens={len(tokens)} "
            f"text_chars={len(text)}"
        )

    decisions = _base_ner_predict(
        sentences=sentences,
        tokens=tokens,
        threshold=threshold,
        model=model_name,
        model_path=model_path,
        debug=debug,
    )
    from module.parts.labels import NER_LABEL_SET
    ner_aggregated = _aggregate_decisions(decisions, sorted(NER_LABEL_SET))
    aggregated = _build_full_metadata(text, ner_aggregated)
    return aggregated, len(sentences), len(tokens), len(decisions)


# ═══════════════════════════════════════════════════════════════════════
# public: 텍스트 × 임계값 스윕 (main.py에서 사용)
# ═══════════════════════════════════════════════════════════════════════


def predict_at_thresholds(
    text: str,
    model_name: str,
    thresholds: List[float],
    *,
    model_path: Optional[str] = None,
) -> Dict[float, Dict[str, Any]]:
    """텍스트 → {threshold → 집계 결과} — inference 1회만 실행."""
    struct = text_module.read_text(text)
    sentences = struct.get("sentences", [])
    tokens = struct.get("tokens", [])

    decisions_by_thr = _base_ner_predict_at_thresholds(
        sentences=sentences,
        tokens=tokens,
        thresholds=thresholds,
        model=model_name,
        model_path=model_path,
    )

    from module.parts.labels import NER_LABEL_SET
    ner_labels = sorted(NER_LABEL_SET)
    return {
        thr: _aggregate_decisions(decisions, ner_labels)
        for thr, decisions in decisions_by_thr.items()
    }


# ═══════════════════════════════════════════════════════════════════════
# public: ner_predict 오케스트레이터 (api.py에서 위임)
# ═══════════════════════════════════════════════════════════════════════


def ner_predict(
    *,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    input_path: Optional[str] = None,
    input_text: Optional[str] = None,
    output_path: Optional[str] = None,
    threshold: Optional[float] = None,
    thresholds: Optional[Sequence[float]] = None,
    result_phase: Optional[str] = None,
    log_adapter_status: bool = True,
    debug: bool = False,
    debug_path: Optional[str] = None,
) -> Dict[str, Any]:
    """NER 예측 + regex + post-process + LLM placeholder → 35-라벨 JSON 저장.

    인자 미지정 시 ``configs/labels.yaml`` 의 ``ner`` / ``paths`` 섹션에서 로드.
    - ``input_text``가 주어지면 파일 I/O 건너뛰고 텍스트 집계 반환.
    - ``thresholds``가 주어지면 임계값 스윕 모드.
    """
    # ---- 기본값 로드 (인자 미지정 시 labels.yaml) ----
    _defaults = load_ner_defaults()
    _paths = _defaults.get("_paths", {})
    if model_name is None:
        model_name = _defaults.get("model_name", DEFAULT_MODEL)
    if input_path is None and input_text is None:
        input_path = _paths.get("ner_input", "data/in")
    if output_path is None:
        output_path = _paths.get("metadata_output")
    if threshold is None and thresholds is None:
        threshold = float(_defaults.get("threshold", DEFAULT_THRESHOLD))
    if result_phase is None:
        result_phase = _defaults.get("result_phase")

    ctx_token = None
    session_dir: Optional[Path] = None
    thr_list, sweep_mode = _normalize_predict_thresholds(threshold, thresholds)
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    display = model_display_name(model_name)
    auto_predict_base: Optional[Path] = None
    if output_path is None:
        auto_predict_base = project_root() / "data" / "out" / "results" / display / "predict" / stamp

    debug_threshold_dir = "thr_sweep" if sweep_mode else _threshold_folder_name(
        thr_list[0] if thr_list else None
    )

    if debug:
        session_dir = make_ner_debug_session_dir(
            debug_path,
            model_name,
            debug_kind="predict",
            threshold_dir=debug_threshold_dir,
        )
        ctx_token = NER_DEBUG_SESSION_DIR.set(session_dir)
        configure_ner_debug(True)
        attach_ner_debug_file_logging(session_dir)
        meta = {
            "kind": "ner_predict",
            "model_name": model_name,
            "input_path": input_path,
            "output_path": output_path,
            "results_auto_base": str(auto_predict_base) if auto_predict_base is not None else None,
            "threshold": threshold,
            "thresholds": list(thr_list) if sweep_mode else None,
            "threshold_sweep": sweep_mode,
            "result_phase": result_phase,
            "log_adapter_status": log_adapter_status,
            "debug_path": debug_path,
            "threshold_dir": debug_threshold_dir,
            "session_dir": str(session_dir),
        }
        (session_dir / "session_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ner_debug_print(f"[NER debug][predict] 세션 디렉터리={session_dir}")

    try:
        if debug:
            ner_debug_print(
                f"[NER debug][predict] API 인자 model_name={model_name!r} input_path={input_path!r} "
                f"output_path={output_path!r} threshold={threshold!r} thresholds={list(thr_list)!r} "
                f"sweep={sweep_mode} result_phase={result_phase!r}"
            )

        # ── input_text 직접 입력 fast-path ──────────────────────────────
        if input_text is not None:
            if log_adapter_status:
                _log_ner_adapter_disk(model_name)
            by_threshold_text: Dict[str, Any] = {}
            for thr in thr_list:
                thr_key = _predict_threshold_key(thr)
                aggregated, ns, nt, nd = _predict_one_text(
                    input_text, model_name, thr,
                    model_path=model_path,
                    debug=debug,
                )
                if log_adapter_status and thr == thr_list[0]:
                    _log_ner_adapter_runtime(model_name)
                by_threshold_text[thr_key] = {"threshold": thr, "aggregated": aggregated}
            if not sweep_mode:
                return by_threshold_text[_predict_threshold_key(thr_list[0])]["aggregated"]
            return {
                "threshold_sweep": True,
                "thresholds": list(thr_list),
                "by_threshold": by_threshold_text,
            }

        root = resolve_user_path(input_path)
        if not root.exists():
            raise ValueError(f"input_path가 없습니다: {root}")

        effective_out_resolved: Optional[str] = None
        if output_path is not None:
            effective_out_resolved = resolve_out_dir(
                str(resolve_user_path(output_path)),
                display,
                result_phase,
            )

        phase_note = f", phase={result_phase}" if result_phase else ""

        if debug:
            ner_debug_print(
                f"[NER debug][predict] root={root} "
                f"effective_out_dir={effective_out_resolved or auto_predict_base} "
                f"display={display}"
            )

        if log_adapter_status:
            _log_ner_adapter_disk(model_name)

        adapter_runtime_logged = False
        by_threshold: Dict[str, Any] = {}

        for thr in thr_list:
            thr_key = _predict_threshold_key(thr)
            if output_path is None:
                if auto_predict_base is None:
                    raise RuntimeError("auto_predict_base가 None입니다. 출력 경로를 확인하세요.")
                od_segment = str(auto_predict_base / _threshold_folder_name(thr))
            else:
                if effective_out_resolved is None:
                    raise RuntimeError("effective_out_resolved가 None입니다. 출력 경로를 확인하세요.")
                od_segment = (
                    str(Path(effective_out_resolved) / _threshold_folder_name(thr))
                    if sweep_mode
                    else effective_out_resolved
                )
            print(
                f"NER [{display}{phase_note}] threshold={thr!r} → 출력 {od_segment}"
            )

            if root.is_dir():
                text_exts = ["txt", "md"]
                text_files = set(directory.iter_files_by_ext(root, text_exts))
                doc_files  = set(directory.iter_document_files(root))
                all_files  = sorted(text_files | doc_files, key=str)
                if not all_files:
                    raise ValueError(f"처리할 문서가 없습니다: {root}")

                results: Dict[str, Dict[str, List[str]]] = {}
                out_dir_path = directory.ensure_outdir(od_segment)
                total = len(all_files)
                print(
                    f"NER [{display}{phase_note}]: 디렉터리={root}, 파일={total}개 → {od_segment}"
                )
                for idx, fpath in enumerate(all_files, start=1):
                    rel = str(fpath.relative_to(root))
                    print(f"[{idx}/{total}] {rel}")
                    aggregated, ns, nt, nd = _predict_one_file(
                        fpath, model_name, thr,
                        model_path=model_path,
                        debug=debug,
                    )
                    if log_adapter_status and not adapter_runtime_logged:
                        _log_ner_adapter_runtime(model_name)
                        adapter_runtime_logged = True
                    print(
                        f"    sentences={ns}, tokens={nt}, decisions={nd}"
                    )
                    rel_no_suffix = fpath.relative_to(root).with_suffix("")
                    out_file = out_dir_path / rel_no_suffix.parent / f"{rel_no_suffix.name}_metadata.json"
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(aggregated, f, ensure_ascii=False, indent=2)
                    print(f"    저장: {out_file}")
                    results[rel] = aggregated

                if debug:
                    ner_debug_print(
                        f"[NER debug][predict] threshold={thr!r} 디렉터리 처리 완료 "
                        f"files={len(results)} out_dir={out_dir_path}"
                    )

                by_threshold[thr_key] = {
                    "threshold": thr,
                    "directory": str(root),
                    "output_dir": str(out_dir_path),
                    "results": results,
                }
            else:
                aggregated, ns, nt, nd = _predict_one_file(
                    root, model_name, thr,
                    model_path=model_path,
                    debug=debug,
                )
                if log_adapter_status and not adapter_runtime_logged:
                    _log_ner_adapter_runtime(model_name)
                    adapter_runtime_logged = True
                print(
                    f"NER [{display}{phase_note}]: threshold={thr!r} "
                    f"sentences={ns}, tokens={nt}, decisions={nd}"
                )

                out_dir_path = directory.ensure_outdir(od_segment)
                out_file = directory.default_outfile(
                    file_path=str(root),
                    out_dir=out_dir_path,
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(aggregated, f, ensure_ascii=False, indent=2)
                print(f"NER 결과 저장: {out_file}")

                if debug:
                    non_na = sum(1 for v in aggregated.values() if v != ["N/A"])
                    ner_debug_print(
                        f"[NER debug][predict] threshold={thr!r} 집계 완료 labels={len(aggregated)} "
                        f"non_N/A_labels={non_na} out_file={out_file}"
                    )
                    if session_dir is not None:
                        ner_debug_print(
                            f"[NER debug][predict] debug_session_dir={session_dir}"
                        )

                by_threshold[thr_key] = {
                    "threshold": thr,
                    "aggregated": aggregated,
                    "out_file": str(out_file),
                }

        if not sweep_mode:
            sole = by_threshold[_predict_threshold_key(thr_list[0])]
            if root.is_dir():
                out = {
                    "directory": sole["directory"],
                    "output_dir": sole["output_dir"],
                    "results": sole["results"],
                }
                if debug and session_dir is not None:
                    out["debug_session_dir"] = str(session_dir)
                return out
            return sole["aggregated"]

        out_sweep: Dict[str, Any] = {
            "threshold_sweep": True,
            "thresholds": list(thr_list),
            "by_threshold": by_threshold,
        }
        if debug and session_dir is not None:
            out_sweep["debug_session_dir"] = str(session_dir)
        return out_sweep
    finally:
        if debug and ctx_token is not None:
            configure_ner_debug(False)
            NER_DEBUG_SESSION_DIR.reset(ctx_token)
