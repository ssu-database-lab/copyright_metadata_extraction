"""NER 적재 실패를 실행 로그에서 보이게 — 로그 줄·결과 필드 (docs/v4/M3_console_spec_addendum_logs.md §A2).

적재 논리(기반 모델 선택·load_state_dict)는 바꾸지 않았다 — 실패가 **보이는지만** 본다. 실제 가중치·네트워크 없음(가짜 적재기).
  · 첫 적재 실패: ERROR 한 줄 — 예외 클래스와 첫 줄(+ 가중치 모양 불일치 예), 모델 경로, 한국어 설명.
  · 호출마다 WARNING '모델 미적재 — 정규표현식 대체' + 모델/정규식 엔티티 수. 캐시가 실패를 기억하는 동안에도 그 사유를 적는다.
  · 결과: extract_entities_with_status 의 상태, ner_predict 의 mode · model_loaded · load_error · entity_count.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

_API_DIR = Path(__file__).resolve().parents[3]          # .../api
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

import torch                                            # noqa: E402
from module.ner import ner_system as ns                 # noqa: E402

LOGGER = "module.ner.ner_system"
TEXT = "계약자: 홍길동\n전화번호: 010-1234-5678\n이메일: hong@example.com\n계약금: 5,000,000원"
MISMATCH = ("Error(s) in loading state_dict for BertCrfForNER:\n\tsize mismatch for bert.embeddings.word_embeddings.weight: "
            "copying a param with shape torch.Size([32000, 1024]) from checkpoint, the shape in current model is "
            "torch.Size([119547, 768]).\n\tsize mismatch for bert.pooler.dense.weight: copying a param with shape "
            "torch.Size([1024, 1024]) from checkpoint, the shape in current model is torch.Size([768, 768]).")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("NER_MODEL_CACHE", "NER_MAX_CONCURRENCY", "NER_MODEL_RETRY_SEC"):
        monkeypatch.delenv(var, raising=False)
    ns.clear_ner_model_cache()
    yield
    ns.clear_ner_model_cache()


def _model_dir(tmp_path: Path, config: dict | None = None) -> Path:
    d = tmp_path / "klue-roberta-large"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config if config is not None else {"model_type": "roberta"}),
                                   encoding="utf-8")
    return d


def _failing(model_path, verbose=True):
    raise RuntimeError(MISMATCH)


def _ok(model_path, verbose=True):
    return (object(), object(), {0: "O", 1: "B-NAME", 2: "I-NAME"}, torch.device("cpu"))


def _msgs(caplog, level=None):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and (level is None or r.levelno == level)]


def test_load_failure_is_error_then_warning_per_call(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ns, "load_model_and_tokenizer", _failing)
    d = _model_dir(tmp_path)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        ents, st = ns.extract_entities_with_status(TEXT, "klue/roberta-large", model_path=d)
        ents2, st2 = ns.extract_entities_with_status(TEXT, "klue/roberta-large", model_path=d)   # 캐시가 실패를 기억

    starts = [m for m in _msgs(caplog, logging.INFO) if m.startswith("NER 모델 적재 시작")]
    assert len(starts) == 1 and "klue/roberta-large" in starts[0] and str(d) in starts[0] and "장치 " in starts[0]
    errors = _msgs(caplog, logging.ERROR)
    assert len(errors) == 1                                           # 적재는 한 번만 시도·한 번만 ERROR
    e = errors[0]
    assert e.startswith("NER 모델 적재 실패 — RuntimeError: Error(s) in loading state_dict for BertCrfForNER:")
    assert "크기 불일치 2건" in e and "bert.embeddings.word_embeddings.weight 가중치 [32000, 1024] ≠ 뼈대 [119547, 768]" in e
    assert f"경로 {d}" in e
    assert "모델 폴더 config.json 의 구조와 맞지 않아" in e            # 2026-10-11~: 뼈대는 모델 폴더 config.json 으로 만든다
    warns = [m for m in _msgs(caplog, logging.WARNING) if m.startswith("모델 미적재 — 정규표현식 대체")]
    assert len(warns) == 2                                            # 호출마다
    assert f"(모델 0개 · 정규식 {st['regex_entities']}개 · 최종 {len(ents)}개)" in warns[0]
    assert "사유: RuntimeError: Error(s) in loading state_dict" in warns[0]
    assert "앞선 적재 실패 기억됨(" in warns[1] and "RuntimeError: Error(s) in loading state_dict" in warns[1]

    for s in (st, st2):
        assert s["mode"] == "regex_fallback" and s["model_loaded"] is False and s["model_entities"] == 0
        assert s["load_error"].startswith("RuntimeError: Error(s) in loading state_dict for BertCrfForNER:")
        assert s["entity_count"] == len(ents) and s["model"] == "klue/roberta-large"
    assert ents == ents2 and ("010-1234-5678", "PHONE") in ents
    assert ns.extract_entities_from_text(TEXT, model_path=d) == ents          # 반환 모양은 예전 그대로(목록)


def test_model_mode_when_loader_works(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ns, "load_model_and_tokenizer", _ok)
    monkeypatch.setattr(ns, "extract_entities_by_bio_tagging", lambda *a: {("홍길동", "NAME")})
    d = _model_dir(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ents, st = ns.extract_entities_with_status(TEXT, "klue/roberta-large", model_path=d)
        ns.extract_entities_with_status(TEXT, "klue/roberta-large", model_path=d)
    assert st["mode"] == "model" and st["model_loaded"] is True and st["load_error"] is None
    assert st["model_entities"] == 1 and ("홍길동", "NAME") in ents
    done = [m for m in _msgs(caplog, logging.INFO) if m.startswith("NER 모델 적재 완료")]
    assert len(done) == 1 and "라벨 3개" in done[0]
    assert any(m.startswith("NER 모델 캐시 사용") for m in _msgs(caplog, logging.DEBUG))   # 캐시 적중은 DEBUG
    assert not _msgs(caplog, logging.WARNING) and not _msgs(caplog, logging.ERROR)


def test_cache_off_failure_logs_error_every_call(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("NER_MODEL_CACHE", "0")
    monkeypatch.setattr(ns, "load_model_and_tokenizer", _failing)
    d = _model_dir(tmp_path)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        for _ in range(2):
            _, st = ns.extract_entities_with_status(TEXT, model_path=d)
    assert len(_msgs(caplog, logging.ERROR)) == 2 and st["mode"] == "regex_fallback"
    assert "다시 시도하지 않음" not in _msgs(caplog, logging.ERROR)[0]


def test_missing_model_dir_is_visible(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ns, "load_model_and_tokenizer", _failing)
    import transformers

    def no_download(*a, **k):
        raise OSError("offline (test)")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", no_download)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _, st = ns.extract_entities_with_status(TEXT, "fake/none", model_path=tmp_path / "absent")
    assert st["mode"] == "regex_fallback" and "내려받기 실패" in st["load_error"]
    assert any("NER 모델 내려받기 실패" in m for m in _msgs(caplog, logging.ERROR))
    assert any(m.startswith("모델 미적재 — 정규표현식 대체") for m in _msgs(caplog, logging.WARNING))


def test_ner_predict_result_fields(tmp_path, monkeypatch):
    d = _model_dir(tmp_path)
    monkeypatch.setattr(ns, "get_model_path", lambda name=None: d)
    src = tmp_path / "ocr"
    src.mkdir()
    (src / "a.txt").write_text(TEXT, encoding="utf-8")

    monkeypatch.setattr(ns, "load_model_and_tokenizer", _failing)
    r = ns.ner_predict(str(src), str(tmp_path / "out"), model_name="klue/roberta-large")
    assert r["success"] is True and r["mode"] == "regex_fallback" and r["model_loaded"] is False
    assert r["load_error"].startswith("RuntimeError:") and r["entity_count"] == r["total_entities"] > 0
    assert set(r) >= {"processed_files", "extracted_entities", "statistics", "output_directory"}   # 예전 필드 그대로

    ns.clear_ner_model_cache()
    monkeypatch.setattr(ns, "load_model_and_tokenizer", _ok)
    monkeypatch.setattr(ns, "extract_entities_by_bio_tagging", lambda *a: {("홍길동", "NAME")})
    r = ns.ner_predict(str(src), str(tmp_path / "out2"), model_name="klue/roberta-large")
    assert r["mode"] == "model" and r["model_loaded"] is True and r["load_error"] is None

    r = ns.ner_predict(str(tmp_path / "nowhere"), str(tmp_path / "out3"), model_name="klue/roberta-large")
    assert r["success"] is False and r["mode"] is None and r["model_loaded"] is False and r["entity_count"] == 0


def test_error_brief_and_explanation(tmp_path):
    assert ns._error_brief(RuntimeError(MISMATCH)) == (
        "RuntimeError: Error(s) in loading state_dict for BertCrfForNER: · 크기 불일치 2건 "
        "(예: bert.embeddings.word_embeddings.weight 가중치 [32000, 1024] ≠ 뼈대 [119547, 768])")
    assert ns._error_brief("RuntimeError: x\n\tMissing key(s) in state_dict: \"crf.start\"").endswith(
        "Missing key(s) in state_dict: \"crf.start\"")
    no_name = _model_dir(tmp_path / "a")
    named = _model_dir(tmp_path / "b", {"_name_or_path": "klue/roberta-large"})
    assert "모델 폴더 config.json 의 구조와 맞지 않아" in ns._explain_load_error(RuntimeError(MISMATCH), no_name)
    assert "모델 폴더 config.json 의 구조와 맞지 않아" in ns._explain_load_error(RuntimeError(MISMATCH), named)
    assert "기본 기반 모델" in ns._explain_load_error(RuntimeError(MISMATCH), tmp_path / "no_cfg")
    assert "meta-device" in ns._explain_load_error(RuntimeError("Cannot copy out of meta tensor"), named)
    assert "메모리" in ns._explain_load_error(MemoryError(), named)
    assert ns._explain_load_error(ValueError("?"), named).endswith("정규표현식 추출만 씁니다")


def test_remembered_failure_is_error_once_per_run(caplog):
    """기억된 적재 실패: 시험 실행마다 첫 대체는 ERROR, 같은 실행의 다음 세트는 WARNING, 실행 밖은 WARNING."""
    import logging as _logging
    from module.ner import ner_system as ns
    try:
        from module import log_context as lc
    except ImportError:
        from api.module import log_context as lc
    ns._FALLBACK_RUNS.clear()
    assert ns._first_fallback_in_run() is False                      # 실행 밖
    with lc.bind(run_id="run-A", set_id="1"):
        assert ns._first_fallback_in_run() is True
    with lc.bind(run_id="run-A", set_id="2"):
        assert ns._first_fallback_in_run() is False
    with lc.bind(run_id="run-B", set_id="1"):
        assert ns._first_fallback_in_run() is True


def test_loader_builds_crf_from_model_folder(tmp_path, monkeypatch):
    """2026-10-11 수정: BERT-CRF 뼈대는 모델 폴더(config.json)로 만든다 — _name_or_path 가 없어도 mBERT 로 가지 않는다."""
    import json as _json, sys as _sys, types
    d = tmp_path / "klue-roberta-large"; d.mkdir()
    (d / "config.json").write_text(_json.dumps({"model_type": "roberta"}), encoding="utf-8")
    (d / "label_map.json").write_text(_json.dumps({"id2label": {"0": "O", "1": "B-NAME"}}), encoding="utf-8")
    (d / "model.pt").write_bytes(b"x")
    seen = {}

    class FakeCrf(torch.nn.Module):
        def __init__(self, model_name, num_labels, encoder_config=None):
            super().__init__(); seen.update(model_name=model_name, num_labels=num_labels,
                                            from_config=getattr(encoder_config, "model_type", None))
        def load_state_dict(self, sd, *a, **k):
            seen["loaded"] = True
    fake = types.ModuleType("module.ner.ner_train"); fake.BertCrfForNER = FakeCrf
    monkeypatch.setitem(_sys.modules, "module.ner.ner_train", fake)
    monkeypatch.setattr(ns.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(ns.torch, "load", lambda *a, **k: {})
    ns.load_model_and_tokenizer(d, verbose=False)
    assert seen == {"model_name": str(d), "num_labels": 2, "from_config": "roberta", "loaded": True}


def test_dedup_prefers_model_and_is_order_stable():
    """같은 글자에 라벨이 둘이면 모델 결과가 이기고, 결과가 해시 시드와 무관하다."""
    import subprocess, sys as _sys
    code = ("import sys; sys.path.insert(0, %r); from module.ner import ner_system as ns;"
            "ns.extract_entities_by_regex=lambda t: {('홍길동','NAME'),('홍길동','COMPANY'),('홍길동','DATE'),('서울','ADDRESS'),('서울','COMPANY')};"
            "print(sorted(ns.extract_entities_from_text('홍길동 서울', 'none-model', model_path=__import__('pathlib').Path('/nonexistent'))))") % str(_API_DIR)
    outs = {subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True,
                           env={**__import__('os').environ, "PYTHONHASHSEED": str(seed), "HF_HUB_OFFLINE": "1",
                                "TRANSFORMERS_OFFLINE": "1"}).stdout.strip().splitlines()[-1]
            for seed in (1, 2, 3, 4, 5)}
    assert outs == {"[('서울', 'ADDRESS'), ('홍길동', 'COMPANY')]"}      # 글자 → 라벨 순으로 고정
