"""VLM 체인 — JSON 으로 읽지 못한 응답 (무하유 2026-10-12, 이미지 8건 중 2건). API 호출 없음.

1. 같은 백엔드로 한 번 더 부른 뒤에 폴백한다 (VLM_PARSE_RETRIES, 기본 1)
2. 못 읽은 원문을 결과의 parse_failures 에 남긴다 (파이프라인이 vlm_unparsed.jsonl 로 저장)
3. DashScope 백엔드는 JSON 만 출력하게 강제한다 (VLM_JSON_MODE, 기본 on)
"""
from __future__ import annotations

import os
import sys

_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from module.clip_extraction.vlm import extractor as EX        # noqa: E402
from module.clip_extraction.vlm.client import VLMResult       # noqa: E402


class _Fake:
    def __init__(self, label, replies):
        self.model_label, self.replies, self.calls = label, list(replies), 0

    def _next(self, name):
        self.calls += 1
        text = self.replies.pop(0)
        ok_json = text.startswith("{")
        return VLMResult(model_label=self.model_label, image=name, ok=True, latency_s=1.0, raw_text=text,
                         parsed={"work_type": "사진"} if ok_json else None, parse_ok=ok_json, finish_reason="stop")

    def extract(self, image_path, *a, **k):
        return self._next(image_path.name)

    def extract_frames(self, frame_paths, *a, label="video", **k):
        return self._next(label)


def _chain(*backends):
    x = EX.VLMExtractor.__new__(EX.VLMExtractor)
    x.backends, x._active, x._active_index, x.max_tokens, x.temperature = list(backends), backends[0], 0, 1024, 0.0
    return x


def test_retry_same_model_then_succeed(tmp_path, monkeypatch):
    monkeypatch.delenv("VLM_PARSE_RETRIES", raising=False)
    a, b = _Fake("omni", ["죄송합니다", '{"work_type": "사진"}']), _Fake("plus", ['{"work_type": "사진"}'])
    r = _chain(a, b).extract(tmp_path / "x.png")
    assert r.parse_ok and r.backend_used == "omni" and a.calls == 2 and b.calls == 0
    assert [f["raw"] for f in r.parse_failures] == ["죄송합니다"] and r.parse_failures[0]["backend"] == "omni"


def test_fallback_after_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_PARSE_RETRIES", "1")
    a, b = _Fake("omni", ["깨짐", "또 깨짐"]), _Fake("plus", ['{"work_type": "사진"}'])
    r = _chain(a, b).extract_video([tmp_path / "f1.jpg"], [0.0], label="v.mp4")
    assert r.parse_ok and r.backend_used == "plus" and a.calls == 2 and b.calls == 1
    assert len(r.parse_failures) == 2


def test_all_fail_keeps_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_PARSE_RETRIES", "0")
    r = _chain(_Fake("omni", ["x"]), _Fake("plus", ["y"])).extract(tmp_path / "x.png")
    assert not r.ok and [f["raw"] for f in r.parse_failures] == ["x", "y"]


def test_json_mode_flag(monkeypatch):
    monkeypatch.delenv("VLM_JSON_MODE", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test")
    assert EX.make_omni_backend().json_mode and EX.make_qwen_backend().json_mode and EX.make_omni_backup_backend().json_mode
    assert not EX.make_gemma_backend().json_mode
    monkeypatch.setenv("VLM_JSON_MODE", "off")
    assert not EX.make_omni_backend().json_mode
