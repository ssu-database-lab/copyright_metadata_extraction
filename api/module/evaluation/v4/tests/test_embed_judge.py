"""embed_judge.py 테스트 (spec §8 d) — 스텁 임베더 주입, τ 경계, 내용 토큰 확인 on/off, 캐시. 모델은 올리지 않는다."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from api.module.evaluation.v4.embed_judge import (DEFAULT_MODEL_ID, STOPLIST, EmbedJudge, content_tokens, cosine,
                                                  shared_content_tokens, text_sha1)

# 텍스트 → 고정 벡터 (정규화된 텍스트로 조회한다)
_H = math.sqrt(1 - 0.65 ** 2)
VECS: dict[str, list[float]] = {
    "alpha one": [1.0, 0.0, 0.0],
    "alpha two": [0.65, _H, 0.0],          # cos(alpha one, alpha two) ≈ 0.65 (경계)
    "alpha three": [0.0, 1.0, 0.0],        # cos = 0
    "beta four": [1.0, 0.0, 0.0],          # alpha one 과 동일 벡터, 공유 토큰 없음
    "바다 위의 작은 배": [0.6, 0.8, 0.0],
    "바다 건너는 작은 배": [0.6, 0.8, 0.0],
}


def stub_embedder(text: str) -> np.ndarray:
    if text not in VECS:
        raise KeyError(f"스텁에 없는 텍스트: {text!r}")
    return np.array(VECS[text], dtype=np.float32)


def make_judge(**kw) -> EmbedJudge:
    kw.setdefault("embedder", stub_embedder)
    return EmbedJudge(**kw)


# ----------------------------------------------------------------- 내용 토큰
def test_content_tokens_rules():
    assert content_tokens("Sea Side, 바다 위의 배가 보인다.") == ["sea", "side", "바다", "배가", "위의"]
    assert content_tokens("이 사진 영상 이미지 장면 것") == []      # 모두 불용어
    assert content_tokens("A 1 가") == []                           # 1글자 구간 제외
    assert content_tokens("4K 영상의 2024년") == ["2024년", "4k", "영상의"]
    assert content_tokens(None) == []
    assert "있다" in STOPLIST and "모습" in STOPLIST


def test_shared_content_tokens():
    assert shared_content_tokens("바다 위의 작은 배", "바다 건너는 작은 배") == ["바다", "작은"]
    assert shared_content_tokens("alpha one", "beta four") == []


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 0]) == 0.0


# ----------------------------------------------------------------- 판정
def test_match_requires_cosine_and_shared_token():
    j = make_judge(tau=0.65)
    r = j.decide("바다 위의 작은 배", "바다 건너는 작은 배")
    assert r["match"] is True and r["cosine"] == pytest.approx(1.0)
    assert r["shared_tokens"] == ["바다", "작은"] and r["tau"] == 0.65 and r["content_ok"] is True


def test_tau_boundary_is_inclusive():
    j = make_judge(tau=0.65)
    # 판정기가 실제로 계산한 코사인(float32 벡터 기준)을 τ 로 되돌려 넣어 '≥' 경계를 확인한다
    cos = j.decide("alpha one", "alpha two")["cosine"]
    assert cos == pytest.approx(0.65, abs=1e-6)
    j.tau = cos
    assert j.decide("alpha one", "alpha two")["match"] is True        # cosine == τ → 일치
    j.tau = cos + 1e-9
    assert j.decide("alpha one", "alpha two")["match"] is False       # cosine < τ → 불일치
    assert j.decide("alpha one", "alpha three")["match"] is False     # cos 0


def test_content_check_on_off():
    on = make_judge(tau=0.65, content_check=True)
    r = on.decide("alpha one", "beta four")                          # 동일 벡터, 공유 토큰 없음
    assert r["cosine"] == pytest.approx(1.0) and r["shared_tokens"] == []
    assert r["match"] is False and r["content_ok"] is False
    off = make_judge(tau=0.65, content_check=False)
    r = off.decide("alpha one", "beta four")
    assert r["match"] is True and r["content_ok"] is True and r["content_check"] is False


def test_empty_text_never_calls_embedder():
    j = make_judge()
    r = j.decide("", "alpha one")
    assert r["match"] is False and r["cosine"] == 0.0 and j.n_embed_calls == 0
    r = j.decide(None, None)
    assert r["match"] is False


def test_text_is_normalized_before_embedding():
    j = make_judge()
    r = j.decide("  alpha   one ", "alpha\ntwo")        # 공백 정리 후 스텁 조회
    assert r["cosine"] == pytest.approx(0.65, abs=1e-6)


# ----------------------------------------------------------------- 캐시
def test_cache_roundtrip(tmp_path: Path):
    cache = tmp_path / "emb.jsonl"
    j1 = make_judge(cache_path=cache)
    r1 = j1.decide("alpha one", "alpha two")
    assert j1.n_embed_calls == 2
    assert j1.decide("alpha one", "alpha two") == r1 and j1.n_embed_calls == 2   # 메모리 캐시
    lines = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2 and {l["model"] for l in lines} == {"nlpai-lab/KURE-v1"}

    calls: list[str] = []

    def counting(text: str) -> np.ndarray:
        calls.append(text)
        return stub_embedder(text)

    j2 = EmbedJudge(cache_path=cache, embedder=counting)
    assert j2.decide("alpha one", "alpha two") == r1
    assert calls == [] and j2.n_embed_calls == 0                     # 파일 캐시에서 재사용

    other = EmbedJudge(model_id="other/model", cache_path=cache, embedder=counting)
    other.decide("alpha one", "alpha two")
    assert calls == ["alpha one", "alpha two"]                        # 다른 모델의 캐시는 쓰지 않는다


def test_info_and_defaults():
    j = make_judge()
    info = j.info()
    assert info["model_id"] == "nlpai-lab/KURE-v1" and info["tau"] == 0.65
    assert info["content_check"] is True and info["pooling"] == "cls" and info["max_length"] == 512
    assert info["revision"] is None and info["device"] is None
    assert make_judge(device="cuda").info()["device"] is None     # 요청한 장치는 사실이 아니다 — 모델/캐시가 정해야 기록


# ----------------------------------------------------------------- 캐시 provenance (revision/device) — BUG 회귀
def _cache_line(text: str, revision: str | None, device: str | None, model: str = DEFAULT_MODEL_ID) -> str:
    return json.dumps({"sha1": text_sha1(text), "model": model, "revision": revision, "device": device, "vec": VECS[text]})


def _never(text: str) -> np.ndarray:
    raise AssertionError(f"캐시에 있어야 하는 텍스트를 임베딩하려 했다: {text!r}")


def test_warm_cache_run_records_same_revision_and_device(tmp_path: Path):
    """모든 텍스트가 캐시에서 나와 모델을 올리지 않는 재실행도 처음 실행과 같은 info()(revision/device)를 내야 한다 (§5, §7)."""
    cache = tmp_path / "emb.jsonl"
    cold = make_judge(cache_path=cache, revision="rev-A")
    cold.resolved_device = "cuda"                      # _ensure_model 이 모델을 올리며 채우는 값 — 스텁이라 직접 흉내 낸다
    r_cold = cold.decide("alpha one", "alpha two")
    lines = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [(l["revision"], l["device"]) for l in lines] == [("rev-A", "cuda")] * 2   # 줄마다 provenance 기록

    warm = EmbedJudge(cache_path=cache, embedder=_never)   # revision/device 를 주지 않아도 캐시 줄에서 넘겨받는다
    assert warm.decide("alpha one", "alpha two") == r_cold and warm.n_embed_calls == 0
    assert warm.info() == cold.info()
    assert warm.info()["revision"] == "rev-A" and warm.info()["device"] == "cuda"


def test_cache_provenance_adoption_rules(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cache = tmp_path / "emb.jsonl"
    cache.write_text("\n".join([_cache_line("alpha one", "r1", "cpu"), _cache_line("alpha two", "r2", "cpu"),
                                _cache_line("alpha three", "r1", "cpu", model="other/model"), ""]), encoding="utf-8")
    with caplog.at_level("WARNING", logger="api.module.evaluation.v4.embed_judge"):
        unpinned = EmbedJudge(cache_path=cache, embedder=_never)
    assert unpinned.info()["revision"] is None            # r1/r2 가 섞여 있으면 기록하지 않고 경고
    assert unpinned.info()["device"] == "cpu"             # device 는 하나로 모이므로 채택
    assert any("revision" in m and "섞여" in m for m in caplog.messages)
    assert unpinned.decide("alpha one", "alpha two")["cosine"] == pytest.approx(0.65, abs=1e-6)   # 둘 다 캐시에서

    calls: list[str] = []

    def counting(text: str) -> np.ndarray:
        calls.append(text)
        return stub_embedder(text)

    pinned = EmbedJudge(cache_path=cache, embedder=counting, revision="r1")   # r2 줄은 무시 → alpha two 만 다시 임베딩
    pinned.decide("alpha one", "alpha two")
    assert calls == ["alpha two"] and pinned.info()["revision"] == "r1"
    appended = json.loads(cache.read_text(encoding="utf-8").splitlines()[-1])   # 새 줄: 고정 revision + 캐시에서 넘겨받은 device
    assert appended["revision"] == "r1" and appended["device"] == "cpu" and appended["sha1"] == text_sha1("alpha two")

    explicit = EmbedJudge(cache_path=cache, embedder=_never, revision="r9")   # 고정값이 있으면 캐시 값으로 덮지 않는다
    assert explicit.info()["revision"] == "r9"
