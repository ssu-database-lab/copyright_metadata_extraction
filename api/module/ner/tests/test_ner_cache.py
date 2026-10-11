"""NER 모델 프로세스 캐시 단위 테스트 — 실제 가중치·네트워크 없음 (가짜 적재기/토크나이저/모델).

다루는 것
  · 같은 (경로, 디바이스) 는 같은 객체를 돌려주고 한 번만 적재한다. 경로가 다르면 따로 적재한다.
  · 동시에 들어온 첫 호출들(스레드 + barrier, 느린 가짜 적재기)도 한 번만 적재한다.
    서로 다른 모델의 적재도 겹치지 않는다 (transformers meta-device 경쟁).
  · NER_MODEL_CACHE=0 이면 예전처럼 호출마다 적재하고 캐시에 아무것도 남기지 않는다.
  · clear_ner_model_cache() 뒤에는 다시 적재한다. 비우는 도중에 끝난 적재는 캐시를 되살리지 않는다.
  · 적재 실패는 NER_MODEL_RETRY_SEC 동안 기억되고(적재기를 다시 부르지 않음), 결과는 캐시 끔과 같다(정규식만).
  · 토큰화는 직렬화되고, forward 동시 수는 NER_MAX_CONCURRENCY 로 묶인다 (0 = 제한 없음).

앱과 같은 이름(`module.ner.ner_system`)으로 import 한다 — 다른 이름으로 올리면 모듈 사본과 캐시가 따로 생긴다.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_API_DIR = Path(__file__).resolve().parents[3]          # .../api
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

import torch                                            # noqa: E402
from module.ner import ner_system as ns                 # noqa: E402

N_THREADS = 8


# ---------------------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    for var in ("NER_MODEL_CACHE", "NER_MAX_CONCURRENCY", "NER_MODEL_RETRY_SEC"):
        monkeypatch.delenv(var, raising=False)
    ns.clear_ner_model_cache()
    yield
    ns.clear_ner_model_cache()


def _model_dir(tmp_path: Path, name: str = "fake-model") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")   # extract_entities_from_text 의 존재 확인용
    return d


class FakeLoader:
    """load_model_and_tokenizer 대역 — 호출 수·동시 적재 수를 센다."""

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, model_path, verbose=True):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("Error(s) in loading state_dict for BertCrfForNER (fake)")
            return (object(), object(), {0: "O"}, torch.device("cpu"))
        finally:
            with self._lock:
                self.active -= 1


def _run_threads(n, fn):
    """barrier 로 n 개 스레드를 한꺼번에 출발시키고 (결과, 예외) 목록을 돌려준다."""
    barrier = threading.Barrier(n)
    results, errors = [None] * n, [None] * n

    def worker(i):
        barrier.wait()
        try:
            results[i] = fn(i)
        except BaseException as exc:          # noqa: BLE001 — 테스트에서 전부 모은다
            errors[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "스레드가 끝나지 않음 (교착?)"
    return results, errors


# ---------------------------------------------------------------------------
# 캐시 적재
# ---------------------------------------------------------------------------

def test_cache_returns_same_object_and_loads_once(tmp_path, monkeypatch):
    loader = FakeLoader()
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    d = _model_dir(tmp_path)

    first = ns.get_cached_model_and_tokenizer(d)
    second = ns.get_cached_model_and_tokenizer(d)
    via_other_spelling = ns.get_cached_model_and_tokenizer(d / ".." / d.name)   # 해석된 경로가 키

    assert first is second is via_other_spelling
    assert loader.calls == 1
    info = ns.ner_model_cache_info()
    assert len(info) == 1 and info[0]["status"] == "loaded"
    assert info[0]["model_path"] == str(d.resolve())
    assert info[0]["device"] == str(ns._default_device())


def test_different_models_are_cached_separately(tmp_path, monkeypatch):
    loader = FakeLoader()
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    a = ns.get_cached_model_and_tokenizer(_model_dir(tmp_path, "a"))
    b = ns.get_cached_model_and_tokenizer(_model_dir(tmp_path, "b"))
    assert a is not b
    assert loader.calls == 2
    assert ns.get_cached_model_and_tokenizer(_model_dir(tmp_path, "a")) is a
    assert loader.calls == 2


def test_concurrent_first_calls_load_once(tmp_path, monkeypatch):
    loader = FakeLoader(delay=0.5)                 # 느린 적재 — 다른 스레드가 반드시 기다리게 된다
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    d = _model_dir(tmp_path)

    results, errors = _run_threads(N_THREADS, lambda i: ns.get_cached_model_and_tokenizer(d))

    assert errors == [None] * N_THREADS
    assert loader.calls == 1
    assert all(r is results[0] for r in results)


def test_concurrent_first_calls_through_extract_entities_load_once(tmp_path, monkeypatch):
    """파이프라인이 실제로 타는 경로(extract_entities_from_text)로도 한 번만 적재한다."""
    loader = FakeLoader(delay=0.5)
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    seen_models = []
    seen_lock = threading.Lock()

    def fake_bio(text, tokenizer, model, id2label, device):
        with seen_lock:
            seen_models.append(model)
        return {("홍길동", "NAME")}

    monkeypatch.setattr(ns, "extract_entities_by_bio_tagging", fake_bio)
    d = _model_dir(tmp_path)

    results, errors = _run_threads(
        N_THREADS, lambda i: ns.extract_entities_from_text("계약자: 홍길동", model_path=d))

    assert errors == [None] * N_THREADS
    assert loader.calls == 1
    assert len(seen_models) == N_THREADS and all(m is seen_models[0] for m in seen_models)
    assert all(r == results[0] for r in results)
    assert ("홍길동", "NAME") in results[0]


def test_loads_of_different_models_never_overlap(tmp_path, monkeypatch):
    """transformers 의 from_pretrained 는 동시에 돌면 meta tensor 오류로 둘 다 실패한다 —
    다른 모델끼리도 적재는 한 번에 하나."""
    loader = FakeLoader(delay=0.3)
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    dirs = [_model_dir(tmp_path, f"m{i}") for i in range(3)]

    results, errors = _run_threads(6, lambda i: ns.get_cached_model_and_tokenizer(dirs[i % 3]))

    assert errors == [None] * 6
    assert loader.calls == 3                       # 모델마다 한 번
    assert loader.max_active == 1                  # 겹친 적재 없음
    for i in range(3):
        assert results[i] is results[i + 3]


# ---------------------------------------------------------------------------
# opt-out · clear
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_opt_out_loads_on_every_call(tmp_path, monkeypatch, value):
    monkeypatch.setenv("NER_MODEL_CACHE", value)
    loader = FakeLoader()
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    models = []
    monkeypatch.setattr(ns, "extract_entities_by_bio_tagging",
                        lambda text, tok, model, id2label, device: models.append(model) or set())
    d = _model_dir(tmp_path)

    for _ in range(3):
        ns.extract_entities_from_text("계약자: 홍길동", model_path=d)

    assert loader.calls == 3
    assert len({id(m) for m in models}) == 3        # 호출마다 새 모델
    assert ns.ner_model_cache_info() == []


def test_cache_on_by_default_through_extract_entities(tmp_path, monkeypatch):
    loader = FakeLoader()
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    monkeypatch.setattr(ns, "extract_entities_by_bio_tagging", lambda *a: set())
    d = _model_dir(tmp_path)
    for _ in range(3):
        ns.extract_entities_from_text("계약자: 홍길동", model_path=d)
    assert loader.calls == 1


def test_clear_forces_reload(tmp_path, monkeypatch):
    loader = FakeLoader()
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    d = _model_dir(tmp_path)
    first = ns.get_cached_model_and_tokenizer(d)

    assert ns.clear_ner_model_cache() == 1
    assert ns.ner_model_cache_info() == []
    second = ns.get_cached_model_and_tokenizer(d)

    assert second is not first
    assert loader.calls == 2


def test_clear_during_load_does_not_resurrect_entry(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def blocking_loader(model_path, verbose=True):
        started.set()
        release.wait(5)
        return (object(), object(), {0: "O"}, torch.device("cpu"))

    monkeypatch.setattr(ns, "load_model_and_tokenizer", blocking_loader)
    d = _model_dir(tmp_path)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("bundle", ns.get_cached_model_and_tokenizer(d)))
    t.start()
    assert started.wait(5)
    ns.clear_ner_model_cache()
    release.set()
    t.join(5)

    assert out["bundle"] is not None               # 진행 중이던 호출은 자기 결과를 받는다
    assert ns.ner_model_cache_info() == []         # 하지만 비운 캐시에 들어가지는 않는다


# ---------------------------------------------------------------------------
# 적재 실패
# ---------------------------------------------------------------------------

def test_failed_load_is_remembered(tmp_path, monkeypatch):
    loader = FakeLoader(fail=True)
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    d = _model_dir(tmp_path)

    with pytest.raises(RuntimeError, match="fake") as first:
        ns.get_cached_model_and_tokenizer(d)
    assert not isinstance(first.value, ns.NERModelLoadError)     # 첫 실패는 원래 예외 그대로
    with pytest.raises(ns.NERModelLoadError) as again:
        ns.get_cached_model_and_tokenizer(d)
    assert isinstance(again.value.__cause__, RuntimeError)
    assert loader.calls == 1
    assert ns.ner_model_cache_info()[0]["status"] == "failed"


def test_remembered_failure_does_not_pin_loader_frames(tmp_path, monkeypatch):
    """기억된 실패가 적재 프레임(반쯤 만든 모델·state_dict)을 붙잡으면 재시도 창 내내 GB 가 묶인다."""
    import gc
    import weakref

    class Big:                                     # 반쯤 만든 모델 대역
        pass

    refs = []

    def failing_loader(model_path, verbose=True):
        half_built = Big()
        refs.append(weakref.ref(half_built))
        raise RuntimeError("size mismatch (fake)")

    monkeypatch.setattr(ns, "load_model_and_tokenizer", failing_loader)
    d = _model_dir(tmp_path)
    try:
        ns.get_cached_model_and_tokenizer(d)
    except RuntimeError:
        pass
    gc.collect()

    assert refs and refs[0]() is None
    assert ns.ner_model_cache_info()[0]["status"] == "failed"
    assert "size mismatch" in ns.ner_model_cache_info()[0]["error"]


def test_failed_load_retried_after_retry_window(tmp_path, monkeypatch):
    monkeypatch.setenv("NER_MODEL_RETRY_SEC", "0")
    loader = FakeLoader(fail=True)
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    d = _model_dir(tmp_path)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            ns.get_cached_model_and_tokenizer(d)
    assert loader.calls == 3

    loader.fail = False                            # 일시적 실패가 풀리면 다음 호출에서 적재된다
    bundle = ns.get_cached_model_and_tokenizer(d)
    assert ns.get_cached_model_and_tokenizer(d) is bundle
    assert loader.calls == 4


def test_concurrent_first_calls_with_failing_loader_load_once(tmp_path, monkeypatch):
    loader = FakeLoader(delay=0.3, fail=True)
    monkeypatch.setattr(ns, "load_model_and_tokenizer", loader)
    d = _model_dir(tmp_path)
    _, errors = _run_threads(N_THREADS, lambda i: ns.get_cached_model_and_tokenizer(d))
    assert loader.calls == 1
    assert all(isinstance(e, RuntimeError) for e in errors)


def test_failed_load_output_matches_cache_off(tmp_path, monkeypatch):
    """적재가 실패하면 예전처럼 정규식 결과만 — 캐시를 켜든 끄든 같은 결과."""
    monkeypatch.setattr(ns, "load_model_and_tokenizer", FakeLoader(fail=True))
    d = _model_dir(tmp_path)
    text = "계약자: 홍길동\n전화번호: 010-1234-5678\n이메일: hong@example.com\n계약금: 5,000,000원"

    monkeypatch.setenv("NER_MODEL_CACHE", "0")
    off = ns.extract_entities_from_text(text, model_path=d)
    monkeypatch.setenv("NER_MODEL_CACHE", "1")
    on_first = ns.extract_entities_from_text(text, model_path=d)
    on_cached = ns.extract_entities_from_text(text, model_path=d)

    assert off == on_first == on_cached
    assert ("010-1234-5678", "PHONE") in off and ("홍길동", "NAME") in off


# ---------------------------------------------------------------------------
# 토크나이저 잠금 · forward 세마포어
# ---------------------------------------------------------------------------

class _ConcurrencyProbe:
    def __init__(self, delay: float):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._lock = threading.Lock()

    def enter(self):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1


class FakeTokenizer:
    def __init__(self, probe: _ConcurrencyProbe):
        self.probe = probe

    def __call__(self, sentence, **kwargs):
        self.probe.enter()
        n = 4
        return {
            "input_ids": torch.zeros((1, n), dtype=torch.long),
            "attention_mask": torch.ones((1, n), dtype=torch.long),
            "offset_mapping": torch.tensor([[[0, 0], [0, 1], [1, 2], [0, 0]]]),
        }


class FakeCrfModel:
    """BertCrfForNER 추론 출력 형식({'predictions'}) 을 흉내 낸다 — 전부 'O'."""

    def __init__(self, probe: _ConcurrencyProbe):
        self.probe = probe

    def __call__(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()          # no_grad 안에서 불려야 한다
        self.probe.enter()
        return {"predictions": torch.zeros_like(input_ids)}


def _bio_from_threads(n, tokenizer, model):
    return _run_threads(n, lambda i: ns.extract_entities_by_bio_tagging(
        "계약자 홍길동 서명", tokenizer, model, {0: "O"}, torch.device("cpu")))


def test_tokenizer_calls_are_serialized(monkeypatch):
    monkeypatch.setenv("NER_MAX_CONCURRENCY", "0")
    tok_probe, model_probe = _ConcurrencyProbe(0.05), _ConcurrencyProbe(0.0)
    results, errors = _bio_from_threads(6, FakeTokenizer(tok_probe), FakeCrfModel(model_probe))
    assert errors == [None] * 6
    assert tok_probe.calls == 6
    assert tok_probe.max_active == 1


@pytest.mark.parametrize("limit", [1, 2])
def test_forward_concurrency_is_bounded(monkeypatch, limit):
    monkeypatch.setenv("NER_MAX_CONCURRENCY", str(limit))
    model_probe = _ConcurrencyProbe(0.15)
    results, errors = _bio_from_threads(6, FakeTokenizer(_ConcurrencyProbe(0.0)), FakeCrfModel(model_probe))
    assert errors == [None] * 6
    assert model_probe.calls == 6
    assert model_probe.max_active == limit
    assert results == [set()] * 6


def test_forward_unbounded_when_zero(monkeypatch):
    monkeypatch.setenv("NER_MAX_CONCURRENCY", "0")
    model_probe = _ConcurrencyProbe(0.3)
    _, errors = _bio_from_threads(6, FakeTokenizer(_ConcurrencyProbe(0.0)), FakeCrfModel(model_probe))
    assert errors == [None] * 6
    assert model_probe.max_active >= 3              # 묶지 않으면 겹친다 (스케줄링 여유를 둔 하한)


def test_max_concurrency_env_parsing(monkeypatch):
    default = ns._default_max_concurrency()
    assert default >= 1
    monkeypatch.setenv("NER_MAX_CONCURRENCY", "3")
    assert ns._max_concurrency() == 3
    monkeypatch.setenv("NER_MAX_CONCURRENCY", "0")
    assert ns._max_concurrency() == 0 and ns._forward_semaphore() is None
    monkeypatch.setenv("NER_MAX_CONCURRENCY", "-2")
    assert ns._max_concurrency() == default
    monkeypatch.setenv("NER_MAX_CONCURRENCY", "lots")
    assert ns._max_concurrency() == default
    monkeypatch.setenv("NER_MAX_CONCURRENCY", " ")
    assert ns._max_concurrency() == default
