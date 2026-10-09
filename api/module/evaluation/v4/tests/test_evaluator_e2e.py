"""evaluator 종단 테스트 (spec §8 c) — 합성 6세트 × 23키, 손으로 계산한 TP/FP/FN 과 Macro/Micro 를 소수 3자리까지 확인.

T-002 (키 17~23): 디렉터리 예측, 스텁 설명 판정기, empty/na 상태, 예측 누락 1세트, gold 없는 예측 1건.
T-001 (키 1~16): jsonl 예측, 8가지 권리 표기, 날짜 5형식, 관리번호 접두, other_rights 비null 경고, 평가 외 키 경고.
"""
from __future__ import annotations

import json
import re
from fractions import Fraction as Fr
from pathlib import Path
from typing import Any

import pytest

from api.module.evaluation.v4.embed_judge import shared_content_tokens
from api.module.evaluation.v4.evaluate import evaluate, main
from api.module.evaluation.v4.keys23 import KEY_NAMES, T001_KEYS, T002_KEYS, empty_prediction


# ----------------------------------------------------------------- 도우미
def gv(value: Any, kind: str | None = None) -> dict[str, Any]:
    return {"value": value, "state": "value", "source": "synthetic", "tier": "A", "kind": kind}


def ge(kind: str | None = None) -> dict[str, Any]:
    return {"value": None, "state": "empty", "source": "synthetic", "tier": "A", "kind": kind}


def gna(kind: str | None = None) -> dict[str, Any]:
    return {"value": None, "state": "na", "source": "synthetic", "tier": "-", "kind": kind}


def gold_rec(set_id: str, test: str, media: str, bucket: str, keys: dict[str, Any]) -> dict[str, Any]:
    return {"set_id": set_id, "test": test, "media": media, "bucket": bucket, "source": "공유마당", "keys": keys}


def pred(**kw: Any) -> dict[str, Any]:
    p = empty_prediction()
    p.update(kw)
    return p


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def f1(tp: int, fp: int, fn: int) -> Fr:
    return Fr(2 * tp, 2 * tp + fp + fn)


class StubJudge:
    """텍스트가 같으면 일치(코사인 0.9), 다르면 불일치(0.1). 공유 토큰은 실제 규칙으로 계산."""

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, p: str, g: str) -> dict[str, Any]:
        self.calls += 1
        same = p.strip() == g.strip()
        return {"match": same, "cosine": 0.9 if same else 0.1, "shared_tokens": shared_content_tokens(p, g),
                "tau": 0.65, "content_check": True, "content_ok": True}

    def info(self) -> dict[str, Any]:
        return {"model_id": "stub", "revision": None, "device": "cpu", "tau": 0.65, "content_check": True,
                "max_length": 512, "pooling": "cls", "cache_path": None}


# =================================================================== T-002
HASH = {c: c * 64 for c in "acdef"}

T002_GOLD = [
    gold_rec("S1", "T-002", "image", "ccl", {
        "keywords": gv(["바다", "해변", "파도"]), "hashtags": gv(["#바다", "#해변"], "site"),
        "file_format": gv("JPG"), "file_size": gv(123456), "resolution": gv("1920x1080"),
        "description": gv("해변에 파도가 밀려온다", "caption"), "file_hash": gv(HASH["a"])}),
    gold_rec("S2", "T-002", "image", "ccl", {
        "keywords": gv(["서울", "야경", "한강"]), "hashtags": gna(),
        "file_format": gv("PNG"), "file_size": gv(1000), "resolution": gv("800x600"),
        "description": gv("서울 야경", "generated"), "file_hash": gv(HASH["c"])}),
    gold_rec("S3", "T-002", "video", "kogl", {
        "keywords": ge(), "hashtags": ge(),
        "file_format": gv("MP4"), "file_size": gv(5000), "resolution": gna(),
        "description": ge(), "file_hash": gna()}),
    gold_rec("S4", "T-002", "video", "kogl", {          # 예측 누락
        "keywords": gv(["a", "b", "c"]), "hashtags": gv(["a"], "site"),
        "file_format": gv("MP4"), "file_size": gv(1), "resolution": ge(),
        "description": gv("D", "caption"), "file_hash": gv(HASH["d"])}),
    gold_rec("S5", "T-002", "image", "donated", {
        "keywords": gv(["꽃", "봄"]), "hashtags": gv(["#꽃"], "generated"),
        "file_format": gv("TIFF"), "file_size": gv(42), "resolution": gv("100x100"),
        "description": gv("꽃밭", "generated"), "file_hash": gv(HASH["e"])}),
    gold_rec("S6", "T-002", "image", "donated", {
        "keywords": gv(["a"]), "hashtags": gna(),
        "file_format": gv("JPG"), "file_size": gv(7), "resolution": gna(),
        "description": gna("label"), "file_hash": gv(HASH["f"])}),
]

T002_PRED = {
    "S1": pred(keywords=["바다", "해변", "파도"], hashtags=["바다", "해변"], file_format="jpeg",
               file_size="123,456", resolution="1920 x 1080", description="해변에 파도가 밀려온다",
               file_hash="sha256:" + "A" * 64),
    "S2": pred(keywords=["서울", "한강대교", "불빛"], hashtags=["x"], file_format="JPG", file_size=None,
               resolution="800*600", description="다른 설명", file_hash="zz"),
    "S3": pred(keywords=["a", "b"], hashtags=[], file_format="mpeg4", file_size="5 KB", resolution="1x1",
               description="무언가", file_hash=None),
    "S5": pred(keywords="꽃, 봄, 나무", hashtags=["#꽃"], file_format="tif", file_size=42,
               resolution="100×100", description="꽃밭", file_hash="E" * 64),
    "S6": pred(keywords=None, hashtags=None, file_format="GIF", file_size="7 bytes", resolution=None,
               description="whatever", file_hash=None, work_title="계약서에서 복사된 제목",
               _meta={"request_id": "r6"}, foo=1),
    "S9": pred(keywords=["gold 없음"]),                 # gold 없는 예측 → 경고만
}

# 손으로 계산한 키별 (TP, FP, FN, 평가 건수) — 각 세트의 기여는 모듈 docstring/gold 정의를 따라 적었다.
# keywords   S1 TP3 · S2 TP2 FP1 FN1 (서울=, 한강⊂한강대교, 불빛 FP, 야경 FN) · S3 empty+2개 → FP2 · S4 누락 → FN3 · S5 TP2 FP1 · S6 null → FN1
# hashtags   S1 TP2 · S2 na · S3 empty+[] → 0 · S4 FN1 · S5 TP1 · S6 na
# file_format S1 TP · S2 FP+FN · S3 TP · S4 FN · S5 TP · S6 FP+FN
# file_size  S1 TP · S2 null → FN · S3 '5 KB' 해석 불가 → FP+FN · S4 FN · S5 TP · S6 TP
# resolution S1 TP · S2 TP · S3 na · S4 empty+null → 0 · S5 TP · S6 na
# description S1 TP · S2 FP+FN · S3 empty+값 → FP · S4 FN · S5 TP · S6 na
# file_hash  S1 TP · S2 'zz' 해석 불가 → FP+FN · S3 na · S4 FN · S5 TP · S6 null → FN
T002_EXPECT = {
    "keywords": (7, 4, 5, 6),
    "hashtags": (3, 0, 1, 4),
    "file_format": (3, 2, 3, 6),
    "file_size": (3, 1, 3, 6),
    "resolution": (3, 0, 0, 4),
    "description": (2, 2, 2, 5),
    "file_hash": (2, 1, 3, 5),
}


@pytest.fixture()
def t002_run(tmp_path: Path):
    gold = write_jsonl(tmp_path / "gold23_t002.jsonl", T002_GOLD)
    pdir = tmp_path / "test_predictions"
    pdir.mkdir()
    for sid, p in T002_PRED.items():
        (pdir / f"{sid}.json").write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    judge = StubJudge()
    m = evaluate("T-002", gold, pdir, out, judge=judge, cells=True)
    return m, out, judge


def test_t002_per_key_counts(t002_run):
    m, _, judge = t002_run
    assert m["test"] == "T-002" and m["n_sets"] == 6 and m["n_predictions"] == 6
    for key, (tp, fp, fn, n) in T002_EXPECT.items():
        st = m["keys"][key]
        assert (st["tp"], st["fp"], st["fn"], st["n_scored"]) == (tp, fp, fn, n), key
        assert st["f1"] == pytest.approx(float(f1(tp, fp, fn)))
    assert set(m["keys"]) == set(T002_KEYS)
    assert judge.calls == 3                      # 판정기는 gold value + 예측 non-null 일 때만 호출: S1, S2, S5 (S3 empty, S4 누락, S6 na)
    assert m["excluded_keys"] == []


def test_t002_macro_micro(t002_run):
    m, _, _ = t002_run
    f1s = [f1(tp, fp, fn) for tp, fp, fn, _ in T002_EXPECT.values()]
    macro = sum(f1s) / len(f1s)
    TP = sum(v[0] for v in T002_EXPECT.values())
    FP = sum(v[1] for v in T002_EXPECT.values())
    FN = sum(v[2] for v in T002_EXPECT.values())
    assert (TP, FP, FN) == (23, 10, 17)
    assert round(m["macro_f1"], 3) == round(float(macro), 3) == 0.659
    assert round(m["micro_f1"], 3) == round(float(f1(TP, FP, FN)), 3) == 0.630
    assert m["micro"]["precision"] == pytest.approx(23 / 33) and m["micro"]["recall"] == pytest.approx(23 / 40)
    assert m["macro"]["n_keys"] == 7 and m["passed"] is False and m["pass_mark"] == 0.85


def test_t002_by_kind_and_cells(t002_run):
    m, _, _ = t002_run
    desc = m["by_kind"]["description"]
    assert (desc["caption"]["tp"], desc["caption"]["fp"], desc["caption"]["fn"]) == (1, 0, 1)
    assert desc["caption"]["f1"] == pytest.approx(2 / 3)
    assert (desc["generated"]["tp"], desc["generated"]["fp"], desc["generated"]["fn"]) == (1, 1, 1)
    assert desc["generated"]["f1"] == pytest.approx(0.5)
    assert desc["combined"]["f1"] == pytest.approx(0.5) and desc["combined"]["n_scored"] == 5
    assert desc["label"]["n_scored"] == 0 and desc["label"]["f1"] is None
    tags = m["by_kind"]["hashtags"]
    assert tags["site"]["f1"] == pytest.approx(0.8) and tags["generated"]["f1"] == pytest.approx(1.0)
    assert tags["combined"]["f1"] == pytest.approx(6 / 7)
    assert set(m["cells"]) == {"image/ccl", "video/kogl", "image/donated"}
    cell = m["cells"]["image/ccl"]
    assert cell["n_sets"] == 2
    kw = cell["keys"]["keywords"]
    assert (kw["tp"], kw["fp"], kw["fn"]) == (5, 1, 1) and kw["f1"] == pytest.approx(10 / 12)


def test_t002_warnings_and_states(t002_run):
    m, _, _ = t002_run
    w = m["warnings"]
    assert w["missing_predictions"] == ["S4"]
    assert w["predictions_without_gold"] == ["S9"]
    assert [(x["set_id"], x["key"]) for x in w["unparseable"]] == [("S2", "file_hash"), ("S3", "file_size")]
    assert w["unscored_nonnull"] == []          # T-002 에서는 1~16 복사 허용
    assert m["warning_counts"]["unparseable"] == 2
    assert m["keys"]["hashtags"]["states"] == {"value": 3, "empty": 1, "na": 2}
    assert m["keys"]["description"]["states"] == {"value": 4, "empty": 1, "na": 1}


def test_t002_outputs_and_determinism(t002_run, tmp_path: Path):
    m, out, _ = t002_run
    assert (out / "test_metrics.json").exists() and (out / "report.md").exists() and (out / "per_set.jsonl").exists()
    rows = [json.loads(l) for l in (out / "per_set.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 * 7
    assert [r["set_id"] for r in rows[:7]] == ["S1"] * 7 and [r["key"] for r in rows[:7]] == list(T002_KEYS)
    d = next(r for r in rows if r["set_id"] == "S1" and r["key"] == "description")
    assert d["tp"] == 1 and d["detail"]["cosine"] == 0.9 and "해변에" in d["detail"]["shared_tokens"]
    s4 = [r for r in rows if r["set_id"] == "S4"]
    assert sum(r["fn"] for r in s4) == 3 + 1 + 1 + 1 + 1 + 1 and all(r["pred"] is None for r in s4)
    na = next(r for r in rows if r["set_id"] == "S6" and r["key"] == "description")
    assert (na["state"], na["kind"], na["tp"], na["fp"], na["fn"], na["matched"]) == ("na", "label", 0, 0, 0, None)
    for st in m["keys"].values():
        lo, hi = st["ci95_f1"]
        assert 0.0 <= lo <= hi <= 1.0
    assert m["macro"]["ci95_f1"] and m["micro"]["ci95_f1"]
    report = (out / "report.md").read_text(encoding="utf-8")
    for token in ("최종 지표", "정밀도", "재현율", "F1-score", "Macro 평균", "Micro 평균", "image/ccl", "gold 종류별", "경고"):
        assert token in report
    # 같은 입력 → 바이트 단위로 같은 test_metrics.json (고정 seed, 정렬 키, 타임스탬프 없음)
    out2 = tmp_path / "out2"
    evaluate("T-002", m["inputs"]["gold"], m["inputs"]["pred"], out2, judge=StubJudge(), cells=True)
    a = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    b = json.loads((out2 / "test_metrics.json").read_text(encoding="utf-8"))
    assert "outputs" not in a                     # 파일에는 출력 경로를 적지 않는다 (반환 dict 에만 있음)
    assert a == b


# =================================================================== T-001
def rights(*vals: Any) -> dict[str, Any]:
    names = ("reproduction_right", "performance_right", "public_transmission_right", "exhibition_right",
             "distribution_right", "rental_right", "derivative_work_right")
    return dict(zip(names, vals))


def c_gold(set_id: str, media: str, bucket: str, **keys: Any) -> dict[str, Any]:
    return gold_rec(set_id, "T-001", media, bucket, keys)


T001_GOLD = [
    c_gold("C1", "image", "ccl", work_title=gv("운수 자판기"), file_name=gv("13387752 운수 자판기.pdf"), work_type=gv("미술"),
           creator=gv(["홍길동"]), copyright_holder=gv(["한국저작권위원회"]),
           **{k: gv(v) for k, v in rights("허용", "불허", "허용", "불허", "허용", "불허", "허용").items()},
           other_rights=gna(), license_start_date=gv("2002-08-31"), license_end_date=gv("2033-12-23"),
           contract_validity_period=gv("2002-08-31~2033-12-23")),
    c_gold("C2", "image", "ccl", work_title=gv("A"), file_name=gv("x y.pdf"), work_type=gv("사진"),
           creator=gv(["김민수"]), copyright_holder=gv(["a", "b"]),
           reproduction_right=gv("허용"), performance_right=gv("허용"), public_transmission_right=gv("불허"),
           exhibition_right=gv("허용"), distribution_right=ge(), rental_right=ge(), derivative_work_right=gv("불허"),
           other_rights=gna(), license_start_date=gv("2020-01-05"), license_end_date=gv("2021-01-04"),
           contract_validity_period=gv("2020-01-05~2021-01-04")),
    c_gold("C3", "text", "donated", work_title=gv("C3"), file_name=gv("c3.pdf"), work_type=gv("영상"),      # 예측 누락
           creator=gv(["c"]), copyright_holder=gna(),
           **{k: gv(v) for k, v in rights(*(["허용"] * 7)).items()},
           other_rights=gna(), license_start_date=gv("2021-01-01"), license_end_date=gv("2021-12-31"),
           contract_validity_period=gv("2021-01-01~2021-12-31")),
    c_gold("C4", "video", "kogl", work_title=gv("C4"), file_name=gv("c4.pdf"), work_type=gv("영상"),
           creator=gv(["a", "b"]), copyright_holder=gv(["c"]),
           **{k: gv(v) for k, v in rights(*(["허용"] * 7)).items()},
           other_rights=gna(), license_start_date=gv("2022-02-02"), license_end_date=gv("2023-02-01"),
           contract_validity_period=gv("2022-02-02~2023-02-01")),
    c_gold("C5", "text", "donated", work_title=gv("제목"), file_name=gv("10001 제목.pdf"), work_type=gv("어문"),
           creator=gv(["박지성", "김연아"]), copyright_holder=gv(["한국저작권위원회"]),
           **{k: gv(v) for k, v in rights(*(["불허"] * 7)).items()},
           other_rights=gna(), license_start_date=gv("2024-03-01"), license_end_date=gv("2025-02-28"),
           contract_validity_period=gv("2024-03-01~2025-02-28")),
    c_gold("C6", "video", "kogl", work_title=gv("C6"), file_name=gv("c6.pdf"), work_type=gv("기타"),
           creator=gv(["a"]), copyright_holder=gna(),
           **{k: gv(v) for k, v in rights("허용", "허용", "허용", "불허", "불허", "불허", "허용").items()},
           other_rights=gna(), license_start_date=gv("2019-12-31"), license_end_date=gv("2020-12-30"),
           contract_validity_period=gv("2019-12-31~2020-12-30")),
]

T001_PRED = [
    {"set_id": "C1", **pred(work_title="13387752 운수 자판기", file_name="13387752  운수 자판기.pdf", work_type="미술저작물",
                             creator="홍길동 (저작물 3 건)", copyright_holder=["한국저작권위원회"],
                             **rights(True, "X", "O", "N", "Y", False, "허용"),
                             license_start_date="2002년 8월 31일", license_end_date="20331223",
                             contract_validity_period="2002년 08월 31일부터 2033년 12월 23일까지")},
    {"set_id": "C2", **pred(work_title="B", file_name="x_y.pdf", work_type="그림", creator=["김민"], copyright_holder="a, b",
                             **rights("불허", "허용", None, "maybe", "허용", None, "X"),
                             other_rights="방송권", license_start_date="2020.01.05", license_end_date="2021/1/4",
                             contract_validity_period="2020.01.05 – 2021.01.04", keywords=["x"])},
    {"_meta": {"set_id": "C4", "request_id": "r4"},
     **pred(work_title="C4", file_name="c4.pdf", work_type="동영상", creator=["b", "a"], copyright_holder=None,
            **rights("허용", "허용", "허용", "허용", "허용", None, "허용"),
            license_start_date="2022-02-02", license_end_date=None, contract_validity_period=None)},
    {"set_id": "C5", **pred(work_title="10001  제목", file_name="10001 제목.PDF", work_type="어문 저작물",
                             creator="박지성 · 김연아", copyright_holder="한국 저작권 위원회",
                             **rights("불허", "X", "N", "false", False, "x", "n"),
                             license_start_date="2024-03-01", license_end_date="2025년 2월 28일",
                             contract_validity_period="2024-03-01 ~ 2025-02-28")},
    {"set_id": "C6", **pred(work_title="C6", file_name=None, work_type="기타저작물", creator=["a", "b"], copyright_holder=["z"],
                             **rights(True, "Y", "O", "X", "N", False, "불허"),
                             license_start_date="2019-12-31", license_end_date="2020-13-30",
                             contract_validity_period="2019-12-31~2020-12-30")},
]

# 손으로 계산한 키별 (TP, FP, FN, 평가 건수). C3 는 예측 누락(gold 값 있는 키 모두 FN), copyright_holder 는 C3·C6 에서 na.
T001_EXPECT = {
    "work_title": (4, 1, 2, 6),            # C2 오답
    "file_name": (3, 1, 3, 6),             # C2 '_' 오답, C6 null
    "work_type": (4, 1, 2, 6),             # C2 '그림' 해석 불가
    "creator": (3, 2, 3, 6),               # C2 김민≠김민수, C6 [a,b]≠[a]
    "copyright_holder": (3, 0, 1, 4),      # C4 null; C3·C6 na
    "reproduction_right": (4, 1, 2, 6),    # C2 불허≠허용
    "performance_right": (5, 0, 1, 6),
    "public_transmission_right": (4, 0, 2, 6),   # C2 null
    "exhibition_right": (4, 1, 2, 6),      # C2 'maybe' 해석 불가
    "distribution_right": (4, 1, 1, 6),    # C2 empty + '허용' → FP
    "rental_right": (3, 0, 2, 6),          # C2 empty + null → 0, C4 null
    "derivative_work_right": (4, 1, 2, 6), # C6 불허≠허용
    "license_start_date": (5, 0, 1, 6),
    "license_end_date": (3, 1, 3, 6),      # C4 null, C6 2020-13-30 해석 불가
    "contract_validity_period": (4, 0, 2, 6),    # C4 null
}


@pytest.fixture()
def t001_run(tmp_path: Path):
    gold = write_jsonl(tmp_path / "gold23_t001.jsonl", T001_GOLD)
    predf = write_jsonl(tmp_path / "test_predictions.jsonl", T001_PRED)
    out = tmp_path / "out"
    m = evaluate("T-001", gold, predf, out)        # 설명 키가 없으니 판정기(모델) 불필요
    return m, out


def test_t001_per_key_counts(t001_run):
    m, _ = t001_run
    assert m["n_sets"] == 6 and m["n_predictions"] == 5
    for key, (tp, fp, fn, n) in T001_EXPECT.items():
        st = m["keys"][key]
        assert (st["tp"], st["fp"], st["fn"], st["n_scored"]) == (tp, fp, fn, n), key
    other = m["keys"]["other_rights"]
    assert (other["tp"], other["fp"], other["fn"], other["n_scored"]) == (0, 0, 0, 0)
    assert other["states"] == {"value": 0, "empty": 0, "na": 6}
    assert m["excluded_keys"] == [{"key": "other_rights", "reason": "n_scored=0"}]
    assert set(m["keys"]) == set(T001_KEYS) and m["settings"]["embed"] is None


def test_t001_macro_micro(t001_run):
    m, _ = t001_run
    f1s = [f1(tp, fp, fn) for tp, fp, fn, _ in T001_EXPECT.values()]
    macro = sum(f1s) / len(f1s)
    TP = sum(v[0] for v in T001_EXPECT.values())
    FP = sum(v[1] for v in T001_EXPECT.values())
    FN = sum(v[2] for v in T001_EXPECT.values())
    assert (TP, FP, FN) == (57, 10, 29)
    assert m["macro"]["n_keys"] == 15
    assert round(m["macro_f1"], 3) == round(float(macro), 3) == 0.747
    assert round(m["micro_f1"], 3) == round(float(f1(TP, FP, FN)), 3) == 0.745
    assert m["passed"] is False


def test_t001_warnings(t001_run):
    m, _ = t001_run
    w = m["warnings"]
    assert w["missing_predictions"] == ["C3"]
    assert [(x["set_id"], x["key"]) for x in w["unparseable"]] == [
        ("C2", "exhibition_right"), ("C2", "work_type"), ("C6", "license_end_date")]
    assert [(x["set_id"], x["key"]) for x in w["unscored_nonnull"]] == [("C2", "keywords"), ("C2", "other_rights")]
    assert w["predictions_without_gold"] == [] and w["duplicate_predictions"] == []
    assert m["by_kind"] == {}                      # T-001 에는 설명·해시태그가 없다
    assert set(m["cells"]) == {"image/ccl", "text/donated", "video/kogl"}


def test_t001_per_set_rows(t001_run):
    _, out = t001_run
    rows = [json.loads(l) for l in (out / "per_set.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 * 16
    c1 = {r["key"]: r for r in rows if r["set_id"] == "C1"}
    assert all(c1[k]["tp"] == 1 for k in T001_KEYS if k != "other_rights")
    assert c1["other_rights"]["state"] == "na" and c1["other_rights"]["matched"] is None
    c3 = [r for r in rows if r["set_id"] == "C3"]
    assert sum(r["fn"] for r in c3) == 14 and sum(r["tp"] + r["fp"] for r in c3) == 0


def test_cli_main_t001(tmp_path: Path, capsys):
    gold = write_jsonl(tmp_path / "gold.jsonl", T001_GOLD)
    predf = write_jsonl(tmp_path / "pred.jsonl", T001_PRED)
    out = tmp_path / "cli_out"
    rc = main(["--test", "T-001", "--gold", str(gold), "--pred", str(predf), "--out", str(out), "--cells", "--n-boot", "50"])
    assert rc == 0
    cap = capsys.readouterr()
    captured = cap.out
    assert "Macro-F1 74.71%" in captured and "불합격" in captured
    assert re.search(r"CI \[\d+\.\d{2}%, \d+\.\d{2}%\]", captured)      # CI 도 F1 처럼 퍼센트로
    assert "채점 시작" in cap.err and "Macro-F1" in cap.err                 # INFO 로그가 CLI(stderr)에 실제로 보인다
    m = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert m["bootstrap"]["n_resamples"] == 50
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "### 5.image/ccl" in report                 # --cells → 셀별 항목 표


def test_predictions_without_known_keys_and_case_insensitive_json(tmp_path: Path):
    """23키가 하나도 없는 예측(필드가 감싸인 API 응답)은 조용히 0점이 아니라 경고로 드러나야 하고, 'X.JSON' 도 예측으로 읽어야 한다."""
    gold = write_jsonl(tmp_path / "gold.jsonl", T002_GOLD[:2])
    pdir = tmp_path / "preds"
    pdir.mkdir()
    (pdir / "S1.JSON").write_text(json.dumps(T002_PRED["S1"], ensure_ascii=False), encoding="utf-8")
    (pdir / "S2.json").write_text(json.dumps({"metadata": T002_PRED["S2"], "_meta": {"request_id": "r2"}}, ensure_ascii=False),
                                  encoding="utf-8")
    (pdir / "notes.txt").write_text("예측 아님", encoding="utf-8")
    out = tmp_path / "out"
    m = evaluate("T-002", gold, pdir, out, judge=StubJudge(), n_boot=10)
    w = m["warnings"]
    assert m["n_predictions"] == 2 and w["missing_predictions"] == [] and w["unreadable_predictions"] == []
    assert w["predictions_without_known_keys"] == ["S2"] and w["ignored_files"] == ["notes.txt"]
    assert m["warning_counts"]["predictions_without_known_keys"] == 1 and m["warning_counts"]["ignored_files"] == 1
    for key in ("file_format", "file_size", "resolution", "description", "file_hash"):
        assert (m["keys"][key]["tp"], m["keys"][key]["fp"], m["keys"][key]["fn"]) == (1, 0, 1), key   # S1 TP, S2 전부 null → FN
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "23키가 하나도 없는 예측: 1건" in report and "S2" in report and "무시한 파일(.json 아님): 1건" in report
    # 같은 set_id 가 'S1.json' 과 'S1.JSON' 으로 둘 있으면 중복으로 알린다
    (pdir / "S1.json").write_text(json.dumps(T002_PRED["S1"], ensure_ascii=False), encoding="utf-8")
    m2 = evaluate("T-002", gold, pdir, tmp_path / "out2", judge=StubJudge(), n_boot=10)
    assert m2["warnings"]["duplicate_predictions"] == ["S1"]


def test_reinterpreted_empty_gold_counts_as_state_empty(tmp_path: Path):
    """gold 가 'value' 상태인데 정규화 후 비면 채점은 ②(FP only) — 상태 표도 ② 로 세어야 한다 (per_set.state 포함)."""
    gold = write_jsonl(tmp_path / "gold.jsonl", [gold_rec("E1", "T-002", "image", "ccl", {
        "keywords": gv([]), "hashtags": gna(), "file_format": gv(""), "file_size": gv(1), "resolution": gna(),
        "description": gv("   "), "file_hash": gna()})])
    predf = write_jsonl(tmp_path / "pred.jsonl", [{"set_id": "E1", **pred(keywords=["a"], file_format="JPG", file_size=1,
                                                                            description="x")}])
    out = tmp_path / "o"
    m = evaluate("T-002", gold, predf, out, judge=StubJudge(), n_boot=10)
    for key in ("keywords", "file_format", "description"):
        assert m["keys"][key]["states"] == {"value": 0, "empty": 1, "na": 0}, key
        assert (m["keys"][key]["tp"], m["keys"][key]["fp"], m["keys"][key]["fn"], m["keys"][key]["n_scored"]) == (0, 1, 0, 1), key
    assert m["keys"]["file_size"]["states"] == {"value": 1, "empty": 0, "na": 0}
    rows = [json.loads(l) for l in (out / "per_set.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["state"] for r in rows if r["key"] in ("keywords", "file_format", "description")} == {"empty"}
    assert next(r for r in rows if r["key"] == "file_size")["state"] == "value"


def test_warm_embed_cache_rerun_is_byte_identical(tmp_path: Path):
    """BUG 회귀(평가기 수준): 모든 설명이 임베딩 캐시에서 나오는 재실행도 settings.embed(revision/device)까지 같은
    test_metrics.json 과 report.md 를 내야 한다 — 리뷰어가 실데이터로 본 'revision=null/device=null' 재현 시나리오."""
    import hashlib

    import numpy as np

    from api.module.evaluation.v4.embed_judge import EmbedJudge

    def embedder(text: str) -> np.ndarray:                 # 결정적 가짜 임베더 — 같은 텍스트면 같은 벡터(코사인 1.0)
        return np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest(), dtype=np.uint8).astype(np.float32)

    def never(text: str) -> np.ndarray:
        raise AssertionError(f"따뜻한 캐시 재실행이 임베딩을 다시 계산했다: {text!r}")

    gold = write_jsonl(tmp_path / "gold.jsonl", T002_GOLD[:2])
    pdir = tmp_path / "preds"
    pdir.mkdir()
    for sid in ("S1", "S2"):
        (pdir / f"{sid}.json").write_text(json.dumps(T002_PRED[sid], ensure_ascii=False), encoding="utf-8")
    cache = tmp_path / "emb.jsonl"
    cold = EmbedJudge(cache_path=cache, embedder=embedder, revision="rev-A")
    cold.resolved_device = "cuda"                          # 실제 모델 적재가 채우는 값을 흉내 낸다
    m1 = evaluate("T-002", gold, pdir, tmp_path / "run1", judge=cold, n_boot=10)
    assert cold.n_embed_calls == 3 and m1["settings"]["embed"] == {**m1["settings"]["embed"], "revision": "rev-A", "device": "cuda"}
    assert m1["keys"]["description"]["tp"] == 1            # S1 같은 텍스트 → 코사인 1.0 + 공유 토큰

    warm = EmbedJudge(cache_path=cache, embedder=never)    # revision/device 를 주지 않은 재실행
    m2 = evaluate("T-002", gold, pdir, tmp_path / "run2", judge=warm, n_boot=10)
    assert warm.n_embed_calls == 0
    a = json.loads((tmp_path / "run1" / "test_metrics.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "run2" / "test_metrics.json").read_text(encoding="utf-8"))
    assert a == b and b["settings"]["embed"]["revision"] == "rev-A" and b["settings"]["embed"]["device"] == "cuda"
    r1 = (tmp_path / "run1" / "report.md").read_text(encoding="utf-8")
    r2 = (tmp_path / "run2" / "report.md").read_text(encoding="utf-8")
    assert r1 == r2 and "revision rev-A" in r2 and "미기록" not in r2
    assert (tmp_path / "run1" / "per_set.jsonl").read_bytes() == (tmp_path / "run2" / "per_set.jsonl").read_bytes()


def test_gold_test_mismatch_and_bad_pred_path(tmp_path: Path):
    gold = write_jsonl(tmp_path / "gold.jsonl", T001_GOLD + [T002_GOLD[0]])
    predf = write_jsonl(tmp_path / "pred.jsonl", T001_PRED)
    m = evaluate("T-001", gold, predf, tmp_path / "o", n_boot=10)
    assert m["n_sets"] == 6 and m["warnings"]["gold_test_mismatch"] == ["S1"]
    with pytest.raises(ValueError):
        evaluate("T-001", gold, tmp_path / "nope.txt", tmp_path / "o2")
    with pytest.raises(ValueError):
        evaluate("T-003", gold, predf, tmp_path / "o3")


# =================================================================== --only-predicted (M2b 수정 4)
def test_only_predicted_restricts_gold_to_predicted_sets(tmp_path: Path):
    """gold 5세트 · 예측 2세트: 기본(M1)은 누락 3세트가 전부 FN, --only-predicted 면 그 2세트만 채점(n_scored 2, 누락 0)."""
    import logging
    from api.module.evaluation.v4 import evaluator as ev_core

    gold = write_jsonl(tmp_path / "gold.jsonl", T001_GOLD[:5])                                   # C1~C5
    preds = [{"set_id": sid, **pred(**{k: v["value"] for k, v in rec["keys"].items() if v["state"] == "value"})}
             for sid, rec in (("C1", T001_GOLD[0]), ("C5", T001_GOLD[4]))]
    predf = write_jsonl(tmp_path / "pred.jsonl", preds)

    base = evaluate("T-001", gold, predf, tmp_path / "base", n_boot=10)                           # 기본 = M1 그대로
    assert base["n_sets"] == 5 and base["n_gold_dropped"] == 0 and base["settings"]["only_predicted"] is False
    assert base["warnings"]["missing_predictions"] == ["C2", "C3", "C4"]
    wt = base["keys"]["work_title"]
    assert (wt["n_scored"], wt["tp"], wt["fp"], wt["fn"]) == (5, 2, 0, 3)

    caught: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            caught.append(record.getMessage())

    h = _Catch()
    old = ev_core.logger.level
    ev_core.logger.addHandler(h)
    ev_core.logger.setLevel(logging.INFO)
    try:
        m = evaluate("T-001", gold, predf, tmp_path / "only", n_boot=10, only_predicted=True)
    finally:
        ev_core.logger.removeHandler(h)
        ev_core.logger.setLevel(old)
    assert m["n_sets"] == 2 and m["n_predictions"] == 2 and m["n_gold_dropped"] == 3
    assert m["warnings"]["missing_predictions"] == [] and m["settings"]["only_predicted"] is True
    for key in T001_KEYS:
        if key == "other_rights":
            continue
        st = m["keys"][key]
        assert (st["n_scored"], st["tp"], st["fp"], st["fn"]) == (2, 2, 0, 0), (key, st)
    assert m["macro_f1"] == pytest.approx(1.0) and m["micro_f1"] == pytest.approx(1.0) and m["passed"] is True
    assert set(m["cells"]) == {"image/ccl", "text/donated"} and m["cells"]["image/ccl"]["n_sets"] == 1   # 셀도 남은 세트만
    assert any("gold 5세트 중 3세트 제외, 2세트 대조" in msg for msg in caught)
    assert "gold 제외 3개 (--only-predicted" in (tmp_path / "only" / "report.md").read_text(encoding="utf-8")
    per_set = {json.loads(l)["set_id"] for l in (tmp_path / "only" / "per_set.jsonl").read_text(encoding="utf-8").splitlines()}
    assert per_set == {"C1", "C5"}
    # 제한은 gold 쪽에만 — gold 없는 예측은 여전히 경고로 남는다
    predf2 = write_jsonl(tmp_path / "pred2.jsonl", preds + [{"set_id": "ZZ", **pred(work_title="z")}])
    m2 = evaluate("T-001", gold, predf2, tmp_path / "only2", n_boot=10, only_predicted=True)
    assert m2["n_sets"] == 2 and m2["warnings"]["predictions_without_gold"] == ["ZZ"] and m2["n_gold_dropped"] == 3


def test_cli_only_predicted_flag(tmp_path: Path, capsys):
    gold = write_jsonl(tmp_path / "gold.jsonl", T001_GOLD[:5])
    predf = write_jsonl(tmp_path / "pred.jsonl", [{"set_id": "C1", **pred(**{k: v["value"] for k, v in T001_GOLD[0]["keys"].items()
                                                                              if v["state"] == "value"})}])
    out = tmp_path / "cli_only"
    rc = main(["--test", "T-001", "--gold", str(gold), "--pred", str(predf), "--out", str(out), "--n-boot", "10", "--only-predicted"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[T-001] 세트 1 · 예측 누락 0 · gold 제외 4 (--only-predicted)" in captured and "Macro-F1 100.00%" in captured
    m = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert m["n_sets"] == 1 and m["n_gold_dropped"] == 4 and m["settings"]["only_predicted"] is True
    # 플래그 없이는 예전 출력 그대로 (gold 제외 문구 없음)
    rc = main(["--test", "T-001", "--gold", str(gold), "--pred", str(predf), "--out", str(tmp_path / "cli_all"), "--n-boot", "10"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[T-001] 세트 5 · 예측 누락 4" in captured and "gold 제외" not in captured
