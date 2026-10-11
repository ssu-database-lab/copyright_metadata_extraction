"""matchers.py 단위 테스트 (spec §8 b) — 단일값 match/mismatch/null/empty/na, 리스트 포함 1:1 매칭, 이름 집합 동일성."""
from __future__ import annotations

from typing import Any

import pytest

from api.module.evaluation.v4 import matchers as M
from api.module.evaluation.v4.keys23 import BY_NAME


def G(value: Any = None, state: str | None = None, kind: str | None = None) -> dict[str, Any]:
    if state is None:
        state = "value" if value is not None else "na"
    return {"value": value, "state": state, "kind": kind, "source": "test", "tier": "A"}


def counts(res: dict[str, Any]) -> tuple[int, int, int, Any]:
    return res["tp"], res["fp"], res["fn"], res["matched"]


# ----------------------------------------------------------------- 단일값
class TestScalar:
    KEY = BY_NAME["file_format"]

    def test_match(self):
        assert counts(M.match_key(self.KEY, "jpeg", G("JPG"))) == (1, 0, 0, True)

    def test_mismatch_is_fp_and_fn(self):
        assert counts(M.match_key(self.KEY, "png", G("JPG"))) == (0, 1, 1, False)

    def test_null_pred_with_gold_value_is_fn(self):
        assert counts(M.match_key(self.KEY, None, G("JPG"))) == (0, 0, 1, False)
        assert counts(M.match_key(self.KEY, "", G("JPG"))) == (0, 0, 1, False)

    def test_empty_gold(self):
        assert counts(M.match_key(self.KEY, "png", G(None, "empty"))) == (0, 1, 0, False)
        assert counts(M.match_key(self.KEY, None, G(None, "empty"))) == (0, 0, 0, None)

    def test_na_gold_skipped(self):
        assert counts(M.match_key(self.KEY, "png", G(None, "na"))) == (0, 0, 0, None)
        assert counts(M.match_key(self.KEY, "png", {})) == (0, 0, 0, None)
        assert counts(M.match_key(self.KEY, "png", None)) == (0, 0, 0, None)

    def test_unparseable_pred_counts_as_wrong(self):
        res = M.match_key(self.KEY, "???", G("JPG"))
        assert counts(res) == (0, 1, 1, False)
        assert "unparseable" in res["detail"]
        res = M.match_key(self.KEY, "???", G(None, "empty"))
        assert counts(res) == (0, 1, 0, False)

    def test_unparseable_gold_is_skipped_and_flagged(self):
        res = M.match_key(self.KEY, "JPG", G("???"))
        assert counts(res) == (0, 0, 0, None)
        assert "gold_unparseable" in res["detail"]

    def test_both_sides_normalized(self):
        assert counts(M.match_key("license_start_date", "2024년 1월 5일", G("2024.01.05"))) == (1, 0, 0, True)
        assert counts(M.match_key("file_size", "123,456", G("123456.0"))) == (1, 0, 0, True)
        assert counts(M.match_key("resolution", "1920*1080", G("1920 x 1080"))) == (1, 0, 0, True)
        assert counts(M.match_key("file_hash", "sha256:" + "A" * 64, G("a" * 64))) == (1, 0, 0, True)
        assert counts(M.match_key("work_type", "미술저작물", G("미술"))) == (1, 0, 0, True)
        assert counts(M.match_key("contract_validity_period", "2002년 08월 31일부터 2033년 12월 23일까지",
                                  G("2002-08-31~2033-12-23"))) == (1, 0, 0, True)

    def test_rights_value_matching(self):
        assert counts(M.match_key("reproduction_right", "O", G("허용"))) == (1, 0, 0, True)
        assert counts(M.match_key("reproduction_right", "X", G("불허"))) == (1, 0, 0, True)   # 맞는 불허도 TP
        assert counts(M.match_key("reproduction_right", "X", G("허용"))) == (0, 1, 1, False)
        assert counts(M.match_key("reproduction_right", None, G("허용"))) == (0, 0, 1, False)
        assert counts(M.match_key("reproduction_right", "null", G("허용"))) == (0, 0, 1, False)


class TestText:
    def test_work_title_strips_id_prefix(self):
        assert counts(M.match_key("work_title", "13387752 운수 자판기", G("운수 자판기"))) == (1, 0, 0, True)
        assert counts(M.match_key("work_title", "운수  자판기", G("운수 자판기"))) == (1, 0, 0, True)
        assert counts(M.match_key("work_title", "13387752 운수 자판기", G("13387752 운수 자판기"))) == (1, 0, 0, True)
        # 관리번호 접두는 예측 쪽만 뗀다(§1 #1) — gold 에 접두가 있으면 접두 없는 예측은 오답
        assert counts(M.match_key("work_title", "운수 자판기", G("13387752 운수 자판기"))) == (0, 1, 1, False)
        assert counts(M.match_key("work_title", "운수 자판기2", G("운수 자판기"))) == (0, 1, 1, False)
        assert counts(M.match_key("work_title", "20240909 Arirang", G("Arirang"))) == (1, 0, 0, True)   # 8자리 날짜형 토큰도 예측 쪽만

    def test_file_name_exact_after_whitespace_and_case(self):
        gold = G("13387752 운수 자판기.pdf")
        assert counts(M.match_key("file_name", "13387752  운수 자판기.PDF", gold)) == (1, 0, 0, True)
        assert counts(M.match_key("file_name", "13387752_운수_자판기.pdf", gold)) == (0, 1, 1, False)
        assert counts(M.match_key("file_name", ["13387752 운수 자판기.pdf"], gold)) == (1, 0, 0, True)


class TestOtherRights:
    def test_always_excluded_never_fp(self):
        res = M.match_key("other_rights", ["방송권"], G(None, "na"))
        assert counts(res) == (0, 0, 0, None)
        assert res["detail"]["unscored_nonnull"] is True
        res = M.match_key("other_rights", None, G(None, "na"))
        assert res["detail"]["unscored_nonnull"] is False
        assert counts(M.match_key("other_rights", "없음", G("없음"))) == (0, 0, 0, None)


# ----------------------------------------------------------------- 리스트
class TestTagLists:
    def test_containment_one_to_one(self):
        tp, fp, fn, pairs = M.match_tag_lists(["서울", "한강대교", "불빛"], ["서울", "야경", "한강"])
        assert (tp, fp, fn) == (2, 1, 1)
        assert ("서울", "서울") in pairs and ("한강", "한강대교") in pairs

    def test_each_gold_matched_once(self):
        tp, fp, fn, _ = M.match_tag_lists(["한강"], ["한강", "한강대교"])
        assert (tp, fp, fn) == (1, 0, 1)
        tp, fp, fn, _ = M.match_tag_lists(["한강", "한강"], ["한강"])   # 중복 예측은 split_tags 에서 제거되지만 직접 호출 시 FP
        assert (tp, fp, fn) == (1, 1, 0)

    def test_exact_preferred_over_containment(self):
        tp, fp, fn, pairs = M.match_tag_lists(["한강", "한강대교"], ["한강대교"])
        assert (tp, fp, fn) == (1, 1, 0)
        assert pairs == [("한강대교", "한강대교")]

    def test_single_char_only_exact(self):
        assert M.tag_match("강", "강")
        assert not M.tag_match("강", "한강")
        assert M.tag_match("한강", "한강대교")
        assert not M.tag_match("바다", "하늘")

    def test_match_key_keywords(self):
        res = M.match_key("keywords", "꽃, 봄, 나무", G(["꽃", "봄"]))
        assert counts(res) == (2, 1, 0, False)
        assert res["detail"]["set_f1"] == pytest.approx(0.8)
        assert res["detail"]["unmatched_pred"] == ["나무"]
        assert counts(M.match_key("keywords", ["꽃", "봄"], G(["꽃", "봄"]))) == (2, 0, 0, True)
        assert counts(M.match_key("keywords", None, G(["a", "b", "c"]))) == (0, 0, 3, False)
        assert counts(M.match_key("keywords", ["a", "b"], G(None, "empty"))) == (0, 2, 0, False)
        assert counts(M.match_key("keywords", [], G(None, "empty"))) == (0, 0, 0, None)
        assert counts(M.match_key("keywords", ["a"], G(None, "na"))) == (0, 0, 0, None)
        assert counts(M.match_key("keywords", ["a"], G([], "value"))) == (0, 1, 0, False)   # 빈 value 는 empty 로

    def test_hashtags_strip_hash_and_dedupe(self):
        assert counts(M.match_key("hashtags", ["#바다", "바다"], G(["바다"]))) == (1, 0, 0, True)
        assert counts(M.match_key("hashtags", "#바다 #해변", G(["#해변", "#바다"]))) == (2, 0, 0, True)


# ----------------------------------------------------------------- 이름
class TestNames:
    def test_names_equal(self):
        assert M.names_equal("홍길동 (저작물 3 건)", ["홍길동"])
        assert M.names_equal("a, b", ["b", "a"])
        assert M.names_equal("한국 저작권 위원회", "한국저작권위원회")
        assert not M.names_equal("김민", "김민수")          # 이름은 포함 일치 없음
        assert not M.names_equal(["a"], ["a", "b"])

    def test_match_key_creator(self):
        assert counts(M.match_key("creator", "박지성 · 김연아", G(["김연아", "박지성"]))) == (1, 0, 0, True)
        assert counts(M.match_key("creator", ["김민"], G(["김민수"]))) == (0, 1, 1, False)
        assert counts(M.match_key("creator", None, G(["a"]))) == (0, 0, 1, False)
        assert counts(M.match_key("creator", ["z"], G(None, "na"))) == (0, 0, 0, None)
        assert counts(M.match_key("creator", "x", G([], "value"))) == (0, 1, 0, False)
        assert counts(M.match_key("copyright_holder", ["x"], G(None, "empty"))) == (0, 1, 0, False)


# ----------------------------------------------------------------- 설명
class StubJudge:
    def __init__(self, match: bool):
        self.match = match
        self.calls: list[tuple[str, str]] = []

    def decide(self, p: str, g: str) -> dict[str, Any]:
        self.calls.append((p, g))
        return {"match": self.match, "cosine": 0.7, "shared_tokens": ["x"], "tau": 0.65, "content_check": True, "content_ok": True}


class TestDescription:
    def test_uses_judge_and_records_detail(self):
        j = StubJudge(True)
        res = M.match_key("description", " 설명 A ", G("설명 B", kind="caption"), judge=j)
        assert counts(res) == (1, 0, 0, True)
        assert res["detail"]["cosine"] == 0.7 and res["detail"]["shared_tokens"] == ["x"]
        assert j.calls == [("설명 A", "설명 B")]
        assert counts(M.match_key("description", "설명 A", G("설명 B"), judge=StubJudge(False))) == (0, 1, 1, False)

    def test_null_empty_na_skip_the_judge(self):
        j = StubJudge(True)
        assert counts(M.match_key("description", None, G("설명"), judge=j)) == (0, 0, 1, False)
        assert counts(M.match_key("description", "x", G(None, "empty"), judge=j)) == (0, 1, 0, False)
        assert counts(M.match_key("description", "x", G(None, "na"), judge=j)) == (0, 0, 0, None)
        assert j.calls == []

    def test_without_judge_falls_back_to_exact(self):
        assert counts(M.match_key("description", "같은 설명", G("같은  설명"))) == (1, 0, 0, True)
        assert counts(M.match_key("description", "다른 설명", G("같은 설명"))) == (0, 1, 1, False)


class TestEffectiveState:
    """'value' gold 가 정규화 후 비면 empty 로 다시 해석하고 detail["state"] 로 알린다 — 평가기 상태 표(②)와 채점이 일치해야 한다."""

    def test_reinterpreted_empty_is_marked(self):
        for key, pred, gold in [("keywords", ["a"], G([], "value")), ("creator", "x", G([], "value")),
                                ("file_format", "png", G("", "value")), ("file_name", "a.pdf", G("  ", "value")),
                                ("description", "x", G("", "value"))]:
            res = M.match_key(key, pred, gold)
            assert counts(res) == (0, 1, 0, False), key
            assert res["detail"]["state"] == "empty", key

    def test_regular_states_are_not_marked(self):
        assert "state" not in M.match_key("file_format", "jpeg", G("JPG"))["detail"]
        assert "state" not in M.match_key("file_format", "png", G(None, "empty"))["detail"]
        assert "state" not in M.match_key("keywords", ["a"], G(["a"]))["detail"]
        assert "state" not in M.match_key("file_format", "png", G(None, "na"))["detail"]


def test_set_f1_and_gold_state():
    assert M.set_f1(0, 0, 0) is None
    assert M.set_f1(2, 1, 0) == pytest.approx(0.8)
    assert M.gold_state(None) == "na"
    assert M.gold_state({"value": "x"}) == "value"
    assert M.gold_state({"value": None}) == "na"
    assert M.gold_state({"value": None, "state": "empty"}) == "empty"
