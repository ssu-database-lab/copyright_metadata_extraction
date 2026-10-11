"""pilot_tau 의 순수 함수 단위 시험 (모델·네트워크 없음)."""
import numpy as np
import pandas as pd

from api.module.evaluation.v4 import pilot_tau as pt


def test_binary_stats_perfect_and_kappa():
    y = np.array([1, 1, 0, 0], bool)
    s = pt.binary_stats(y, y)
    assert s["tpr"] == 1 and s["fpr"] == 0 and s["youden_j"] == 1 and s["kappa"] == 1


def test_select_tau_picks_middle_of_plateau():
    sw = pd.DataFrame({"tau": [0.60, 0.61, 0.62, 0.63, 0.64], "strict_youden_j": [0.5, 0.8, 0.8, 0.8, 0.6]})
    r = pt.select_tau(sw, "strict_youden_j")
    assert r["tau"] == 0.62 and r["plateau"] == [0.61, 0.63] and r["value"] == 0.8


def test_tau_sweep_separates_labels():
    cos = np.array([0.80, 0.75, 0.70, 0.50, 0.45])
    ok = np.ones(5, bool)
    y = {"strict": np.array([1.0, 1.0, 1.0, 0.0, 0.0])}
    sw = pt.tau_sweep(cos, ok, None, y)
    best = pt.select_tau(sw, "strict_youden_j")
    assert best["value"] == 1.0 and 0.51 <= best["tau"] <= 0.70


def test_content_check_blocks_pass():
    sw = pt.tau_sweep(np.array([0.9]), np.array([False]), None, None)
    assert sw.pass_rate.max() == 0.0


def test_ab_order_is_stable_and_mixed():
    a = [pt.ab_order(f"stage2|{i}") for i in range(200)]
    assert a == [pt.ab_order(f"stage2|{i}") for i in range(200)]
    assert 60 < sum(a) < 140


def test_final_gold_rules():
    drafts = {"1": {"draft_text": "초안"}, "2": {"draft_text": "초안2"}, "3": {"draft_text": "초안3"}, "4": {}}
    s1 = {"1": {"verdict": "ACCEPT"}, "2": {"verdict": "edit", "final_text": " 고친 문장 "}, "3": {"verdict": "REJECT"}}
    assert pt.final_gold("1", s1, drafts, {"kind": "generated"}) == ("초안", "generated_accept")
    assert pt.final_gold("2", s1, drafts, {"kind": "generated"}) == ("고친 문장", "generated_edit")
    assert pt.final_gold("3", s1, drafts, {"kind": "generated"}) == (None, "generated_reject")
    assert pt.final_gold("4", s1, drafts, {"kind": "generated"}) == (None, "pending")
    assert pt.final_gold("4", s1, drafts, {"kind": "caption", "gold": "사이트 캡션"}) == ("사이트 캡션", "caption")


def test_parse_label():
    assert pt._parse_label('```json\n{"label": "부분일치", "reason": "x"}\n```')["label"] == "부분일치"
    assert pt._parse_label('{"label": "같음"}') is None
