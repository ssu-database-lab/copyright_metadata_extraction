"""통합검증(중재자) 견고성 — 2026-10-19/20 부하 시험에서 본 실패 4가지 (API 호출 없음).

1. 응답이 8,192 토큰 상한에 닿아 잘림 → 값이 있는 필드만 보내고 받는다 (프롬프트)
2. 빈 응답·래퍼 없는 응답이 성공으로 저장 → 비면 실패로 보고 폴백
3. 중재자가 빼먹은 필드가 사라짐 → LLM 값을 지키고, 근거를 든 REMOVED 만 비운다
4. decisions 에 글자만 있는 항목 → 'str' object has no attribute 'copy' 로 통합 전체 실패
"""
from __future__ import annotations

import os
import sys

_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from module.consolidator import consolidation_agent as CA          # noqa: E402
from module.consolidator.reasoning_generator import ReasoningGenerator   # noqa: E402

LLM = {"work_title": "운수 자판기", "author": "김은혜", "copyright_holder": "한국저작권위원회",
       "personal_info": "010-0000-0000", "license_start": "2024-01-01", "description": None, "keyword": []}


def _agent():
    a = CA.ConsolidationAgent.__new__(CA.ConsolidationAgent)
    a.reasoner = ReasoningGenerator()
    a.model_name = "alibaba-qwen3.5-122b-a10b"
    return a


def test_prompt_sends_only_fields_with_values():
    p = _agent()._create_consolidation_prompt(LLM, [("김은혜", "PERSON")], {"author": [("김은혜", "PERSON")], "x": []},
                                             "본문", "계약서")
    assert '"work_title": "운수 자판기"' in p and '"description"' not in p and '"keyword"' not in p
    assert "only the 5 fields" in p and "Never list empty fields" in p and '"x"' not in p
    assert "Include ALL fields" not in p


def test_merge_keeps_llm_values_and_only_removes_with_reason():
    res = {"consolidated_metadata": {"work_title": "운수 자판기 (개정)", "author": None, "invented_key": "x",
                                     "license_end": "2027-12-31"},
           "decisions": [{"field": "work_title", "decision": "CONFLICT"},
                         {"field": "personal_info", "decision": "REMOVED", "final_value": None}]}
    m = CA.ConsolidationAgent._merge_with_llm(LLM, res, {"license_end": [("2027-12-31", "DATE")]})
    md = m["consolidated_metadata"]
    assert md["work_title"] == "운수 자판기 (개정)"                     # 중재자 판정 반영
    assert md["author"] == "김은혜" and md["copyright_holder"] == "한국저작권위원회"   # 빠뜨린 필드는 LLM 값 유지
    assert md["personal_info"] is None and m["removed_by_arbiter"] == ["personal_info"]
    assert md["license_end"] == "2027-12-31"                           # NER 대응 필드는 새로 받아들인다
    assert "invented_key" not in md                                    # 지어낸 키는 버린다
    assert set(m["kept_from_llm"]) == {"author", "copyright_holder", "license_start"}
    diag = CA._consolidation_diagnostics({"metadata": LLM}, m)
    assert diag["consolidation_lost_fields"] == [] and diag["consolidation_removed_fields"] == ["personal_info"]


def test_empty_or_missing_result_is_not_usable():
    U = CA.ConsolidationAgent._usable
    assert not U(None, LLM) and not U({}, LLM)
    assert not U({"consolidated_metadata": {}}, LLM)                   # LLM 에 값이 있는데 비었다 → 폴백
    assert U({"consolidated_metadata": {"work_title": "운수 자판기"}}, LLM)
    assert U({"consolidated_metadata": {}}, {"work_title": None})      # 원래 비어 있었으면 빈 결과도 정상


def test_post_process_survives_text_decisions_and_fills_evidence():
    res = {"consolidated_metadata": dict(LLM), "llm_confidence": 0.9,
           "decisions": ["work_title 은 일치", {"field": "author", "llm_value": "김은혜", "ner_value": "김은혜",
                                                "final_value": "김은혜", "decision": "AGREED", "confidence": 0.95}]}
    out = _agent()._post_process(res, {"metadata": LLM}, {"extracted_entities": []}, {}, [], ocr_text="저작자 : 김은혜")
    ds = out["validation_report"]["decisions"]
    assert len(ds) == 1 and ds[0]["field"] == "author"
    assert ds[0]["evidence"].get("reasoning") and ds[0]["evidence"].get("ocr_excerpt")   # 예전에는 늘 {}
    assert CA._norm_decisions({"author": {"decision": "AGREED"}, "x": "y"}) == [
        {"field": "author", "decision": "AGREED"}, {"field": "x", "final_value": "y"}]
    assert CA._norm_decisions("문자열") == []


def test_thinking_switch(monkeypatch):
    from module.llm_extraction.models import cloud_extractor as CE
    from module.evaluation import batch_runner as BR
    for mod in (CA, CE):
        monkeypatch.delenv("CONSOLIDATION_THINKING", raising=False)
        assert mod._thinking_kwargs("CONSOLIDATION_THINKING") == {"extra_body": {"enable_thinking": False}}   # 기본 끔 (2026-10-20)
        monkeypatch.setenv("CONSOLIDATION_THINKING", "on")
        assert mod._thinking_kwargs("CONSOLIDATION_THINKING") == {}
    # 비용 추정도 생각 설정을 따른다 (텍스트층 계약서 · 중재 포함)
    resp = {"ocr_provider_used": "direct_text", "consolidate": True}
    monkeypatch.delenv("CONSOLIDATION_THINKING", raising=False); monkeypatch.delenv("EXTRACTION_THINKING", raising=False)
    assert BR._estimate_cost(resp, "image", True) == BR.COST_LLM_EXTRACT_NO_THINK + BR.COST_CONSOLIDATION_DOC_NO_THINK
    monkeypatch.setenv("CONSOLIDATION_THINKING", "on"); monkeypatch.setenv("EXTRACTION_THINKING", "on")
    assert BR._estimate_cost(resp, "image", True) == BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC


def test_consolidation_timeout(monkeypatch):
    monkeypatch.delenv("CONSOLIDATION_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("CONSOLIDATION_THINKING", raising=False)
    assert CA.consolidation_timeout_sec() == 90.0                 # 생각 끔: 멈춘 호출을 90초에 끊고 다시
    assert CA.consolidation_timeout_sec(fallback=True) == 240.0   # 폴백(qwen3.5-plus)은 끔에서 재 보지 않았다
    monkeypatch.setenv("CONSOLIDATION_THINKING", "on")
    assert CA.consolidation_timeout_sec() == 240.0
    monkeypatch.setenv("CONSOLIDATION_TIMEOUT_SEC", "120")
    assert CA.consolidation_timeout_sec() == 120.0 and CA.consolidation_timeout_sec(fallback=True) == 120.0
