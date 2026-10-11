"""시험 콘솔 — 원본 위치 표시 테스트 (docs/v4/M3_console_spec_addendum_highlight.md §1–§3, §5.4).

모델·API 호출 없음: Qwen 은 가짜 클라이언트(set_client_factory)로 바꿔 끼운다. 스캔 쪽은 실제 계약서 PDF 를 그림으로 그려
'이미지만 있는 PDF' 를 만들고, 가짜 Qwen 응답은 원본 텍스트 레이어의 줄 상자(0–1000)로 만든다. 텍스트 레이어 쪽은
dataset/new_contracts/ 의 계약서를 읽기만 한다. 엔드포인트는 디스크의 실제 T-001 실행을 임시 폴더로 복사해 쓴다(없으면 건너뜀).
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .test_m3_console import (CA, ROOT, Console, _httpx_testclient_compat, console, logged_in,  # noqa: F401
                              no_internal)
from .test_m3_console_logs import REAL_RUN, _copy_real_run, needs_real_run

from web import console_locate as CL                         # noqa: E402

CONTRACT = ROOT / "dataset/new_contracts/00003_13333800 F!RE 가사.pdf"
needs_contract = pytest.mark.skipif(not CONTRACT.is_file(), reason="신규 양식 계약서 PDF 없음")


def _record(**over) -> Dict[str, Any]:
    rec = {
        "set_id": "13333800", "test": "T-001", "work_title": "F!RE 가사", "file_name": "00003_13333800 F!RE 가사.pdf",
        "work_type": "어문", "creator": ["양지웅", "이영지", "신혜진"], "copyright_holder": ["한국저작권위원회"],
        "reproduction_right": "불허", "performance_right": "불허", "public_transmission_right": "허용",
        "exhibition_right": "불허", "distribution_right": "불허", "rental_right": "불허", "derivative_work_right": "허용",
        "other_rights": None, "license_start_date": "2007-03-22", "license_end_date": "2028-06-30",
        "contract_validity_period": "2007-03-22~2028-06-30", "keywords": None, "hashtags": None, "file_format": None,
        "file_size": None, "resolution": None, "description": None, "file_hash": None,
        "_meta": {"pipeline_meta": {"work_title_source": "rules", "work_type_source": "rules",
                                    "copyright_holder_source": "licensee_party",
                                    "licensee_rule": {"source": "signature"}}},
    }
    rec.update(over)
    return rec


DECISIONS = [
    {"field": "author", "final_value": '["양지웅", "이영지", "신혜진"]', "key23": "creator"},
    {"field": "payment_amount", "final_value": 266464758},
    {"field": "signature_date", "final_value": "2007-03-10"},
    {"field": "contact_info.address", "final_value": "서울특별시 중구 합성로 93"},
    {"field": "economic_rights", "final_value": '["공중송신권", "2 차적저작물작성권"]'},
    {"field": "special_terms", "final_value": '["이용료는 일시금으로 지급", ...]'},
    {"field": "parties", "final_value": '[{"name": "양지웅", ...}, ...]'},
    {"field": "payment_currency", "final_value": "원"},
]


def _locate(path, rec=None, decisions=DECISIONS, cache=None, stem="s", **kw):
    return CL.locate(path, rec or _record(), decisions, "T-001", cache, stem, key23_of=CA.field_key23,
                     label_of=CA.field_label_ko, **kw)


def _marks(out, target):
    return [m for m in out["marks"] if m["target"] == target]


# ---------------------------------------------------------------------------
# 정규화 · 표기 변형 · 근사 찾기
# ---------------------------------------------------------------------------
def test_norm_text_whitespace_nfkc_quotes():
    assert CL.norm_text("２００７ 년  ０３ 월​") == "2007년03월"
    assert CL.norm_text("F!RE  가사") == CL.norm_text("f!re가사") == "f!re가사"
    assert CL.norm_text("“권리자”") == '"권리자"'
    assert CL.norm_text("어문\n저작물") == "어문저작물"
    assert CL.norm_text(None) == ""


def test_value_variants():
    v = CL.date_variants("2007-03-22")
    assert v[0] == "2007-03-22"
    for f in ("2007년 3월 22일", "2007년 03월 22일", "2007. 3. 22.", "2007.03.22"):
        assert f in v
    assert CL.date_variants("2007년 3월 2일")[1] == "2007-03-02"
    assert CL.date_variants("다음 달") == ["다음 달"]
    assert CL.parse_date("2007-13-01") is None
    assert CL.amount_variants(266464758) == ["266,464,758원", "266,464,758", "266464758원", "266464758"]
    assert CL.amount_variants("266,464,758 원") == CL.amount_variants(266464758)
    assert CL.amount_variants(12) is None and CL.amount_variants(1.5) is None and CL.amount_variants("약 3만") is None
    assert CL.work_type_variants("어문") == ["어문", "어문저작물"]
    assert CL.work_type_variants("어문저작물") == ["어문저작물", "어문"]


def test_fuzzy_find_tolerance():
    assert CL.fuzzy_tolerance(3) == 1 and CL.fuzzy_tolerance(12) == 2 and CL.fuzzy_tolerance(18) == 3
    hay = CL.norm_text("저작자 : 양지용 (저작물 3 건)")
    hits = CL.fuzzy_find(hay, "양지웅")
    assert len(hits) == 1
    s, e, d = hits[0]
    assert hay[s:e] == "양지용" and d == 1
    assert CL.fuzzy_find(hay, "홍길동") == []
    long_hay = CL.norm_text("저작재산권 독점적 이용허락 계약서")
    s, e, d = CL.fuzzy_find(long_hay, CL.norm_text("저작권 독점적 이용허락 계약서"))[0]
    assert d == 2 and long_hay[s:e].endswith("계약서")
    # 숫자 경계 — 2007 이 20071 안에서 찾히지 않는다
    assert CL.find_all("20071년2007년", "2007") == [(6, 10)]


def test_norm_rect_and_clamp():
    assert CL.norm_rect((59.5, 84.2, 119, 168.4), 595, 842) == pytest.approx((0.1, 0.1, 0.2, 0.2))
    assert CL.norm_rect((119, 168.4, 59.5, 84.2), 595, 842) == pytest.approx((0.1, 0.1, 0.2, 0.2))   # 뒤집힌 상자
    assert CL.norm_rect((-10, -5, 700, 900), 595, 842) == (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        CL.norm_rect((0, 0, 1, 1), 0, 842)


def test_parse_ocr_lines_coords_rotate_rect_and_salvage():
    rel = json.dumps([{"text": "제목(제호) : 1 F!RE 가사", "bbox_2d": [100, 500, 400, 520]}], ensure_ascii=False)
    out = CL.parse_ocr_lines("```json\n" + rel + "\n```", 1653, 2339)
    assert out == [{"text": "제목(제호) : 1 F!RE 가사", "box": [0.1, 0.5, 0.4, 0.52]}]
    px = json.dumps([{"text": "가나다", "bbox_2d": [165.3, 233.9, 1653, 1169.5]}])        # 1000 을 넘으면 픽셀 좌표
    assert CL.parse_ocr_lines(px, 1653, 2339)[0]["box"] == pytest.approx([0.1, 0.1, 1.0, 0.5])
    # qwen-vl-ocr 의 rotate_rect — 각도 ~90° 면 가로·세로가 바뀌어 온다
    rr = json.dumps([{"text": "가로줄", "rotate_rect": [500, 500, 20, 300, 90]},
                     {"text": "그대로", "rotate_rect": [500, 600, 300, 20, 0]}])
    a, b = CL.parse_ocr_lines(rr, 1000, 1000)
    assert a["box"] == pytest.approx([0.35, 0.49, 0.65, 0.51]) and b["box"] == pytest.approx([0.35, 0.59, 0.65, 0.61])
    # 8점 다각형 · 잘린 응답(배열이 닫히지 않음)에서 온전한 객체만 건진다
    quad = '[{"text": "넷", "bbox_2d": [10, 10, 90, 12, 90, 30, 10, 28]}, {"text": "잘린'
    assert CL.parse_ocr_lines(quad, 2000, 2000)[0]["box"] == pytest.approx([0.01, 0.01, 0.09, 0.03])
    for bad in ("죄송합니다. 이미지를 읽을 수 없습니다.", "[]", '[{"text": "", "bbox_2d": [1, 2, 3, 4]}]'):
        with pytest.raises(CL.OcrParseError):
            CL.parse_ocr_lines(bad, 1000, 1000)


def test_ocr_stream_char_span_inside_line_box():
    st = CL.ocr_stream([{"text": "저작자 : 양지웅", "box": [0.1, 0.5, 0.5, 0.52]}])
    (s, e), = CL.find_all(st.text, "양지웅")
    (r,) = st.rects(s, e)
    assert r[1] == 0.5 and r[3] == 0.52
    assert 0.3 < r[0] < r[2] <= 0.5 + 1e-9                  # 줄의 뒤쪽 글자 구간
    assert st.raw_text(s, e) == "양지웅"


# ---------------------------------------------------------------------------
# 대상 만들기
# ---------------------------------------------------------------------------
def test_build_targets_kinds_aliases_and_reasons():
    rec = _record(file_size=1234, description="요약 문장", keywords=["가", "나"])
    ts = {t.target: t for t in CL.build_targets(rec, DECISIONS, "T-001", key23_of=CA.field_key23)}
    assert "k:other_rights" not in ts                       # 항상 NULL — 대상 아님
    assert ts["k:file_name"].skip_reason == CL.R_OUTSIDE and ts["k:file_size"].skip_reason == CL.R_OUTSIDE
    assert ts["k:description"].skip_reason == CL.R_GENERATED and ts["k:keywords"].skip_reason == CL.R_GENERATED
    assert ts["k:creator"].values == ["양지웅", "이영지", "신혜진"] and ts["k:creator"].kind == "name"
    assert ts["k:copyright_holder"].primary == {"mode": "last"}             # 서명란 규칙 → 마지막 자리
    assert ts["k:work_title"].primary["mode"] == "anchor"
    assert ts["k:reproduction_right"].kind == "right"
    assert ts["k:contract_validity_period"].kind == "period"
    assert ts["f:author"].alias == "k:creator"                              # 23키로 이어지는 통합 필드
    assert ts["f:payment_amount"].kind == "amount" and ts["f:signature_date"].kind == "date"
    assert ts["f:economic_rights"].kind == "right_names"
    assert ts["f:special_terms"].skip_reason == CL.R_GENERATED
    assert ts["f:parties"].skip_reason == CL.R_GENERATED                    # 중재자가 줄여 적은 값 ('...')
    assert ts["f:payment_currency"].skip_reason == CL.R_NORMALISED          # 한 글자 값
    # T-001 키가 비어 있으면(T-002 레코드 등) 대상이 아니다
    t2 = CL.build_targets({"test": "T-002", "work_title": "제목", "file_name": "a.jpg", "file_hash": "ab" * 32},
                          [], "T-002")
    assert [t.target for t in t2] == ["k:work_title", "k:file_name", "k:file_hash"]


# ---------------------------------------------------------------------------
# 텍스트 레이어 (실제 계약서)
# ---------------------------------------------------------------------------
@needs_contract
def test_text_layer_locate_contract():
    rec = _record(creator=["양지웅", "이영지", "신혜진", "홍길동"], license_end_date="2099-01-01",
                  description="계약 요약")
    out = _locate(CONTRACT, rec)
    assert out["available"] is True and out["reason"] is None
    assert [g["source"] for g in out["pages"]] == ["text_layer"] * 5
    assert out["pages"][0]["width"] == 595.0 and out["engine"]["ocr"] == "none" and out["engine"]["cost_krw"] == 0.0
    for m in out["marks"]:
        assert m["source"] == "text_layer" and m["match"] in ("exact", "variant")
        for r in m["rects"]:
            assert 0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1
        assert set(m) >= {"id", "target", "key", "no", "field", "label_ko", "value", "page", "rects", "match", "primary",
                          "source"}
    title = _marks(out, "k:work_title")
    pri = [m for m in title if m["primary"]]
    assert len(title) == 2 and len(pri) == 1 and pri[0]["page"] == 1 and pri[0]["rects"][0][1] > 0.45   # 제목(제호) 줄
    creators = _marks(out, "k:creator")
    assert {m["value"] for m in creators} == {"양지웅", "이영지", "신혜진"}
    assert sum(m["primary"] for m in creators) == 3                         # 이름마다 하나 — 제2조 저작자 줄
    assert all(m["page"] == 1 and 0.5 < m["rects"][0][1] < 0.6 for m in creators if m["primary"])
    holder = [m for m in _marks(out, "k:copyright_holder") if m["primary"]]
    assert holder[0]["page"] == 5                                           # 서명란
    wt, = _marks(out, "k:work_type")
    assert wt["text"] == "■어문저작물" and wt["match"] == "variant" and wt["primary"]
    rights = {m["key"]: m for m in out["marks"] if m["key"] in CL.RIGHT_LABELS}
    assert set(rights) == set(CL.RIGHT_LABELS)
    assert rights["reproduction_right"]["text"] == "□복제권"
    assert rights["public_transmission_right"]["text"].startswith("■공중송신권(")
    assert rights["derivative_work_right"]["text"] == "■2차적저작물1)작성권"           # 각주 표시를 건너뛴다
    assert len(rights["exhibition_right"]["rects"]) == 2                     # '□' 와 '전시권' 이 두 줄에 걸친다
    start = [m for m in _marks(out, "k:license_start_date") if m["primary"]]
    assert start[0]["text"] == "2007년 03월 22일" and start[0]["match"] == "variant" and start[0]["page"] == 1
    period, = _marks(out, "k:contract_validity_period")
    assert period["text"] == "2007년 03월 22일부터 2028년 06월 30일"
    amount = _marks(out, "f:payment_amount")
    assert amount and all(m["page"] == 2 and m["text"] == "266,464,758원" for m in amount)
    assert _marks(out, "f:signature_date")[0]["text"] == "2007년 3월 10일"
    assert len(_marks(out, "f:economic_rights")) == 2
    nf = {(n["target"], n["value"]): n["reason"] for n in out["not_found"]}
    assert nf[("k:creator", "홍길동")] == CL.R_NOT_FOUND
    assert nf[("k:license_end_date", "2099-01-01")] == CL.R_NORMALISED
    assert nf[("k:file_name", "00003_13333800 F!RE 가사.pdf")] == CL.R_OUTSIDE
    assert nf[("k:description", "계약 요약")] == CL.R_GENERATED
    tg = {t["target"]: t for t in out["targets"]}
    assert tg["f:author"] == {"target": "f:author", "key": None, "no": None, "field": "author",
                              "label_ko": tg["f:author"]["label_ko"], "alias": "k:creator"}
    assert tg["k:creator"]["primary"]["page"] == 1 and tg["k:creator"]["pages"] == [1, 5]
    assert tg["k:file_name"]["reason"] == CL.R_OUTSIDE and tg["k:file_name"]["marks"] == 0


def test_rotated_page_rects_follow_the_rendered_page(tmp_path):
    import fitz
    import numpy as np
    doc = fitz.open()
    pg = doc.new_page(width=595, height=842)
    pg.insert_text((100, 100), "TITLE LINE HELLO WORLD 2024", fontsize=18)
    pg.insert_text((100, 300), "Author : KIMCHULSU and other text to pass thirty chars", fontsize=12)
    pg.set_rotation(90)
    p = tmp_path / "rot.pdf"
    doc.save(str(p))
    out = CL.locate(p, {"test": "T-001", "work_title": "HELLO WORLD"}, [], "T-001", None, "r")
    m, = _marks(out, "k:work_title")
    with fitz.open(str(p)) as d:
        pix = d[0].get_pixmap(dpi=72)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        x0, y0, x1, y1 = (int(round(v)) for v in (m["rects"][0][0] * pix.width, m["rects"][0][1] * pix.height,
                                                   m["rects"][0][2] * pix.width, m["rects"][0][3] * pix.height))
        assert out["pages"][0]["width"] == 842 and out["pages"][0]["height"] == 595     # 회전한 쪽 크기
        assert (a[y0:y1, x0:x1, 0] < 128).sum() > 50                       # 표시 상자 안에 글자 잉크가 있다
        assert x1 - x0 < y1 - y0                                            # 90° 회전 — 세로로 긴 상자


# ---------------------------------------------------------------------------
# 스캔 쪽 — 가짜 Qwen
# ---------------------------------------------------------------------------
def _scan_pdf(src: Path, dst: Path, pages=(0,), dpi=100) -> List[List[Dict[str, Any]]]:
    """원본 쪽을 그림으로 그린 '이미지만 있는 PDF' + 가짜 OCR 이 돌려줄 줄 목록(0–1000 좌표)."""
    import fitz
    lines_per_page = []
    with fitz.open(str(src)) as d:
        out = fitz.open()
        for i in pages:
            pg = d[i]
            pix = pg.get_pixmap(dpi=dpi)
            np_ = out.new_page(width=pg.rect.width, height=pg.rect.height)
            np_.insert_image(np_.rect, stream=pix.tobytes("png"))
            W, H = pg.rect.width, pg.rect.height
            lines = []
            for b in pg.get_text("rawdict")["blocks"]:
                for ln in b.get("lines", []):
                    t = "".join(c["c"] for sp in ln["spans"] for c in sp["chars"])
                    x0, y0, x1, y1 = ln["bbox"]
                    if t.strip():
                        lines.append({"text": t, "bbox_2d": [round(x0 / W * 1000), round(y0 / H * 1000),
                                                             round(x1 / W * 1000), round(y1 / H * 1000)]})
            lines_per_page.append(lines)
        out.save(str(dst))
    return lines_per_page


class FakeQwen:
    """chat.completions.create 대역 — 모델마다 정한 행동(lines | raise | junk)."""

    def __init__(self, lines_by_call, behaviour=None):
        self.lines_by_call = lines_by_call
        self.behaviour = behaviour or {}
        self.calls: List[Dict[str, Any]] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        how = self.behaviour.get(kw["model"], "lines")
        if how == "raise":
            raise RuntimeError("APIConnectionError: 연결 끊김 key=sk-ws-abcdefghijklmnop123456")
        content = "죄송합니다 — 줄을 찾지 못했습니다" if how == "junk" else json.dumps(self.lines_by_call(), ensure_ascii=False)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
                                     usage=types.SimpleNamespace(prompt_tokens=1375, completion_tokens=1500))


@pytest.fixture
def fake_qwen(monkeypatch):
    holder = {}

    def install(lines_by_call, behaviour=None):
        f = FakeQwen(lines_by_call, behaviour)
        holder["f"] = f
        CL.set_client_factory(lambda: f)
        return f
    monkeypatch.delenv("CONSOLE_LOCATE_OCR", raising=False)
    monkeypatch.delenv("CONSOLE_LOCATE_MODELS", raising=False)
    yield install
    CL.set_client_factory(None)


@needs_contract
def test_scan_page_qwen_positions_fuzzy_and_cache(tmp_path, fake_qwen, caplog):
    scan = tmp_path / "13333800_scan.pdf"
    lines, = _scan_pdf(CONTRACT, scan)
    misread = [dict(L, text=L["text"].replace("이영지", "이영자")) for L in lines]   # OCR 오독 → 근사 일치
    fq = fake_qwen(lambda: misread)
    caplog.set_level(logging.INFO, logger="web.console_locate")
    cache = tmp_path / "loc"
    gt = _locate(CONTRACT)
    one_page = _record()
    out = _locate(scan, one_page, cache=cache, stem="13333800")
    assert out["available"] is True and len(fq.calls) == 1
    assert fq.calls[0]["model"] == "qwen3.8-omni-flash" and "extra_body" not in fq.calls[0]
    assert fq.calls[0]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    pg, = out["pages"]
    assert pg["source"] == "ocr_qwen" and pg["model"] == "qwen3.8-omni-flash" and pg["lines"] == len(lines)
    e = out["engine"]
    assert e["ocr"] == "qwen" and e["model"] == "qwen3.8-omni-flash" and e["pages_ocr"] == 1
    assert e["cost_krw"] == pytest.approx((1375 * 0.15 + 1500 * 0.47) / 1e6 * 1400, abs=0.01)
    assert all(m["source"] == "ocr_qwen" for m in out["marks"])

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u else 0
    for tgt in ("k:work_title", "k:work_type", "k:reproduction_right", "k:license_start_date"):
        g = next(m for m in _marks(gt, tgt) if m["primary"])
        o = next(m for m in _marks(out, tgt) if m["primary"])
        assert o["page"] == g["page"] == 1 and iou(o["rects"][0], g["rects"][0]) > 0.5, tgt
    fz = [m for m in _marks(out, "k:creator") if m["value"] == "이영지"]
    assert fz and all(m["match"] == "fuzzy" and "이영자" in m["text"] for m in fz)
    msgs = [r.getMessage() for r in caplog.records if r.name == "web.console_locate"]
    assert any(m.startswith("원본 위치 — 스캔 1쪽 qwen3.8-omni-flash (₩") for m in msgs)
    assert any(m.startswith("원본 위치 — 스캔 1쪽(OCR 좌표) · 표시 ") for m in msgs)
    assert {getattr(r, "component", None) for r in caplog.records if r.name == "web.console_locate"} <= {"OCR", "측정"}

    # 캐시 — 같은 요청은 다시 계산하지 않고, refresh=1 은 표시만 다시(OCR 줄 재사용), refresh=ocr 은 OCR 도 다시
    assert (cache / "13333800.json").is_file() and (cache / "13333800.ocr.json").is_file()
    again = _locate(scan, one_page, cache=cache, stem="13333800")
    assert again["engine"]["cached"] is True and len(fq.calls) == 1 and again["marks"] == out["marks"]
    re1 = _locate(scan, one_page, cache=cache, stem="13333800", refresh="1")
    assert re1["engine"]["cached"] is False and len(fq.calls) == 1 and re1["engine"]["pages_ocr_fresh"] == 0
    re2 = _locate(scan, one_page, cache=cache, stem="13333800", refresh="ocr")
    assert len(fq.calls) == 2 and re2["engine"]["pages_ocr_fresh"] == 1
    # 레코드 값이 바뀌면 (대상 서명) 캐시를 쓰지 않는다
    changed = _locate(scan, _record(work_title="다른 제목"), cache=cache, stem="13333800")
    assert changed["engine"]["cached"] is False and not _marks(changed, "k:work_title")


@needs_contract
def test_scan_qwen_fallback_on_error_and_bad_json(tmp_path, fake_qwen, caplog, monkeypatch):
    scan = tmp_path / "s.pdf"
    lines, = _scan_pdf(CONTRACT, scan)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ws-abcdefghijklmnop123456")
    caplog.set_level(logging.INFO, logger="web.console_locate")
    fq = fake_qwen(lambda: lines, {"qwen3.8-omni-flash": "raise"})
    out = _locate(scan)
    assert [c["model"] for c in fq.calls] == ["qwen3.8-omni-flash", "qwen3.6-flash"]
    assert fq.calls[1]["extra_body"] == {"enable_thinking": False}             # 3.6-flash 는 생각 끔
    assert out["available"] and out["pages"][0]["model"] == "qwen3.6-flash" and out["engine"]["model"] == "qwen3.6-flash"
    warn = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "web.console_locate"]
    assert any("원본 위치 OCR 실패 — 1쪽 qwen3.8-omni-flash" in r.getMessage() and "qwen3.6-flash 로 다시" in r.getMessage()
               for r in warn)
    assert all("abcdefghijklmnop" not in r.getMessage() for r in caplog.records)     # 비밀값 가림
    # JSON 이 아닌 응답 → 다음 모델
    fq2 = fake_qwen(lambda: lines, {"qwen3.8-omni-flash": "junk"})
    out2 = _locate(scan)
    assert [c["model"] for c in fq2.calls] == ["qwen3.8-omni-flash", "qwen3.6-flash"] and out2["available"]
    assert any("응답을 해석하지 못했습니다" in r.getMessage() for r in caplog.records)
    # 모델 목록은 환경 변수로
    monkeypatch.setenv("CONSOLE_LOCATE_MODELS", "qwen3.6-flash")
    fq3 = fake_qwen(lambda: lines)
    _locate(scan)
    assert [c["model"] for c in fq3.calls] == ["qwen3.6-flash"]


@needs_contract
def test_scan_all_models_fail_then_retry_later(tmp_path, fake_qwen, monkeypatch):
    scan = tmp_path / "s.pdf"
    lines, = _scan_pdf(CONTRACT, scan)
    fq = fake_qwen(lambda: lines, {"qwen3.8-omni-flash": "raise", "qwen3.6-flash": "raise"})
    cache = tmp_path / "loc"
    out = _locate(scan, cache=cache, stem="x")
    assert out["available"] is False and "실패" in out["reason"] and out["marks"] == []
    assert out["pages"][0]["source"] == "none" and "실패" in out["pages"][0]["note"]
    assert len(fq.calls) == 2
    _locate(scan, cache=cache, stem="x")
    assert len(fq.calls) == 2                                                   # 실패도 잠시 캐시
    monkeypatch.setattr(CL, "FAIL_RETRY_SEC", 0.0)
    fake_qwen(lambda: lines)
    ok = _locate(scan, cache=cache, stem="x")
    assert ok["available"] is True and ok["pages"][0]["source"] == "ocr_qwen"


@needs_contract
def test_ocr_off_and_missing_key(tmp_path, fake_qwen, monkeypatch):
    scan = tmp_path / "s.pdf"
    lines, = _scan_pdf(CONTRACT, scan)
    fq = fake_qwen(lambda: lines)
    monkeypatch.setenv("CONSOLE_LOCATE_OCR", "off")
    out = _locate(scan)
    assert out["available"] is False and "CONSOLE_LOCATE_OCR=off" in out["reason"]
    assert out["engine"]["ocr"] == "off" and out["pages"][0]["source"] == "none" and "꺼져" in out["pages"][0]["note"]
    assert fq.calls == []
    # 텍스트 레이어 쪽은 OCR 이 꺼져 있어도 그대로
    txt = _locate(CONTRACT)
    assert txt["available"] is True and txt["engine"]["ocr"] == "none"
    # 키가 없으면 — 호출 없이 위치 정보 없음
    monkeypatch.delenv("CONSOLE_LOCATE_OCR")
    CL.set_client_factory(None)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: pytest.fail("키 없이 클라이언트를 만들면 안 된다"))
    out = _locate(scan)
    assert out["available"] is False and "DASHSCOPE_API_KEY" in json.dumps(out["pages"], ensure_ascii=False)


def test_image_contract_uses_raw_orientation(tmp_path, fake_qwen):
    from PIL import Image
    p = tmp_path / "contract.png"
    Image.new("RGB", (800, 1200), "white").save(p)
    fq = fake_qwen(lambda: [{"text": "저작자 : 홍길동", "bbox_2d": [100, 200, 500, 230]}])
    out = CL.locate(p, {"test": "T-001", "creator": ["홍길동"]}, [], "T-001", None, "img")
    assert out["pages"] == [{"page": 1, "width": 800.0, "height": 1200.0, "source": "ocr_qwen",
                             "model": "qwen3.8-omni-flash", "lines": 1}]
    m, = out["marks"]
    assert m["rects"][0][1] == pytest.approx(0.2) and 0.1 < m["rects"][0][0] < m["rects"][0][2] <= 0.5 + 1e-9
    assert len(fq.calls) == 1


def test_unavailable_contracts(tmp_path):
    rec = _record()
    assert CL.locate(None, rec, [], "T-001", None, "a")["reason"] == "계약서 파일이 없습니다"
    assert CL.locate(tmp_path / "없음.pdf", rec, [], "T-001", None, "a")["reason"] == "계약서 원본 파일을 찾을 수 없습니다"
    d = tmp_path / "c.docx"
    d.write_bytes(b"PK\x03\x04")
    out = CL.locate(d, rec, [], "T-001", None, "a")
    assert out["available"] is False and "DOCX·HWP" in out["reason"] and out["marks"] == []
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    assert CL.locate(bad, rec, [], "T-001", None, "a")["available"] is False


# ---------------------------------------------------------------------------
# 엔드포인트 (로그인 · 실제 실행 복사본)
# ---------------------------------------------------------------------------
@needs_real_run
def test_locate_endpoint_auth_payload_cache(console, fake_qwen):
    dst = _copy_real_run(console.runs)
    shutil.rmtree(dst / "locate", ignore_errors=True)
    fq = fake_qwen(lambda: [])
    url = f"/api/test/runs/{REAL_RUN.name}/sets/13333800/locate"
    console.set_password()
    anon = console.client()
    assert anon.get(url).status_code == 401
    c = logged_in(console)
    r = c.get(url)
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"available", "reason", "pages", "marks", "not_found", "targets", "engine"}
    assert d["available"] is True and d["engine"]["cached"] is False and fq.calls == []
    assert len(d["pages"]) == 5 and all(g["source"] == "text_layer" for g in d["pages"])
    keys_found = {m["key"] for m in d["marks"] if m["target"].startswith("k:")}
    assert keys_found >= {"work_title", "work_type", "creator", "copyright_holder", "reproduction_right",
                          "performance_right", "public_transmission_right", "exhibition_right", "distribution_right",
                          "rental_right", "derivative_work_right", "license_start_date", "license_end_date",
                          "contract_validity_period"}
    assert {(n["target"], n["reason"]) for n in d["not_found"]} >= {("k:file_name", CL.R_OUTSIDE)}
    assert any(t.get("alias") == "k:creator" for t in d["targets"])
    no_internal(d)
    assert (dst / "locate" / "13333800.json").is_file()
    assert c.get(url).json()["engine"]["cached"] is True
    assert c.get(url + "?refresh=1").json()["engine"]["cached"] is False
    assert c.get(f"/api/test/runs/{REAL_RUN.name}/sets/99999999/locate").status_code == 404
    assert c.get(f"/api/test/runs/없는실행/sets/13333800/locate").status_code == 404
    # 계약서 원본이 사라진 세트 → available false (오류 아님)
    rec_p = dst / "out" / "test_predictions" / "13387752.json"
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    rec["_meta"]["files"]["contract"] = str(dst / "없는파일.pdf")
    rec_p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    gone = c.get(f"/api/test/runs/{REAL_RUN.name}/sets/13387752/locate").json()
    assert gone["available"] is False and gone["reason"] == "계약서 원본 파일을 찾을 수 없습니다"


@needs_real_run
def test_locate_endpoint_survives_internal_error(console, monkeypatch):
    _copy_real_run(console.runs)
    c = logged_in(console)

    def boom(*a, **k):
        raise RuntimeError("계산 실패 sk-ws-abcdefghijklmnop123456")
    monkeypatch.setattr(CL, "locate_set", boom)
    r = c.get(f"/api/test/runs/{REAL_RUN.name}/sets/13333800/locate")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is False and d["reason"] == "위치 계산 중 오류 — RuntimeError" and d["marks"] == []
    assert "abcdefghijklmnop" not in r.text
