"""계약서 생성기 v5 (dataset_builder/generate_contracts_v5) — 값 정리 규칙, 템플릿 채움, 원장→Gold 일치."""
from __future__ import annotations

import re
import zipfile

import pytest

from api.module.dataset_builder import generate_contracts_v5 as G
from api.module.dataset_builder import hwpx_kit as K
from api.module.evaluation.v4 import gold23


def _sheet_row(**over):
    row = {"제목": "The Great Train Robbery 1903", "저작자": "한국저작권위원회", "권리자": "에드윈 S. 포터", "이용자": "한국저작권위원회",
           "권리자_생년월일": "19700101", "권리자_주소": "서울특별시 중구 합성로 12",
           "이용자_생년월일": "436-66-70705", "이용자_주소": "대구광역시 중구 가상로 188",
           "이용허락기간_SY": 2020, "이용허락기간_SM": "11", "이용허락기간_SD": "24",
           "이용허락기간_EY": 2023, "이용허락기간_EM": "01", "이용허락기간_ED": "21", "Ls_1": "■", "Ls_2": "■"}
    for i in range(1, 11):
        row[f"C_{i}"] = "■" if i == 7 else "□"
    for c, _ in G.RIGHTS + G.SUB_RIGHTS:
        row[c] = "□"
    row["R_1"] = row["R_5"] = "■"
    row.update(over)
    return row


SEL = {"work_id": "13222347", "media": "video", "license_bucket": "expired", "cell": "video/expired",
       "work_file": "/x/13222347_1_a.wmv", "in_te1": "True", "in_te2": "True"}


def test_clean_names_strips_annotation_and_joins_lines():
    assert G.clean_names("신애리 (저작물 197 건)\n강은서") == "신애리, 강은서"
    assert G.clean_names("김홍도(金弘道)") == "김홍도(金弘道)"


def test_expired_video_swaps_creator_and_holder_and_replaces_licensee():
    v = G.values_from_sheet(SEL, _sheet_row())
    assert (v["저작자"], v["권리자"]) == ("에드윈 S. 포터", G.KCC)
    assert v["권리자_번호"] == "436-66-70705"           # 위원회 번호는 시트의 이용자 칸에서 옮겨 온다
    assert v["이용자"] not in (G.KCC, v["저작자"])        # 권리자와 같았던 이용자는 새로 만든다
    assert "swap_creator_holder" in v["fixes"] and "new_licensee" in v["fixes"]
    assert "합성로" not in v["권리자_주소"] + v["이용자_주소"] and "가상로" not in v["권리자_주소"] + v["이용자_주소"]
    assert v["contract_date"] < v["start"]


def test_values_are_deterministic():
    assert G.values_from_sheet(SEL, _sheet_row()) == G.values_from_sheet(SEL, _sheet_row())


def test_korean_amount():
    assert G.korean_amount(1_250_000) == "일백이십오만"
    assert G.korean_amount(30_000_000) == "삼천만"
    assert G.korean_amount(2_020_000) == "이백이만"


@pytest.mark.parametrize("tpl", ["A", "B", "C"])
def test_templates_fill_every_value(tpl, tmp_path):
    v = G.values_from_sheet(SEL, _sheet_row())
    v.update(template=tpl, pdf_name="13222347 The Great Train Robbery 1903.pdf")
    parts = G.Templates().render(v)
    out = tmp_path / "c.hwpx"
    K.write_parts(parts, out)
    with zipfile.ZipFile(out) as z:
        assert z.namelist()[0] == "mimetype" and z.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        sec = z.read("Contents/section0.xml").decode("utf-8")
        header = z.read("Contents/header.xml").decode("utf-8")
    assert "{{" not in sec
    assert "linesegarray" not in sec.split("</hp:p>", 1)[1]   # 첫 문단(쪽 설정) 뒤에는 저장된 줄 배치가 없다
    text = re.sub(r"<[^>]+>", "", sec)
    for needle in (v["제목"], v["저작자"], v["권리자"], v["이용자"]):
        assert needle in text
    assert 'textColor="#FF0000"' not in header
    if tpl == "A":
        assert "“권리자”라 함" in text and "이라 함" not in text and "사업자등록번호" in text
        assert f"{v['fee']:,}" in text


def test_ledger_row_gives_gold_that_matches_printed_values():
    v = G.values_from_sheet(SEL, _sheet_row())
    v["pdf_name"] = "13222347 The Great Train Robbery 1903.pdf"
    keys = gold23.ledger_keys(G.ledger_row(v))
    assert keys["creator"]["value"] == ["에드윈 S. 포터"]
    assert keys["copyright_holder"]["value"] == [G.KCC]
    assert keys["work_type"]["value"] == "영상"
    assert keys["license_start_date"]["value"] == "2020-11-24"
    assert keys["contract_validity_period"]["value"] == "2020-11-24~2023-01-21"
    assert keys["reproduction_right"]["value"] == "허용" and keys["performance_right"]["value"] == "불허"


def test_assign_templates_is_balanced_per_cell():
    rows = [{"work_id": str(i), "cell": "image/ccl"} for i in range(400)] + \
           [{"work_id": str(1000 + i), "cell": "text/kogl"} for i in range(50)]
    t = G.assign_templates(rows)
    from collections import Counter
    c = Counter(t[r["work_id"]] for r in rows if r["cell"] == "image/ccl")
    assert c == {"A": 200, "B": 100, "C": 100}


def test_check_review_flags_rule_breaks():
    item = {"task": "write_description_and_hashtags"}
    ok = {"description": "철조망에 색색의 리본이 빽빽하게 매달려 있다. 파란 하늘 아래 리본이 화면을 채운다.",
          "hashtags": ["#철조망", "#리본", "#울타리", "#파란하늘", "#야외"]}
    assert G.check_review(ok, item) == []
    bad = {"description": "이 이미지는 아름다운 풍경입니다", "hashtags": ["#사진", "#풍경"]}
    probs = G.check_review(bad, item)
    assert any("종결" in p for p in probs) and any("금지 해시태그" in p for p in probs) and any("해시태그 2개" in p for p in probs)
