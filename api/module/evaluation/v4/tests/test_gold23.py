"""gold23 테스트 (M1 spec §8(e)) — 합성 10행 인덱스 + 원장 + 초안으로 Gold 상태(value/empty/na)와 레코드 형식을 검증한다.

실데이터·네트워크 접근 없음: 모든 입력은 tmp_path 에 실제 파일과 같은 열 이름으로 만든다.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from .. import gold23 as g
from ..keys23 import KEY_NAMES, T001_KEYS, T002_KEYS

INDEX_COLUMNS = [
    "source", "work_id", "media", "title", "license_bucket", "work_file", "work_저자", "work_설명", "work_설명_tier",
    "work_키워드", "work_키워드_tier", "work_키워드_batch_level", "work_키워드_내용태그", "work_해시태그", "work_해시태그_tier",
    "work_해상도", "work_해상도_tier", "work_파일크기", "work_파일크기_tier", "work_파일포맷", "work_파일포맷_tier",
]
IDS_COLUMNS = ["test", "cell", "sub_source", "work_id", "media", "license_bucket", "source", "title", "work_file",
               "has_contract", "contract_pdf", "work_설명_tier", "work_해시태그_tier", "work_해상도_tier"]
LEDGER_HEADER = [
    "공유형태", "저작물형태", "파일명", "권리자", "이용자", "저작권자명", "제목", "저작자",
    *[f"C_{i}" for i in range(1, 11)], "C_기타", "R_1", "R_2", "R_3", "R_3_1", "R_3_2", "R_3_3", "R_4", "R_5", "R_6", "R_7",
    "이용허락기간_SY", "이용허락기간_SM", "이용허락기간_SD", "이용허락기간_EY", "이용허락기간_EM", "이용허락기간_ED",
    "양도_SY", "양도_SM", "양도_SD", "Ls_1", "Ls_2", "Ls_3", "정기지급액", "Po_1", "Po_2", "Po_3", "분할1_MO", "분할2_MO",
    "분할3_MO", "정기지급_M", "정기지급_B", "정기지급_Y", "지급시기_ETC", "권리자_생년월일", "권리자_주소", "이용자_생년월일",
    "이용자_주소", "NEW_COL",  # 모르는 열 하나 — 로그만 남고 무시돼야 한다
]
LABEL = "컬렉션 라벨 텍스트"


def _row(work_id: str, media: str, bucket: str, source: str, **over: str) -> dict[str, str]:
    base = {c: "" for c in INDEX_COLUMNS}
    base.update({"source": source, "work_id": work_id, "media": media, "license_bucket": bucket,
                 "title": f"제목 {work_id}", "work_file": f"/nonexistent/{work_id}.bin",
                 "work_키워드_batch_level": "False", "work_파일크기": "1000.0", "work_파일크기_tier": "B",
                 "work_파일포맷": "JPG", "work_파일포맷_tier": "B", "work_해상도": "100x100", "work_해상도_tier": "A"})
    base.update(over)
    return base


def _index_rows(tmp: Path) -> list[dict[str, str]]:
    w1 = tmp / "w1.jpg"
    w1.write_bytes(b"jpeg-bytes-1")
    w6 = tmp / "w6.mp4"
    w6.write_bytes(b"mp4-bytes-6" * 100)
    return [
        _row("90000001", "image", "ccl", "공유마당", work_file=str(w1), work_설명="고유한 캡션 문장", work_설명_tier="A",
             work_키워드="바다 | 해변 | 바다 ", work_키워드_tier="A", work_해상도="1920x1080", work_파일크기="123.0", work_파일포맷="jpg"),
        _row("90000002", "image", "ccl", "공유마당", work_설명=LABEL, work_설명_tier="A", work_키워드="a | b",
             work_키워드_tier="A", work_키워드_batch_level="True"),
        _row("90000003", "image", "ccl", "공유마당", work_설명=LABEL, work_설명_tier="A", work_키워드="a | b",
             work_키워드_tier="A", work_키워드_batch_level="True", work_키워드_내용태그="서울 | #서울"),
        _row("90000004", "image", "kogl", "공공누리(KOGL)", work_설명="https://example.org/x", work_설명_tier="A",
             work_키워드="", work_키워드_tier="A", work_해시태그="", work_해시태그_tier="A", work_해상도="", work_해상도_tier="",
             work_파일크기="", work_파일포맷=""),
        _row("90000005", "image", "ccl", "공유마당", work_설명=LABEL, work_설명_tier="A", work_키워드="", work_키워드_tier="제외"),
        _row("90000006", "video", "expired", "공유마당", work_file=str(w6), work_설명="영화 줄거리 텍스트", work_설명_tier="A",
             work_해상도="1280 x 720", work_해상도_tier="B", work_파일포맷="MP4"),
        _row("90000007", "video", "kogl", "공공누리(KOGL)", work_설명="기관 배경 설명 텍스트", work_설명_tier="A",
             work_해시태그="문화 | 역사 | 문화", work_해시태그_tier="A", work_파일포맷="MKV"),
        _row("90000008", "image", "kogl", "공유마당", work_설명="아무개의 국외 일러스트 작품", work_설명_tier="A",
             work_키워드="그림", work_키워드_tier="A"),
        _row("90000009", "text", "donated", "공유마당", work_해상도="", work_해상도_tier="N/A", work_파일포맷="PDF"),
        _row("90000010", "text", "ccl", "공유마당", work_해상도="", work_해상도_tier="N/A", work_파일포맷="TXT"),
    ]


def _ledger_row(**over: object) -> list[object]:
    base: dict[str, object] = {c: "□" if c.startswith(("C_", "R_")) else "-" for c in LEDGER_HEADER}
    base.update({"C_기타": "-", "이용허락기간_SY": 2009, "이용허락기간_SM": "09", "이용허락기간_SD": "27",
                 "이용허락기간_EY": 2033, "이용허락기간_EM": "04", "이용허락기간_ED": "14"})
    base.update(over)
    return [base[c] for c in LEDGER_HEADER]


@pytest.fixture
def data(tmp_path: Path) -> dict[str, Path]:
    import openpyxl

    rows = _index_rows(tmp_path)
    index = tmp_path / "gt_merged_index.csv"
    with open(index, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)

    t002_ids = tmp_path / "t002_ids.csv"
    with open(t002_ids, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=IDS_COLUMNS)
        wr.writeheader()
        for r in rows[:8]:
            wr.writerow({"test": "T-002", "cell": f"{r['media']}/{r['license_bucket']}", "work_id": r["work_id"],
                         "media": r["media"], "license_bucket": r["license_bucket"], "source": r["source"],
                         "title": r["title"], "work_file": r["work_file"]})
        wr.writerow({"test": "T-002", "work_id": "90000099"})  # 인덱스에 없는 id

    t001_ids = tmp_path / "t001_ids.txt"
    t001_ids.write_text("id\n90000009\n90000010\n90000001\n", encoding="utf-8")
    t001_ids_noheader = tmp_path / "t001_ids_noheader.txt"
    t001_ids_noheader.write_text("# 주석\n90000009\n90000010 어떤 제목\n90000009\n", encoding="utf-8")

    ledger = tmp_path / "ledger.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = g.DEFAULT_LEDGER_SHEET
    ws.append(LEDGER_HEADER)
    ws.append(_ledger_row(공유형태="기증", 저작물형태="어문", 파일명="90000009 운수 자판기", 권리자="한국저작권위원회",
                          제목="운수 자판기", 저작자="양지웅 (저작물 3 건)\n이영지 (저작물 2 건)\n신혜진", C_1="v",
                          R_1="□", R_2="v", R_3="■", R_4="V", R_5="□", R_6="□", R_7="v"))
    ws.append(_ledger_row(공유형태="CCL", 저작물형태="어문", 파일명="90000010 악보집", 권리자="-", 제목="악보집",
                          저작자="김중우, 전재승", C_10="v", C_기타="악보", R_1="x", 이용허락기간_SM="13",
                          이용허락기간_EY=2030, 이용허락기간_EM="01", 이용허락기간_ED="31"))
    ws.append([None] * len(LEDGER_HEADER))  # 빈 행은 건너뛴다
    # 장세영 전달 5건의 파일명 형식 — 앞 생성 순번 '00002_' + 뒤 다운로드 중복 표시 ' (1)' 이 붙어도 id 를 읽어야 한다
    ws.append(_ledger_row(공유형태="기증", 저작물형태="어문", 파일명="00002_13387752 운수 자판기 (1)", 권리자="한국저작권위원회",
                          제목="운수 자판기", 저작자="홍길동", C_1="v", R_1="v"))
    wb.save(ledger)

    t001_ids_csv = tmp_path / "t001_ids.csv"   # frozen 목록 형식 — contract_pdf 열이 file_name gold 의 기본 출처
    with open(t001_ids_csv, "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=IDS_COLUMNS)
        wr.writeheader()
        wr.writerow({"test": "T-001", "cell": "text/donated", "work_id": "90000009", "media": "text", "license_bucket": "donated",
                     "source": "공유마당", "has_contract": "True",
                     "contract_pdf": "dataset/new_contracts/00002_90000009 운수 자판기 (1).pdf"})
        wr.writerow({"test": "T-001", "cell": "text/ccl", "work_id": "90000010", "media": "text", "license_bucket": "ccl",
                     "source": "공유마당", "has_contract": "True", "contract_pdf": ""})

    def draft(work_id: str, status: str, text: str, tags: list[str]) -> dict:
        return {"work_id": work_id, "status": status, "draft_text": text, "draft_hashtags": tags,
                "site_gold_kind": "none", "tier": "A-검수(대기)"}

    drafts = tmp_path / "drafts.jsonl"
    with open(drafts, "w", encoding="utf-8") as fh:
        for rec in [draft("90000001", "ok", "생성 설명 1", ["#바다", "#해변"]),
                    draft("90000002", "unsupported_format", "", []),
                    draft("90000002", "ok", "생성 설명 2", ["#라벨", "#사진"]),
                    draft("90000003", "needs_review", "검토 필요 3", ["#x"]),
                    draft("90000006", "ok", "생성 설명 6", ["#영화"]),
                    draft("90000007", "ok", "생성 설명 7", ["#새우젓"]),
                    draft("90000008", "ok", "생성 설명 8", ["#그림"])]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    verified = tmp_path / g.VERIFIED_BASENAME
    with open(verified, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"work_id": "90000007", "description": "검수된 설명 7", "hashtags": ["#검수", "#새우젓"],
                             "verification_outcome": "accepted"}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"work_id": "90000008", "description": "거절된 설명", "hashtags": ["#x"],
                             "verification_outcome": "rejected"}, ensure_ascii=False) + "\n")
    return {"index": index, "t002_ids": t002_ids, "t001_ids": t001_ids, "t001_ids_noheader": t001_ids_noheader,
            "t001_ids_csv": t001_ids_csv,
            "ledger": ledger, "drafts": drafts, "verified": verified, "cache": tmp_path / "_probe" / "file_sha256.jsonl",
            "w1": tmp_path / "w1.jpg", "w6": tmp_path / "w6.mp4", "tmp": tmp_path}


def _build_t002(data: dict[str, Path], mode: str = "draft", hash_flag: bool = False) -> dict[str, dict]:
    recs = g.build_gold("T-002", data["t002_ids"], data["index"], None, data["drafts"], data["verified"],
                        mode, hash_flag, data["cache"])
    return {r["set_id"]: r for r in recs}


def _build_t001(data: dict[str, Path], ids_key: str = "t001_ids", **kw: object) -> dict[str, dict]:
    recs = g.build_gold("T-001", data[ids_key], data["index"], data["ledger"], None, None, "none", False, data["cache"], **kw)
    return {r["set_id"]: r for r in recs}


# ---------------------------------------------------------------- 레코드 형식 (§2)
def test_record_format_both_tests(data: dict[str, Path]) -> None:
    for recs in (_build_t002(data), _build_t001(data)):
        for rec in recs.values():
            assert list(rec) == ["set_id", "test", "media", "bucket", "source", "keys"]
            assert list(rec["keys"]) == list(KEY_NAMES)
            for cell in rec["keys"].values():
                assert list(cell) == ["value", "state", "source", "tier", "kind"]
                assert cell["state"] in {g.STATE_VALUE, g.STATE_EMPTY, g.STATE_NA}
                assert (cell["value"] is None) == (cell["state"] == g.STATE_NA)
                assert cell["source"]
    assert all(r["test"] == "T-002" for r in _build_t002(data).values())
    json.dumps(list(_build_t002(data).values()), ensure_ascii=False)  # 직렬화 가능


def test_unscored_keys_are_na(data: dict[str, Path]) -> None:
    t2 = _build_t002(data)["90000001"]["keys"]
    assert all(t2[k]["state"] == g.STATE_NA for k in T001_KEYS if k != "file_name")
    assert t2["file_name"] == {"value": "w1.jpg", "state": "value", "source": "work_file basename", "tier": "B", "kind": None}
    t1 = _build_t001(data)["90000009"]["keys"]
    assert all(t1[k]["state"] == g.STATE_NA for k in T002_KEYS)
    assert t1["file_name"]["value"] == "90000009 운수 자판기.pdf"


# ---------------------------------------------------------------- T-002 상태
def test_t002_description_kinds_and_states(data: dict[str, Path]) -> None:
    recs = _build_t002(data)
    d = {w: recs[w]["keys"]["description"] for w in recs}
    assert d["90000001"] == {"value": "고유한 캡션 문장", "state": "value", "source": "index:work_설명 (사이트 캡션)", "tier": "A", "kind": "caption"}
    assert d["90000002"]["state"] == "value" and d["90000002"]["kind"] == "generated" and d["90000002"]["value"] == "생성 설명 2"
    assert d["90000002"]["tier"] == g.TIER_DRAFT
    assert d["90000003"]["state"] == "na" and d["90000003"]["kind"] == "label" and "needs_review" in d["90000003"]["source"]
    assert d["90000004"]["state"] == "na" and d["90000004"]["kind"] is None  # URL 만 있는 설명 → none
    assert d["90000005"]["state"] == "na" and d["90000005"]["kind"] == "label"
    assert d["90000006"]["kind"] == "generated"  # plot → 생성 Gold
    assert d["90000007"]["kind"] == "generated"  # background → 생성 Gold
    assert d["90000008"]["kind"] == "generated"  # 귀속 문구 → none → 생성 Gold
    assert d["90000099"]["state"] == "na" and "인덱스에 work_id 없음" in d["90000099"]["source"]


def test_t002_keywords_states(data: dict[str, Path]) -> None:
    kw = {w: r["keys"]["keywords"] for w, r in _build_t002(data).items()}
    assert kw["90000001"]["value"] == ["바다", "해변"] and kw["90000001"]["state"] == "value"  # 중복 제거
    assert kw["90000002"]["state"] == "na" and "배치" in kw["90000002"]["source"]
    assert kw["90000003"]["value"] == ["서울"] and kw["90000003"]["source"] == "index:work_키워드_내용태그"  # '#서울' 중복 제거
    assert kw["90000004"] == {"value": [], "state": "empty", "source": "index:work_키워드 (수집됨, 비어 있음)", "tier": "A", "kind": None}
    assert kw["90000005"]["state"] == "na"


def test_t002_hashtags_states(data: dict[str, Path]) -> None:
    ht = {w: r["keys"]["hashtags"] for w, r in _build_t002(data).items()}
    assert ht["90000007"] == {"value": ["문화", "역사"], "state": "value", "source": "index:work_해시태그 (KOGL 원본)", "tier": "A", "kind": "site"}
    assert ht["90000004"]["state"] == "empty" and ht["90000004"]["kind"] == "site"
    assert ht["90000001"]["value"] == ["바다", "해변"] and ht["90000001"]["kind"] == "generated"
    assert ht["90000005"]["state"] == "na" and ht["90000003"]["state"] == "na"
    assert ht["90000008"]["kind"] == "generated"  # 공유마당 kogl 버킷 → 사이트 해시태그 없음 → 생성


def test_t002_technical_keys(data: dict[str, Path]) -> None:
    recs = _build_t002(data)
    k1, k4, k6, k7 = (recs[w]["keys"] for w in ("90000001", "90000004", "90000006", "90000007"))
    assert k1["file_format"]["value"] == "JPG" and k1["file_size"]["value"] == 123 and k1["resolution"]["value"] == "1920x1080"
    assert k4["file_format"]["state"] == "na" and k4["file_size"]["state"] == "na" and k4["resolution"]["state"] == "na"
    assert k6["resolution"]["value"] == "1280x720" and k6["resolution"]["tier"] == "B" and k6["file_format"]["value"] == "MP4"
    assert k7["file_format"]["value"] == "MKV" and k7["file_name"]["value"] == "90000007.bin"
    assert all(recs[w]["keys"]["file_hash"] == g.na_cell("미계산 (--hash 필요)") for w in recs if w != "90000099")
    assert recs["90000099"]["keys"]["file_hash"]["state"] == "na"


def test_generated_none_and_verified(data: dict[str, Path]) -> None:
    none = _build_t002(data, "none")
    assert none["90000002"]["keys"]["description"]["state"] == "na" and none["90000001"]["keys"]["description"]["kind"] == "caption"
    assert none["90000001"]["keys"]["hashtags"]["state"] == "na" and none["90000007"]["keys"]["hashtags"]["kind"] == "site"
    ver = _build_t002(data, "verified")
    d7 = ver["90000007"]["keys"]["description"]
    assert d7["value"] == "검수된 설명 7" and d7["tier"] == g.TIER_VERIFIED and d7["source"].startswith("gold_descriptions.jsonl")
    assert ver["90000007"]["keys"]["hashtags"]["kind"] == "site"  # KOGL 원본은 사이트 해시태그 우선
    assert ver["90000008"]["keys"]["description"]["state"] == "na" and "검수 거절" in ver["90000008"]["keys"]["description"]["source"]
    assert ver["90000008"]["keys"]["hashtags"]["state"] == "na"
    d2 = ver["90000002"]["keys"]["description"]  # 미검수 → 초안 보완
    assert d2["value"] == "생성 설명 2" and d2["tier"] == g.TIER_DRAFT and "미검수" in d2["source"]


def test_verified_mode_requires_file(data: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError):
        g.build_gold("T-002", data["t002_ids"], data["index"], None, data["drafts"], data["tmp"] / "missing.jsonl",
                     "verified", False, data["cache"])


# ---------------------------------------------------------------- T-001 상태 (원장)
def test_t001_ledger_mapping(data: dict[str, Path]) -> None:
    k = _build_t001(data)["90000009"]["keys"]
    assert k["work_title"] == {"value": "운수 자판기", "state": "value", "source": "ledger:제목", "tier": "A", "kind": None}
    assert k["work_type"]["value"] == "어문"
    assert k["creator"]["value"] == ["양지웅", "이영지", "신혜진"]
    assert k["copyright_holder"]["value"] == ["한국저작권위원회"]
    rights = {r: k[r]["value"] for r in g.RIGHTS_LEDGER_COLUMNS}
    assert rights == {"reproduction_right": "불허", "performance_right": "허용", "public_transmission_right": "허용",
                      "exhibition_right": "허용", "distribution_right": "불허", "rental_right": "불허", "derivative_work_right": "허용"}
    assert k["other_rights"]["state"] == "na"
    assert k["license_start_date"]["value"] == "2009-09-27" and k["license_end_date"]["value"] == "2033-04-14"
    assert k["contract_validity_period"]["value"] == "2009-09-27~2033-04-14"


def test_t001_na_paths(data: dict[str, Path]) -> None:
    recs = _build_t001(data)
    k = recs["90000010"]["keys"]
    assert k["copyright_holder"]["state"] == "na" and "'-'" in k["copyright_holder"]["source"]
    assert k["creator"]["value"] == ["김중우", "전재승"]
    assert k["work_type"]["value"] == "기타" and "C_기타=악보" in k["work_type"]["source"]
    assert k["reproduction_right"]["state"] == "na" and "'x'" in k["reproduction_right"]["source"]
    assert k["performance_right"]["value"] == "불허"
    assert k["license_start_date"]["state"] == "na" and k["license_end_date"]["value"] == "2030-01-31"
    assert k["contract_validity_period"]["state"] == "na"
    missing = recs["90000001"]["keys"]  # 원장 행 없음 → 1–16 전부 na
    assert all(missing[key]["state"] == "na" and "원장" in missing[key]["source"] for key in T001_KEYS)
    assert recs["90000001"]["media"] == "image" and recs["90000001"]["bucket"] == "ccl"  # 인덱스에서 보완


def test_t001_file_name_follows_delivered_contract_pdf(data: dict[str, Path]) -> None:
    """§1 #2: file_name gold = 시스템에 주는 계약서 PDF 의 basename. auto = contract_pdf 우선, 없으면 '<원장 파일명>.pdf'."""
    auto = _build_t001(data, "t001_ids_csv")
    assert auto["90000009"]["keys"]["file_name"] == {"value": "00002_90000009 운수 자판기 (1).pdf", "state": "value",
                                                      "source": "ids:contract_pdf basename", "tier": "B", "kind": None}
    assert auto["90000010"]["keys"]["file_name"] == {"value": "90000010 악보집.pdf", "state": "value",
                                                      "source": "ledger:파일명+'.pdf'", "tier": "A", "kind": None}
    assert auto["90000009"]["keys"]["work_title"]["value"] == "운수 자판기"   # 나머지 키는 원장 그대로
    ledger = _build_t001(data, "t001_ids_csv", file_name_from="ledger")
    assert ledger["90000009"]["keys"]["file_name"]["value"] == "90000009 운수 자판기.pdf"
    ids = _build_t001(data, "t001_ids_csv", file_name_from="ids")
    assert ids["90000009"]["keys"]["file_name"]["value"] == "00002_90000009 운수 자판기 (1).pdf"
    assert ids["90000010"]["keys"]["file_name"]["state"] == "na" and "contract_pdf" in ids["90000010"]["keys"]["file_name"]["source"]
    with pytest.raises(ValueError):
        _build_t001(data, "t001_ids_csv", file_name_from="bogus")


@pytest.mark.parametrize("stem, work_id", [
    ("13387752 운수 자판기", "13387752"),
    ("00002_13387752 운수 자판기 (1)", "13387752"),            # 장세영 전달 5건 형식
    ("00003_13333800 F!RE 가사", "13333800"),
    ("00004_13333561 국제기업협력센터를 중심으로 한 비지니스 허브 도시 구축 전략", "13333561"),
    ("13304492 빛의 세상으로(희망가) 가사", "13304492"),      # 제목 안의 괄호는 그대로
    ("100388 20240909_12_55Arirang News", "100388"),         # 제목에 '_' 가 있어도 id 는 앞 토큰
    ("  13387752   운수 자판기 ", "13387752"),
    ("운수 자판기", ""), ("2024 신년 음악회", ""), ("13387752", ""), ("", ""),
])
def test_ledger_work_id_forms(stem: str, work_id: str) -> None:
    assert g.ledger_work_id(stem) == work_id


def _write_ledger(path: Path, file_names: list[str]) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = g.DEFAULT_LEDGER_SHEET
    ws.append(LEDGER_HEADER)
    for fn in file_names:
        ws.append(_ledger_row(파일명=fn, 제목="t", 저작자="a", C_1="v"))
    wb.save(path)
    return path


def test_load_ledger_fails_instead_of_silent_all_na(tmp_path: Path) -> None:
    """파일명 형식이 달라져 id 를 못 읽으면 경고 후 전부 na 가 되는 대신 ValueError 로 멈춘다 (0건 또는 절반 미만)."""
    with pytest.raises(ValueError, match="id 를 읽은 행이 0/2건"):
        g.load_ledger(_write_ledger(tmp_path / "none.xlsx", ["운수 자판기", "악보집"]))
    with pytest.raises(ValueError, match="1/3건"):
        g.load_ledger(_write_ledger(tmp_path / "third.xlsx", ["90000009 a", "x", "y"]))
    with pytest.raises(ValueError, match="0/0건"):
        g.load_ledger(_write_ledger(tmp_path / "empty.xlsx", []))
    ok = g.load_ledger(_write_ledger(tmp_path / "ok.xlsx", ["90000009 a", "00002_90000010 b (1)", "junk", "90000011 c"]))
    assert set(ok) == {"90000009", "90000010", "90000011"}   # 3/4 ≥ 50% → 경고만


def test_plain_id_file_without_header_and_duplicates(data: dict[str, Path]) -> None:
    rows = g.load_ids(data["t001_ids_noheader"])
    assert [r.work_id for r in rows] == ["90000009", "90000010"]
    recs = _build_t001(data, "t001_ids_noheader")
    assert list(recs) == ["90000009", "90000010"]
    assert recs["90000010"]["media"] == "text" and recs["90000010"]["bucket"] == "ccl"


# ---------------------------------------------------------------- 해시 캐시
def test_hash_flag_computes_and_caches(data: dict[str, Path]) -> None:
    first = _build_t002(data, hash_flag=True)
    h1 = first["90000001"]["keys"]["file_hash"]
    assert h1["value"] == hashlib.sha256(data["w1"].read_bytes()).hexdigest() and h1["source"].endswith("[computed]")
    assert first["90000006"]["keys"]["file_hash"]["value"] == hashlib.sha256(data["w6"].read_bytes()).hexdigest()
    assert first["90000002"]["keys"]["file_hash"]["state"] == "na" and "파일 없음" in first["90000002"]["keys"]["file_hash"]["source"]
    cache_lines = [json.loads(l) for l in data["cache"].read_text(encoding="utf-8").splitlines()]
    assert len(cache_lines) == 2 and set(cache_lines[0]) == {"path", "size", "mtime", "sha256"}
    second = _build_t002(data, hash_flag=True)
    assert second["90000001"]["keys"]["file_hash"]["source"].endswith("[cache]")
    assert len(data["cache"].read_text(encoding="utf-8").splitlines()) == 2


# ---------------------------------------------------------------- 커버리지 표 + CLI
def test_coverage_table(data: dict[str, Path]) -> None:
    recs = list(_build_t002(data).values())
    keys = ("file_name",) + T002_KEYS
    cov = g.coverage(recs, keys)
    assert set(cov) == {"image/ccl", "image/kogl", "video/expired", "video/kogl", "?/?"}
    assert cov["image/ccl"]["description"] == Counter({"value": 2, "na": 2})
    assert cov["image/kogl"]["keywords"] == Counter({"empty": 1, "value": 1})
    text = g.format_coverage(cov, keys)
    assert text.splitlines()[1].split()[:2] == ["key", "?/?"] and "합계" in text.splitlines()[1]
    desc_line = next(l for l in text.splitlines() if l.startswith("description"))
    assert desc_line.split()[-1] == "5/0/4"  # 9세트: caption 1 + generated 4 = value 5 / empty 0 / na 4


def test_cli_main_writes_jsonl(data: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    out = data["tmp"] / "out" / "gold23_t002.jsonl"
    rc = g.main(["--test", "T-002", "--ids", str(data["t002_ids"]), "--out", str(out), "--index", str(data["index"]),
                 "--drafts", str(data["drafts"]), "--generated", "draft", "--hash-cache", str(data["cache"])])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 9 and json.loads(lines[0])["set_id"] == "90000001"
    assert "셀 × 키 커버리지" in capsys.readouterr().out
    out1 = data["tmp"] / "out" / "gold23_t001.jsonl"
    rc = g.main(["--test", "T-001", "--ids", str(data["t001_ids"]), "--out", str(out1), "--index", str(data["index"]),
                 "--ledger", str(data["ledger"]), "--hash-cache", str(data["cache"])])
    assert rc == 0 and len(out1.read_text(encoding="utf-8").splitlines()) == 3
    out1b = data["tmp"] / "out" / "gold23_t001_ids.jsonl"
    rc = g.main(["--test", "T-001", "--ids", str(data["t001_ids_csv"]), "--out", str(out1b), "--index", str(data["index"]),
                 "--ledger", str(data["ledger"]), "--hash-cache", str(data["cache"]), "--file-name-from", "ids"])
    recs = [json.loads(l) for l in out1b.read_text(encoding="utf-8").splitlines()]
    assert rc == 0 and recs[0]["keys"]["file_name"]["value"] == "00002_90000009 운수 자판기 (1).pdf"


# ---------------------------------------------------------------- 도우미 단위 테스트
def test_split_names_and_tags() -> None:
    assert g.split_names("양지웅 (저작물 3 건)\n이영지 (2건)\n신혜진; 신혜진 ") == ["양지웅", "이영지", "신혜진"]
    assert g.split_names("山內正瞭 原著, 야마노우치 마사아키 원저") == ["山內正瞭 原著", "야마노우치 마사아키 원저"]
    assert g.split_names(None) == []
    assert g.split_tags("#서울 | 서울 | 한 강 | 한강 | ") == ["서울", "한 강"]
    assert g.split_tags(["#바다", "#Bada", "#바다"]) == ["바다", "Bada"]


def test_ledger_date_and_glyphs() -> None:
    assert g.ledger_date({"이용허락기간_SY": "2009", "이용허락기간_SM": "9", "이용허락기간_SD": "27.0"}, "S") == "2009-09-27"
    assert g.ledger_date({"이용허락기간_EY": "2030", "이용허락기간_EM": "02", "이용허락기간_ED": "30"}, "E") == ""
    assert g.ledger_date({}, "S") == ""
    assert [g.glyph_state(x) for x in ("v", "V", "■", "□", "x", "", None)] == [True, True, True, False, None, None, None]


def test_site_kind_rule() -> None:
    def row(media: str, bucket: str, source: str, desc: str, tier: str = "A") -> dict[str, str]:
        return {"media": media, "license_bucket": bucket, "source": source, "work_설명": desc, "work_설명_tier": tier}

    index = {"1": row("image", "ccl", "공유마당", "같은 라벨"), "2": row("image", "ccl", "공유마당", "같은 라벨"),
             "3": row("image", "ccl", "공유마당", "같은 라벨"), "4": row("image", "kogl", "공유마당", "같은 라벨"),
             "5": row("video", "expired", "공유마당", "줄거리"), "6": row("video", "kogl", "공공누리(KOGL)", "배경"),
             "7": row("image", "ccl", "공유마당", "캡션"), "8": row("image", "ccl", "공유마당", "캡션", "제외"),
             "9": row("image", "ccl", "공유마당", "http://x"), "10": row("image", "ccl", "공유마당", "홍길동의 사진 자료")}
    kinds = g.classify_site_kinds(index)
    assert [kinds[str(i)] for i in range(1, 11)] == ["label", "label", "label", "caption", "plot", "background",
                                                     "caption", "none", "none", "none"]


def test_load_ledger_reports_unknown_columns(data: dict[str, Path], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="v4.gold23"):
        ledger = g.load_ledger(data["ledger"])
    assert set(ledger) == {"90000009", "90000010", "13387752"}   # '00002_13387752 운수 자판기 (1)' 도 읽힌다
    assert ledger["13387752"]["제목"] == "운수 자판기"
    assert any("모르는 열" in m and "NEW_COL" in m for m in caplog.messages)
