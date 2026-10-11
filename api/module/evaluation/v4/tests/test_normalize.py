"""normalize.py 단위 테스트 (spec §8 a) — 날짜 5형식, 권리 8형식, 해상도 4형식, 이름 주석, 태그 #·공백, 관리번호 접두."""
from __future__ import annotations

import unicodedata

import pytest

from api.module.evaluation.v4 import normalize as N


# ----------------------------------------------------------------- 텍스트
def test_norm_text_collapses_whitespace_and_invisible_chars():
    assert N.norm_text("  운수​ 자판기 ‪ ") == "운수 자판기"
    assert N.norm_text("a\n\n  b\tc") == "a b c"
    assert N.norm_text(None) is None
    assert N.norm_text("   ") is None
    assert N.norm_text([]) is None
    assert N.norm_text(123) == "123"


def test_norm_text_nfc():
    decomposed = unicodedata.normalize("NFD", "한글")
    assert decomposed != "한글"
    assert N.norm_text(decomposed) == "한글"


def test_text_key_is_case_insensitive_for_latin():
    assert N.text_key("Photo.JPG") == "photo.jpg"
    assert N.text_key("사진") == "사진"


@pytest.mark.parametrize("raw, expected", [
    ("13387752 운수 자판기", "운수 자판기"),
    ("13387752   운수 자판기", "운수 자판기"),
    ("12345 abc", "abc"),
    ("2024 신년 음악회", "2024 신년 음악회"),       # 4자리(연도)는 관리번호가 아니다
    ("123456789 제목", "123456789 제목"),           # 9자리도 아니다
    ("13387752", "13387752"),                       # 뒤에 공백+제목이 없으면 그대로
    (None, None),
])
def test_strip_id_prefix(raw, expected):
    assert N.strip_id_prefix(raw) == expected


# ----------------------------------------------------------------- 이름
def test_norm_name_removes_spaces_and_annotations():
    assert N.norm_name("홍 길동 (저작물 3 건)") == "홍길동"
    assert N.norm_name("홍길동(3건)") == "홍길동"
    assert N.norm_name("John Smith") == "johnsmith"
    assert N.norm_name(None) is None


def test_split_names_from_string_and_list():
    assert N.split_names("홍길동, 김철수 (2건)") == ["홍길동", "김철수"]
    assert N.split_names("박지성 · 김연아") == ["박지성", "김연아"]
    assert N.split_names("a/b;c\nd") == ["a", "b", "c", "d"]
    assert N.split_names(["홍길동", "홍 길동", " "]) == ["홍길동"]
    assert N.split_names(None) == []
    assert N.split_names([]) == []


# ----------------------------------------------------------------- 태그
def test_norm_tag_strips_hash_spaces_punctuation():
    assert N.norm_tag("#경복궁 단청") == "경복궁단청"
    assert N.norm_tag(" Sea-Side_01 ") == "seaside01"
    assert N.norm_tag("##") is None
    assert N.norm_tag(None) is None


def test_split_tags_forms():
    assert N.split_tags("#바다 #해변") == ["바다", "해변"]
    assert N.split_tags("바다, 해변|노을; 석양") == ["바다", "해변", "노을", "석양"]
    assert N.split_tags(["#바다", "바다", " ", "경복궁 단청"]) == ["바다", "경복궁단청"]
    assert N.split_tags(None) == []
    assert N.split_tags("") == []


# ----------------------------------------------------------------- 날짜·기간
@pytest.mark.parametrize("raw", [
    "2024-01-05", "2024.01.05", "2024/01/05", "2024년 1월 5일", "20240105",
    "2024.1.5", " 2024년1월5일 ", "2024-01-05T00:00:00", "2024. 01. 05.",
])
def test_norm_date_five_forms(raw):
    assert N.norm_date(raw) == "2024-01-05"


def test_norm_date_null_and_invalid():
    assert N.norm_date(None) is None
    assert N.norm_date("") is None
    for bad in ("2024-13-01", "2024-02-30", "abc", "2024-01", "01-05-2024"):
        with pytest.raises(ValueError):
            N.norm_date(bad)


@pytest.mark.parametrize("raw", [
    "2002-08-31~2033-12-23",
    "2002-08-31 ~ 2033-12-23",
    "2002.08.31 - 2033.12.23",
    "2002/08/31 – 2033/12/23",
    "2002년 08월 31일부터 2033년 12월 23일까지",
    "20020831~20331223",
    "2002-08-31-2033-12-23",
    ["2002-08-31", "2033-12-23"],
    {"start": "2002.08.31", "end": "2033.12.23"},
])
def test_norm_period_forms(raw):
    assert N.norm_period(raw) == "2002-08-31~2033-12-23"


def test_norm_period_invalid():
    assert N.norm_period(None) is None
    for bad in ("2002-08-31", "없음", ["2002-08-31"], {"start": "2002-08-31"}):
        with pytest.raises(ValueError):
            N.norm_period(bad)


# ----------------------------------------------------------------- 권리 (8형식)
@pytest.mark.parametrize("raw, expected", [
    ("허용", "허용"), ("불허", "불허"),
    (True, "허용"), (False, "불허"),
    ("true", "허용"), ("FALSE", "불허"),
    ("O", "허용"), ("X", "불허"),
    ("Y", "허용"), ("n", "불허"),
    (None, None), ("null", None), ("", None), (" 허용 ", "허용"),
    # 공중송신권의 하위 권리 괄호(§1 #8)는 떼고 본다 — 새 템플릿 '공중송신권(v방송권, □전송권, v디지털음성송신권)'
    ("허용(방송권, 디지털음성송신권)", "허용"), ("불허 (전송권)", "불허"), ("O (방송권)", "허용"),
    ("허용(V방송권, □전송권, V디지털음성송신권)", "허용"),
])
def test_norm_rights_forms(raw, expected):
    assert N.norm_rights(raw) == expected


def test_norm_rights_unparseable():
    # 체크 글리프(v/V/■/□)는 예측값으로 받지 않는다 (§1 Rights values — 추출기는 허용/불허 로 내야 한다)
    for bad in ("maybe", "■", "□", "v", "V", "허용함", "yes", "(허용)", "방송권(허용)"):
        with pytest.raises(ValueError):
            N.norm_rights(bad)


# ----------------------------------------------------------------- 저작물유형
@pytest.mark.parametrize("raw, expected", [
    ("미술저작물", "미술"), ("사진", "사진"), ("컴퓨터 프로그램 저작물", "컴퓨터프로그램"),
    ("프로그램", "컴퓨터프로그램"), ("동영상", "영상"), ("영상 저작물", "영상"), ("기타저작물", "기타"),
    ("어문(소설)", "어문"), (None, None), ("", None),
])
def test_norm_work_type(raw, expected):
    assert N.norm_work_type(raw) == expected


def test_norm_work_type_unknown():
    with pytest.raises(ValueError):
        N.norm_work_type("그림")


# ----------------------------------------------------------------- 파일 기술정보
@pytest.mark.parametrize("raw, expected", [
    ("jpeg", "JPG"), ("JPG", "JPG"), (".JPG", "JPG"), ("image/jpeg", "JPG"), ("photo.tiff", "TIF"),
    ("tif", "TIF"), ("mpeg4", "MP4"), ("MPEG-4", "MP4"), ("mp4", "MP4"), ("video/mp4", "MP4"),
    ("png", "PNG"), ("PDF", "PDF"), (None, None),
    ("application/x-hwp", "HWP"), ("application/pdf", "PDF"), ("image/x-tiff", "TIF"),   # MIME 비표준 접두 'x-'
])
def test_norm_format(raw, expected):
    assert N.norm_format(raw) == expected


def test_norm_format_unparseable():
    with pytest.raises(ValueError):
        N.norm_format("???")


@pytest.mark.parametrize("raw, expected", [
    (1234, 1234), ("1,234", 1234), ("1234 bytes", 1234), ("1234B", 1234), ("1234 byte", 1234),
    ("1234.0", 1234), (1234.0, 1234), ("11553825.0", 11553825), ("7 바이트", 7), (None, None), ("", None),
])
def test_norm_int_bytes(raw, expected):
    assert N.norm_int_bytes(raw) == expected


@pytest.mark.parametrize("bad", ["5 KB", "1.2MB", "1.5", True, -3, "abc"])
def test_norm_int_bytes_rejects_units_and_junk(bad):
    with pytest.raises(ValueError):
        N.norm_int_bytes(bad)


@pytest.mark.parametrize("raw", [
    "1920x1080", "1920 x 1080", "1920*1080", "1920×1080", "1920X1080 px", "1920 × 1080 픽셀",
    [1920, 1080], ["1920", "1080"], ("1920", " 1080 "), [1920, "1080"], {"width": 1920, "height": "1080"},
    ["1920 x 1080"], ["1920 x 1080", ""],   # 문자열 하나짜리 리스트는 일반 경로(빈 원소 무시)
])
def test_norm_resolution_forms(raw):
    assert N.norm_resolution(raw) == "1920x1080"


def test_norm_resolution_invalid():
    assert N.norm_resolution(None) is None
    for bad in ("1920", "axb", "0x100", [1920], {"width": 1920}, ["a", "b"], ["1920", "0"]):
        with pytest.raises(ValueError):
            N.norm_resolution(bad)


def test_norm_hash():
    h = "AB" * 32
    assert N.norm_hash(h) == "ab" * 32
    assert N.norm_hash("sha256:" + h) == "ab" * 32
    assert N.norm_hash("SHA-256: " + h) == "ab" * 32
    assert N.norm_hash(None) is None
    for bad in ("abc", "g" * 64, "ab" * 31):
        with pytest.raises(ValueError):
            N.norm_hash(bad)


def test_is_blank_and_scalar_input():
    assert N.is_blank(None) and N.is_blank("  ") and N.is_blank([]) and N.is_blank({})
    assert not N.is_blank(0) and not N.is_blank("x") and not N.is_blank(False)
    assert N.scalar_input(["a"]) == "a"
    assert N.scalar_input(["a", "b"]) == "a, b"
    assert N.scalar_input([]) is None
    assert N.scalar_input("a") == "a"
