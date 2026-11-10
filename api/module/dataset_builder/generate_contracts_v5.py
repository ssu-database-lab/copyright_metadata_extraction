#!/usr/bin/env python
"""시험 계약서 v5 생성기 — 장세영 연구원 GOLD 시트 값 + 템플릿 3종 → HWPX → PDF → 검사 → 정답 파일.

입력
  dataset/test_set_v5/contracts.csv · te1_ids.csv · te2_ids.csv   (build_test_set_v5.py 가 고른 4,200 / 4,000 / 4,000)
  docs/1._생성계약서_GOLD 데이터.xlsx                              (계약서 값, 4,114건)
  dataset/gt_merged_index.csv                                     (시트에 없는 86건의 제목·저자)
  docs/2. 계약서.hwpx                                              (장세영 템플릿 = 템플릿 A)

값 정리 (사용자 승인 2026-11-03, 네 가지 모두 "yes")
  1. 이름: '(저작물 N 건)' 제거, 줄바꿈으로 붙은 두 이름은 'A, B'.
     만료 영상: 시트의 저작자(한국저작권위원회)와 권리자(감독·제작자)가 뒤바뀌어 있어 맞바꾼다.
     이용자가 '-'·'소속없음'이거나 권리자와 같으면 가상의 이용자(회사·개인)를 새로 만든다.
  2. 템플릿 A 최소 수정: 제목(제호) ← {{제목}} (파일명 아님), 저작물명 = 제목, 계약 체결일 = 이용허락 시작 1~30일 전,
     값 글자색 검정, '이라 함' → '라 함', 번호 칸 이름 = 생년월일 / 사업자등록번호, 이용료 천 단위 쉼표.
     원본 인쇄 설정(2쪽 모아 찍기)은 PDF 가 가로 한 장에 두 쪽씩 나오게 하므로 기본 인쇄로 바꾼다.
  3. 템플릿 B(저작재산권 비독점적 이용허락 계약서, 조항식) · C(저작물 이용허락 계약서, 표 양식)를 더해
     A 50 % · B 25 % · C 25 % 를 칸(매체×유형)마다 고르게 배정한다.
  4. 튜닝에 쓴 58건은 목록 단계(build_test_set_v5)에서 이미 빠져 있다.
  계약서와 상관없는 표시값(주소 도로명, 이용료)은 실제 계약서처럼 보이게 다시 만든다 — 시험 항목이 아니다.
  정답은 늘 계약서에 인쇄된 값이다 (같은 values 레코드에서 계약서와 정답을 함께 만든다).

단계
  values  → dataset/test_set_v5/values.jsonl          (계약서 1건 = 1행, 템플릿 배정 포함)
  hwpx    → dataset/test_set_v5/hwpx/<id> <제목>.hwpx
  pdf     → 한글 자동화(32비트 PowerShell, tools/convert_hwpx_dir.ps1) → dataset/test_set_v5/계약서/
  verify  → PDF 글자에서 제목·저작자·권리자·날짜·체크 칸을 다시 읽어 values 와 대조 → verify.json
  gold    → dataset/test_set_v5/answers/gold_v5.jsonl (Te-1 4,000 + Te-2 4,000; Te-2 의 1~16 은 같은 계약서 값)

  python -m api.module.dataset_builder.generate_contracts_v5 --stage values hwpx pdf verify --samples
  python -m api.module.dataset_builder.generate_contracts_v5 --stage all
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from . import hwpx_kit as K

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dataset/test_set_v5"
GOLD_XLSX = ROOT / "docs/1._생성계약서_GOLD 데이터.xlsx"
SHEET = "생성 계약서 데이터(전체)"
TEMPLATE_A = ROOT / "docs/2. 계약서.hwpx"
INDEX = ROOT / "dataset/gt_merged_index.csv"
SEED = "20261103"
TEMPLATE_SHARE = (("A", 0.50), ("B", 0.25), ("C", 0.25))

WIN_TMP = Path("/mnt/c/Users/user/AppData/Local/Temp/hwpx_v5")
WIN_TMP_W = r"C:\Users\user\AppData\Local\Temp\hwpx_v5"
PS32 = "/mnt/c/Windows/SysWOW64/WindowsPowerShell/v1.0/powershell.exe"
PS_SCRIPT_W = r"api\module\dataset_builder\tools\convert_hwpx_dir.ps1"

CHECKED = {"■", "v", "V"}
TYPE_COLS = [f"C_{i}" for i in range(1, 11)]
TYPE_LABELS = ["어문저작물", "음악저작물", "연극저작물", "미술저작물", "건축저작물", "사진저작물", "영상저작물",
               "도형저작물", "컴퓨터프로그램저작물", "기타"]
RIGHTS = [("R_1", "복제권"), ("R_2", "공연권"), ("R_3", "공중송신권"), ("R_4", "전시권"), ("R_5", "배포권"),
          ("R_6", "대여권"), ("R_7", "2차적저작물작성권")]
SUB_RIGHTS = [("R_3_1", "방송권"), ("R_3_2", "전송권"), ("R_3_3", "디지털음성송신권")]
MEDIA_TYPE_COL = {"text": "C_1", "image": "C_6", "video": "C_7"}
KCC = "한국저작권위원회"

ANNOTATION_RE = re.compile(r"\(\s*(?:저작물\s*)?\d+\s*건\s*\)")
BIZNO_RE = re.compile(r"^\d{3}-\d{2}-\d{5}$")
BIRTH_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")

# 시트 주소는 시·구는 실제, 도로명은 '표본로'·'테스트로' 같은 자리표시다 → 그 구의 실제 도로명으로 바꾼다
ROADS = {
    "광주광역시 북구": ["우치로", "서암대로", "설죽로", "북문대로"],
    "부산광역시 연제구": ["중앙대로", "월드컵대로", "과정로", "연수로"],
    "서울특별시 중구": ["세종대로", "을지로", "퇴계로", "남대문로"],
    "대구광역시 중구": ["국채보상로", "달구벌대로", "동덕로", "중앙대로"],
    "대전광역시 서구": ["둔산로", "대덕대로", "계룡로", "한밭대로"],
}
EXTRA_DISTRICTS = {
    "서울특별시 마포구": ["월드컵북로", "양화로", "독막로"], "서울특별시 강남구": ["테헤란로", "도산대로", "봉은사로"],
    "서울특별시 종로구": ["율곡로", "삼일대로", "대학로"], "경기도 성남시 분당구": ["판교역로", "황새울로", "성남대로"],
    "경기도 고양시 일산동구": ["중앙로", "정발산로", "백석로"], "경기도 수원시 팔달구": ["효원로", "경수대로", "매산로"],
    "인천광역시 남동구": ["인주대로", "구월로", "예술로"], "강원특별자치도 춘천시": ["중앙로", "공지로", "후석로"],
    "충청북도 청주시 상당구": ["상당로", "대성로"], "전북특별자치도 전주시 완산구": ["팔달로", "효자로", "전주천동로"],
    "경상남도 창원시 성산구": ["중앙대로", "원이대로", "용지로"], "제주특별자치도 제주시": ["문연로", "연북로", "중앙로"],
}
SURNAMES = list("김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허유남심노하곽성차주우구민나진지엄채원천방공현")
GIVEN = ["민준", "서연", "지훈", "하은", "도윤", "예린", "현우", "수빈", "정호", "은영", "성민", "지아", "태윤", "다은",
         "상우", "유진", "재원", "소희", "동현", "미경", "영수", "혜진", "준호", "가영", "승현", "나연", "우진", "보람",
         "경민", "주희", "시우", "윤서", "건우", "채원", "형준", "수정", "지성", "은지", "병철", "선희"]
ORG_STEMS = ["한빛", "다온", "누리", "청람", "소담", "미르", "가람", "하람", "별빛", "새움", "푸른숲", "늘봄", "한결",
             "바른", "해솔", "단비", "도담", "라온", "온새미로", "나래", "솔빛", "그린나래", "이음", "다올"]
ORG_KINDS = ["미디어", "출판", "콘텐츠", "디자인", "에듀", "필름", "아카이브", "커뮤니케이션즈", "스튜디오", "북스",
             "프로덕션", "교육연구소", "문화기획", "영상", "아트"]


# ================================================================== 공용
def rng_for(*parts: Any) -> random.Random:
    h = hashlib.sha256(":".join([SEED, *map(str, parts)]).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def nfc(s: Any) -> str:
    return unicodedata.normalize("NFC", "" if s is None else str(s)).strip()


def clean_names(raw: Any) -> str:
    """'신애리 (저작물 197 건)\\n강은서' → '신애리, 강은서'."""
    text = ANNOTATION_RE.sub("", nfc(raw))
    names = [re.sub(r"\s+", " ", n).strip(" ,") for n in re.split(r"[\r\n]+", text)]
    out: list[str] = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return ", ".join(out)


def bizno(rng: random.Random) -> str:
    d = [rng.randint(0, 9) for _ in range(9)]
    w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    s = sum(a * b for a, b in zip(d, w)) + (d[8] * 5) // 10
    n = "".join(map(str, d + [(10 - s % 10) % 10]))
    return f"{n[:3]}-{n[3:5]}-{n[5:]}"


def birth(rng: random.Random) -> str:
    day = dt.date(1955, 1, 1) + dt.timedelta(days=rng.randint(0, 40 * 365))
    return day.strftime("%Y%m%d")


def real_address(addr: str, rng: random.Random) -> str:
    """'광주광역시 북구 표본로 114' → '광주광역시 북구 우치로 114'. 모르는 형식이면 그대로."""
    for district, roads in ROADS.items():
        if addr.startswith(district):
            m = re.match(rf"^{re.escape(district)}\s+\S+로\s+(\d+)(.*)$", addr)
            if m:
                return f"{district} {rng.choice(roads)} {m.group(1)}{m.group(2)}"
    return addr


def new_address(rng: random.Random) -> str:
    pool = {**ROADS, **EXTRA_DISTRICTS}
    district = rng.choice(sorted(pool))
    num = rng.randint(1, 320)
    tail = rng.choice(["", "", f", {rng.randint(2, 12)}층", f", {rng.randint(101, 1204)}호"])
    return f"{district} {rng.choice(pool[district])} {num}{tail}"


def person_name(rng: random.Random) -> str:
    return rng.choice(SURNAMES) + rng.choice(GIVEN)


def new_party(rng: random.Random, avoid: set[str]) -> dict[str, str]:
    """가상의 이용자 — 회사 60 %, 개인 40 %. 실재 기관 이름은 쓰지 않는다."""
    for _ in range(20):
        if rng.random() < 0.6:
            stem, kind = rng.choice(ORG_STEMS), rng.choice(ORG_KINDS)
            name = rng.choice([f"주식회사 {stem}{kind}", f"(주){stem}{kind}", f"{stem}{kind} 주식회사",
                               f"{stem}{kind}", f"(유){stem}{kind}"])
            party = {"name": name, "id": bizno(rng), "addr": new_address(rng)}
        else:
            party = {"name": person_name(rng), "id": birth(rng), "addr": new_address(rng)}
        if party["name"] not in avoid:
            return party
    raise RuntimeError("이용자 이름을 만들지 못했다")


def fee_amount(rng: random.Random) -> int:
    """30만~3,000만 원, 로그 균등, 만 원 단위."""
    return int(round(math.exp(rng.uniform(math.log(300_000), math.log(30_000_000))), -4))


def korean_amount(n: int) -> str:
    """1250000 → '일백이십오만' (계약서 금액 한글 표기)."""
    digits, units, big = "영일이삼사오육칠팔구", ["", "십", "백", "천"], ["", "만", "억", "조"]
    parts, i = [], 0
    while n > 0:
        chunk, n = n % 10000, n // 10000
        if chunk:
            s = "".join(digits[(chunk // 10 ** p) % 10] + units[p] for p in range(3, -1, -1) if (chunk // 10 ** p) % 10)
            parts.append(s + big[i])
        i += 1
    return "".join(reversed(parts)) or "영"


def id_label(value: str) -> str:
    return "사업자등록번호" if BIZNO_RE.match(value) else "생년월일"


def fmt_id(value: str, style: str) -> str:
    m = BIRTH_RE.match(value)
    if not m:
        return value
    y, mo, d = m.groups()
    return f"{y}년 {int(mo)}월 {int(d)}일" if style == "B" else f"{y}. {mo}. {d}."


def safe_title(title: str, limit: int = 80) -> str:
    t = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", title)
    t = re.sub(r"\s+", " ", t).strip(" .")
    return t[:limit].rstrip(" .") or "제목없음"


# ================================================================== 1. values
def load_sheet() -> dict[str, dict[str, Any]]:
    import openpyxl
    wb = openpyxl.load_workbook(GOLD_XLSX, read_only=True, data_only=True)
    it = wb[SHEET].iter_rows(values_only=True)
    head = [nfc(h) for h in next(it)]
    rows: dict[str, dict[str, Any]] = {}
    for r in it:
        if r[2] in (None, ""):
            continue
        wid = nfc(r[2])
        if wid not in rows:
            rows[wid] = {h: v for h, v in zip(head, r) if h}  # 겹친 열(권리자_생년월일 등)은 값이 같다 (확인 2026-11-03)
    return rows


def assign_templates(rows: list[dict[str, str]]) -> dict[str, str]:
    by_cell: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r["work_id"])
    out: dict[str, str] = {}
    for cell, wids in by_cell.items():
        wids = sorted(wids, key=lambda w: hashlib.sha256(f"{SEED}:tpl:{w}".encode()).hexdigest())
        n = len(wids)
        n_a = round(n * TEMPLATE_SHARE[0][1])
        n_b = round(n * TEMPLATE_SHARE[1][1])
        for i, w in enumerate(wids):
            out[w] = "A" if i < n_a else "B" if i < n_a + n_b else "C"
    return out


def values_from_sheet(sel: dict[str, str], row: dict[str, Any]) -> dict[str, Any]:
    wid = sel["work_id"]
    rng = rng_for("values", wid)
    creator, holder = clean_names(row["저작자"]), clean_names(row["권리자"])
    holder_id, holder_addr = nfc(row["권리자_생년월일"]), nfc(row["권리자_주소"])
    user_raw = nfc(row["이용자"])
    user, user_id, user_addr = clean_names(user_raw), nfc(row["이용자_생년월일"]), nfc(row["이용자_주소"])
    fixes: list[str] = []
    if sel["media"] == "video" and sel["license_bucket"] == "expired" and creator == KCC and holder != KCC:
        # 만료 영상 499건: 저작자 = 한국저작권위원회, 권리자 = 감독 으로 뒤바뀌어 있다
        creator, holder = holder, KCC
        holder_id, holder_addr = (user_id, user_addr) if user == KCC else (bizno(rng), new_address(rng))
        fixes.append("swap_creator_holder")
    if user in ("", "-", "소속없음") or user == holder:
        p = new_party(rng, {holder, creator})
        user, user_id, user_addr = p["name"], p["id"], p["addr"]
        fixes.append("new_licensee")
    if raw_has_annotation(row):
        fixes.append("clean_names")
    v = base_values(sel, rng)
    v.update({
        "제목": nfc(row["제목"]), "저작자": creator, "권리자": holder, "이용자": user,
        "권리자_번호": holder_id, "권리자_주소": real_address(holder_addr, rng),
        "이용자_번호": user_id, "이용자_주소": real_address(user_addr, rng),
        "start": dt.date(int(row["이용허락기간_SY"]), int(row["이용허락기간_SM"]), int(row["이용허락기간_SD"])).isoformat(),
        "end": dt.date(int(row["이용허락기간_EY"]), int(row["이용허락기간_EM"]), int(row["이용허락기간_ED"])).isoformat(),
        "source": "gold_sheet", "fixes": fixes,
    })
    for col in TYPE_COLS + [c for c, _ in RIGHTS] + [c for c, _ in SUB_RIGHTS] + ["Ls_1", "Ls_2"]:
        v[col] = nfc(row[col])
    v["contract_date"] = contract_date(v["start"], rng)
    return v


def raw_has_annotation(row: dict[str, Any]) -> bool:
    return any(ANNOTATION_RE.search(nfc(row.get(c))) or "\n" in str(row.get(c) or "") for c in ("저작자", "권리자", "이용자"))


def values_new(sel: dict[str, str], irow: dict[str, str]) -> dict[str, Any]:
    """시트에 없는 86건: 제목·저작자는 사이트 메타데이터, 계약 조건은 시트와 같은 방식으로 뽑는다."""
    wid = sel["work_id"]
    rng = rng_for("values", wid)
    title = nfc(irow.get("work_제목") or irow.get("title"))
    author = clean_names(irow.get("work_저자"))
    holder_site = clean_names(irow.get("contract_meta_copyright_holder_name"))
    if sel["media"] == "video" and sel["license_bucket"] == "expired":
        holder = KCC
    else:
        holder = holder_site if holder_site and holder_site != "소속없음" else author
    glyph = rng.choice(sorted(CHECKED))
    v = base_values(sel, rng)
    for col in TYPE_COLS:
        v[col] = glyph if col == MEDIA_TYPE_COL[sel["media"]] else "□"
    picks = {c: rng.random() < 0.45 for c, _ in RIGHTS}
    if not any(picks.values()):
        picks[rng.choice([c for c, _ in RIGHTS])] = True
    for c, _ in RIGHTS:
        v[c] = glyph if picks[c] else "□"
    for c, _ in SUB_RIGHTS:
        v[c] = v["R_3"]
    v["Ls_1"] = v["Ls_2"] = glyph
    start = dt.date(2000, 1, 1) + dt.timedelta(days=rng.randint(0, 25 * 365))
    end = min(start + dt.timedelta(days=rng.randint(180, 3650)), dt.date(2035, 12, 31))
    holder_is_org = holder == KCC or any(m in holder for m in ("원", "회", "청", "관", "공사", "센터", "재단", "주식회사", "(주)", "방송"))
    user = new_party(rng, {holder, author})
    v.update({
        "제목": title, "저작자": author, "권리자": holder, "이용자": user["name"],
        "권리자_번호": bizno(rng) if holder_is_org else birth(rng), "권리자_주소": new_address(rng),
        "이용자_번호": user["id"], "이용자_주소": user["addr"],
        "start": start.isoformat(), "end": end.isoformat(), "source": "new", "fixes": ["new_row"],
    })
    v["contract_date"] = contract_date(v["start"], rng)
    return v


def base_values(sel: dict[str, str], rng: random.Random) -> dict[str, Any]:
    return {"work_id": sel["work_id"], "media": sel["media"], "bucket": sel["license_bucket"], "cell": sel["cell"],
            "in_te1": sel.get("in_te1") == "True", "in_te2": sel.get("in_te2") == "True",
            "work_file": sel["work_file"], "fee": fee_amount(rng)}


def contract_date(start_iso: str, rng: random.Random) -> str:
    return (dt.date.fromisoformat(start_iso) - dt.timedelta(days=rng.randint(1, 30))).isoformat()


def stage_values() -> list[dict[str, Any]]:
    sheet = load_sheet()
    index = {r["work_id"]: r for r in csv.DictReader(open(INDEX, encoding="utf-8-sig"))}
    sel = list(csv.DictReader(open(OUT / "contracts.csv", encoding="utf-8-sig")))
    tpl = assign_templates(sel)
    out = []
    for s in sel:
        v = values_from_sheet(s, sheet[s["work_id"]]) if s["work_id"] in sheet else values_new(s, index[s["work_id"]])
        v["template"] = tpl[s["work_id"]]
        v["pdf_name"] = f"{s['work_id']} {safe_title(v['제목'])}.pdf"
        out.append(v)
    names = Counter(v["pdf_name"] for v in out)
    assert max(names.values()) == 1, "PDF 이름 중복"
    with open(OUT / "values.jsonl", "w", encoding="utf-8") as f:
        for v in out:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    fx = Counter(x for v in out for x in v["fixes"])
    print(json.dumps({"contracts": len(out), "templates": Counter(v["template"] for v in out), "fixes": fx},
                     ensure_ascii=False, default=dict))
    return out


def load_values() -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(OUT / "values.jsonl", encoding="utf-8")]


# ================================================================== 2. 템플릿
def ymd(iso: str) -> tuple[str, str, str]:
    d = dt.date.fromisoformat(iso)
    return str(d.year), f"{d.month:02d}", f"{d.day:02d}"


def date_ko(iso: str, pad: bool = True) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.year}년 {d.month:02d}월 {d.day:02d}일" if pad else f"{d.year}년 {d.month}월 {d.day}일"


def date_dot(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.year}. {d.month:02d}. {d.day:02d}."


class Templates:
    """원본 HWPX 를 한 번 읽어 A/B/C 의 바탕을 만든다."""

    # B·C 에 덧붙이는 글자 모양·테두리 번호 (원본 header 는 charPr 0–19, borderFill 1–5)
    CH_TITLE_B, CH_BOLD_B, CH_BOLD_C, CH_TITLE_C, CH_SMALL_C = 20, 21, 22, 23, 24
    BF_SHADE = 6

    def __init__(self, path: Path = TEMPLATE_A):
        base = K.read_parts(path)
        base.pop("Preview/PrvImage.png", None)  # 미리보기에는 자리표시 원문이 들어 있다 — 한글이 저장할 때 다시 만든다
        base["Preview/PrvText.txt"] = b""
        base["settings.xml"] = K.set_print_method(base["settings.xml"].decode("utf-8")).encode("utf-8")
        header = base["Contents/header.xml"].decode("utf-8")
        header = K.set_char_color(header, [18, 19])                                  # 값 빨강 → 검정
        header = K.clone_char_pr(header, 16, self.CH_TITLE_B, height=1700)           # 휴먼명조 굵게 17pt
        header = K.clone_char_pr(header, 0, self.CH_BOLD_B, bold=True)               # 함초롬바탕 굵게 10pt
        header = K.clone_char_pr(header, 1, self.CH_BOLD_C, bold=True)               # 함초롬돋움 굵게 10pt
        header = K.clone_char_pr(header, 8, self.CH_TITLE_C, height=1800)            # HY헤드라인M 18pt
        header = K.clone_char_pr(header, 1, self.CH_SMALL_C, height=900)             # 함초롬돋움 9pt
        header = K.add_border_fill(header, self.BF_SHADE, face="#E7E6E6")
        base["Contents/header.xml"] = header.encode("utf-8")
        sec = base["Contents/section0.xml"].decode("utf-8")
        self.head, self.first, self.tail = K.section_shell(sec)
        self.base = base
        self.sec_a = self._prepare_a(sec)

    @staticmethod
    def _prepare_a(sec: str) -> str:
        fixes = [("{{파일명}}", "{{제목}}"), ("“권리자”이라 함", "“권리자”라 함")]
        for old, new in fixes:
            assert old in sec, old
            sec = sec.replace(old, new)
        # 원본에 저장된 줄 배치(자리표시 글자 기준)를 지운다 — 남겨 두면 긴 제목이 들어간 문단의 마지막 줄이
        # 옛 줄 위치에 맞춰 한 줄로 짓눌려 찍힌다. 한글이 열 때 다시 배치한다.
        sec = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", sec, flags=re.S)
        label = "<hp:t>생년월일 : </hp:t>"
        assert sec.count(label) == 2
        sec = sec.replace(label, "<hp:t>{{권리자_번호라벨}} : </hp:t>", 1).replace(label, "<hp:t>{{이용자_번호라벨}} : </hp:t>", 1)
        return sec

    # ---------------------------------------------------------------- A
    def render_a(self, v: dict[str, Any]) -> dict[str, bytes]:
        sy, sm, sd = ymd(v["start"])
        ey, em, ed = ymd(v["end"])
        ly, lm, ld = ymd(v["contract_date"])
        subs = {
            "제목": v["제목"], "저작물명": v["제목"], "저작자": v["저작자"], "권리자": v["권리자"], "이용자": v["이용자"],
            "C_기타": "-", "Ls_1": v["Ls_1"], "Ls_2": v["Ls_2"], "Ls_3": f"{v['fee']:,}",
            "이용허락기간_SY": sy, "이용허락기간_SM": sm, "이용허락기간_SD": sd,
            "이용허락기간_EY": ey, "이용허락기간_EM": em, "이용허락기간_ED": ed,
            "계약시기_LY": ly, "계약시기_LM": lm, "계약시기_LD": ld,
            "권리자_번호라벨": id_label(v["권리자_번호"]), "권리자_생년월일": fmt_id(v["권리자_번호"], "A"),
            "권리자_주소": v["권리자_주소"],
            "이용자_번호라벨": id_label(v["이용자_번호"]), "이용자_생년월일": fmt_id(v["이용자_번호"], "A"),
            "이용자_주소": v["이용자_주소"],
            **{c: v[c] for c in TYPE_COLS + [c for c, _ in RIGHTS] + [c for c, _ in SUB_RIGHTS]},
        }
        missing: set[str] = set()

        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key not in subs:
                missing.add(key)
                return m.group(0)
            return K.esc(subs[key])

        sec = re.sub(r"\{\{([^}]+)\}\}", sub, self.sec_a)
        if missing:
            raise KeyError(f"템플릿 A 자리표시 값 없음: {sorted(missing)}")
        return {**self.base, "Contents/section0.xml": sec.encode("utf-8")}

    def _assemble(self, body: list[str]) -> dict[str, bytes]:
        sec = self.head + self.first + "".join(body) + self.tail
        return {**self.base, "Contents/section0.xml": sec.encode("utf-8")}

    @staticmethod
    def type_lines(v: dict[str, Any], split: tuple[int, ...], sep: str = "   ") -> list[str]:
        """저작물 종류 체크 목록을 split 개수씩 줄로 — 한 줄이 넘쳐 '건축저/작물'처럼 낱말이 갈리지 않게."""
        items = [f"{v[c]} {lab}" for c, lab in zip(TYPE_COLS, TYPE_LABELS)]
        items[-1] = f"{v['C_10']} 기타(   )"
        out, i = [], 0
        for n in split:
            out.append(sep.join(items[i:i + n]))
            i += n
        return out

    @staticmethod
    def rights_lines(v: dict[str, Any], sep: str = "   ") -> list[str]:
        subs = ", ".join(f"{v[c]} {lab}" for c, lab in SUB_RIGHTS)
        first = sep.join(f"{v[c]} {lab}" for c, lab in RIGHTS[:2]) + f"{sep}{v['R_3']} 공중송신권({subs})"
        return [first, sep.join(f"{v[c]} {lab}" for c, lab in RIGHTS[3:])]

    @staticmethod
    def type_line(v: dict[str, Any], sep: str = "  ") -> str:
        items = [f"{v[c]} {lab}" for c, lab in zip(TYPE_COLS, TYPE_LABELS)]
        items[-1] = f"{v['C_10']} 기타(   )"
        return sep.join(items)

    @staticmethod
    def rights_line(v: dict[str, Any], sep: str = "  ") -> str:
        subs = ", ".join(f"{v[c]} {lab}" for c, lab in SUB_RIGHTS)
        items = []
        for c, lab in RIGHTS:
            items.append(f"{v[c]} {lab}" + (f"({subs})" if c == "R_3" else ""))
        return sep.join(items)

    # ---------------------------------------------------------------- B: 비독점 이용허락, 조항식
    def render_b(self, v: dict[str, Any]) -> dict[str, bytes]:
        T, BOLD, BODY = self.CH_TITLE_B, self.CH_BOLD_B, 0
        fee = v["fee"]
        start, end = date_ko(v["start"], pad=False), date_ko(v["end"], pad=False)
        art = []

        def article(title: str, *paras: str) -> None:
            art.append(K.para(title, pr=13, char=BOLD))
            for p in paras:
                art.append(K.para(p, pr=20, char=BODY))

        body = [K.para("저작재산권 비독점적 이용허락 계약서", pr=21, char=T), K.blank(BODY),
                K.para([(BODY, "저작재산권자 "), (BOLD, v["권리자"]), (BODY, "(이하 “권리자”라 한다)와 이용자 "),
                        (BOLD, v["이용자"]), (BODY, "(이하 “이용자”라 한다)는 아래에 표시한 저작물의 저작재산권 "
                                                   "비독점적 이용허락에 관하여 다음과 같이 계약을 체결한다.")], pr=20)]
        article("제1조 (목적)", "이 계약은 권리자가 이용자에게 대상저작물의 이용을 비독점적으로 허락하고 이용자가 그 대가를 "
                             "지급하는 것과 관련하여 당사자 사이의 권리와 의무를 정하는 것을 목적으로 한다.")
        art.append(K.para("제2조 (대상저작물)", pr=13, char=BOLD))
        art.append(K.para("이 계약의 대상이 되는 저작물(이하 “대상저작물”이라 한다)은 다음과 같다.", pr=20, char=BODY))
        art.append(K.para([(BODY, "1. 제호 : "), (BODY, v["제목"])], pr=2))
        art.append(K.para([(BODY, "2. 저작자 : "), (BODY, v["저작자"])], pr=2))
        art.append(K.para("3. 저작물의 종류 :", pr=2, char=BODY))
        for line in self.type_lines(v, (5, 5)):
            art.append(K.para(line, pr=16, char=BODY))
        art.append(K.blank(BODY))
        article("제3조 (이용허락의 범위)",
                "① 권리자는 이용자에게 대상저작물에 관한 다음 저작재산권 가운데 표시한 권리를 비독점적으로 행사하도록 허락한다.")
        for line in self.rights_lines(v):
            art.append(K.para(line, pr=16, char=BODY))
        art.append(K.para("② 권리자는 이 계약의 기간 중에도 제3자에게 대상저작물의 이용을 허락할 수 있다.", pr=20, char=BODY))
        art.append(K.para("③ 이용자는 권리자의 사전 서면 동의 없이 제3자에게 대상저작물의 이용을 다시 허락하거나 이 계약상의 "
                          "지위를 양도할 수 없다.", pr=20, char=BODY))
        article("제4조 (이용허락 기간)",
                f"대상저작물의 이용허락 기간은 {start}부터 {end}까지로 한다. 기간을 연장하려는 당사자는 기간이 끝나기 "
                "1개월 전까지 상대방에게 서면으로 알리고, 연장 조건은 당사자가 서면으로 합의하여 정한다.")
        article("제5조 (이용료)",
                f"① 이용자는 권리자에게 이용료로 금 {korean_amount(fee)}원정(₩{fee:,})을 {start}까지 일시금으로 지급한다.",
                "② 권리자는 이용료를 받은 날부터 7일 이내에 대상저작물의 이용에 필요한 자료를 이용자에게 인도한다.")
        article("제6조 (저작인격권의 존중)",
                "이용자는 대상저작물을 이용할 때 저작자의 실명 또는 이명을 표시하여야 하며, 대상저작물의 내용·형식 및 "
                "제호의 동일성을 해치지 아니한다. 다만, 이용 매체의 특성상 불가피한 범위의 수정은 권리자에게 미리 알린 후 할 수 있다.")
        article("제7조 (권리자의 보증)",
                "권리자는 이 계약을 체결할 정당한 권한을 가지고 있으며, 대상저작물이 제3자의 저작권 그 밖의 권리를 침해하지 "
                "아니함을 보증한다. 이를 위반하여 이용자에게 손해가 생긴 경우 권리자가 그 손해를 배상한다.")
        article("제8조 (계약의 해지)",
                "① 당사자 일방이 이 계약을 위반한 경우 상대방은 14일 이상의 기간을 정하여 서면으로 시정을 요구할 수 있으며, "
                "그 기간 안에 시정되지 아니하면 이 계약을 해지할 수 있다.",
                "② 계약의 해지는 손해배상 청구에 영향을 미치지 아니한다.")
        article("제9조 (분쟁의 해결)",
                "이 계약과 관련하여 분쟁이 생긴 경우 당사자는 협의로 해결하도록 노력하며, 협의가 이루어지지 아니하면 "
                "한국저작권위원회에 조정을 신청하거나 민사소송법에 따른 관할법원에 소를 제기할 수 있다.")
        article("제10조 (그 밖의 사항)",
                "이 계약에서 정하지 아니한 사항은 저작권법, 민법 등 관계 법령과 일반 관례에 따른다.")
        sign = [K.blank(BODY),
                K.para("이 계약의 성립을 증명하기 위하여 계약서 2통을 작성하여 당사자가 서명 또는 날인한 후 각 1통씩 보관한다.",
                       pr=20, char=BODY),
                K.blank(BODY), K.para(date_ko(v["contract_date"], pad=False), pr=21, char=BODY), K.blank(BODY)]
        for role, who in (("권리자", "권리자"), ("이용자", "이용자")):
            sign += [K.para(f"{role}", pr=12, char=BOLD),
                     K.para(f"성명(상호) : {v[who]}          (서명 또는 인)", pr=2, char=BODY),
                     K.para(f"{id_label(v[who + '_번호'])} : {fmt_id(v[who + '_번호'], 'B')}", pr=2, char=BODY),
                     K.para(f"주소 : {v[who + '_주소']}", pr=2, char=BODY), K.blank(BODY)]
        return self._assemble(body + art + sign)

    # ---------------------------------------------------------------- C: 표 양식
    def render_c(self, v: dict[str, Any]) -> dict[str, bytes]:
        T, BOLD, BODY, SMALL = self.CH_TITLE_C, self.CH_BOLD_C, 1, self.CH_SMALL_C
        shade = self.BF_SHADE
        fee = v["fee"]

        def label(text: str) -> K.Cell:
            return K.Cell([K.para(text, pr=21, char=BOLD)], border_fill=shade)

        def value(*lines: str) -> K.Cell:
            return K.Cell([K.para(t, pr=12, char=BODY) for t in lines])

        rights1, rights2 = self.rights_lines(v)
        types = self.type_lines(v, (4, 4, 2))
        work = K.Table(rows=[
            [label("제호(제목)"), value(v["제목"])],
            [label("저작자"), value(v["저작자"])],
            [label("저작재산권자"), value(v["권리자"])],
            [label("저작물의 종류"), value(*types)],
            [label("이용허락 권리"), value(rights1, rights2)],
            [label("이용허락 기간"), value(f"{date_dot(v['start'])} ~ {date_dot(v['end'])}")],
            [label("이용료"), value(f"금 {fee:,}원 (일시금, 부가가치세 별도)",
                                    f"지급기한 : {date_dot(v['start'])}")],
        ], col_widths=[10200, 37600], row_height=1800, table_id=1900000001)
        sign = K.Table(rows=[
            [label("계약일"), K.Cell([K.para(date_ko(v["contract_date"], pad=False), pr=12, char=BODY)], col_span=2)],
            [label("구분"), label("갑 (권리자)"), label("을 (이용자)")],
            [label("성명(상호)"), value(v["권리자"]), value(v["이용자"])],
            [label("생년월일 /\n사업자등록번호".replace("\n", " ")), value(fmt_id(v["권리자_번호"], "C")), value(fmt_id(v["이용자_번호"], "C"))],
            [label("주소"), value(v["권리자_주소"]), value(v["이용자_주소"])],
            [label("서명"), value("                     (인)"), value("                     (인)")],
        ], col_widths=[10200, 18800, 18800], row_height=1800, table_id=1900000002, page_break="NONE")
        clauses = [
            ("제1조(목적)", "이 계약은 갑이 을에게 위 표에 적은 대상저작물의 이용을 허락하고, 을이 그 대가를 지급하는 데 필요한 "
                          "사항을 정함을 목적으로 한다."),
            ("제2조(이용허락)", "갑은 을에게 위 표의 이용허락 권리를 이용허락 기간 동안 행사하도록 허락한다. 표에 표시하지 아니한 "
                             "권리는 허락하지 아니한 것으로 본다."),
            ("제3조(이용료)", "을은 위 표의 이용료를 지급기한까지 갑이 지정하는 계좌로 지급한다."),
            ("제4조(성명표시)", "을은 대상저작물을 이용할 때 저작자와 저작재산권자의 성명을 표시하고, 대상저작물의 동일성을 "
                             "해치는 변경을 하지 아니한다."),
            ("제5조(보증)", "갑은 대상저작물에 관하여 이 계약을 체결할 권한이 있음을 보증한다."),
            ("제6조(재이용허락 금지)", "을은 갑의 서면 동의 없이 대상저작물의 이용을 제3자에게 다시 허락하지 아니한다."),
            ("제7조(해지)", "당사자 일방이 이 계약을 위반하고 상대방의 시정 요구를 받은 날부터 14일 안에 시정하지 아니하면 "
                          "상대방은 이 계약을 해지할 수 있다."),
            ("제8조(분쟁해결)", "이 계약에 관한 분쟁은 당사자가 협의하여 해결하고, 협의가 되지 아니하면 갑의 주소지를 관할하는 "
                             "법원을 관할법원으로 한다."),
        ]
        body = [K.para("저작물 이용허락 계약서", pr=21, char=T), K.blank(BODY),
                K.para([(BOLD, v["권리자"]), (BODY, "(이하 “갑”이라 한다)과 "), (BOLD, v["이용자"]),
                        (BODY, "(이하 “을”이라 한다)은 아래 표의 저작물(이하 “대상저작물”이라 한다)의 이용허락에 관하여 "
                               "다음과 같이 계약을 체결한다.")], pr=20),
                K.para("1. 대상저작물 및 이용 조건", pr=13, char=BOLD), work.xml(anchor_char=BODY), K.blank(SMALL),
                K.para("2. 계약 조항", pr=12, char=BOLD)]
        body += [K.para([(BOLD, h + " "), (BODY, t)], pr=0) for h, t in clauses]
        body += [K.para("위 계약을 증명하기 위하여 계약서 2부를 작성하고 갑과 을이 서명 또는 날인하여 각 1부씩 보관한다.",
                        pr=20, char=BODY),
                 sign.xml(anchor_char=BODY)]
        return self._assemble(body)

    def render(self, v: dict[str, Any]) -> dict[str, bytes]:
        return {"A": self.render_a, "B": self.render_b, "C": self.render_c}[v["template"]](v)


def stage_hwpx(values: list[dict[str, Any]], hwpx_dir: Path) -> None:
    t = Templates()
    hwpx_dir.mkdir(parents=True, exist_ok=True)
    for v in values:
        K.write_parts(t.render(v), hwpx_dir / (Path(v["pdf_name"]).stem + ".hwpx"))
    print(f"hwpx {len(values)} → {hwpx_dir}")


# ================================================================== 3. pdf (한글 자동화)
def stage_pdf(values: list[dict[str, Any]], hwpx_dir: Path, pdf_dir: Path, batch: int = 100, force: bool = False) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    todo = [v for v in values if force or not (pdf_dir / v["pdf_name"]).exists()]
    print(f"pdf: {len(todo)} to convert ({len(values) - len(todo)} already done)")
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        tag = f"b{b // batch:03d}"
        src, dst = WIN_TMP / tag / "in", WIN_TMP / tag / "out"
        shutil.rmtree(WIN_TMP / tag, ignore_errors=True)
        src.mkdir(parents=True)
        for v in chunk:
            shutil.copy2(hwpx_dir / (Path(v["pdf_name"]).stem + ".hwpx"), src)
        cmd = [PS32, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS_SCRIPT_W,
               "-In", f"{WIN_TMP_W}\\{tag}\\in", "-Out", f"{WIN_TMP_W}\\{tag}\\out"]
        try:  # 한 건 ~1.5 초 + 시작 ~15 초. 한글 창을 따로 열어 두면 자동화가 멈춘다 — 묶음째 건너뛰고 다음 실행에서 다시 한다
            res = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=120 + 20 * len(chunk))
            log = res.stdout.decode("utf-8", "replace").strip().splitlines()
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or b"").decode("utf-8", "replace").strip().splitlines() + [
                f"TIMEOUT — 한글이 멈췄다 (열린 한글 창·대화상자 확인). 이 묶음의 남은 파일은 다음 실행에서 다시 변환"]
        print(f"  {tag}: {log[-1] if log else '?'}", flush=True)
        for line in log:
            if line.startswith("FAIL"):
                print("   ", line)
        for v in chunk:
            out = dst / v["pdf_name"]
            if out.exists() and out.stat().st_size > 0:
                shutil.copy2(out, pdf_dir / v["pdf_name"])
        shutil.rmtree(WIN_TMP / tag, ignore_errors=True)


# ================================================================== 4. verify
def pdf_read(path: Path) -> tuple[list[str], int]:
    """(글자 두 벌: 읽기 순서 · 배치 유지, 쪽수). 가로 쪽(2쪽 모아 찍기)이면 쪽수를 음수로 돌려 실패로 본다."""
    texts = [subprocess.run(["pdftotext", *mode, str(path), "-"], capture_output=True).stdout.decode("utf-8", "replace")
             for mode in ([], ["-layout"])]
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True).stdout.decode("utf-8", "replace")
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1)) if "Pages:" in info else 0
    size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
    if size and float(size.group(1)) > float(size.group(2)):
        pages = -pages
    return [unicodedata.normalize("NFC", t) for t in texts], pages


def compact(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def verify_one(v: dict[str, Any], pdf: Path) -> tuple[list[str], int]:
    if not pdf.exists():
        return ["pdf 없음"], 0
    texts, pages = pdf_read(pdf)
    flats = [compact(t) for t in texts]
    has = lambda needle: any(compact(needle) in f for f in flats)
    problems = []
    if pages <= 0 or pages > 8:
        problems.append(f"쪽수/방향 이상: {pages}")
    for key in ("제목", "저작자", "권리자"):
        for name in [v[key]] if key == "제목" else v[key].split(", "):
            if not has(name):
                problems.append(f"{key} 없음: {name}")
    fmt = {"A": lambda d: date_ko(d), "B": lambda d: date_ko(d, pad=False), "C": date_dot}[v["template"]]
    for key in ("start", "end"):
        if not has(fmt(v[key])):
            problems.append(f"{key} 날짜 없음: {fmt(v[key])}")
    labels = list(zip(TYPE_COLS, [l if l != "기타" else "기타(" for l in TYPE_LABELS])) + RIGHTS + SUB_RIGHTS
    for col, lab in labels:
        if v["template"] == "A" and col == "R_7":
            lab = "2차적저작물"
        found = [g for f in flats for g in re.findall(rf"([■vV□]){re.escape(compact(lab))}", f)]
        if not found:
            problems.append(f"체크 칸 없음: {lab}")
            continue
        want = v[col] in CHECKED
        if any((g in CHECKED) != want for g in found):
            problems.append(f"체크 불일치: {lab} 기대 {v[col]} 실제 {sorted(set(found))}")
    return problems, pages


def stage_verify(values: list[dict[str, Any]], pdf_dir: Path) -> dict[str, Any]:
    bad = {}
    pages = Counter()
    for v in values:
        p, n = verify_one(v, pdf_dir / v["pdf_name"])
        if p:
            bad[v["pdf_name"]] = p
        pages[(v["template"], n)] += 1
    rep = {"checked": len(values), "failed": len(bad), "pages": {f"{t}:{n}": c for (t, n), c in sorted(pages.items())},
           "problems": bad}
    name = "verify.json" if len(values) > 50 else "verify_samples.json"
    (OUT / name).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("checked", "failed", "pages")}, ensure_ascii=False))
    for nm, p in list(bad.items())[:10]:
        print("  ", nm, p[:4])
    return rep


# ================================================================== 5. gold
def ledger_row(v: dict[str, Any]) -> dict[str, str]:
    """gold23.ledger_keys 가 읽는 원장 행 모양으로 바꾼다 (값은 계약서에 인쇄한 그대로)."""
    sy, sm, sd = ymd(v["start"])
    ey, em, ed = ymd(v["end"])
    row = {"파일명": Path(v["pdf_name"]).stem, "제목": v["제목"], "저작자": v["저작자"], "권리자": v["권리자"],
           "이용허락기간_SY": sy, "이용허락기간_SM": sm, "이용허락기간_SD": sd,
           "이용허락기간_EY": ey, "이용허락기간_EM": em, "이용허락기간_ED": ed}
    row.update({c: v[c] for c in TYPE_COLS + [c for c, _ in RIGHTS]})
    return row


def stage_gold(values: list[dict[str, Any]], pdf_dir: Path) -> None:
    from api.module.evaluation.v4 import gold23 as G
    from api.module.evaluation.v4.keys23 import CONTRACT_ITEM_KEYS, KEY_NAMES, T002_KEYS

    G.configure_logging("WARNING")
    by_id = {v["work_id"]: v for v in values}
    contract_keys = {}
    for v in values:
        keys = G.ledger_keys(ledger_row(v))
        keys["file_name"] = G.make_cell(v["pdf_name"], G.STATE_VALUE, "generated contract pdf", G.TIER_FILE)
        contract_keys[v["work_id"]] = keys
    # 목록 파일에 계약서 경로를 적어 둔다 (gold23·콘솔이 쓰는 contract_pdf 열)
    for name in ("te1_ids.csv", "te2_ids.csv"):
        rows = list(csv.DictReader(open(OUT / name, encoding="utf-8-sig")))
        for r in rows:
            r["contract_pdf"] = str((pdf_dir / by_id[r["work_id"]]["pdf_name"]).relative_to(ROOT))
            r["title"] = by_id[r["work_id"]]["제목"]
        with open(OUT / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    verified = OUT / "answers" / "gold_verified_v5.jsonl"
    if not verified.exists():
        raise FileNotFoundError(f"{verified} 없음 — 먼저 --stage review 로 93건 검수 결과를 합친다")
    te2 = G.build_gold("T-002", OUT / "te2_ids.csv", INDEX, None, DRAFTS, verified, "verified",
                       True, ROOT / "dataset/_probe/file_sha256.jsonl")
    # 이미지 해상도 = 실제 파일의 크기 (시스템과 같이 Pillow 원본 크기, 회전 정보 미적용). 사이트 게시값이 시험에 주는
    # 파일과 다른 이미지가 60건 있었다 (2026-11-10 G0050 내부 시험: 13239516 게시 496x280 ↔ 파일 3591x2166).
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    res_fixed = 0
    for rec in te2:
        cell = rec["keys"]["resolution"]
        if rec["media"] != "image" or cell["state"] != "value":
            continue
        with Image.open(by_id[rec["set_id"]]["work_file"]) as im:
            measured = f"{im.width}x{im.height}"
        if measured != cell["value"]:
            rec["keys"]["resolution"] = G.make_cell(measured, G.STATE_VALUE,
                                                    f"file:Pillow 원본 크기 (사이트 게시값 {cell['value']} 과 다름)", G.TIER_FILE)
            res_fixed += 1
    print(f"  이미지 해상도를 파일 실측으로 고침: {res_fixed}건")
    te1_rows = list(csv.DictReader(open(OUT / "te1_ids.csv", encoding="utf-8-sig")))
    records = []
    for r in te1_rows:
        v = by_id[r["work_id"]]
        keys = {k: contract_keys[r["work_id"]][k] for k in G.T001_KEYS}
        keys.update({k: G.na_cell("T-002 전용 키 — T-001 비채점") for k in T002_KEYS})
        records.append({"set_id": r["work_id"], "test": "T-001", "media": v["media"], "bucket": v["bucket"],
                        "source": r.get("source", ""), "template": v["template"],
                        "keys": {k: keys[k] for k in KEY_NAMES}})
    for rec in te2:
        ck = contract_keys[rec["set_id"]]
        for k in CONTRACT_ITEM_KEYS:
            rec["keys"][k] = ck[k]
        rec["template"] = by_id[rec["set_id"]]["template"]
        records.append(rec)
    out = OUT / "answers" / "gold_v5.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    states = Counter((rec["test"], k, rec["keys"][k]["state"]) for rec in records for k in KEY_NAMES)
    print(f"gold → {out}: Te-1 {len(te1_rows)}, Te-2 {len(te2)}")
    for test in ("T-001", "T-002"):
        na = {k: states[(test, k, "na")] for k in KEY_NAMES if states[(test, k, "na")] and states[(test, k, "value")]}
        print(f"  {test} 값이 일부 na 인 키: {na}")


# ================================================================== 5b. 검수 결과 합치기 (review93)
DRAFTS = ROOT / "dataset/gold_4200/drafts_combined.jsonl"
VERIFIED_OCT = ROOT / "dataset/gold_4200/description_review/gold_verified_all.jsonl"
REVIEW = OUT / "review93"
BANNED_TAGS = {"사진", "이미지", "영상", "동영상", "비디오", "한국정책방송원", "한국학중앙연구원"}


def check_review(rec: dict[str, Any], item: dict[str, Any]) -> list[str]:
    """검수 출력 한 줄을 규칙(GUIDE.md)으로 검사한다."""
    out = []
    d = rec.get("description")
    if d is not None:
        if not (30 <= len(d) <= 200):
            out.append(f"설명 길이 {len(d)}")
        if not d.rstrip().endswith("다."):
            out.append("설명 종결")
        if re.search(r"입니다|이 (이미지|영상)은|로 보인다|아름다운|멋진|평화로운", d):
            out.append("설명 금지 표현")
    tags = rec.get("hashtags")
    if tags is not None:
        if not (5 <= len(tags) <= 10):
            out.append(f"해시태그 {len(tags)}개")
        if any(not t.startswith("#") or " " in t for t in tags):
            out.append("해시태그 형식")
        if any(t.lstrip("#") in BANNED_TAGS for t in tags):
            out.append("금지 해시태그")
    if item["task"] == "write_description_and_hashtags" and (d is None or tags is None):
        out.append("설명·해시태그 둘 다 필요")
    if item["task"] == "write_hashtags" and tags is None:
        out.append("해시태그 필요")
    return out


def stage_review() -> None:
    items = {json.loads(l)["work_id"]: json.loads(l) for l in open(REVIEW / "items.jsonl", encoding="utf-8")}
    outs = {}
    for k in (1, 2, 3):
        for l in open(REVIEW / f"out_{k}.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                outs[str(r["work_id"])] = r
    missing = sorted(set(items) - set(outs))
    problems = {w: check_review(outs[w], items[w]) for w in outs if w in items}
    problems = {w: p for w, p in problems.items() if p}
    if missing or problems:
        print(json.dumps({"missing": missing, "problems": problems}, ensure_ascii=False, indent=1))
        raise SystemExit("검수 결과에 빠진 항목이나 규칙 위반이 있다")
    drafts = {}
    for l in open(DRAFTS, encoding="utf-8"):
        j = json.loads(l)
        drafts[str(j.get("work_id"))] = j
    index = {r["work_id"]: r for r in csv.DictReader(open(INDEX, encoding="utf-8-sig"))}
    add = []
    for w, it in items.items():
        r = outs[w]
        desc = r["description"] if r.get("description") is not None else (
            it["site_description"] or it["pending_description"] or nfc(index[w].get("work_설명")))
        tags = r["hashtags"] if r.get("hashtags") is not None else (it["pending_hashtags"] or [])
        tags = [t if t.startswith("#") else f"#{t}" for t in tags]
        d = drafts.get(w, {})
        add.append({
            "work_id": w, "media": it["media"], "source": it["source"], "bucket": it["cell"].split("/")[1],
            "title": it["title"], "description": desc, "hashtags": tags,
            "review_scope": "hashtags" if r.get("description") is None else ("description" if r.get("hashtags") is None else "both"),
            "gold_source": {"new": "claude_written_v5", "accept": "ai_draft_human_verified",
                            "edit": "ai_draft_human_verified"}[r["outcome"]],
            "tier": "A-검수", "verification_outcome": "accept" if r["outcome"] == "new" else r["outcome"],
            "edited": r["outcome"] == "edit", "draft_text": it["pending_description"], "draft_hashtags": it["pending_hashtags"],
            "edit_reason_codes": r.get("edit_reason_codes") or [], "verifier_id": "claude-review-v5",
            "verified_at": dt.date.today().isoformat(), "notes": r.get("notes"),
            "draft_model": d.get("model") if it["pending_description"] or it["pending_hashtags"] else None,
            "guideline_version": "desc-guideline-v0.3",
        })
    (OUT / "answers").mkdir(parents=True, exist_ok=True)
    (OUT / "answers" / "gold_verified_v5_addendum.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in add), encoding="utf-8")
    base = [json.loads(l) for l in open(VERIFIED_OCT, encoding="utf-8")]
    ids = {a["work_id"] for a in add}
    merged = [b for b in base if str(b.get("work_id")) not in ids] + add
    (OUT / "answers" / "gold_verified_v5.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in merged), encoding="utf-8")
    print(f"review: {len(add)} 건 합침 ({Counter(outs[w]['outcome'] for w in items)}) → gold_verified_v5.jsonl {len(merged)} 행")


# ================================================================== 6. Excel (장세영 시트 열 + Te-2 열 채움)
def stage_excel(values: list[dict[str, Any]]) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    gold = [json.loads(l) for l in open(OUT / "answers" / "gold_v5.jsonl", encoding="utf-8")]
    te2 = {g["set_id"]: g["keys"] for g in gold if g["test"] == "T-002"}
    bk_ko = {"expired": "만료", "donated": "기증", "ccl": "CCL", "kogl": "KOGL"}
    med_ko = {"text": "어문", "image": "이미지", "video": "영상"}
    head = (["공유형태", "저작물형태", "인덱스번호", "인덱스번호+파일명", "파일명", "파일명(확장자 미포함)", "권리자", "이용자",
             "제목", "저작자"] + TYPE_COLS + ["C_기타"] + [c for c, _ in RIGHTS[:3]] + [c for c, _ in SUB_RIGHTS]
            + [c for c, _ in RIGHTS[3:]]
            + [f"이용허락기간_{p}{u}" for p in "SE" for u in "YMD"] + ["Ls_1", "Ls_2", "Ls_3", "권리자_생년월일", "권리자_주소",
                                                                    "이용자_생년월일", "이용자_주소"]
            + ["키워드(keywords)", "해시태그(hashtags)", "파일포맷(file_format)", "파일크기(file_size)", "해상도(resolution)",
               "설명(description)", "해시데이터(file_hash)"]
            + ["Te-1", "Te-2", "계약서 파일명", "템플릿", "계약 체결일", "값 출처", "정리 내역", "설명 출처", "해시태그 출처"])
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "생성 계약서 데이터(v5)"
    ws.append(head)

    def cell_val(keys: Optional[dict], k: str):
        if not keys or keys[k]["state"] != "value":
            return None
        v = keys[k]["value"]
        if isinstance(v, list):
            return " | ".join(("#" + t.lstrip("#")) if k == "hashtags" else t for t in v)
        return v

    for v in sorted(values, key=lambda x: (x["bucket"], x["media"], x["work_id"])):
        sy, sm, sd = ymd(v["start"])
        ey, em, ed = ymd(v["end"])
        wf = Path(v["work_file"]).name
        k2 = te2.get(v["work_id"])
        row = ([bk_ko[v["bucket"]], med_ko[v["media"]], v["work_id"], f"{v['work_id']} {v['제목']}", wf, Path(wf).stem,
                v["권리자"], v["이용자"], v["제목"], v["저작자"]] + [v[c] for c in TYPE_COLS] + ["-"]
               + [v[c] for c, _ in RIGHTS[:3]] + [v[c] for c, _ in SUB_RIGHTS] + [v[c] for c, _ in RIGHTS[3:]]
               + [int(sy), sm, sd, int(ey), em, ed, v["Ls_1"], v["Ls_2"], v["fee"],
                  v["권리자_번호"], v["권리자_주소"], v["이용자_번호"], v["이용자_주소"]]
               + [cell_val(k2, k) for k in ("keywords", "hashtags", "file_format", "file_size", "resolution",
                                            "description", "file_hash")]
               + ["O" if v["in_te1"] else "", "O" if v["in_te2"] else "", v["pdf_name"], v["template"], v["contract_date"],
                  "장세영 GOLD 시트" if v["source"] == "gold_sheet" else "신규(사이트 메타데이터)", ", ".join(v["fixes"]),
                  (k2["description"].get("kind") or "") + "/" + (k2["description"].get("tier") or "") if k2 else "",
                  (k2["hashtags"].get("kind") or "") + "/" + (k2["hashtags"].get("tier") or "") if k2 else ""])
        ws.append(row)
    bold = Font(bold=True)
    for c in ws[1]:
        c.font = bold
        c.fill = PatternFill("solid", fgColor="E7E6E6")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "C2"
    for col, width in (("E", 40), ("I", 36), ("J", 20), ("G", 20), ("H", 20)):
        ws.column_dimensions[col].width = width

    s2 = wb.create_sheet("요약")
    s2.append(["항목", "값"])
    cnt = Counter((v["bucket"], v["media"]) for v in values)
    s2.append(["계약서 수", len(values)])
    s2.append(["Te-1 / Te-2", f"{sum(v['in_te1'] for v in values)} / {sum(v['in_te2'] for v in values)}"])
    s2.append(["템플릿 A / B / C", " / ".join(str(sum(v['template'] == t for v in values)) for t in "ABC")])
    for (b, m), n in sorted(cnt.items()):
        s2.append([f"{bk_ko[b]} {med_ko[m]}", n])
    for k in ("keywords", "hashtags", "file_format", "file_size", "resolution", "description", "file_hash"):
        st = Counter(g[k]["state"] for g in te2.values())
        s2.append([f"Te-2 {k} 값 있음 / 없음(na)", f"{st['value']} / {st['na']}"])
    s2.append(["정리 규칙", "만료 영상 저작자·권리자 맞바꿈; '(저작물 N 건)' 제거·두 이름 'A, B'; 이용자 '-'·'소속없음'·권리자와 같음 → 가상 이용자; "
                         "주소 도로명 실제화; 이용료 30만~3,000만 원; 계약 체결일 = 시작 1~30일 전"])
    s2.append(["키워드 na", "사이트 내용 태그가 없는 저작물 — 규칙상 키워드는 채점하지 않음"])
    s2.append(["해상도 na", "PPT 템플릿(.pptx) 9건 — 픽셀 해상도 비적용"])
    s2.column_dimensions["A"].width = 32
    s2.column_dimensions["B"].width = 120
    out = OUT / "GOLD_v5.xlsx"
    wb.save(out)
    print(f"excel → {out} ({len(values)} 행)")


# ================================================================== CLI
def pick_samples(values: list[dict[str, Any]], ids: Optional[list[str]]) -> list[dict[str, Any]]:
    if ids:
        want = set(ids)
        return [v for v in values if v["work_id"] in want]
    out, seen = [], set()
    # 템플릿마다 하나씩, 매체를 섞어서 (A=만료 영상 — 맞바꾼 당사자 확인, B=이미지, C=어문)
    prefs = [("A", "video", "expired"), ("B", "image", None), ("C", "text", None)]
    for tpl, media, bucket in prefs:
        for v in values:
            if v["template"] == tpl and v["media"] == media and (bucket is None or v["bucket"] == bucket) and tpl not in seen:
                out.append(v)
                seen.add(tpl)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["all"],
                    choices=["values", "hwpx", "pdf", "verify", "review", "gold", "excel", "all"])
    ap.add_argument("--samples", action="store_true", help="템플릿마다 1건씩만 → dataset/test_set_v5/samples/")
    ap.add_argument("--ids", nargs="*", help="이 work_id 만")
    ap.add_argument("--force", action="store_true", help="이미 있는 PDF 도 다시 변환")
    a = ap.parse_args(argv)
    stages = ["values", "hwpx", "pdf", "verify", "review", "gold", "excel"] if "all" in a.stage else a.stage
    values = stage_values() if "values" in stages else load_values()
    subset = a.samples or bool(a.ids)
    if subset:
        values = pick_samples(values, a.ids)
        sub = OUT / ("samples" if a.samples else "check")
        hwpx_dir, pdf_dir = sub / "hwpx", sub
    else:
        hwpx_dir, pdf_dir = OUT / "hwpx", OUT / "계약서"
    if "hwpx" in stages:
        stage_hwpx(values, hwpx_dir)
    if "pdf" in stages:
        stage_pdf(values, hwpx_dir, pdf_dir, force=a.force or subset)
    if "verify" in stages:
        stage_verify(values, pdf_dir)
    if "review" in stages:
        stage_review()
    if "gold" in stages and not subset:
        stage_gold(values, pdf_dir)
    if "excel" in stages and not subset:
        stage_excel(values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
