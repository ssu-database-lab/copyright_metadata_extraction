"""KOGL(공공누리) 원본 1,500건의 평가 정답셋 → E:/kogl_originals/ground_truth.jsonl

공유마당 정답(build_ground_truth.py → D:/copyright_dataset_metadata/{media}/{bucket}/
ground_truth.jsonl)과 같은 레코드 형식이다. 전에는 build_eval_manifest.py --source kogl 이
평가 세트를 만들 때마다 즉석에서 조립했고 저장된 정답 파일이 없었다. 이제 정답은 여기서
한 번 만들어 원본 색인(originals_index.xlsx) 옆에 두고, 평가 매니페스트는 이 파일을 읽는다.

입력
----
  docs/붙임1_원문저작물 메타데이터.xlsx       제목·저작권자·공공누리 유형·주제어·해시태그
  E:/kogl_originals/originals_index.xlsx     저장 파일 · 해상도(이미지 PIL / 영상 ffprobe 실측) · fps · 재생시간
  dataset/kogl_gold/kogl_detail_1500_*.jsonl 상세페이지 '저작물 설명'·화면 태그 (2026-10-02 수집)
  dataset/kogl_gold/kogl_gold.xlsx           이전 수집분 79건 — 상세페이지 수집이 없을 때만

설명 (2026-10-02 확인)
---------------------
붙임1 에는 설명 열이 없다(0/1,500). 상세페이지의 '저작물 설명' 이 유일한 출처다.
수집값을 전부 정답 파일에 남기되, 채점 가능한 것만 value 에 넣는다 — 채점기는 tier 와
상관없이 value 가 있으면 채점하기 때문이다. 나머지는 value=None + raw(원문) + 사유:
  - 원문명만 기재 ('원문명 : 箋註四家詩')   어문 323건 — 원어 제목이지 설명이 아니다
  - 촬영 장비·일시(EXIF)                     'Canon EOS 100D / 2019/04/15 …'
  - 공유마당과 같은 기준 (build_ground_truth.description_gt): 15자 미만('원본 없음',
    '입선'), 출처·제공 문구, 귀속 표기만, 제목/저자 복사
  - 상세페이지 빈 응답: 이미지 99건(전부 한국학중앙연구원)이 동일한 87,458바이트 빈
    페이지를 돌려준다. 붙임1 은 공개·미삭제로 적고 있어 '설명 없음' 이 아니라 '미확인' 이다.
설명 정답이 있는 것은 이미지 83 · 영상 131 · 어문 39 = 253건 (전체의 약 17%).

키워드 · 해시태그
-----------------
KOGL 의 주제어와 해시태그는 같은 태그 목록을 구분자만 달리 적은 것이다(1,483/1,500 일치,
나머지는 해시태그 쪽 오타 '#제주, 영상' 16건과 붙임1 에 두 번 실린 98362). 그래서:
  - 키워드 = 주제어를 '#' 단위로 파싱한 태그 목록 (채점 대상, tier A).
    예전 파서는 공백으로만 잘라 '#제주 영상' 을 '제주'·'영상' 으로 쪼개고 '충남,' 처럼
    쉼표를 붙인 토큰을 1,834개 만들었다(채점기가 구두점을 지워 점수엔 영향이 없었다).
  - 해시태그 = 해시태그 열 그대로 파싱 (채점 대상 아님 — 11속성·TTA 어디에도 없고
    파이프라인에 해시태그 출력 필드도 없다). 화면에 실제로 보이는 태그(page_value)도 함께.
태그 대부분은 내용어가 아니다 — 64~73% 의 저작물이 내용 태그를 하나도 갖지 않는다
(이미지 '#문화 #역사', 영상 '#4K #항공촬영 #제주 영상' + 기관명 + 제목 단어). 그래서 태그마다
분류(tag_classes)를 달고, 공유마당과 같은 규칙으로 5건 이상이 똑같은 태그 집합을 공유하면
batch_level 로 표시한다(채점기가 리포트에 따로 집계한다).

사용
----
  python -m api.module.dataset_builder.build_kogl_ground_truth
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .build_ground_truth import description_gt, norm

REPO = Path(__file__).resolve().parents[3]
KOGL_ROOT = Path("/mnt/e/kogl_originals")
INDEX = KOGL_ROOT / "originals_index.xlsx"
OUT = KOGL_ROOT / "ground_truth.jsonl"
EXPORT = REPO / "docs" / "붙임1_원문저작물 메타데이터.xlsx"
GOLD_DIR = REPO / "dataset" / "kogl_gold"

MEDIA = {"이미지": "image", "영상": "video", "어문": "text"}
STUB_MAX_BYTES = 130_000        # 정상 상세페이지 ≈ 178~191KB, 빈 페이지 = 87,458B
BATCH_MIN = 5                   # build_ground_truth 와 같은 기준
COLLECTION_SHARE = 0.10         # 같은 매체 저작물의 10% 이상에 붙은 태그 = 컬렉션 공통 태그

# 형식·매체·장비를 말할 뿐 화면 내용이 아닌 태그 (실측 상위 빈도에서 추림)
TECH_TAGS = {"4K", "항공촬영", "일반촬영", "드론", "드론영상", "필드영상", "영상", "동영상",
             "제주 영상", "사진", "이미지", "간행물", "어문저작물", "보육자료", "카드뉴스", "원문", "자료"}
_ORIGINAL_TITLE_ONLY = re.compile(r"^원문명\s*[:：]\s*\S.{0,80}$", re.S)
_EXIF = re.compile(r"Canon|Nikon|Sony|EOS|f/\d|\d+\s*mm\b|ISO\s?\d|\d{4}/\d{2}/\d{2}")


def attr(value, field, source, tier, note=None, **extra) -> dict:
    d = {"value": value, "schema_field": field, "source": source, "tier": tier}
    if note:
        d["note"] = note
    d.update({k: v for k, v in extra.items() if v not in (None, "", [], {})})
    return d


def parse_tags(s) -> list:
    """'#a #b', '#a,#b', '#a, #b' → ['a', 'b']. 여러 단어 태그('#제주 영상')는 그대로 둔다."""
    out = []
    for t in str(s or "").split("#"):
        t = norm(t).strip(" ,;")
        if t and t not in ("-", "nan", "None") and t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------- 입력
def load_export(ids: set) -> dict:
    """붙임1 에서 우리 1,500건만 스트리밍으로 읽는다. 중복 원문인덱스는 첫 행(예전 iloc[0]과 동일)."""
    import openpyxl
    wb = openpyxl.load_workbook(EXPORT, read_only=True)
    ws = wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    hdr = {h: i for i, h in enumerate(next(it))}
    keep = ("원문인덱스", "분류", "제목", "저작권자명", "원본소유자", "공공누리 유형", "주제어", "해시태그")
    out = {}
    for r in it:
        k = str(r[hdr["원문인덱스"]])
        if k in ids and k not in out:
            out[k] = {c: r[hdr[c]] for c in keep}
    wb.close()
    return out


def load_detail_scan() -> dict:
    files = sorted(glob.glob(str(GOLD_DIR / "kogl_detail_1500_*.jsonl")))
    out = {}
    for p in files:                                   # 최신 파일이 마지막에 덮어쓴다
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                out[str(r["원문인덱스"])] = r
    return out, (Path(files[-1]).name if files else None)


def load_old_gold() -> dict:
    import pandas as pd
    p = GOLD_DIR / "kogl_gold.xlsx"
    if not p.exists():
        return {}
    g = pd.read_excel(p)
    return {str(r["recommendIdx"]): str(r["gold_desc"]) for _, r in g.iterrows()
            if str(r.get("status")) == "ok" and isinstance(r.get("gold_desc"), str)}


# ---------------------------------------------------------------- 속성별 정답
def description_attr(scan: dict | None, old_gold: str | None, title: str, holder: str,
                     scan_file: str | None) -> dict:
    src = f"kogl:상세페이지 저작물 설명 ({scan_file})"
    if scan and scan.get("bytes", 0) < STUB_MAX_BYTES and not scan.get("desc_found"):
        return attr(None, "description", src, "제외",
                    f"상세페이지가 빈 페이지({scan['bytes']:,}B)로 응답 — 설명 미확인. "
                    "붙임1 은 공개·미삭제로 기재")
    raw = (scan or {}).get("desc") or ""
    if not raw and old_gold and not scan:
        raw, src = old_gold, "kogl_gold.xlsx:gold_desc"
    raw = raw.strip()
    if not raw:
        return attr(None, "description", src if scan else "—", "제외",
                    "상세페이지 저작물 설명 비어 있음" if scan else "상세페이지 미수집")
    flat = re.sub(r"\s+", " ", raw)
    if _ORIGINAL_TITLE_ONLY.match(flat):
        return attr(None, "description", src, "제외", "원문명(원어 제목)만 기재 — 설명 아님", raw=raw)
    if _EXIF.search(flat) and len(flat) < 200:
        return attr(None, "description", src, "제외", "촬영 장비·일시(EXIF) 정보 — 설명 아님", raw=raw)
    val, why = description_gt(raw, title, holder)
    if not val:
        return attr(None, "description", src, "제외", why, raw=raw)
    note = (f"본문 전문에 가까움({len(val):,}자) — 요약형 설명과 분량이 다름"
            if len(val) > 1500 else None)
    return attr(val, "description", src, "A", note)


def classify_tags(tags, media, freq_by_media, n_by_media, owners, title) -> dict:
    out = {}
    t_title = re.sub(r"\s", "", title or "")
    for t in tags:
        if t in owners or (len(t) >= 4 and any(t in o or o in t for o in owners)):
            out[t] = "기관명"
        elif t in TECH_TAGS:
            out[t] = "형식·매체"
        elif freq_by_media[media][t] / n_by_media[media] >= COLLECTION_SHARE:
            out[t] = "컬렉션 공통"
        elif re.sub(r"\s", "", t) in t_title:
            out[t] = "제목 단어"
        else:
            out[t] = "내용"
    return out


# ---------------------------------------------------------------- 빌드
def build() -> tuple[list, dict]:
    import pandas as pd

    oi = pd.read_excel(INDEX)
    oi = oi[oi["status"].astype(str).str.lower() == "downloaded"]
    ids = set(oi["원문인덱스"].astype(str))
    meta = load_export(ids)
    scan, scan_file = load_detail_scan()
    old_gold = load_old_gold()

    # 태그 분류용 집계 (매체별 빈도, 기관명 사전)
    owners = set()
    freq_by_media, n_by_media = defaultdict(Counter), Counter()
    for k, m in meta.items():
        owners.update(o for o in (str(m.get("원본소유자") or ""), str(m.get("저작권자명") or ""))
                      if len(o) > 1 and o not in ("None", "-"))
        med = MEDIA.get(str(m.get("분류")), "image")
        n_by_media[med] += 1
        freq_by_media[med].update(parse_tags(m.get("주제어") or m.get("해시태그")))

    out, cov = [], defaultdict(Counter)
    for _, r in oi.iterrows():
        oid = str(r["원문인덱스"])
        m = meta.get(oid)
        if not m:
            continue
        cls = str(r.get("분류") or "이미지")
        media = MEDIA.get(cls, "image")
        saved = KOGL_ROOT / cls / str(r.get("saved_file"))
        exists = saved.is_file()
        visual_na = media == "text"
        title = norm(m.get("제목"))
        holder = norm(m.get("저작권자명") or m.get("원본소유자"))

        kw = parse_tags(m.get("주제어")) or parse_tags(m.get("해시태그"))
        ht = parse_tags(m.get("해시태그"))
        sc = scan.get(oid)
        page_tags = [t.lstrip("#").strip() for t in (sc or {}).get("page_tags", [])] \
            if sc and sc.get("bytes", 0) >= STUB_MAX_BYTES else None
        kw_src = "kogl:주제어" if parse_tags(m.get("주제어")) else "kogl:해시태그(주제어 비어 있음)"

        width, height = r.get("width"), r.get("height")
        has_dims = pd.notna(width) and pd.notna(height) and not visual_na
        rec = {
            "id": oid, "media": media, "license_bucket": "kogl",
            "file": str(saved), "file_exists": exists,
            "attributes": {
                "제목": attr(title or None, "work_title", "kogl:제목", "A"),
                "저자": attr(holder or None, "copyright_holder", "kogl:저작권자명", "A"),
                "설명": description_attr(sc, old_gold.get(oid), title, holder, scan_file),
                "라이선스 유형": attr(f"제{m.get('공공누리 유형')}유형" if m.get("공공누리 유형") else None,
                                "kogl_type", "kogl:공공누리 유형", "A"),
                "키워드": attr(kw or None, "keyword", kw_src, "A" if kw else "제외",
                             None if kw else "주제어·해시태그 모두 비어 있음",
                             tag_classes=classify_tags(kw, media, freq_by_media, n_by_media,
                                                       owners, title)),
                "해시태그": attr(ht or None, "hashtags", "kogl:해시태그", "A" if ht else "제외",
                              ("채점 대상 아님(11속성·TTA 모두 해당 없음, 파이프라인에 출력 필드 없음). "
                               + ("주제어와 같은 목록" if ht == kw else
                                  f"주제어와 다름 — 주제어 {kw}")) if ht else "해시태그 비어 있음",
                              page_value=page_tags),
                "해상도": attr(f"{int(width)}x{int(height)}" if has_dims else None,
                            "resolution", f"originals_index:{r.get('dims_source') or '실측'}",
                            "N/A" if visual_na else "B",
                            None if visual_na else "KOGL 미게시 — 파일 실측. 추출기와 같은 경로(순환 주의)"),
                "주요 색상": attr(None, "dominant_colors", "—", "N/A" if visual_na else "C", "외부 정답 없음"),
                "개체 범주": attr(None, "main_subjects", "—", "N/A" if visual_na else "C", "외부 정답 없음"),
                "파일크기": attr(saved.stat().st_size if exists else None, "file_size", "파일:os.stat", "B"),
                "파일포맷": attr(saved.suffix.lstrip(".").upper() or None, "digital_format", "파일:확장자", "B"),
                "파일 생성 날짜": attr(None, "file_created_date", "—", "D",
                                  "붙임1 제작일자가 '조선' 등 비날짜 — 정답 불가"),
            },
        }
        if media == "video" and pd.notna(r.get("fps")):
            rec["extra_reference"] = {"실측_fps": r.get("fps"),
                                      "실측_재생시간_sec": r.get("duration_sec"),
                                      "실측_출처": "ffprobe"}
        out.append(rec)
        c = cov[media]
        c["n"] += 1
        for name, a in rec["attributes"].items():
            c[f"{name}|ok"] += a["value"] not in (None, "", [])

    # 컬렉션 단위 태그 판정 — build_ground_truth 와 같은 규칙
    sig = Counter(tuple(g["attributes"]["키워드"]["value"]) for g in out
                  if g["attributes"]["키워드"]["value"])
    for g in out:
        a = g["attributes"]["키워드"]
        kw = a.get("value")
        if kw and sig[tuple(kw)] >= BATCH_MIN:
            a["batch_level"] = True
            a["batch_size"] = sig[tuple(kw)]
            a["note"] = (f"컬렉션 공통 태그({sig[tuple(kw)]}건 동일) — 개별 저작물 화면 내용이 아님. "
                         "채점 시 별도 취급 권장")
            cov[g["media"]]["키워드|batch"] += 1
    return out, cov


def main() -> int:
    ap = argparse.ArgumentParser(description="KOGL 원본 1,500건 정답셋 빌더")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    recs, cov = build()
    out = Path(a.out)
    with out.open("w", encoding="utf-8") as f:
        for g in recs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    names = ["제목", "저자", "설명", "라이선스 유형", "키워드", "해시태그", "해상도", "파일크기", "파일포맷"]
    print(f"정답 {len(recs):,}건 → {out}")
    print(f"  {'매체':<6}{'건수':>6}" + "".join(f"{n[:5]:>8}" for n in names) + f"{'키워드배치':>10}")
    for med in ("image", "video", "text"):
        c = cov[med]
        print(f"  {med:<6}{c['n']:>6}" + "".join(f"{c[n + '|ok']:>8}" for n in names)
              + f"{c['키워드|batch']:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
