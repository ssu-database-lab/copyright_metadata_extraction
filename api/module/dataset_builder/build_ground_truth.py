"""평가 정답셋(ground truth) 빌더 — 연구개발계획서 2-4 '메타데이터 속성 추출' 11속성.

입력 3종을 합쳐 저작물 1건당 정답 레코드 1건을 만든다:

  1) records.jsonl      수집 시 확보한 서지정보 (제목·저작권자·라이선스·요약정보·분류)
  2) detail_meta.jsonl  재수집한 상세페이지 전체 라벨 (해상도·화질·화면비율·재생시간·창작년도)
                        → rescrape_details.py 산출물
  3) 실제 파일          파일크기·파일포맷

**티어**가 이 빌더의 핵심이다. 11속성은 정답의 출처가 근본적으로 다르다:

  A  외부 정답 — 사이트가 게시한 값. 우리 추출기와 독립이므로 그대로 채점 가능.
  B  파일 산출 — 우리가 직접 잰 값. 추출기도 같은 파일을 읽으므로 '검증'이지 '대조'가 아님.
  C  정답 없음 — 주요 색상·개체 범주. 어디에도 정답이 없어 사람 라벨링/루브릭 필요.
  D  의미 불일치 — 계획서는 '파일 생성 날짜'인데 사이트는 '창작년도'(저작물 창작 시점).
  N/A 해당 없음 — 어문 저작물의 시각적 속성(해상도·주요색상·개체범주).
      사이트 어문 상세페이지에도 해상도 필드 자체가 없다.

정답이 없는 속성은 값을 지어내지 않고 tier 와 사유를 남긴다 — 채점에서 제외할지
사람 라벨링할지는 시험 절차서에서 결정할 사항이다.

사용:
  python -m api.module.dataset_builder.build_ground_truth            # 전체 + 커버리지 리포트
  python -m api.module.dataset_builder.build_ground_truth --cells image/expired
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/mnt/d/copyright_dataset_metadata")
MEDIA = ("text", "image", "video")
BUCKETS = ("expired", "donated", "ccl", "kogl")

# 요약정보가 설명 정답으로 쓸 수 없는 경우들 (실측: 전체의 27%)
_ADMIN = re.compile(r"기증식별번호|기증신청|신청인구분|저작재산권자 본인|기증자")
_HOLDER_SUFFIX = re.compile(r"\s*\(저작물\s*[\d,]+\s*건\)\s*$")
_UNKNOWN = {"미상", "-", "없음", "미기재", "불명"}
# 구 records.jsonl 의 분류_장르 값에는 UCI 위젯 텍스트가 붙어 있다
# ("회화 일반,회화,미술 UCI 로고 G905-… UCI 코드 도움말"). 재수집분은 이미 정리돼 있으나
# 재수집 미완 레코드를 위해 빌더에서도 한 번 더 걷어낸다.
_UCI = re.compile(r"\s*UCI\s*(로고|코드).*$", re.S)
# 사이트 해상도 값은 양방향 제어문자로 감싸여 있다: '‪723 x 1680‬'
# 눈에 보이지 않아 문자열 비교가 조용히 실패한다. 반드시 제거.
_BIDI = dict.fromkeys(range(0x202A, 0x202F), None)


def norm(s) -> str:
    """유니코드 정규화 + 양방향 제어문자 제거 + 공백 정리."""
    if s is None:
        return ""
    txt = unicodedata.normalize("NFKC", str(s)).translate(_BIDI)
    return re.sub(r"\s+", " ", txt).strip()


def norm_resolution(s) -> str | None:
    """'723 x 1680' / '600 * 498' / '‪320 x 240‬' → '723x1680'. 형식이 아니면 None.

    ⚠️ 사이트는 구분자를 **두 가지**로 쓴다 — 'x'(영문자)와 '*'(별표).
       영상/만료는 주로 'x', 이미지/KOGL·CCL 은 전부 '*' 다.
       'x'만 받으면 image/kogl 500건·image/ccl 332건이 통째로 0으로 집계된다
       (라벨은 100% 존재하는데 값 파싱에서 조용히 탈락).
    """
    t = norm(s)
    m = re.match(r"^(\d+)\s*[xX×*]\s*(\d+)$", t)
    return f"{m.group(1)}x{m.group(2)}" if m else None


def norm_unknown(s) -> str | None:
    """'미상' 등 부재 표기를 None 으로."""
    t = norm(s)
    return None if (not t or t in _UNKNOWN) else t


def description_gt(summary, title, holder) -> tuple[str | None, str]:
    """요약정보를 설명 정답으로 쓸 수 있는지 판정."""
    s = norm(summary)
    if not s:
        return None, "빈값"
    if _ADMIN.search(s):
        return None, "기증 행정정보(시각 설명 아님)"
    if len(s) < 15:
        return None, "15자 미만(저자명 복사 등)"
    if s in norm(title) + norm(holder):
        return None, "제목/저자 복사"
    return s, "ok"


def attr(value, field, source, tier, note=None) -> dict:
    d = {"value": value, "schema_field": field, "source": source, "tier": tier}
    if note:
        d["note"] = note
    return d


def build_record(rec: dict, detail: dict | None, media: str, bucket: str) -> dict:
    fields = (detail or {}).get("fields", {}) or {}
    path = (rec.get("saved_paths") or "").split(";")[0].strip()
    exists = bool(path) and os.path.isfile(path)

    title = norm(fields.get("저작물명") or rec.get("제목"))
    holder = _HOLDER_SUFFIX.sub("", norm(fields.get("저작(권)자") or rec.get("저작권자"))).strip()
    desc_val, desc_note = description_gt(rec.get("요약정보"), title, holder)
    # ── 키워드와 분류는 서로 다른 것이다 ──────────────────────────────────
    # 분류(장르)는 3단 분류체계(세부·장르·대분류)이고 대분류는 영상/미술/사진/어문
    # 4개뿐이다. 이것을 자유 키워드 정답으로 쓰면 내용 키워드와 어휘 자체가 달라
    # 정확한 추출도 0점이 된다. 사이트는 **별도의 키워드(검색 태그)**를 제공하며
    # 실측 98% 보유·평균 6.9개로 이쪽이 키워드 정답이다.
    genre_raw = fields.get("분류(장르)") or fields.get("분류") or rec.get("분류_장르")
    genre = [g.strip() for g in _UCI.sub("", norm(genre_raw)).split(",") if g.strip()][:3]
    site_kw = [norm(k) for k in ((detail or {}).get("keywords") or []) if norm(k)]

    # --- 해상도: 사이트 게시값이 최우선(외부 정답 = 순환 없음) ------------------
    site_res = norm_resolution(fields.get("이미지저작물 해상도") or fields.get("영상저작물 화질"))
    if media == "text":
        res_a = attr(None, "resolution", "—", "N/A",
                     "어문 저작물 — 시각적 속성 비해당(사이트에도 해상도 필드 없음)")
    elif site_res:
        res_a = attr(site_res, "resolution", "gongu:상세페이지 게시값", "A",
                     "사이트 게시값 — 우리 추출기와 독립")
    else:
        res_a = attr(None, "resolution", "—", "B",
                     "사이트 미게시 — 파일 실측으로 대체 필요(순환 주의)")

    # --- 파일 생성 날짜: 계획서 항목명과 사이트 필드의 의미가 다르다 -------------
    created = norm_unknown(fields.get("창작년도")) or norm_unknown(fields.get("공표년도"))
    date_a = attr(created, "file_created_date",
                  "gongu:창작년도/공표년도" if created else "—", "D",
                  "계획서는 '파일 생성 날짜'이나 사이트 값은 저작물 창작 시점 — 의미 불일치"
                  if created else "사이트 값도 미상 — 정답 없음")

    visual_na = media == "text"
    out = {
        "id": str(rec.get("wrtSn")),
        "media": media,
        "license_bucket": bucket,
        "file": path,
        "file_exists": exists,
        "detail_rescraped": bool(detail and detail.get("ok")),
        "attributes": {
            "제목": attr(title or None, "work_title", "gongu:저작물명", "A"),
            "저자": attr(holder or None, "copyright_holder", "gongu:저작(권)자", "A"),
            "설명": attr(desc_val, "description", "gongu:요약정보",
                        "A" if desc_val else "제외", None if desc_val else desc_note),
            "라이선스 유형": attr(norm(rec.get("license_name")) or None, "kogl_type",
                             f"gongu:license_code={rec.get('license_code')}", "A"),
            "키워드": attr(site_kw or None, "keyword",
                         "gongu:저작물 키워드(검색 태그)" if site_kw else "—",
                         "A" if site_kw else "제외",
                         None if site_kw else "키워드 태그 미보유"),
            "해상도": res_a,
            "주요 색상": attr(None, "dominant_colors", "—",
                           "N/A" if visual_na else "C",
                           "어문 — 시각적 속성 비해당" if visual_na
                           else "외부 정답 없음 — 사람 라벨링 또는 루브릭 필요"),
            "개체 범주": attr(None, "main_subjects", "—",
                           "N/A" if visual_na else "C",
                           "어문 — 시각적 속성 비해당" if visual_na
                           else "외부 정답 없음 — 사람 라벨링 또는 루브릭 필요"),
            "파일크기": attr(os.path.getsize(path) if exists else None, "file_size",
                          "파일:os.stat", "B"),
            "파일포맷": attr((os.path.splitext(path)[1].lstrip(".").upper() or None)
                          if path else None, "digital_format", "파일:확장자", "B"),
            "파일 생성 날짜": date_a,
        },
    }
    # 분류(장르) — 우리 11속성이 아니라 '공유저작물 유형 분류' 지표(에이치엠컴퍼니) 영역.
    # 폐쇄집합이라 정확 일치로 채점 가능하다: 대분류는 영상/미술/사진/어문 4개뿐.
    if genre:
        out["classification"] = {
            "detail": genre[0] if len(genre) > 0 else None,
            "genre": genre[1] if len(genre) > 1 else None,
            "category": genre[2] if len(genre) > 2 else None,
            "raw": genre, "source": "gongu:분류(장르)", "closed_set": True,
        }

    # 11속성 밖이지만 확보되면 기술 메타 검증에 쓸 수 있는 값
    extra = {}
    if norm_resolution(fields.get("영상저작물 화질")):
        ar = norm(fields.get("영상저작물 화면비율"))
        if ar:
            extra["화면비율"] = ar
    dur = norm(fields.get("음원저작물 재생시간") or fields.get("영상저작물 재생시간"))
    if dur:
        extra["재생시간"] = dur
    if extra:
        out["extra_reference"] = extra
    return out


def load_details(cell_dir: Path) -> dict:
    """detail_meta.jsonl → {wrtSn: record}. 재수집 미완이면 빈 dict."""
    p = cell_dir / "detail_meta.jsonl"
    out = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                out[str(r.get("wrtSn"))] = r
    return out


def run_cell(media: str, bucket: str) -> tuple[list, dict]:
    cell_dir = DATA_ROOT / media / bucket
    src = cell_dir / "records.jsonl"
    if not src.exists():
        return [], {}
    details = load_details(cell_dir)

    gts, cov = [], defaultdict(int)
    with src.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "ok":
                continue
            gt = build_record(rec, details.get(str(rec.get("wrtSn"))), media, bucket)
            gts.append(gt)
            cov["n"] += 1
            cov["rescraped"] += gt["detail_rescraped"]
            for name, a in gt["attributes"].items():
                if a["tier"] == "N/A":
                    cov[f"{name}|na"] += 1
                elif a["value"] not in (None, "", []):
                    cov[f"{name}|ok"] += 1

    return gts, cov, cell_dir


ATTRS = ["제목", "저자", "설명", "라이선스 유형", "키워드", "해상도",
         "주요 색상", "개체 범주", "파일크기", "파일포맷", "파일 생성 날짜"]


def main() -> int:
    ap = argparse.ArgumentParser(description="평가 정답셋 빌더")
    ap.add_argument("--cells", default="all")
    args = ap.parse_args()

    if args.cells == "all":
        cells = [(m, b) for m in MEDIA for b in BUCKETS]
    else:
        cells = [tuple(t.strip().split("/")) for t in args.cells.split(",")]

    allcov, total = {}, defaultdict(int)
    collected = []          # (cell_dir, [records]) — 배치 판정 후 한 번에 기록
    for media, bucket in cells:
        gts, cov, cell_dir = run_cell(media, bucket)
        if not gts:
            continue
        collected.append((cell_dir, gts))
        allcov[f"{media}/{bucket}"] = cov
        for k, v in cov.items():
            total[k] += v

    # ── 컬렉션 단위 키워드 판정 ────────────────────────────────────────────
    # 기증 배치(예: 상명대 2026 프로젝트)는 소속 저작물 전체가 **동일한 키워드
    # 집합**을 공유한다. 그 태그는 배치를 설명할 뿐 개별 저작물의 화면 내용이
    # 아니므로 모델이 픽셀에서 만들어낼 수 없다. 조용히 감점하지 말고 표시한다.
    sig = defaultdict(int)
    for _, gts in collected:
        for g in gts:
            kw = (g["attributes"]["키워드"] or {}).get("value")
            if kw:
                sig[tuple(kw)] += 1
    BATCH_MIN = 5           # 5건 이상이 완전히 같은 키워드 집합 → 컬렉션 태그로 본다
    n_batch = 0
    for _, gts in collected:
        for g in gts:
            a = g["attributes"]["키워드"]
            kw = a.get("value")
            if kw and sig[tuple(kw)] >= BATCH_MIN:
                a["batch_level"] = True
                a["batch_size"] = sig[tuple(kw)]
                a["note"] = (f"컬렉션 공통 태그({sig[tuple(kw)]}건 동일) — "
                             "개별 저작물 화면 내용이 아님. 채점 시 별도 취급 권장")
                n_batch += 1

    for cell_dir, gts in collected:
        with (cell_dir / "ground_truth.jsonl").open("w", encoding="utf-8") as f:
            for g in gts:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("=" * 108)
    print("정답셋 커버리지 — 속성별 정답 확보 건수 (na = 해당 없음)")
    print("=" * 108)
    head = f"  {'셀':<15}{'건수':>6}{'재수집':>7}"
    for a in ATTRS:
        head += f"{a[:5]:>8}"
    print(head)
    print("  " + "-" * 104)
    for cell, c in allcov.items():
        row = f"  {cell:<15}{c['n']:>6}{c['rescraped']:>7}"
        for a in ATTRS:
            row += f"{(str(c[f'{a}|ok']) if not c[f'{a}|na'] else 'na'):>8}"
        print(row)
    print("  " + "-" * 104)
    row = f"  {'합계':<15}{total['n']:>6}{total['rescraped']:>7}"
    for a in ATTRS:
        ok, na = total[f"{a}|ok"], total[f"{a}|na"]
        row += f"{ok:>8}" if not na else f"{str(ok)+'/na':>8}"
    print(row)
    kw_ok = total["키워드|ok"]
    print(f"\n  키워드: {kw_ok}건 확보 · 그중 컬렉션 공통 태그 {n_batch}건 "
          f"({n_batch/kw_ok*100:.0f}%) 은 batch_level 로 표시" if kw_ok else "")
    cats = defaultdict(int)
    for _, gts in collected:
        for g in gts:
            c = (g.get("classification") or {}).get("category")
            if c:
                cats[c] += 1
    print(f"  분류(대분류) 폐쇄집합: {dict(sorted(cats.items(), key=lambda x:-x[1]))}")
    print(f"\n  산출물: <셀>/ground_truth.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
