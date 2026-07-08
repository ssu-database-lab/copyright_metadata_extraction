"""
Build a stratified evaluation manifest from the KOGL 원문저작물 메타데이터 Excel.

Reads the (≈144k-row, 59MB) metadata workbook *read-only*, groups rows by
정보유형 (사진/문서/음성/어문/영상/동영상/미술/3D/복합), samples up to N rows per
group deterministically, builds the public thumbnail URL for each row, and
writes a UTF-8-BOM CSV with ground-truth labels for VLM/CLIP evaluation.

The public thumbnail URL is:
    https://www.kogl.or.kr  +  urllib.parse.quote(쎔네일웹경로, safe="/")
(confirmed returning HTTP 200 image/jpeg, 330×220 px).

Usage:
    # Default: 50 rows per 정보유형, rights-only, write manifest CSV
    python -m api.module.clip_extraction.build_eval_manifest

    # Smaller sample + validate 20 random URLs
    python -m api.module.clip_extraction.build_eval_manifest --per-type 20 --validate-urls 20

    # Also download the thumbnails into test_data/kogl_eval_images/
    python -m api.module.clip_extraction.build_eval_manifest --download

Outputs:
    api/module/clip_extraction/test_data/kogl_eval_manifest.csv   (default --out)
    api/module/clip_extraction/test_data/kogl_eval_images/{원문인덱스}.jpg  (with --download)
"""

from __future__ import annotations

import argparse
import csv
import random
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import openpyxl

# 프로젝트 루트를 import 경로에 추가 (스크립트로 직접 실행될 때) ----------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XLSX = PROJECT_ROOT / "docs" / "붙임1_원문저작물 메타데이터.xlsx"
DEFAULT_OUT = THIS_DIR / "test_data" / "kogl_eval_manifest.csv"
DEFAULT_IMAGE_DIR = THIS_DIR / "test_data" / "kogl_eval_images"

KOGL_BASE = "https://www.kogl.or.kr"

# 출력 CSV 컬럼 (요청 스펙 그대로) ------------------------------------------------
OUT_COLUMNS = [
    "원문인덱스",
    "분류",
    "정보유형",
    "제목",
    "thumbnail_url",
    "post_url",
    "주제어",
    "상업적이용허락",
    "비보호저작물",
    "원본소유자",
]

# Excel 헤더에서 읽어올 컬럼 (오타 주의: 썸이 아니라 '쎔'네일웹경로) ----------------
SRC_COLUMNS = [
    "원문인덱스",
    "분류",
    "정보유형",
    "제목",
    "쎔네일웹경로",   # 오타가 원본 그대로임 (썸 아님)
    "게시글URL",
    "주제어",
    "상업적이용허락",
    "비보호저작물",
    "원본소유자",
    "저작권자명",     # rights-only 필터용 (출력에는 미포함)
]

# 일부 셀 값이 작은따옴표로 감싸져 저장돼 있어(예: "'강릉문화원'") 벗겨낸다.
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) build_eval_manifest/1.0"}


# ---------------------------------------------------------------- helpers ---
def clean(value: object) -> str:
    """셀 값을 문자열로 정규화하고, 작은따옴표로 감싼 경우 한 겹 벗긴다."""
    if value is None:
        return ""
    s = str(value).strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1].strip()
    return s


def build_thumbnail_url(thumb_path: str) -> str:
    """쎔네일웹경로 -> 공개 썸네일 URL (검증된 패턴)."""
    return KOGL_BASE + urllib.parse.quote(thumb_path, safe="/")


def _ssl_context() -> ssl.SSLContext:
    """KOGL 인증서 검증 무시 (요청 스펙: ssl unverified is fine)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------- loading ---
def read_rows(xlsx_path: Path, rights_only: bool):
    """
    워크북을 read-only로 스트리밍하며 정보유형별로 그룹화한다.

    반환: (groups, stats)
      groups: dict[str(정보유형) -> list[dict(SRC_COLUMNS -> cleaned str)]]
      stats:  dict (총 행수, 썸네일 없는 행수, rights-only로 걸러진 행수 등)
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    header = [clean(h) for h in next(rows_iter)]

    # 필요한 컬럼의 인덱스 매핑 (없으면 즉시 오류)
    col_idx: dict[str, int] = {}
    missing: list[str] = []
    for name in SRC_COLUMNS:
        if name in header:
            col_idx[name] = header.index(name)
        else:
            missing.append(name)
    if missing:
        wb.close()
        raise KeyError(
            f"Excel 헤더에서 다음 컬럼을 찾지 못했습니다: {missing}\n"
            f"실제 헤더: {header}"
        )

    groups: dict[str, list[dict[str, str]]] = {}
    stats = {
        "total_rows": 0,
        "skipped_no_thumb": 0,
        "skipped_no_rights": 0,
        "kept": 0,
    }

    for raw in rows_iter:
        stats["total_rows"] += 1
        rec = {name: clean(raw[col_idx[name]]) for name in SRC_COLUMNS}

        # 썸네일 경로가 비면 평가 대상에서 제외
        if not rec["쎔네일웹경로"]:
            stats["skipped_no_thumb"] += 1
            continue

        # rights-only: 저작권자명(정답 권리 라벨)이 있는 행만
        if rights_only and not rec["저작권자명"]:
            stats["skipped_no_rights"] += 1
            continue

        info_type = rec["정보유형"] or "(미분류)"
        groups.setdefault(info_type, []).append(rec)
        stats["kept"] += 1

    wb.close()
    return groups, stats


def stratified_sample(
    group_rows: list[dict[str, str]], n: int
) -> list[dict[str, str]]:
    """
    그룹에서 최대 n개를 결정론적으로 추출한다.

    무작위가 아니라 매 k번째 행을 골라 파일 전체에 고르게 퍼지게 한다
    (k = floor(group_size / n), 최소 1). 재현성을 위해 random 미사용.
    """
    size = len(group_rows)
    if size <= n:
        return list(group_rows)
    step = size // n  # >= 1 (size > n 이므로)
    sampled = [group_rows[i * step] for i in range(n)]
    return sampled


# ---------------------------------------------------------------- writing ---
def write_manifest(out_path: Path, sampled_by_type: dict[str, list[dict[str, str]]]) -> int:
    """매니페스트 CSV를 utf-8-sig(BOM)로 기록하고 기록한 행 수를 반환한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for info_type in sorted(sampled_by_type):
            for rec in sampled_by_type[info_type]:
                writer.writerow({
                    "원문인덱스": rec["원문인덱스"],
                    "분류": rec["분류"],
                    "정보유형": rec["정보유형"],
                    "제목": rec["제목"],
                    "thumbnail_url": build_thumbnail_url(rec["쎔네일웹경로"]),
                    "post_url": rec["게시글URL"],
                    "주제어": rec["주제어"],
                    "상업적이용허락": rec["상업적이용허락"],
                    "비보호저작물": rec["비보호저작물"],
                    "원본소유자": rec["원본소유자"],
                })
                n_written += 1
    return n_written


# ---------------------------------------------------------------- download --
def download_thumbnails(
    sampled_by_type: dict[str, list[dict[str, str]]], image_dir: Path
) -> tuple[int, list[tuple[str, str]]]:
    """각 썸네일을 {원문인덱스}.jpg로 저장. (성공 수, [(원문인덱스, 오류)]) 반환."""
    image_dir.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context()
    n_ok = 0
    failures: list[tuple[str, str]] = []
    for info_type in sorted(sampled_by_type):
        for rec in sampled_by_type[info_type]:
            idx = rec["원문인덱스"]
            url = build_thumbnail_url(rec["쎔네일웹경로"])
            dest = image_dir / f"{idx}.jpg"
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                    data = resp.read()
                dest.write_bytes(data)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                failures.append((idx, f"{type(e).__name__}: {e}"))
    return n_ok, failures


# ---------------------------------------------------------------- validate --
def validate_urls(
    sampled_by_type: dict[str, list[dict[str, str]]], n: int
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    무작위로 n개 URL을 받아 HTTP 200 + image/* 인지 검사.
    재현성을 위해 고정 시드 사용. (image 200 수, 검사 수, [(url, 결과)]) 반환.
    """
    all_recs: list[dict[str, str]] = []
    for info_type in sorted(sampled_by_type):
        all_recs.extend(sampled_by_type[info_type])
    if not all_recs:
        return 0, 0, []

    rng = random.Random(42)  # 검증 샘플만 고정 시드 무작위
    pick = rng.sample(all_recs, min(n, len(all_recs)))
    ctx = _ssl_context()

    n_ok = 0
    details: list[tuple[str, str]] = []
    for rec in pick:
        url = build_thumbnail_url(rec["쎔네일웹경로"])
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                status = resp.status
                ctype = resp.headers.get("Content-Type", "")
                ok = (status == 200) and ctype.startswith("image/")
                if ok:
                    n_ok += 1
                details.append((url, f"HTTP {status} {ctype}"))
        except Exception as e:  # noqa: BLE001
            details.append((url, f"ERROR {type(e).__name__}: {e}"))
    return n_ok, len(pick), details


# ----------------------------------------------------------------- main -----
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a stratified KOGL evaluation manifest CSV "
                    "(thumbnail URLs + ground-truth labels)."
    )
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX),
                        help="KOGL 메타데이터 Excel 경로")
    parser.add_argument("--per-type", type=int, default=50,
                        help="정보유형별 추출 행 수 (기본 50)")
    parser.add_argument("--rights-only", dest="rights_only",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="저작권자명이 있는 행만 포함 (정답 권리 라벨 보유). 기본 True. "
                             "끄려면 --no-rights-only")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="출력 매니페스트 CSV 경로")
    parser.add_argument("--download", action="store_true",
                        help="썸네일을 test_data/kogl_eval_images/ 에 내려받기")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR),
                        help="--download 시 이미지 저장 디렉터리")
    parser.add_argument("--validate-urls", dest="validate_urls", type=int, default=0,
                        help="무작위 N개 URL을 HEAD/GET 검사하고 성공률 출력")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        print(f"입력 Excel을 찾을 수 없습니다: {xlsx_path}")
        return 1

    print(f"Excel 읽는 중 (read-only): {xlsx_path}")
    print(f"rights-only={args.rights_only}, per-type={args.per_type}")
    try:
        groups, stats = read_rows(xlsx_path, rights_only=args.rights_only)
    except KeyError as e:
        print(f"컬럼 오류: {e}")
        return 2

    print(
        f"스캔 완료: 총 {stats['total_rows']}행, "
        f"썸네일없음 {stats['skipped_no_thumb']} 제외, "
        f"권리없음 {stats['skipped_no_rights']} 제외, "
        f"후보 {stats['kept']}행"
    )

    if not groups:
        print("샘플링할 행이 없습니다. (rights-only 필터를 끄거나 입력을 확인하세요)")
        return 3

    # 정보유형별 결정론적 샘플링
    sampled_by_type: dict[str, list[dict[str, str]]] = {}
    for info_type, rows in groups.items():
        sampled_by_type[info_type] = stratified_sample(rows, args.per_type)

    # 매니페스트 기록
    out_path = Path(args.out)
    n_written = write_manifest(out_path, sampled_by_type)

    # 요약 출력
    print()
    print(f"매니페스트 작성: {out_path}")
    print(f"총 {n_written}행 기록")
    print("정보유형별 행 수:")
    for info_type in sorted(sampled_by_type):
        print(f"  {info_type}: {len(sampled_by_type[info_type])}  "
              f"(전체 후보 {len(groups[info_type])})")

    # 선택: URL 검증
    if args.validate_urls and args.validate_urls > 0:
        print()
        print(f"URL 검증 중 (무작위 {args.validate_urls}개, 고정 시드)...")
        n_ok, n_checked, details = validate_urls(sampled_by_type, args.validate_urls)
        for url, res in details:
            print(f"  [{res}] {url}")
        rate = (n_ok / n_checked * 100) if n_checked else 0.0
        print(f"URL 검증 결과: {n_ok}/{n_checked} 가 HTTP 200 image/* "
              f"({rate:.1f}%)")

    # 선택: 다운로드
    if args.download:
        image_dir = Path(args.image_dir)
        print()
        print(f"썸네일 다운로드 중 -> {image_dir}")
        n_ok, failures = download_thumbnails(sampled_by_type, image_dir)
        print(f"다운로드 완료: {n_ok}/{n_written} 성공")
        if failures:
            print(f"실패 {len(failures)}건:")
            for idx, err in failures[:20]:
                print(f"  원문인덱스 {idx}: {err}")
            if len(failures) > 20:
                print(f"  ... 외 {len(failures) - 20}건")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
