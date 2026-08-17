"""v3 배치 콘솔 테스트용 ZIP 생성기.

`/v3` 에 그대로 업로드할 수 있는 자족적 세트를 만든다. 각 ZIP 은
manifest.jsonl · ground_truth.jsonl · README.md · works/ (· contracts/) 를 담는다.

시나리오를 나눠 만드는 이유 — 하나로는 확인이 안 되는 것들이 있다:
  quick   3건   업로드~리포트 왕복만 2분 안에 확인
  images  25건  진행률·ETA·동시실행·새로고침 재연결을 충분한 시간 동안 확인
  video   6건   **전부 실패하는 게 정상** — 영상 트랙이 P3 가드로 미구현이라
                실패 표시·사유·재개 대상 처리를 확인하는 용도
  mixed   12건  이미지·영상·어문 혼합 → 리포트의 미디어별 분해 확인
  contract 8건  유일하게 실제 계약서가 있는 세트(63155) 포함 → 11속성 전 경로

크기·시간 고려:
  - 영상 평균 433MB, 어문 평균 93페이지(1건 23분)라 무작위로 담으면 못 쓴다.
    영상은 소용량, 어문은 4페이지 이하만 고른다.
  - 이미지는 세트당 약 1~3분.

사용:
  python -m module.dataset_builder.make_test_zips --out /경로 [--only images,video]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve()
_API_ROOT = _HERE.parents[2]
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

GONGU = Path("/mnt/d/copyright_dataset_metadata")
BUCKETS = ("expired", "donated", "ccl", "kogl")
SEED = 20260817

# 실제 생성된 계약서 (현재 1건뿐)
CONTRACT_PDF = _API_ROOT.parent / "0034_다산박물관_소장유물_이학편(천자).pdf"
CONTRACT_WORK = Path("/mnt/e/kogl_originals/이미지/63155.jpg")


def _load(media: str, max_bytes: int, need_desc: bool = False) -> List[Dict]:
    out = []
    for b in BUCKETS:
        p = GONGU / media / b / "ground_truth.jsonl"
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            f = g.get("file", "")
            if not (g.get("file_exists") and os.path.isfile(f)):
                continue
            if os.path.getsize(f) > max_bytes:
                continue
            if need_desc and not g["attributes"]["설명"]["value"]:
                continue
            out.append(g)
    return out


def _short_text(limit: int, max_bytes: int = 1_500_000) -> List[Dict]:
    """어문은 페이지 수가 곧 시간·비용이다. 4페이지 이하만 고른다."""
    from module.evaluation.batch_runner import _pdf_page_count
    picked = []
    for g in _load("text", max_bytes):
        f = g["file"]
        if f.lower().endswith(".txt") or _pdf_page_count(f) <= 4:
            picked.append(g)
        if len(picked) >= limit * 4:
            break
    return picked


def _stratify(pool: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """권리유형이 한쪽으로 쏠리지 않게 고른다."""
    by = {b: [] for b in BUCKETS}
    for g in pool:
        by.setdefault(g["license_bucket"], []).append(g)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        b = BUCKETS[i % len(BUCKETS)]
        if by.get(b):
            out.append(by[b].pop())
        i += 1
        if i > n * 8:
            break
    return out[:n]


def build_zip(name: str, gts: List[Dict], out_dir: Path, readme: str,
              with_contract: bool = False) -> Optional[Path]:
    if not gts and not with_contract:
        print(f"  ⚠️ {name}: 대상 없음 — 건너뜀")
        return None
    stage = out_dir / f"_stage_{name}"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "works").mkdir(parents=True)

    manifest, kept = [], []
    if with_contract and CONTRACT_PDF.is_file() and CONTRACT_WORK.is_file():
        (stage / "contracts").mkdir(exist_ok=True)
        shutil.copy2(CONTRACT_PDF, stage / "contracts" / "63155.pdf")
        shutil.copy2(CONTRACT_WORK, stage / "works" / "63155.jpg")
        manifest.append({"set_id": "63155", "contract": "contracts/63155.pdf",
                         "work": "works/63155.jpg", "media": "image",
                         "license_bucket": "kogl"})
        A = lambda v, f, s, t, nt=None: {"value": v, "schema_field": f, "source": s,
                                         "tier": t, **({"note": nt} if nt else {})}
        kept.append({"set_id": "63155", "id": "63155", "media": "image",
                     "license_bucket": "kogl", "file": str(stage / "works" / "63155.jpg"),
                     "file_exists": True, "attributes": {
            "제목": A("이학편(천자)", "work_title", "kogl:제목", "A"),
            "저자": A("전라남도 강진군", "copyright_holder", "kogl:원본소유자", "A"),
            "설명": A(None, "description", "—", "제외", "붙임1에 설명 열 없음"),
            "라이선스 유형": A("제1유형", "kogl_type", "kogl:공공누리 유형", "A"),
            "키워드": A(["소장품", "유물"], "keyword", "kogl:주제어", "A"),
            "해상도": A("1757x1172", "resolution", "originals_index", "B"),
            "주요 색상": A(None, "dominant_colors", "—", "C", "외부 정답 없음"),
            "개체 범주": A(None, "main_subjects", "—", "C", "외부 정답 없음"),
            "파일크기": A(CONTRACT_WORK.stat().st_size, "file_size", "파일", "B"),
            "파일포맷": A("JPG", "digital_format", "파일:확장자", "B"),
            "파일 생성 날짜": A(None, "file_created_date", "—", "D", "제작일자 비날짜")}})

    for g in gts:
        ext = Path(g["file"]).suffix.lower() or ".bin"
        dst = stage / "works" / f"{g['id']}{ext}"
        shutil.copy2(g["file"], dst)
        manifest.append({"set_id": g["id"], "work": f"works/{dst.name}",
                         "media": g["media"], "license_bucket": g["license_bucket"]})
        kept.append({**g, "set_id": g["id"]})

    with (stage / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with (stage / "ground_truth.jsonl").open("w", encoding="utf-8") as f:
        for g in kept:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    (stage / "README.md").write_text(readme, encoding="utf-8")

    zpath = out_dir / f"{name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(stage))
    shutil.rmtree(stage, ignore_errors=True)

    cb = Counter(m["license_bucket"] for m in manifest)
    cm = Counter(m["media"] for m in manifest)
    print(f"  ✔ {zpath.name:<28} {len(manifest):>3}세트 · {zpath.stat().st_size/1e6:>6.1f}MB "
          f"· {dict(cm)} · {dict(cb)}")
    return zpath


def main() -> int:
    ap = argparse.ArgumentParser(description="v3 테스트 ZIP 생성기")
    ap.add_argument("--out", default=str(_API_ROOT.parent / "test_zips"))
    ap.add_argument("--only", default=None, help="쉼표 구분: quick,images,video,mixed,contract")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    want = set(args.only.split(",")) if args.only else None
    def on(k): return want is None or k in want

    img = _load("image", 1_500_000, need_desc=True)
    vid = _load("video", 3_000_000)
    txt = _short_text(6)
    print(f"후보 — 이미지 {len(img)} · 영상 {len(vid)} · 단문 어문 {len(txt)}\n")

    if on("quick"):
        build_zip("v3_test_quick_3", _stratify(img, 3, rng), out,
                  "# 빠른 왕복 확인 (3세트)\n\n이미지 3건. 업로드→진행률→리포트까지 2~5분.\n"
                  "계약서가 없으므로 제목·저자·라이선스는 비어 정확도가 낮게 나옵니다 — 정상입니다.\n")
    if on("images"):
        build_zip("v3_test_images_25", _stratify(img, 25, rng), out,
                  "# 이미지 25세트\n\n진행률·ETA·동시실행·**새로고침 재연결**을 충분한 시간 동안 확인하는 용도.\n"
                  "4 workers 기준 약 15~25분, 추정 비용 약 ₩900.\n\n"
                  "실행 중 F5 를 눌러 작업에 자동 재연결되는지 꼭 확인해 보세요.\n")
    if on("video"):
        build_zip("v3_test_video_6", _stratify(vid, 6, rng), out,
                  "# 영상 6세트 — **전부 실패하는 것이 정상**\n\n"
                  "영상 트랙은 아직 P3 가드로 미구현이라 `저작물 처리 실패(media=video)` 로 기록됩니다.\n"
                  "확인 목적: 실패가 **조용히 0점으로 집계되지 않고** 실패로 분리되는지,\n"
                  "사유가 표시되는지, 재개 시 재시도 대상이 되는지.\n\n"
                  "수초 내 종료되며 비용은 거의 0입니다.\n")
    if on("mixed"):
        mixed = _stratify(img, 4, rng) + _stratify(vid, 4, rng) + _stratify(txt, 4, rng)
        build_zip("v3_test_mixed_12", mixed, out,
                  "# 혼합 12세트 (이미지 4 · 영상 4 · 어문 4)\n\n"
                  "리포트의 **미디어별 분해**를 확인하는 용도.\n"
                  "- 이미지: 정상 채점\n- 영상: 전부 실패(P3 미구현)\n"
                  "- 어문: 4페이지 이하 단문만 선별(장문은 1건 23분이라 테스트 부적합).\n"
                  "  어문은 해상도·주요색상·개체범주가 '해당없음' 으로 빠집니다.\n")
    if on("contract"):
        build_zip("v3_test_contract_8", _stratify(img, 7, rng), out,
                  "# 계약서 포함 8세트\n\n"
                  "`63155` 만 실제 계약서를 갖고 있어 **11속성 전 경로**(계약서→저작물 상속)를 탑니다.\n"
                  "나머지 7건은 저작물 단독이라 제목·저자·라이선스가 구조적으로 비어 있습니다.\n"
                  "계약서 유무가 정확도를 얼마나 가르는지 대비해 보세요 (실측: 86% vs 41%).\n",
                  with_contract=True)

    print(f"\n산출 위치: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
