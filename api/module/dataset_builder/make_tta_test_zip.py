"""TTA 채점용 평가 ZIP 생성기 — /v3 에 그대로 올릴 수 있는 자족 세트.

기존 make_test_zips.py 는 연구개발계획서 11속성 기준의 정답셋을 만든다. TTA 채점은
속성 이름 자체가 달라서(저작물명/저작재산권자/복제권…) 그 ZIP 으로 돌리면 전 항목이
skipped_no_gt 로 빠지고 결과가 0% 처럼 보인다. 이 스크립트는 매니페스트에서
TTA 이름의 정답셋을 만들어 담는다.

사용:
  python -m module.dataset_builder.make_tta_test_zip --per-media 2 --out ../test_zips
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_API_ROOT = _HERE.parents[2]
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from module.evaluation.tta_ground_truth import build as build_attrs   # noqa: E402

ROOT = _API_ROOT.parent
MANIFEST = ROOT / "dataset" / "contract_work_manifest.csv"
# 파이프라인이 열 수 없는 형식은 애초에 담지 않는다.
UNPROCESSABLE = {".zip", ".pptx", ".ppt", ".xlsx", ".xls", ".hwpx"}
# 미디어별 상한 — 영상은 크고 어문은 페이지 수가 곧 시간이라 작은 것부터 고른다.
CAPS = {"image": 2_000_000, "video": 15_000_000, "text": 600_000}


def _rows():
    with MANIFEST.open(encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def pick(per_media: int):
    """미디어별로 계약서가 있는 세트를 라이선스 버킷이 겹치지 않게 고른다."""
    buckets: dict[str, list] = {"image": [], "video": [], "text": []}
    for r in _rows():
        if r.get("eval_ready") not in ("True", "true"):
            continue
        if not r.get("contract_pdf") or int(r.get("contract_pdf_bytes") or 0) == 0:
            continue
        m, wp = r.get("media"), r.get("work_path")
        if m not in buckets or not wp or not os.path.isfile(wp):
            continue
        size = int(r.get("work_bytes") or 0)
        if not (0 < size <= CAPS[m]):
            continue
        if Path(wp).suffix.lower() in UNPROCESSABLE:
            continue
        buckets[m].append(r)

    out = []
    for m, rows in buckets.items():
        rows.sort(key=lambda r: int(r.get("work_bytes") or 0))
        seen, taken = set(), []
        for r in rows:                      # 라이선스 버킷 분산 우선
            if r.get("license_bucket") in seen:
                continue
            taken.append(r); seen.add(r.get("license_bucket"))
            if len(taken) == per_media:
                break
        for r in rows:                      # 모자라면 크기순으로 채운다
            if len(taken) >= per_media:
                break
            if r not in taken:
                taken.append(r)
        out += taken[:per_media]
    return out


def build(per_media: int, out_dir: Path, name: str) -> Path:
    rows = pick(per_media)
    if not rows:
        raise SystemExit("조건에 맞는 세트가 없습니다 — 매니페스트를 확인하세요.")
    stage = out_dir / f"_stage_{name}"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "works").mkdir(parents=True)
    (stage / "contracts").mkdir()

    manifest, gt = [], []
    for r in rows:
        sid = r["set_id"]
        wext = Path(r["work_path"]).suffix.lower() or ".bin"
        shutil.copy2(r["work_path"], stage / "works" / f"{sid}{wext}")
        shutil.copy2(ROOT / r["contract_pdf"], stage / "contracts" / f"{sid}.pdf")
        manifest.append({"set_id": sid, "work": f"works/{sid}{wext}",
                         "contract": f"contracts/{sid}.pdf",
                         "media": r["media"], "license_bucket": r["license_bucket"]})
        gt.append({"set_id": sid, "id": sid, "media": r["media"],
                   "license_bucket": r["license_bucket"],
                   "file": f"works/{sid}{wext}", "file_exists": True,
                   "attributes": build_attrs(r)})

    with (stage / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with (stage / "ground_truth.jsonl").open("w", encoding="utf-8") as f:
        for g in gt:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    (stage / "README.md").write_text(
        f"# TTA 표준 채점 세트 ({len(rows)}건)\n\n"
        f"미디어별 {per_media}건 — 이미지·영상·어문. 전부 계약서를 갖고 있습니다.\n\n"
        "**/v3 업로드 시 '채점 기준'을 반드시 `TTA 표준 14속성` 으로 선택하세요.**\n"
        "정답셋이 TTA 속성 이름(저작물명·저작재산권자·복제권…)으로 되어 있어,\n"
        "계획서 11속성으로 돌리면 전 항목이 미채점으로 빠집니다.\n\n"
        "채점 대상 14속성: 저작물명·저작물 유형·저작자·저작재산권자·이용허락자·\n"
        "복제권·공연권·공중송신권·전시권·배포권·대여권·2차적저작물작성권·\n"
        "이용허락 시작일·이용허락 종료일\n", encoding="utf-8")

    zpath = out_dir / f"{name}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage))
    shutil.rmtree(stage, ignore_errors=True)

    from collections import Counter
    print(f"  ✔ {zpath.name}  {len(rows)}세트 · {zpath.stat().st_size/1e6:.1f}MB "
          f"· {dict(Counter(m['media'] for m in manifest))}")
    return zpath


def main() -> int:
    ap = argparse.ArgumentParser(description="TTA 채점용 평가 ZIP 생성")
    ap.add_argument("--per-media", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "test_zips"))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    build(a.per_media, out, a.name or f"v3_tta_{a.per_media * 3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
