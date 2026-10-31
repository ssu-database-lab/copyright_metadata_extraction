#!/usr/bin/env python
"""키워드 개발 세트 — 시험 목록(Te-1·Te-2) 밖의 공유마당 저작물을 키워드 정답이 몰린 셀에서 새로 모은다.

Te-2 키워드 정답은 사이트 키워드(검색 태그)이고, 정답이 있는 세트는 만료 영상 461 · 기증 영상 355 · 기증 이미지 210 ·
CCL 이미지 172 에 몰려 있다. 기존 개발 세트(tag_dev_v4)는 KOGL 이미지 70/96 이라 키워드를 잴 수 없고, 시험 목록 밖에
이 셀들의 키워드 정답이 있는 저작물은 6건뿐이라 새로 받는다. 시험 저작물은 절대 넣지 않는다 (t001_ids · t002_ids 제외).

  python -m api.module.evaluation.v4.kw_dev build --out dataset/kw_dev_v4 --per-cell 16
  → dataset/kw_dev_v4/dev_ids.csv (tag_lab 과 같은 열) · gold.jsonl (사이트 키워드) · 파일은 --files/<media>/<bucket>/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

log = logging.getLogger("v4.kw_dev")

# (media, licenseCd, bucket) — 공유마당 목록의 이용조건 코드
CELLS = (("video", "97", "expired"), ("video", "98,99", "donated"),
         ("image", "98,99", "donated"), ("image", "20,21,22,23,24,25,26,27", "ccl"))
VIDEO_EXT = {".mp4", ".avi", ".wmv", ".mov", ".mkv", ".mpg", ".mpeg", ".flv", ".m4v", ".asf"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
COLS = ["test", "cell", "sub_source", "work_id", "media", "license_bucket", "source", "title", "work_file",
        "has_contract", "contract_pdf"]


def test_ids(root: Path) -> set:
    out: set = set()
    for name in ("t001_ids.csv", "t002_ids.csv"):
        with (root / "dataset/test_set_v4" / name).open(encoding="utf-8-sig") as f:
            out |= {r["work_id"] for r in csv.DictReader(f)}
    return out


def cmd_build(a) -> None:
    from api.module.dataset_builder import gongu_downloader as G
    from api.module.dataset_builder.rescrape_details import make_session, parse_all_labels, parse_keywords
    root = Path(a.repo)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    banned = test_ids(root)
    log.info("시험 목록 %d건은 제외합니다", len(banned))
    rng = random.Random(a.seed)
    gold_path, ids_path = out / "gold.jsonl", out / "dev_ids.csv"
    have = {}
    if gold_path.exists():
        for line in gold_path.open(encoding="utf-8"):
            r = json.loads(line)
            have[r["work_id"]] = r
    s = make_session()
    only = set(filter(None, a.cells.split(",")))
    for media, lic, bucket in CELLS:
        cell = f"{media}/{bucket}"
        if only and cell not in only:
            continue
        kept = [r for r in have.values() if r["cell"] == cell]
        if len(kept) >= a.per_cell:
            continue
        total = G.fetch_total(media, lic, None) or 1000
        pages_total = max(1, total // a.page_unit)
        starts = sorted({1 + (pages_total * i) // a.strata for i in range(a.strata)})
        cands = G.crawl_listing(media, lic, a.per_cell * a.cand_mult, starts, a.page_unit, None, 0.3)
        rng.shuffle(cands)
        log.info("%s: 전체 %d건 · 후보 %d건 (시작 페이지 %s)", cell, total, len(cands), starts)
        ep, menu_no = G.MENU[media]
        for sn in cands:
            if len(kept) >= a.per_cell:
                break
            if sn in banned or sn in have:
                continue
            try:
                r = s.get(f"{G.BASE}/gongu/wrt/wrt/view.do", params={"wrtSn": sn, "menuNo": menu_no}, timeout=40)
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                log.warning("%s 상세 실패: %s", sn, e)
                continue
            kws = parse_keywords(r.text)
            labels = parse_all_labels(r.text)
            det = G.parse_detail(r.text)
            if len(kws) < 2 or not det.get("hosted"):
                continue
            wdir = Path(a.files) / media / bucket
            wdir.mkdir(parents=True, exist_ok=True)
            got = G.download_work(sn, wdir, a.max_mb * 1024 * 1024 if media == "video" else 80 * 1024 * 1024, 3, 0.3)
            ok_ext = VIDEO_EXT if media == "video" else IMAGE_EXT
            files = [g for g in got if g.get("status") == "ok" and Path(str(g.get("path"))).suffix.lower() in ok_ext]
            if not files:
                for g in got:
                    if g.get("status") == "ok":
                        Path(str(g["path"])).unlink(missing_ok=True)
                log.info("%s 건너뜀 — 받을 수 있는 %s 파일 없음 (%s)", sn, media,
                         ", ".join(sorted({str(g.get('issue') or g.get('status')) for g in got})) or "-")
                continue
            rec = {"work_id": sn, "cell": cell, "media": media, "license_bucket": bucket, "title": det.get("제목") or "",
                   "work_file": str(files[0]["path"]), "keywords": kws, "분류": labels.get("분류(장르)") or labels.get("분류"),
                   "저작권자": det.get("저작권자"), "source": "공유마당"}
            have[sn] = rec
            kept.append(rec)
            with gold_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.info("%s %d/%d — %s · 키워드 %s", cell, len(kept), a.per_cell, rec["title"][:30], ", ".join(kws[:8]))
    with ids_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in have.values():
            w.writerow({"test": "dev", "cell": r["cell"], "sub_source": "공유마당", "work_id": r["work_id"], "media": r["media"],
                        "license_bucket": r["license_bucket"], "source": "공유마당", "title": r["title"],
                        "work_file": r["work_file"], "has_contract": "False", "contract_pdf": ""})
    log.info("개발 세트 %d건 → %s", len(have), ids_path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="dataset/kw_dev_v4")
    b.add_argument("--repo", default=".")
    b.add_argument("--files", default="/mnt/e/kw_dev_v4", help="받은 저작물 파일 (영상이 커서 OneDrive 밖에 둔다)")
    b.add_argument("--per-cell", type=int, default=16)
    b.add_argument("--strata", type=int, default=6, help="목록 시작 페이지 수 (층화)")
    b.add_argument("--page-unit", type=int, default=100)
    b.add_argument("--max-mb", type=int, default=1500, help="영상 파일 상한(MB) — 시험 목록 밖에 남은 영상은 대부분 400 MB 를 넘는다")
    b.add_argument("--cand-mult", type=int, default=30, help="셀마다 목록에서 모을 후보 수 = per-cell × 이 값 (시험 저작물이 대부분이라 넉넉히)")
    b.add_argument("--seed", type=int, default=20261031)
    b.add_argument("--cells", default="", help="이 셀만 (쉼표, 예: image/donated,image/ccl) — 셀마다 다른 --out 으로 나눠 받을 수 있다")
    a = ap.parse_args(argv)
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
