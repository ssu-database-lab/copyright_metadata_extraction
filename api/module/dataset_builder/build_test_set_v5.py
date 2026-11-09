#!/usr/bin/env python
"""시험 목록 v5 — 장세영 연구원의 GOLD 시트(계약서 값)를 바탕으로 Te-1 · Te-2 목록과 만들 계약서 목록을 정한다.

요구 (2610028 최종 절차서 To-1 · To-2, 저작물 유형별 1,000건):
  Te-1  4,000 = 유형(만료·기증·CCL·KOGL)마다 어문 50 + 이미지 475 + 영상 475
  Te-2  4,000 = 유형마다 이미지 500 + 영상 500
규칙:
  - GOLD 시트에 있는 저작물(계약서 값이 채워진 행)을 먼저 쓴다. 같은 인덱스번호가 여러 행이면 첫 행.
  - 모자라면 디스크에 원본 파일이 있는 같은 칸의 저작물로 채운다 (개발 세트 tag_dev_v4 는 넣지 않는다 — 튜닝에 썼다).
  - 넘치면 고정 씨앗으로 고른다. Te-1 의 이미지·영상은 Te-2 목록의 부분집합 (계약서 공유 · Te-2 가 Te-1 결과를 다시 쓴다)이고
    GOLD 시트 행을 먼저 고른다 (모자라는 칸만 Te-2 에 새로 넣은 저작물로).
  - 만들 계약서 = 두 목록의 합집합 (이미지·영상 4,000 + 어문 200 = 4,200).

  python -m api.module.dataset_builder.build_test_set_v5 --out dataset/test_set_v5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

GOLD_XLSX = "docs/1._생성계약서_GOLD 데이터.xlsx"
SHEET = "생성 계약서 데이터(전체)"
MED = {"이미지": "image", "영상": "video", "어문": "text"}
BK = {"만료": "expired", "기증": "donated", "CCL": "ccl", "KOGL": "kogl"}
BK_KO = {v: k for k, v in BK.items()}
MED_KO = {v: k for k, v in MED.items()}
BUCKETS = ("expired", "donated", "ccl", "kogl")
TE1 = {"text": 50, "image": 475, "video": 475}
TE2 = {"image": 500, "video": 500}
COLS = ["test", "cell", "sub_source", "work_id", "media", "license_bucket", "source", "title", "work_file",
        "has_contract", "contract_pdf", "contract_source"]


def order(seed: str, work_id: str) -> str:
    return hashlib.sha256(f"{seed}:{work_id}".encode()).hexdigest()


def load_gold_sheet(path: str) -> dict[str, dict]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[SHEET]
    it = ws.iter_rows(values_only=True)
    head = list(next(it))
    rows: dict[str, dict] = {}
    for no, r in enumerate(it, start=2):
        if r[2] in (None, ""):
            continue
        wid = str(r[2]).strip()
        if wid in rows:
            rows[wid]["dup_rows"].append(no)
            continue
        rows[wid] = {"row": no, "media": MED[r[1]], "bucket": BK[r[0]], "title": r[head.index("제목")],
                     "dup_rows": []}
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset/test_set_v5")
    ap.add_argument("--gold-xlsx", default=GOLD_XLSX)
    ap.add_argument("--seed", default="20261103")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    his = load_gold_sheet(a.gold_xlsx)
    idx = {r["work_id"]: r for r in csv.DictReader(open("dataset/gt_merged_index.csv", encoding="utf-8-sig"))}
    dev = {r["work_id"] for r in csv.DictReader(open("dataset/tag_dev_v4/dev_ids.csv", encoding="utf-8-sig"))}
    ondisk = {w: r for w, r in idx.items()
              if r.get("work_file") and os.path.exists(r["work_file"]) and w not in dev}
    missing_file = [w for w in his if w not in ondisk]

    def pool(media: str, bucket: str):
        h = sorted((w for w, v in his.items() if (v["media"], v["bucket"]) == (media, bucket) and w in ondisk),
                   key=lambda w: order(a.seed, w))
        extra = sorted((w for w, r in ondisk.items() if (r["media"], r["license_bucket"]) == (media, bucket)
                        and w not in his), key=lambda w: order(a.seed, w))
        return h, extra

    te2: dict[str, list[str]] = {}
    te1: dict[str, list[str]] = {}
    table = []
    for b in BUCKETS:
        for m, need in TE2.items():
            h, extra = pool(m, b)
            pick = h[:need] + extra[:max(0, need - len(h))]
            te2[f"{m}/{b}"] = pick
            table.append({"test": "Te-2", "bucket": b, "media": m, "need": need, "gold_sheet": len(h),
                          "disk_extra": len(extra), "from_sheet": min(need, len(h)),
                          "added": max(0, need - len(h)), "picked": len(pick)})
        for m, need in TE1.items():
            if m == "text":
                h, extra = pool(m, b)
                pick = h[:need] + extra[:max(0, need - len(h))]
                avail_sheet, avail_extra = len(h), len(extra)
            else:                                     # Te-2 의 부분집합 — GOLD 시트 행을 먼저, 모자라면 Te-2 에 새로 넣은 행
                sel2 = sorted(te2[f"{m}/{b}"], key=lambda w: order(a.seed + ":te1", w))
                cand = [w for w in sel2 if w in his]
                pick = cand[:need] + [w for w in sel2 if w not in his][:max(0, need - len(cand))]
                avail_sheet, avail_extra = len(cand), len(sel2) - len(cand)
            te1[f"{m}/{b}"] = pick
            table.append({"test": "Te-1", "bucket": b, "media": m, "need": need, "gold_sheet": avail_sheet,
                          "disk_extra": avail_extra, "from_sheet": sum(1 for w in pick if w in his),
                          "added": sum(1 for w in pick if w not in his), "picked": len(pick)})

    contracts = sorted({w for v in list(te1.values()) + list(te2.values()) for w in v})

    def row(test: str, w: str) -> dict:
        r = idx[w]
        return {"test": test, "cell": f"{r['media']}/{r['license_bucket']}", "sub_source": r.get("source", ""),
                "work_id": w, "media": r["media"], "license_bucket": r["license_bucket"], "source": r.get("source", ""),
                "title": (his[w]["title"] if w in his else r.get("title")) or "", "work_file": r["work_file"],
                "has_contract": "True", "contract_pdf": "",
                "contract_source": f"gold_sheet_row_{his[w]['row']}" if w in his else "new"}

    for name, sel, test in (("te1_ids.csv", te1, "T-001"), ("te2_ids.csv", te2, "T-002")):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=COLS)
            wr.writeheader()
            for cell in sorted(sel):
                for w in sel[cell]:
                    wr.writerow(row(test, w))
    with (out / "contracts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS + ["in_te1", "in_te2"])
        wr.writeheader()
        s1 = {w for v in te1.values() for w in v}
        s2 = {w for v in te2.values() for w in v}
        for w in contracts:
            wr.writerow({**row("", w), "in_te1": w in s1, "in_te2": w in s2})
    summary = {
        "seed": a.seed, "gold_sheet_works": len(his), "gold_sheet_duplicates": {w: v["dup_rows"] for w, v in his.items() if v["dup_rows"]},
        "gold_sheet_without_file": missing_file,
        "te1": {c: len(v) for c, v in sorted(te1.items())}, "te1_total": sum(len(v) for v in te1.values()),
        "te2": {c: len(v) for c, v in sorted(te2.items())}, "te2_total": sum(len(v) for v in te2.values()),
        "contracts_total": len(contracts), "contracts_new": sum(1 for w in contracts if w not in his),
        "gold_sheet_rows_unused": sorted(w for w in his if w not in set(contracts)),
        "table": table,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("te1_total", "te2_total", "contracts_total", "contracts_new")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
