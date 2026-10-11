#!/usr/bin/env python
"""시험 세트 확정 (v4) — T-002 저작물 4,000건과 T-001 계약서 4,000건의 id 목록을 재현 가능하게 추첨한다.

규칙 (docs/test_execution_plan_20261006.md Plan v2 1.4, 결정 0.7b):
  T-002  이미지 2,000 = ccl/donated/expired 각 500 (공유마당) + kogl 500 = KOGL 원본 250 + 공유마당 kogl 250
         영상   2,000 = ccl/donated/expired 각 500 (공유마당) + kogl 500 (KOGL 원본)
  T-001  어문 200 = 권리유형별 50 ;  이미지 1,900 + 영상 1,900 = 권리유형별 475 — T-002 저작물 안에서 추첨
제외: 내용 중복(61132) · 파일 없음/정답 없음. T-001 에서는 제목 정답 불가(contract_gt_title_scorable=False) 행도 제외.
      튜닝 세트 19건 / pinned 43건 / 개발 세트 29건은 --exclude 로 선택 (기본 none: 셀 pool 이 500~503건이라 제외하면 500건을 못 채운다 —
      대신 셀별 겹침 수를 summary 에 보고하고 절차서에 명시한다).
계약서는 선정 뒤 장세영 연구원이 생성하므로 has_contract 는 조건이 아니라 보고 항목이다 (contracts_needed.csv).

사용:  python api/module/dataset_builder/build_test_set.py [--dry-run] [--seed 20261009] [--out dataset/test_set_v4]
출력:  t002_ids.csv · t001_ids.csv · contracts_needed.csv · exclusions.csv · summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "dataset" / "gt_merged_index.csv"
DEV_SET = ["13387752", "13333800", "13333561", "13333560", "13304492", "13388086", "13387748", "13388087", "13239555", "13239534", "13304427", "13315042", "13315719", "13266561",
           "63814", "82169", "13071064", "13308364", "13311525", "13333578", "13072954", "13313264", "13274337", "13222345", "13223164", "13222923", "87658", "87345", "87395"]
CONTENT_DUPES = ["61132"]  # 61120 과 동일 영상 (바리메오름 여름 드론영상)
BUCKETS = ["ccl", "donated", "expired", "kogl"]
KOGL = "공공누리(KOGL)"
GONGU = "공유마당"


def exclusion_ids() -> dict[str, list[str]]:
    ex = {"tuning_19": [], "pinned_43": [], "dev_29": DEV_SET, "content_dup": CONTENT_DUPES}
    z = ROOT / "test_zips" / "v3_tta_18.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist() if n.endswith("manifest.jsonl")), None)
            if name:
                ex["tuning_19"] = [json.loads(l)["set_id"] for l in zf.read(name).decode("utf-8").splitlines() if l.strip()]
    p = ROOT / "test_zips" / "_pinned_ids.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        ex["pinned_43"] = [str(x) for x in (j if isinstance(j, list) else j.get("ids") or sum((v for v in j.values() if isinstance(v, list)), []))]
    return ex


def draw(pool: pd.DataFrame, n: int, seed: int, cell: str) -> pd.DataFrame:
    """셀마다 독립된 난수열 (seed+cell) — 한 셀의 pool 이 바뀌어도 다른 셀의 추첨은 그대로."""
    rng = random.Random(f"{seed}|{cell}")
    ids = sorted(pool.work_id.tolist())
    rng.shuffle(ids)
    return pool.set_index("work_id").loc[ids[:n]].reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", default=str(ROOT / "dataset" / "test_set_v4"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude", choices=["none", "tuning", "all"], default="none", help="none: 겹침만 보고 | tuning: 19건 튜닝 세트 제외 | all: 튜닝+pinned+개발 세트 제외")
    args = ap.parse_args()
    df = pd.read_csv(INDEX, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    tf = lambda s: s.str.upper() == "TRUE"
    ex = exclusion_ids()
    hard = {"content_dup": ex["content_dup"]}
    if args.exclude == "tuning":
        hard["tuning_19"] = ex["tuning_19"]
    elif args.exclude == "all":
        hard = ex
    excluded = set(sum(hard.values(), []))
    base = df[tf(df.work_file_exists) & tf(df.has_work_gt) & ~df.work_id.isin(excluded)].copy()

    # ---------------- T-002
    cells, short = [], {}
    for media in ("image", "video"):
        for b in BUCKETS:
            if b == "kogl":
                if media == "image":
                    specs = [("KOGL 원본", base[(base.media == media) & (base.source == KOGL)], 250), ("공유마당 kogl", base[(base.media == media) & (base.source == GONGU) & (base.license_bucket == "kogl")], 250)]
                else:
                    specs = [("KOGL 원본", base[(base.media == media) & (base.source == KOGL)], 500)]
            else:
                specs = [(GONGU, base[(base.media == media) & (base.source == GONGU) & (base.license_bucket == b)], 500)]
            for sub, pool, n in specs:
                got = draw(pool, n, args.seed, f"T002/{media}/{b}/{sub}")
                got = got.assign(test="T-002", cell=f"{media}/{b}", sub_source=sub)
                cells.append(got)
                if len(got) < n:
                    short[f"{media}/{b} ({sub})"] = {"need": n, "have": len(got)}
    t002 = pd.concat(cells, ignore_index=True)

    # ---------------- T-001 (contracts): texts + image/video subsets of T-002
    t1 = []
    for b in BUCKETS:
        if b == "kogl":
            pool = base[(base.media == "text") & (base.source == KOGL)]
        else:
            pool = base[(base.media == "text") & (base.source == GONGU) & (base.license_bucket == b)]
        pool = pool[pool.contract_gt_title_scorable.str.upper() != "FALSE"]
        got = draw(pool, 50, args.seed, f"T001/text/{b}").assign(test="T-001", cell=f"text/{b}", sub_source=pool.source.iloc[0] if len(pool) else "")
        t1.append(got)
        if len(got) < 50:
            short[f"text/{b}"] = {"need": 50, "have": len(got)}
    for media in ("image", "video"):
        for b in BUCKETS:
            pool = t002[(t002.cell == f"{media}/{b}") & (t002.contract_gt_title_scorable.str.upper() != "FALSE")]
            got = draw(pool, 475, args.seed, f"T001/{media}/{b}").assign(test="T-001", cell=f"{media}/{b}")
            t1.append(got)
            if len(got) < 475:
                short[f"T-001 {media}/{b}"] = {"need": 475, "have": len(got)}
    t001 = pd.concat(t1, ignore_index=True)

    cols = ["test", "cell", "sub_source", "work_id", "media", "license_bucket", "source", "title", "work_file", "has_contract", "contract_pdf", "work_설명_tier", "work_해시태그_tier", "work_해상도_tier"]
    cols = [c for c in cols if c in t002.columns or c in ("test", "cell", "sub_source")]
    need = pd.concat([t002, t001]).drop_duplicates("work_id")
    need = need[need.has_contract.str.upper() != "TRUE"][["work_id", "media", "license_bucket", "source", "title", "work_file"]]
    summary = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed, "index_sha256": hashlib.sha256(INDEX.read_bytes()).hexdigest(),
               "exclude_policy": args.exclude, "excluded": {k: len(v) for k, v in hard.items()},
               "overlap_in_t002": {k: int(t002.work_id.isin(v).sum()) for k, v in ex.items() if k not in hard}, "t002": t002.groupby("cell").size().to_dict(), "t002_kogl_image_mix": t002[t002.cell == "image/kogl"].groupby("sub_source").size().to_dict(),
               "t001": t001.groupby("cell").size().to_dict(), "shortfalls": short, "unique_works": int(pd.concat([t002, t001]).work_id.nunique()),
               "contracts_needed": int(len(need)), "contracts_needed_by_cell": {f"{m}/{b}": int(n) for (m, b), n in need.groupby(["media", "license_bucket"]).size().items()} if len(need) else {}}
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    if args.dry_run:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t002[cols].to_csv(out / "t002_ids.csv", index=False, encoding="utf-8-sig")
    t001[cols].to_csv(out / "t001_ids.csv", index=False, encoding="utf-8-sig")
    need.to_csv(out / "contracts_needed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([(k, i, "excluded" if k in hard else "kept (overlap)") for k, v in ex.items() for i in v], columns=["set", "work_id", "handling"]).to_csv(out / "exclusions.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("written:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
