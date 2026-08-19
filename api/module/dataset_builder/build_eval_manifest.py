"""평가 세트 매니페스트 빌더 — 계약서 + 저작물 + 정답을 하나로 묶는다.

두 개의 출처를 지원한다. 정답 품질이 크게 다르므로 선택이 중요하다:

  --source gongu  공유마당 수집분 4,834건
                  정답: 이미 구축된 ground_truth.jsonl
                  제목·저자·라이선스·키워드 100% · 설명 73% · 해상도 Tier-A 64%
                  ※ 계약서는 없다 — 생성해서 붙여야 한다

  --source kogl   KOGL 원문 메타데이터(143,946건) + kogl_originals 1,500건
                  정답: 붙임1 엑셀
                  제목·저자·라이선스·키워드 확보 · **설명 정답 없음**(kogl_gold 79건뿐)
                  해상도도 엑셀에 없어 파일 실측(Tier B, 순환 주의)

즉 계약서를 새로 만든다면 **공유마당 메타데이터로 만드는 편이 정답이 훨씬 좋다.**
KOGL 쪽으로 가면 11속성 중 설명·해상도가 채점 불가/약체가 된다.

산출물:
  <out>/manifest.jsonl      세트 목록
  <out>/ground_truth.jsonl  세트별 정답(11속성 + tier)
  --stage 지정 시 works/ contracts/ 로 파일을 복사해 자족적 세트를 만든다
  (시험기관 납품용 ZIP 및 ext4 스테이징에 사용 — /mnt/* 9p 는 소파일 I/O가 10~39배 느리다)

사용:
  python -m module.dataset_builder.build_eval_manifest --source gongu \
      --contracts /path/to/contracts --per-bucket 100 --out /home/mbmk92/eval_staging/pilot400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve()
_API_ROOT = _HERE.parents[2]
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

GONGU_ROOT = Path("/mnt/d/copyright_dataset_metadata")
# 파이프라인이 처리할 수 없는 확장자 — 평가 대상에서 제외한다.
# image 버킷에 pptx·zip 이 섞여 있어 media=image 로 문서 경로에 들어가 실패한다.
UNPROCESSABLE = {".zip", ".pptx", ".ppt", ".xlsx", ".xls", ".hwpx"}
MEDIA = ("text", "image", "video")
BUCKETS = ("expired", "donated", "ccl", "kogl")


def _index_contracts(contracts_dir: Optional[str]) -> Dict[str, str]:
    """계약서 폴더를 {set_id: path} 로 색인.

    파일명이 '<set_id>.pdf' 인 경우와 '<앞자리>_...' 처럼 set_id 가 앞에 오는 경우를 모두 받는다.
    ⚠️ '0034_' 같은 접두는 20건이 공유하므로 접두 매칭은 쓰지 않는다 — stem 완전일치만.
    """
    idx: Dict[str, str] = {}
    if not contracts_dir:
        return idx
    root = Path(contracts_dir)
    if not root.is_dir():
        return idx
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".hwp", ".png", ".jpg"):
            idx.setdefault(p.stem, str(p.resolve()))
    return idx


def _from_gongu(per_bucket: Optional[int], seed: int) -> List[Dict]:
    """구축된 ground_truth.jsonl 을 그대로 정답으로 쓴다."""
    rng = random.Random(seed)
    by_bucket: Dict[str, List[Dict]] = {b: [] for b in BUCKETS}
    for media in MEDIA:
        for bucket in BUCKETS:
            p = GONGU_ROOT / media / bucket / "ground_truth.jsonl"
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if g.get("file_exists") and \
                            Path(g.get("file", "")).suffix.lower() not in UNPROCESSABLE:
                        by_bucket[bucket].append(g)
    out: List[Dict] = []
    for bucket, rows in by_bucket.items():
        rng.shuffle(rows)
        out.extend(rows[:per_bucket] if per_bucket else rows)
    return out


def _from_kogl(per_bucket: Optional[int], seed: int) -> List[Dict]:
    """붙임1 엑셀 + originals_index 로 정답을 조립한다."""
    import pandas as pd

    idx_path = Path("/mnt/e/kogl_originals/originals_index.xlsx")
    if not idx_path.exists():
        raise SystemExit(f"originals_index 를 찾을 수 없습니다: {idx_path}")
    oi = pd.read_excel(idx_path)
    oi = oi[oi["status"].astype(str).str.lower() == "downloaded"]

    meta = pd.read_excel(_API_ROOT.parent / "docs" / "붙임1_원문저작물 메타데이터.xlsx")
    meta = meta.set_index("원문인덱스", drop=False)

    # 설명 정답(있으면) — recommendIdx 기준이라 매칭률이 낮다
    gold: Dict[str, str] = {}
    gpath = _API_ROOT.parent / "dataset" / "kogl_gold" / "kogl_gold.xlsx"
    if gpath.exists():
        gd = pd.read_excel(gpath)
        for _, r in gd.iterrows():
            if str(r.get("status")) == "ok" and r.get("gold_desc"):
                gold[str(r.get("recommendIdx"))] = str(r["gold_desc"])

    rng = random.Random(seed)
    rows = oi.to_dict("records")
    rng.shuffle(rows)
    if per_bucket:
        rows = rows[: per_bucket * len(BUCKETS)]

    def A(v, field, source, tier, note=None):
        d = {"value": v, "schema_field": field, "source": source, "tier": tier}
        if note:
            d["note"] = note
        return d

    out: List[Dict] = []
    for r in rows:
        oid = r.get("원문인덱스")
        if oid not in meta.index:
            continue
        m = meta.loc[oid]
        if hasattr(m, "iloc") and getattr(m, "ndim", 1) > 1:
            m = m.iloc[0]
        saved = Path("/mnt/e/kogl_originals") / str(r.get("분류") or "이미지") / str(r.get("saved_file"))
        if not saved.is_file():
            continue
        media = {"이미지": "image", "영상": "video", "어문": "text"}.get(str(r.get("분류")), "image")
        kw = str(m.get("주제어") or m.get("해시태그") or "").replace("#", " ").split()
        desc = gold.get(str(oid))
        visual_na = media == "text"
        out.append({
            "id": str(oid), "media": media, "license_bucket": "kogl",
            "file": str(saved), "file_exists": True,
            "attributes": {
                "제목": A(str(m.get("제목") or "") or None, "work_title", "kogl:제목", "A"),
                "저자": A(str(m.get("저작권자명") or m.get("원본소유자") or "") or None,
                         "copyright_holder", "kogl:저작권자명", "A"),
                "설명": A(desc, "description", "kogl_gold:gold_desc" if desc else "—",
                         "A" if desc else "제외",
                         None if desc else "붙임1에 설명 열이 없음 — kogl_gold 79건만 보유"),
                "라이선스 유형": A(f"제{m.get('공공누리 유형')}유형" if m.get("공공누리 유형") else None,
                               "kogl_type", "kogl:공공누리 유형", "A"),
                "키워드": A(kw or None, "keyword", "kogl:주제어/해시태그", "A"),
                "해상도": A(f"{int(r['width'])}x{int(r['height'])}"
                          if (r.get("width") and r.get("height") and not visual_na) else None,
                          "resolution", "originals_index 실측", "N/A" if visual_na else "B",
                          None if visual_na else "엑셀에 해상도 열이 없어 파일 실측 — 추출기와 같은 경로(순환 주의)"),
                "주요 색상": A(None, "dominant_colors", "—", "N/A" if visual_na else "C",
                            "외부 정답 없음"),
                "개체 범주": A(None, "main_subjects", "—", "N/A" if visual_na else "C",
                            "외부 정답 없음"),
                "파일크기": A(saved.stat().st_size, "file_size", "파일:os.stat", "B"),
                "파일포맷": A(saved.suffix.lstrip(".").upper() or None, "digital_format",
                           "파일:확장자", "B"),
                "파일 생성 날짜": A(None, "file_created_date", "—", "D",
                               "붙임1 제작일자가 '조선' 등 비날짜 — 정답 불가"),
            },
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="평가 세트 매니페스트 빌더")
    ap.add_argument("--source", choices=["gongu", "kogl"], default="gongu")
    ap.add_argument("--contracts", default=None, help="계약서 폴더 (파일명 stem = set_id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=None,
                    help="권리유형별 상한 (층화 표본). 미지정 시 전체")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", action="store_true",
                    help="works/ contracts/ 로 파일 복사 (납품 ZIP·ext4 스테이징용)")
    ap.add_argument("--require-contract", action="store_true",
                    help="계약서가 있는 세트만 포함")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts = _index_contracts(args.contracts)
    gts = _from_gongu(args.per_bucket, args.seed) if args.source == "gongu" \
        else _from_kogl(args.per_bucket, args.seed)

    entries, kept_gt = [], []
    n_with_contract = 0
    for g in gts:
        sid = str(g["id"])
        cpath = contracts.get(sid)
        if args.require_contract and not cpath:
            continue
        n_with_contract += bool(cpath)
        work = g["file"]
        if args.stage:
            wdst = out_dir / "works" / f"{sid}{Path(work).suffix}"
            wdst.parent.mkdir(parents=True, exist_ok=True)
            if not wdst.exists():
                shutil.copy2(work, wdst)
            work = str(Path("works") / wdst.name)
            if cpath:
                cdst = out_dir / "contracts" / f"{sid}{Path(cpath).suffix}"
                cdst.parent.mkdir(parents=True, exist_ok=True)
                if not cdst.exists():
                    shutil.copy2(cpath, cdst)
                cpath = str(Path("contracts") / cdst.name)
        e = {"set_id": sid, "work": work, "media": g["media"],
             "license_bucket": g["license_bucket"]}
        if cpath:
            e["contract"] = cpath
        entries.append(e)
        kept_gt.append({**g, "set_id": sid})

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (out_dir / "ground_truth.jsonl").open("w", encoding="utf-8") as f:
        for g in kept_gt:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    from collections import Counter
    cb = Counter(e["license_bucket"] for e in entries)
    cm = Counter(e["media"] for e in entries)
    print(f"매니페스트 {len(entries)}세트 → {out_dir}/manifest.jsonl")
    print(f"  계약서 보유 {n_with_contract}/{len(entries)}"
          f"{'  ⚠️ 계약서 없는 세트는 제목·저자·라이선스가 비어 과소평가된다' if n_with_contract < len(entries) else ''}")
    print(f"  권리유형 {dict(cb)}")
    print(f"  미디어   {dict(cm)}")
    if args.stage:
        print(f"  파일 복사됨: {out_dir}/works, {out_dir}/contracts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
