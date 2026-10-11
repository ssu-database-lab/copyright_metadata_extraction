"""평가 세트 매니페스트 빌더 — 계약서 + 저작물 + 정답을 하나로 묶는다.

두 개의 출처를 지원한다. 정답 품질이 크게 다르므로 선택이 중요하다:

  --source gongu  공유마당 수집분 4,834건
                  정답: 이미 구축된 ground_truth.jsonl
                  제목·저자·라이선스·키워드 100% · 설명 73% · 해상도 Tier-A 64%
                  ※ 계약서는 없다 — 생성해서 붙여야 한다

  --source kogl   kogl_originals 1,500건
                  정답: 구축된 E:/kogl_originals/ground_truth.jsonl (build_kogl_ground_truth.py)
                  제목·저자·라이선스·키워드 100% · 설명 17%(상세페이지 저작물 설명 253건)
                  해상도는 KOGL 미게시라 파일 실측(Tier B, 순환 주의)
                  키워드는 70%가 컬렉션 공통 태그(batch_level) — 내용 키워드로 보기 어렵다

즉 계약서를 새로 만든다면 **공유마당 메타데이터로 만드는 편이 정답이 훨씬 좋다.**
KOGL 쪽으로 가면 11속성 중 설명·해상도가 약체가 된다.

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
    """구축된 KOGL 정답셋(E:/kogl_originals/ground_truth.jsonl)을 그대로 정답으로 쓴다.

    예전에는 여기서 붙임1 + originals_index 로 매번 조립했고 저장된 정답 파일이 없었다.
    조립은 build_kogl_ground_truth.py 로 옮겼다 — 설명(상세페이지 저작물 설명)·키워드(주제어
    '#' 단위 파싱)·해시태그·영상 실측값이 거기서 채워진다.
    """
    p = Path("/mnt/e/kogl_originals/ground_truth.jsonl")
    if not p.exists():
        raise SystemExit(f"KOGL 정답셋이 없습니다: {p}\n"
                         "  python -m api.module.dataset_builder.build_kogl_ground_truth 를 먼저 실행하세요")
    rows: List[Dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            if g.get("file_exists") and \
                    Path(g.get("file", "")).suffix.lower() not in UNPROCESSABLE:
                rows.append(g)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: per_bucket * len(BUCKETS)] if per_bucket else rows


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
