"""v4 매니페스트 빌더 — 고정 시험 목록 csv → manifest.jsonl (docs/v4/M2_runner_spec.md §1.B B5).

입력은 `dataset/test_set_v4/t001_ids.csv` · `t001_new5_ids.csv` · `t002_ids.csv` (M1 spec §3 의 열:
test, cell, sub_source, work_id, media, license_bucket, source, title, work_file, has_contract,
contract_pdf, work_설명_tier, work_해시태그_tier, work_해상도_tier). 읽기만 한다.

규칙:
  T-001 → contract = contract_pdf, work 없음 (계약서 단독 세트). 계약서가 없는 행은 만들 수 없어 건너뛰고 센다.
  T-002 → work = work_file, contract = contract_pdf (값이 있을 때 — 저작물 leg 의 제목·종별 문맥에만 쓴다).
set_id = work_id. 상대 경로(dataset/…)는 저장소 루트 기준 절대 경로로 바꿔 쓴다 — 매니페스트는 /tmp 에
놓여도 돌아가야 하고, 매니페스트 폴더 기준으로 다시 풀면 엉뚱한 곳을 가리킨다.

CLI:
  python -m api.module.evaluation.manifest_v4 --ids dataset/test_set_v4/t001_new5_ids.csv --test T-001 \\
      --out /tmp/v4_m2/t001_new5/manifest.jsonl [--limit N] [--work-ids 13385261,87248] [--validate]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .manifest import VALID_TESTS, ManifestEntry, validate_manifest, write_manifest

logger = logging.getLogger(__name__)

# api/module/evaluation/manifest_v4.py → parents[3] = 저장소 루트 (dataset/ 이 여기 있다)
# resolve() 대신 abspath — /home/…/copyright 는 /mnt/c(OneDrive, drvfs) 로 가는 심볼릭 링크라 resolve() 가
# 매니페스트·_meta.files 의 계약서 경로를 전부 /mnt/c 로 적었다. WSL 경로에서 시작했으면 WSL 경로로 남긴다.
REPO_ROOT = Path(os.path.abspath(__file__)).parents[3]
REQUIRED_COLUMNS = ("test", "work_id", "media", "license_bucket", "work_file", "has_contract", "contract_pdf")
EXTRA_COLUMNS = ("title", "cell", "source", "sub_source",
                 "work_설명_tier", "work_해시태그_tier", "work_해상도_tier")


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("true", "1", "y", "yes")


def _abs(p: Optional[str], root: Path) -> Optional[str]:
    p = (p or "").strip()
    if not p:
        return None
    q = Path(p)
    return str(q if q.is_absolute() else (root / q))


def read_ids_csv(ids_csv: str | Path) -> List[Dict[str, str]]:
    """BOM 이 붙은 UTF-8 csv 를 읽고 필수 열을 확인한다."""
    with open(ids_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(f"{ids_csv}: 필요한 열이 없습니다: {missing}")
    return rows


def entry_from_row(row: Dict[str, str], test: str,
                   repo_root: Path = REPO_ROOT) -> Optional[ManifestEntry]:
    """행 1개 → 엔트리. 시험 규칙상 만들 수 없는 행(T-001 인데 계약서 없음, T-002 인데 저작물 없음)은 None."""
    contract = _abs(row.get("contract_pdf"), repo_root)
    work = _abs(row.get("work_file"), repo_root)
    extra = {c: (row.get(c) or "") for c in EXTRA_COLUMNS if c in row}
    extra["work_id"] = str(row.get("work_id") or "")
    base = dict(set_id=str(row.get("work_id") or "").strip(),
                license_bucket=(row.get("license_bucket") or "").strip(),
                extra=extra)
    if test == "T-001":
        if not contract:
            return None
        return ManifestEntry(work=None, contract=contract, media=(row.get("media") or "text").strip(),
                             test="T-001", **base)
    if not work:
        return None
    return ManifestEntry(work=work, contract=contract, media=(row.get("media") or "image").strip(),
                         test="T-002", **base)


def entries_from_ids_csv(ids_csv: str | Path, test: str, *, repo_root: Path = REPO_ROOT,
                         limit: Optional[int] = None,
                         work_ids: Optional[Iterable[str]] = None) -> List[ManifestEntry]:
    """고정 목록 csv → 엔트리 목록. `work_ids` 를 주면 그 id 만, `limit` 은 앞에서 N건."""
    if test not in VALID_TESTS:
        raise ValueError(f"test 는 T-001 또는 T-002 이어야 합니다: {test}")
    wanted: Optional[Set[str]] = {str(w).strip() for w in work_ids} if work_ids is not None else None
    out: List[ManifestEntry] = []
    skipped_test, skipped_rule = 0, []
    for row in read_ids_csv(ids_csv):
        row_test = (row.get("test") or "").strip()
        if row_test and row_test != test:
            skipped_test += 1
            continue
        if wanted is not None and str(row.get("work_id") or "").strip() not in wanted:
            continue
        e = entry_from_row(row, test, repo_root)
        if e is None:
            skipped_rule.append(str(row.get("work_id")))
            continue
        out.append(e)
        if limit and len(out) >= limit:
            break
    if skipped_test:
        logger.info("%s: 시험 구분이 %s 가 아닌 행 %d건 건너뜀", ids_csv, test, skipped_test)
    if skipped_rule:
        logger.warning("%s: %s 규칙으로 만들 수 없는 행 %d건 건너뜀 (예: %s)",
                       ids_csv, test, len(skipped_rule), skipped_rule[:5])
    if wanted is not None:
        missing = sorted(wanted - {e.set_id for e in out})
        if missing:
            logger.warning("%s: 요청한 work_id 중 목록에 없는 것 %d건: %s", ids_csv, len(missing), missing[:10])
    return out


def build_manifest_v4(ids_csv: str | Path, test: str, out: Optional[str | Path] = None, *,
                      repo_root: Path = REPO_ROOT, limit: Optional[int] = None,
                      work_ids: Optional[Iterable[str]] = None) -> Path:
    """고정 목록 csv 로부터 manifest.jsonl 을 쓰고 그 경로를 돌려준다.

    기본 출력은 /tmp/v4_m2/manifest_<csv stem>.jsonl — dataset/ 은 읽기 전용이라 그 옆에 쓰지 않는다.
    """
    entries = entries_from_ids_csv(ids_csv, test, repo_root=repo_root, limit=limit, work_ids=work_ids)
    out_path = Path(out) if out else Path("/tmp/v4_m2") / f"manifest_{Path(ids_csv).stem}.jsonl"
    n = write_manifest(entries, out_path)
    logger.info("매니페스트 %d세트 기록 → %s (test=%s)", n, out_path, test)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="v4 고정 시험 목록 csv → manifest.jsonl")
    ap.add_argument("--ids", required=True, help="dataset/test_set_v4/t00{1,2}_ids.csv 또는 t001_new5_ids.csv")
    ap.add_argument("--test", required=True, choices=list(VALID_TESTS))
    ap.add_argument("--out", required=True, help="출력 manifest.jsonl 경로 (예: /tmp/v4_m2/t001_new5/manifest.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N건만")
    ap.add_argument("--work-ids", default=None, help="쉼표로 구분한 work_id 목록 — 이 세트만 담는다")
    ap.add_argument("--validate", action="store_true", help="파일 존재 여부까지 점검해 출력한다")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ids = [s for s in (args.work_ids or "").split(",") if s.strip()] or None
    path = build_manifest_v4(args.ids, args.test, args.out, limit=args.limit, work_ids=ids)
    from .manifest import load_manifest
    entries = load_manifest(path)
    kinds = {}
    for e in entries:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print(f"{path}: {len(entries)}세트 · 구성 {kinds} · test={args.test}")
    if args.validate:
        v = validate_manifest(entries, test=args.test)
        print(f"  검증 {'통과' if v['ok'] else '실패'} — 계약서단독 {v['contract_only']} · 저작물단독 {v['work_only']} "
              f"· 쌍 {v['pairs']} · 파일없음(저작물 {v['missing_work_count']} / 계약서 {v['missing_contract_count']})")
        for p in v["problems"]:
            print(f"  ⚠️ {p}")
        return 0 if v["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
