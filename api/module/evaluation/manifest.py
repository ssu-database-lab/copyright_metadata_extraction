"""세트 매니페스트 — 계약서 1건 + 저작물 1건을 하나의 평가 단위로 묶는다.

**세트별 압축파일 대신 매니페스트를 쓰는 이유**
  - 4,000개 압축 해제는 순수 오버헤드다. 9p(/mnt/*)에서는 특히 비싸고, 재실행마다 반복된다.
  - 재개하려면 "2,847번 세트"에 바로 접근해야 하는데 압축은 풀어야 찾는다.
  - 파일럿 50건·특정 권리유형만·실패분 재시도가 매니페스트에서는 한 줄 필터다.
  - 짝짓기 근거가 파일시스템 구조가 아니라 **데이터로 남아** 감사 가능하다.

압축이 맞는 곳은 **시험기관 납품**이다 — 권리유형별 4개 아카이브에
contracts/ · works/ · manifest.jsonl · ground_truth.jsonl · checksums.sha256 를 담는다.

세트 ID는 **원문인덱스**를 쓴다. 파일명 접두(0034_)는 20건이 공유하고,
원본파일명 자체도 6,847건이 중복이라 키로 쓸 수 없다.

**v4(M2, docs/v4/M2_runner_spec.md §1.B B1) — 다리 하나짜리 세트**
  T-001 은 계약서 단독, T-002 는 저작물(+문맥용 계약서) 이므로 `work` 와 `contract` 중
  **하나만 있어도** 세트다. 둘 다 없는 줄만 오류다. 시험 구분(`test`)은 엔트리별로 적거나
  작업 수준(--test)으로 준다 — 둘 다 없으면 구성으로 추론한다(계약서만 → T-001, 저작물 있음 → T-002).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_TESTS = ("T-001", "T-002")


@dataclass
class ManifestEntry:
    set_id: str
    work: Optional[str] = None      # 저작물 파일 경로 (매니페스트 기준 상대 또는 절대) — T-001(계약서 단독)이면 None
    contract: Optional[str] = None  # 계약서 경로 — T-001 의 입력, T-002 에서는 제목·종별 규칙 파싱 문맥에만 쓴다
    media: str = "image"           # image | video | text
    license_bucket: str = ""       # expired | donated | ccl | kogl
    document_type: str = "저작재산권 이용허락 계약서"
    test: Optional[str] = None     # "T-001" | "T-002" | None (작업 수준 값 또는 구성에서 추론)
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, root: Path) -> "ManifestEntry":
        """상대 경로를 매니페스트 위치 기준 절대 경로로 바꾼다."""
        def _abs(p):
            if not p:
                return p
            q = Path(p)
            return str(q if q.is_absolute() else (root / q))
        return ManifestEntry(
            set_id=self.set_id, work=_abs(self.work), contract=_abs(self.contract),
            media=self.media, license_bucket=self.license_bucket,
            document_type=self.document_type, test=self.test, extra=self.extra,
        )

    @property
    def kind(self) -> str:
        """구성 — contract_only | work_only | pair | neither."""
        if self.work and self.contract:
            return "pair"
        if self.contract:
            return "contract_only"
        if self.work:
            return "work_only"
        return "neither"


def effective_test(entry: ManifestEntry, job_test: Optional[str] = None) -> Optional[str]:
    """엔트리의 시험 구분 — 엔트리 값 > 작업 수준 값 > 구성에서 추론.

    계약서만 있으면 T-001, 저작물이 있으면 T-002 다. 둘 다 없으면 None(검증에서 걸러진다).
    """
    if entry.test:
        return entry.test
    if job_test:
        return job_test
    if entry.contract and not entry.work:
        return "T-001"
    if entry.work:
        return "T-002"
    return None


def load_manifest(path: str | Path, resolve: bool = True) -> List[ManifestEntry]:
    p = Path(path)
    root = p.parent
    out: List[ManifestEntry] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} JSON 파싱 실패: {e}") from e
            known = {"set_id", "work", "contract", "media", "license_bucket",
                     "document_type", "test", "extra"}
            entry = ManifestEntry(
                set_id=str(d.get("set_id") or d.get("id") or lineno),
                # 저작물이 없는 줄(T-001 계약서 단독)은 None 으로 둔다 — 예전처럼 "" 로 채우면
                # 검증이 '저작물 없음' 오류로 보고 실행기는 open("") 으로 죽는다.
                work=d.get("work") or d.get("work_file") or None,
                contract=d.get("contract") or d.get("contract_file"),
                media=d.get("media") or "image",
                license_bucket=d.get("license_bucket") or "",
                document_type=d.get("document_type") or "저작재산권 이용허락 계약서",
                test=(str(d["test"]).strip() or None) if d.get("test") else None,
                # write_manifest 는 extra 를 중첩 객체로 쓴다 — 그걸 그대로 되살리고, 모르는 최상위 키도 extra 에 담는다
                extra={**(d.get("extra") if isinstance(d.get("extra"), dict) else {}),
                       **{k: v for k, v in d.items() if k not in known}},
            )
            out.append(entry.resolve(root) if resolve else entry)
    return out


def validate_manifest(entries: List[ManifestEntry],
                      test: Optional[str] = None) -> Dict[str, Any]:
    """실행 전 점검 — 없는 파일·중복 ID·다리 없는 세트·시험 구분 불일치를 미리 잡는다.

    4,000건을 돌리다 2,900번째에서 파일 없음으로 죽는 것보다 시작 전에 아는 게 낫다.
    `ok` 가 False 면 실행기(cli.main · batch_api._run_job)는 **시작하지 않는다**(M2 B4) —
    계약서 leg 요금을 쓴 뒤 저작물 open("") 에서 죽고 재개마다 다시 과금되는 일을 막는다.

    `test` 는 작업 수준 시험 구분(엔트리의 `test` 가 우선). 유효 시험이 T-001 인데 계약서가
    없거나 T-002 인데 저작물이 없으면 오류다.
    """
    seen, dup = set(), []
    missing_work, missing_contract = [], []
    neither, invalid_test, t001_no_contract, t002_no_work = [], [], [], []
    kinds = {"contract_only": 0, "work_only": 0, "pair": 0, "neither": 0}
    if test is not None and test not in VALID_TESTS:
        invalid_test.append(f"(작업 수준) {test}")
    for e in entries:
        if e.set_id in seen:
            dup.append(e.set_id)
        seen.add(e.set_id)
        if e.work and not os.path.isfile(e.work):
            missing_work.append(e.set_id)
        if e.contract and not os.path.isfile(e.contract):
            missing_contract.append(e.set_id)
        kinds[e.kind] += 1
        if e.kind == "neither":
            neither.append(e.set_id)
        if e.test and e.test not in VALID_TESTS:
            invalid_test.append(e.set_id)
            continue
        eff = effective_test(e, test if test in VALID_TESTS else None)
        if eff == "T-001" and not e.contract:
            t001_no_contract.append(e.set_id)
        elif eff == "T-002" and not e.work:
            t002_no_work.append(e.set_id)

    problems: List[str] = []
    if dup:
        problems.append(f"중복 set_id {len(dup)}건 (예: {dup[:3]})")
    if missing_work:
        problems.append(f"저작물 파일 없음 {len(missing_work)}건 (예: {missing_work[:3]})")
    if missing_contract:
        problems.append(f"계약서 파일 없음 {len(missing_contract)}건 (예: {missing_contract[:3]})")
    if neither:
        problems.append(f"계약서도 저작물도 없는 세트 {len(neither)}건 (예: {neither[:3]})")
    if invalid_test:
        problems.append(f"시험 구분이 T-001/T-002 가 아닌 세트 {len(invalid_test)}건 (예: {invalid_test[:3]})")
    if t001_no_contract:
        problems.append(f"T-001 인데 계약서가 없는 세트 {len(t001_no_contract)}건 (예: {t001_no_contract[:3]})")
    if t002_no_work:
        problems.append(f"T-002 인데 저작물이 없는 세트 {len(t002_no_work)}건 (예: {t002_no_work[:3]})")

    return {
        "total": len(entries),
        "test": test,
        "duplicate_ids": dup[:20], "duplicate_count": len(dup),
        "missing_work": missing_work[:20], "missing_work_count": len(missing_work),
        "missing_contract": missing_contract[:20],
        "missing_contract_count": len(missing_contract),
        # 구성별 건수 — work_only 는 v3 때부터 있던 키(정수)라 이름을 그대로 둔다.
        "contract_only": kinds["contract_only"],
        "work_only": kinds["work_only"],
        "pairs": kinds["pair"],
        "neither": kinds["neither"], "neither_ids": neither[:20],
        "invalid_test": invalid_test[:20],
        "t001_without_contract": t001_no_contract[:20],
        "t002_without_work": t002_no_work[:20],
        "problems": problems,
        "ok": not problems,
    }


def load_ground_truth(path: str | Path) -> Dict[str, Dict]:
    """ground_truth.jsonl → {set_id: record}. id 또는 set_id 키를 모두 받는다."""
    gt: Dict[str, Dict] = {}
    p = Path(path)
    if not p.exists():
        return gt
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(r.get("set_id") or r.get("id") or "")
            if key:
                gt[key] = r
    return gt


def write_manifest(entries: List[ManifestEntry], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            d = asdict(e)
            if not d.get("extra"):
                d.pop("extra", None)
            for k in ("work", "contract", "test"):
                if d.get(k) is None:
                    d.pop(k, None)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return len(entries)
