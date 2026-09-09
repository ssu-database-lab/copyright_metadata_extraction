"""매니페스트 한 행 → TTA 채점용 정답(attributes) 레코드.

기존 정답셋(/mnt/d/.../ground_truth.jsonl)은 연구개발계획서 11속성 이름으로 키가 잡혀
있어서 TTA 14속성으로는 한 건도 대조되지 않는다. TTA 채점의 정답은 계약서 본문에서
오고, 그 값은 이미 dataset/contract_work_manifest.csv 에 들어 있으므로 여기서 옮겨 담는다.

tier 규칙 — 근거의 강도를 값과 함께 남긴다:
  A : 계약서에 그대로 인쇄되며 전수 검증됨
  제외: 계약서에 근거가 없어 채점하지 않음 (0점이 아니라 미채점으로 리포트할 것)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# 계약서에 인쇄되지 않는 것들 — 이유를 값과 함께 남겨야 리포트에서 0점과 구분된다.
NOT_IN_CONTRACT = {
    "파일명": "계약서에 인쇄되지 않음 (0/300)",
    "저작권자": "저작자·권리자와 다를 때 인쇄되지 않음 (0/57)",
    "공개유형": "라이선스 표기가 계약서에 없음 (0/300)",
}

_RIGHTS = [
    ("복제권", "gt_reproduction_right"),
    ("공연권", "gt_public_performance_right"),
    ("공중송신권", "gt_public_transmission_right"),
    ("전시권", "gt_exhibition_right"),
    ("배포권", "gt_distribution_right"),
    ("대여권", "gt_rental_right"),
    ("2차적저작물작성권", "gt_derivative_work_creation_right"),
]


def _A(value: Any, schema_field: str, source: str,
       tier: str = "A", note: Optional[str] = None) -> Dict[str, Any]:
    d = {"value": value, "schema_field": schema_field, "source": source, "tier": tier}
    if note:
        d["note"] = note
    return d


def _clean(v: Any) -> Any:
    """빈 값·NaN 을 None 으로 통일한다."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "None", "-") else v


def build(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """contract_work_manifest.csv 한 행 → TTA_ATTRIBUTES 이름의 정답 dict."""
    g = row.get
    attrs: Dict[str, Dict[str, Any]] = {
        "저작물명":   _A(_clean(g("gt_title")),        "work_title",  "계약서:제목(제호)"),
        "저작물 유형": _A(_clean(g("gt_work_kind")),    "work_type",   "계약서:종별"),
        "저작자":     _A(_clean(g("gt_author")),       "author",      "계약서:제2조 저작자"),
        # 계약서 전문의 '권리자'가 저작재산권자이자 이용허락자다. 명세서도 저작재산권자를
        # 먼저 적용하고 없으면 이용허락자를 쓰라고 한다.
        "저작재산권자": _A(_clean(g("gt_rights_holder")), "economic_rights_holder", "계약서:전문 권리자"),
        "이용허락자":  _A(_clean(g("gt_rights_holder")), "licensor",   "계약서:전문 권리자"),
        "이용허락 시작일": _A(_clean(g("gt_license_start")), "license_start_date", "계약서:제3조"),
        "이용허락 종료일": _A(_clean(g("gt_license_end")),   "license_end_date",   "계약서:제3조"),
    }
    for name, col in _RIGHTS:
        v = g(col)
        # 체크박스는 False 도 정답이다 — None(파싱 실패)일 때만 미채점으로 둔다.
        if isinstance(v, str):
            v = {"True": True, "False": False}.get(v)
        attrs[name] = (_A(bool(v), name_to_field(name), "계약서:제2조 권리 체크박스")
                       if v is not None else
                       _A(None, name_to_field(name), "—", "제외", "체크박스 판독 불가"))

    # 제목 글리프가 깨진 6건은 OCR 이 원복할 수 없다 — 정답에서 빼고 사유를 남긴다.
    if row.get("gt_title_scorable") in (False, "False"):
        attrs["저작물명"] = _A(None, "work_title", "—", "제외",
                            "PDF 폰트에 악센트 글리프 없음 — OCR 재현 불가")
    return attrs


def name_to_field(name: str) -> str:
    for n, col in _RIGHTS:
        if n == name:
            return col[3:]          # gt_ 접두사 제거
    return name


def build_records(csv_path: str, only_eval_ready: bool = True):
    """매니페스트 전체 → ground_truth.jsonl 과 같은 모양의 레코드 목록."""
    import csv as _csv
    out = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            if only_eval_ready and row.get("eval_ready") not in ("True", "true", True):
                continue
            out.append({
                "set_id": row["set_id"],
                "id": row["set_id"],
                "media": row.get("media") or row.get("media_declared"),
                "license_bucket": row.get("license_bucket"),
                "file": row.get("work_path"),
                "file_exists": bool(row.get("work_path")),
                "contract": row.get("contract_pdf"),
                "attributes": build(row),
            })
    return out
