#!/usr/bin/env python3
"""26-label Free/Regular/Semi-Regular taxonomy 산출 (PLANS §5·§8.3·Appendix A).

source of truth: paper1/paper1.py 의 FORMAT_CLASS (= paper5 와 동일).
라벨 집합(REGEX 9 + NER 17 = 26)은 metadata.module.parts.labels 에서 import 해 교차검증한다.
출력: ../LABEL_TAXONOMY.md, ../data/label_taxonomy.csv
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # kcc2026paper/
sys.path.insert(0, str(ROOT.parent / "metadata"))   # for module.parts.labels (경량, torch 불필요)
from module.parts.labels import REGEX_LABEL_SET, NER_LABEL_SET  # noqa: E402

# --- paper1.py:96 FORMAT_CLASS 미러 (source of truth) ---
FORMAT_CLASS = {
    "format-regular": [
        "phone", "email", "date", "ri_data", "ri_period", "ri_money",
        "address", "copyright_url", "copyright_uci", "copyright_num",
        "copyright_idnum", "copyright_status", "copyright_quantity",
        "copyright_language",
    ],
    "format-semi-regular": [
        "copyright_Keyword", "copyright_kotitle", "ri_law_reference",
        "ri_info", "ri_contract_type", "ri_copyright",
    ],
    "format-free": [
        "name", "company", "department", "position",
        "copyright_description", "copyright_type",
    ],
}

# 간단 glossary (thesis schema 기준)
GLOSS = {
    "phone": "전화번호", "email": "이메일", "date": "날짜",
    "ri_data": "권리 데이터/대상", "ri_period": "이용 기간", "ri_money": "금액",
    "address": "주소", "copyright_url": "원문 URL", "copyright_uci": "UCI 식별자",
    "copyright_num": "저작물 번호", "copyright_idnum": "식별 번호",
    "copyright_status": "저작물 상태", "copyright_quantity": "수량",
    "copyright_language": "언어",
    "copyright_Keyword": "키워드", "copyright_kotitle": "저작물 제목(국문)",
    "ri_law_reference": "법 조항 인용", "ri_info": "권리 정보(설명)",
    "ri_contract_type": "계약 유형", "ri_copyright": "권리/이용조건",
    "name": "인명(저작자 등)", "company": "기관·회사명", "department": "부서명",
    "position": "직책", "copyright_description": "저작물 설명", "copyright_type": "저작물 종별/유형",
}

# KCC paper1 §6.2 — class별 M1→M2 민감도 (mode robustness)
CLASS_SENS = {
    "format-regular": "+54.50 pp (매우 민감)",
    "format-semi-regular": "+7.18 pp (robust)",
    "format-free": "+43.75 pp (매우 민감)",
}
CLASS_NOTE = {
    "format-regular": "regex-tight 표면 형식. 답만(M1)으론 식별 실패 — BIO 는 문맥+토큰 동시 요구.",
    "format-semi-regular": "구조 단서(예: \"저작물명 :\" 직후)가 라벨 토큰에 강하게 인코딩 → mode 에 robust.",
    "format-free": "자유 서술·넓은 어휘. 주변 문맥 의존도가 커 mode 에 매우 민감.",
}


def lane(label: str) -> str:
    if label in REGEX_LABEL_SET:
        return "REGEX"
    if label in NER_LABEL_SET:
        return "NER"
    return "?"


def main() -> None:
    all_labels = [lb for v in FORMAT_CLASS.values() for lb in v]
    # 교차검증: 26 = REGEX 9 + NER 17
    expected = REGEX_LABEL_SET | NER_LABEL_SET
    assert set(all_labels) == expected, (
        f"FORMAT_CLASS({len(all_labels)}) != REGEX|NER({len(expected)}); "
        f"diff={set(all_labels) ^ expected}"
    )
    assert len(all_labels) == 26, f"expected 26 labels, got {len(all_labels)}"

    # CSV
    csv_path = ROOT / "data" / "label_taxonomy.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "format_class", "extraction_lane", "gloss_ko"])
        for cls, labels in FORMAT_CLASS.items():
            for lb in labels:
                w.writerow([lb, cls.replace("format-", ""), lane(lb), GLOSS.get(lb, "")])

    # Markdown (논문 Appendix-ready)
    lines = [
        "# 26-라벨 Format-Regularity Taxonomy",
        "",
        "> KCC2026 / IC-EEECS NER 논문이 사용하는 26개 라벨(= thesis REGEX 9 + NER 17)을",
        "> **Free / Regular / Semi-Regular** 3분류로 정리한 표. source of truth: `paper1/paper1.py` `FORMAT_CLASS`",
        "> (paper5 와 동일). 머신리더블: [`data/label_taxonomy.csv`](data/label_taxonomy.csv).",
        "",
        "## 요약",
        "",
        "| Class | n | M1→M2 민감도 (paper1 §6.2) | 특성 |",
        "|---|---:|---|---|",
    ]
    for cls in ("format-regular", "format-semi-regular", "format-free"):
        lines.append(f"| `{cls}` | {len(FORMAT_CLASS[cls])} | {CLASS_SENS[cls]} | {CLASS_NOTE[cls]} |")
    lines += ["", "**핵심**: semi-regular 는 구조 단서 덕에 mode 에 robust, regular·free 는 답만(M1) 학습 시 붕괴.", ""]

    for cls in ("format-regular", "format-semi-regular", "format-free"):
        lines += [f"## {cls} ({len(FORMAT_CLASS[cls])})", "",
                  "| label | lane | 의미 |", "|---|---|---|"]
        for lb in FORMAT_CLASS[cls]:
            lines.append(f"| `{lb}` | {lane(lb)} | {GLOSS.get(lb, '')} |")
        lines.append("")
    lines += [
        "## 비고",
        "",
        "- **lane**: 통합 메타데이터 파이프라인에서의 추출 경로. 본 NER 실험은 26개를 모두 BIO 로 학습하지만,",
        "  배포 파이프라인에서는 `REGEX` 9개를 결정적 규칙으로, `NER` 17개를 모델로 처리한다(thesis 3-way 분할).",
        "- LLM 위임 9개 라벨(`copyright_id` 등)은 gold 부족으로 본 NER 실험 대상이 아니다.",
        "",
    ]
    (ROOT / "LABEL_TAXONOMY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"OK 26 labels (14/6/6) 검증 통과")
    print(f"  -> {ROOT/'LABEL_TAXONOMY.md'}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
