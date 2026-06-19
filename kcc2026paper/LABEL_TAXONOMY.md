# 26-라벨 Format-Regularity Taxonomy

> KCC2026 / IC-EEECS NER 논문이 사용하는 26개 라벨(= thesis REGEX 9 + NER 17)을
> **Free / Regular / Semi-Regular** 3분류로 정리한 표. source of truth: `paper1/paper1.py` `FORMAT_CLASS`
> (paper5 와 동일). 머신리더블: [`data/label_taxonomy.csv`](data/label_taxonomy.csv).

## 요약

| Class | n | M1→M2 민감도 (paper1 §6.2) | 특성 |
|---|---:|---|---|
| `format-regular` | 14 | +54.50 pp (매우 민감) | regex-tight 표면 형식. 답만(M1)으론 식별 실패 — BIO 는 문맥+토큰 동시 요구. |
| `format-semi-regular` | 6 | +7.18 pp (robust) | 구조 단서(예: "저작물명 :" 직후)가 라벨 토큰에 강하게 인코딩 → mode 에 robust. |
| `format-free` | 6 | +43.75 pp (매우 민감) | 자유 서술·넓은 어휘. 주변 문맥 의존도가 커 mode 에 매우 민감. |

**핵심**: semi-regular 는 구조 단서 덕에 mode 에 robust, regular·free 는 답만(M1) 학습 시 붕괴.

## format-regular (14)

| label | lane | 의미 |
|---|---|---|
| `phone` | REGEX | 전화번호 |
| `email` | REGEX | 이메일 |
| `date` | REGEX | 날짜 |
| `ri_data` | NER | 권리 데이터/대상 |
| `ri_period` | NER | 이용 기간 |
| `ri_money` | REGEX | 금액 |
| `address` | NER | 주소 |
| `copyright_url` | REGEX | 원문 URL |
| `copyright_uci` | REGEX | UCI 식별자 |
| `copyright_num` | REGEX | 저작물 번호 |
| `copyright_idnum` | REGEX | 식별 번호 |
| `copyright_status` | NER | 저작물 상태 |
| `copyright_quantity` | REGEX | 수량 |
| `copyright_language` | NER | 언어 |

## format-semi-regular (6)

| label | lane | 의미 |
|---|---|---|
| `copyright_Keyword` | NER | 키워드 |
| `copyright_kotitle` | NER | 저작물 제목(국문) |
| `ri_law_reference` | NER | 법 조항 인용 |
| `ri_info` | NER | 권리 정보(설명) |
| `ri_contract_type` | NER | 계약 유형 |
| `ri_copyright` | NER | 권리/이용조건 |

## format-free (6)

| label | lane | 의미 |
|---|---|---|
| `name` | NER | 인명(저작자 등) |
| `company` | NER | 기관·회사명 |
| `department` | NER | 부서명 |
| `position` | NER | 직책 |
| `copyright_description` | NER | 저작물 설명 |
| `copyright_type` | NER | 저작물 종별/유형 |

## 비고

- **lane**: 통합 메타데이터 파이프라인에서의 추출 경로. 본 NER 실험은 26개를 모두 BIO 로 학습하지만,
  배포 파이프라인에서는 `REGEX` 9개를 결정적 규칙으로, `NER` 17개를 모델로 처리한다(thesis 3-way 분할).
- LLM 위임 9개 라벨(`copyright_id` 등)은 gold 부족으로 본 NER 실험 대상이 아니다.
