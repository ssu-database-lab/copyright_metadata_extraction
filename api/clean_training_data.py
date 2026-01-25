#!/usr/bin/env python3
"""
대용량 BIO 훈련 데이터 자동 정제
- I-로 시작하는 잘못된 BIO 시퀀스 교정(I- -> B-)
- 형식 강한 엔티티(PHONE/DATE/EMAIL/URL/ID_NUM) 값 검증 후 무효 라벨 제거
- 계약/권리 혼동 최소 교정(키워드 기반 RIGHT_INFO <-> CONTRACT_TYPE 전환)
- 결과는 data/in/cleaned/*.cleaned.txt 로 저장 (원본 유지)
"""
from pathlib import Path
import re
from typing import List, Tuple

INPUT_FILES = [
    Path("data/in/realistic_train.txt"),
    Path("data/in/real_document_train.txt"),
]
OUTPUT_DIR = Path("data/in/cleaned")

# 간단/강건 검증 정규식과 키워드들
PHONE_PATTERNS = [
    re.compile(r"^\d{2,3}-\d{3,4}-\d{4}$"),
    re.compile(r"^010\d{8}$"),
    re.compile(r"^\d{9,11}$"),
]
DATE_PATTERNS = [
    re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$"),
    re.compile(r"^\d{8}$"),
]
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
URL_PATTERN = re.compile(r"^https?://")
ID_PATTERN = re.compile(r"^\d{6}-\d{7}$")

RIGHT_KEYWORDS = ["권", "재산권", "인격권", "인접권", "복제", "배포", "공연", "전시", "송신", "전송", "번역", "각색"]
CONTRACT_KEYWORDS = ["계약", "계약서", "동의서", "협약", "합의서"]

# 헬퍼들

def fix_bio(labels: List[str]) -> List[str]:
    prev = 'O'
    out = []
    for lab in labels:
        if lab.startswith('I-'):
            et = lab[2:]
            if prev not in (f'B-{et}', f'I-{et}'):
                lab = f'B-{et}'
        out.append(lab)
        prev = lab
    return out


def spans_from(labels: List[str]) -> List[Tuple[int,int,str]]:
    spans = []
    i = 0
    while i < len(labels):
        if labels[i].startswith('B-'):
            et = labels[i][2:]
            j = i + 1
            while j < len(labels) and labels[j] == f'I-{et}':
                j += 1
            spans.append((i, j, et))
            i = j
        else:
            i += 1
    return spans


def is_valid(entity_type: str, text: str) -> bool:
    t = text.strip()
    if entity_type == 'PHONE':
        return any(p.fullmatch(t) for p in PHONE_PATTERNS)
    if entity_type == 'DATE':
        return any(p.fullmatch(t) for p in DATE_PATTERNS)
    if entity_type == 'EMAIL':
        return EMAIL_PATTERN.fullmatch(t) is not None
    if entity_type == 'URL':
        # 원본에 콜론/닷이 누락되는 경우가 많아 http 로 시작만 허용
        return t.startswith('http')
    if entity_type == 'ID_NUM':
        return ID_PATTERN.fullmatch(t) is not None
    return True


def relabel_or_drop(entity_type: str, text: str) -> Tuple[str, bool]:
    """라벨 교정 또는 제거 결정
    Returns: (new_type, keep)
    keep=False 이면 라벨을 O로 드랍
    """
    t = text.strip()

    # 강한 형식 엔티티는 유효하지 않으면 드랍
    if entity_type in ('PHONE', 'DATE', 'EMAIL', 'URL', 'ID_NUM'):
        return (entity_type, is_valid(entity_type, t))

    # CONTRACT_TYPE vs RIGHT_INFO 혼동 교정
    if entity_type == 'CONTRACT_TYPE':
        has_contract_kw = any(kw in t for kw in CONTRACT_KEYWORDS)
        has_right_kw = any(kw in t for kw in RIGHT_KEYWORDS)
        if not has_contract_kw and has_right_kw:
            return ('RIGHT_INFO', True)
        if not has_contract_kw and not has_right_kw:
            # 의미 없는 축약/단편이면 드랍
            return ('CONTRACT_TYPE', False)
        return ('CONTRACT_TYPE', True)

    if entity_type == 'RIGHT_INFO':
        has_right_kw = any(kw in t for kw in RIGHT_KEYWORDS)
        has_contract_kw = any(kw in t for kw in CONTRACT_KEYWORDS)
        if not has_right_kw and has_contract_kw:
            return ('CONTRACT_TYPE', True)
        if not has_right_kw and not has_contract_kw:
            return ('RIGHT_INFO', False)
        return ('RIGHT_INFO', True)

    return (entity_type, True)


def process_sample(tokens: List[str], labels: List[str]) -> List[str]:
    # 1) BIO 교정
    labels = fix_bio(labels)

    # 2) 스팬 추출 후 검증/교정
    spans = spans_from(labels)
    labels_out = labels[:]

    for start, end, et in spans:
        text = ''.join(tokens[start:end])
        new_type, keep = relabel_or_drop(et, text)
        if not keep:
            for i in range(start, end):
                labels_out[i] = 'O'
        else:
            if new_type != et:
                labels_out[start] = f'B-{new_type}'
                for i in range(start + 1, end):
                    labels_out[i] = f'I-{new_type}'

    return labels_out


def iter_samples(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens, labels = [], []
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if tokens:
                    yield tokens, labels
                    tokens, labels = [], []
                else:
                    yield [], []
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            tok, lab = parts
            tokens.append(tok)
            labels.append(lab)
        if tokens:
            yield tokens, labels


def write_cleaned(input_path: Path, output_path: Path):
    cnt_samples = 0
    dropped_spans = 0
    relabeled_spans = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        for tokens, labels in iter_samples(input_path):
            if not tokens:
                out.write('\n')
                continue
            # 이전/이후 라벨 비교로 통계 집계
            before_spans = spans_from(labels)
            new_labels = process_sample(tokens, labels)
            after_spans = spans_from(new_labels)

            # 드랍/리라벨 추정
            before_map = {(s, e): et for s, e, et in before_spans}
            after_map = {(s, e): et for s, e, et in after_spans}
            # 드랍된 것은 before에만 존재하는 스팬
            for k in before_map.keys():
                if k not in after_map:
                    dropped_spans += 1
            # 같은 구간인데 타입이 바뀐 경우 리라벨
            for k, et in after_map.items():
                if k in before_map and before_map[k] != et:
                    relabeled_spans += 1

            for t, l in zip(tokens, new_labels):
                if t.strip():
                    out.write(f"{t}\t{l}\n")
            out.write('\n')
            cnt_samples += 1

    return cnt_samples, dropped_spans, relabeled_spans


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for inp in INPUT_FILES:
        if not inp.exists():
            print(f"⚠️  파일 없음: {inp}")
            continue
        outp = OUTPUT_DIR / (inp.stem + ".cleaned.txt")
        print(f"정제 시작: {inp} -> {outp}")
        samples, dropped, relabeled = write_cleaned(inp, outp)
        size_mb = outp.stat().st_size / (1024*1024)
        print(f"  샘플: {samples:,}개, 드랍 스팬: {dropped:,}개, 라벨교정: {relabeled:,}개, 크기: {size_mb:.2f} MB")
    print("✅ 정제 완료")

if __name__ == "__main__":
    main()
