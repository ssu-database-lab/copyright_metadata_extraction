"""sklearn 기반 경량 NER — RandomForest / Logistic Regression.

학습 데이터: BIO JSONL (token_cls.py와 동일 형식)
예측 출력:  {label: [span_text, ...]}  ← api.ner_predict와 동일 형식

특징 추출:
  - 토큰 자체 + 좌우 window 내 토큰들을 하나의 문자열로 이어 붙임
  - TF-IDF (char_wb n-gram 2-4)로 벡터화
  - 토큰별 BIO 라벨 분류 (O / B-xxx / I-xxx)

모델 저장: {model_path}/model.pkl
서명 파일: {model_path}/signature.txt  (재학습 스킵용)
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_WINDOW = 3           # 문맥 윈도우 크기 (좌우 각각)
_MAX_FEATURES = 50000 # TF-IDF 최대 특징 수


class SklearnNER:
    """sklearn 기반 토큰 분류 NER.

    classifier:
        "rf" — RandomForestClassifier (n_estimators=200)
        "lr" — LogisticRegression     (solver=saga)
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._bundle: Optional[dict] = None  # {"clf": Pipeline, "classes_": [...]}

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def train(
        self,
        data_dir: Path,
        *,
        classifier: str = "rf",
        max_per_label: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """BIO JSONL에서 학습 → model.pkl 저장.

        Returns:
            True  = 학습 실행됨
            False = 스킵(재학습 불필요) 또는 데이터 없음
        """
        data_dir = Path(data_dir)
        model_file = self.model_path / "model.pkl"
        sig_file   = self.model_path / "signature.txt"

        current_sig = _data_signature(data_dir)
        if not current_sig:
            log.warning("[SklearnNER] 학습 데이터 없음: %s", data_dir)
            return False

        if not force and model_file.exists() and sig_file.exists():
            if sig_file.read_text(encoding="utf-8").strip() == current_sig:
                print(f"  [SklearnNER] 재학습 불필요 ({self.model_path.name})")
                return False

        records = _load_bio_records(data_dir)
        if not records:
            log.warning("[SklearnNER] 파싱된 레코드 없음: %s", data_dir)
            return False

        if max_per_label is not None:
            records = _sample_by_label(records, max_per_label)

        print(f"  [SklearnNER] 학습 시작 classifier={classifier}  records={len(records)}")
        X, y = _extract_features(records, window=_WINDOW)
        clf  = _build_pipeline(classifier)
        clf.fit(X, y)

        self.model_path.mkdir(parents=True, exist_ok=True)
        bundle = {"clf": clf, "classes_": list(clf.classes_)}
        with open(model_file, "wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        sig_file.write_text(current_sig, encoding="utf-8")

        self._bundle = bundle
        n_labels = sum(1 for c in bundle["classes_"] if not c.startswith("I-") and c != "O")
        print(f"  [SklearnNER] 학습 완료  labels={n_labels}  saved={model_file}")
        return True

    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, List[str]]:
        """텍스트에서 NER 예측 → {entity_label: [span_text, ...]}

        threshold: B- 라벨을 채택하기 위한 최소 확률 (predict_proba 기준).
        """
        self._ensure_loaded()
        if self._bundle is None:
            return {}

        tokens = text.split()
        if not tokens:
            return {}

        clf = self._bundle["clf"]
        X   = [_token_context(tokens, i, _WINDOW) for i in range(len(tokens))]

        try:
            proba   = clf.predict_proba(X)   # (n_tokens, n_classes)
            classes = clf.classes_
        except AttributeError:
            # predict_proba 미지원 (이론상 RF/LR은 지원)
            raw_preds = clf.predict(X)
            return _aggregate_spans(tokens, list(raw_preds))

        # threshold 적용
        predicted: List[str] = []
        for i in range(len(tokens)):
            best_idx   = int(proba[i].argmax())
            best_prob  = float(proba[i][best_idx])
            best_label = classes[best_idx]

            # O 이거나 확률이 threshold 미만이면 O로 처리
            if best_label == "O" or best_prob < threshold:
                predicted.append("O")
            else:
                predicted.append(best_label)

        return _aggregate_spans(tokens, predicted)

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._bundle is not None:
            return
        model_file = self.model_path / "model.pkl"
        if not model_file.exists():
            log.warning("[SklearnNER] model.pkl 없음: %s", model_file)
            return
        with open(model_file, "rb") as fh:
            self._bundle = pickle.load(fh)


# ──────────────────────────────────────────────────────────────────────
# 모듈 레벨 헬퍼 함수
# ──────────────────────────────────────────────────────────────────────

def _data_signature(data_dir: Path) -> str:
    """디렉터리 내 *.jsonl 파일들의 해시 서명."""
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        return ""
    parts = [f"{f.name}:{f.stat().st_mtime:.0f}:{f.stat().st_size}" for f in files]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _load_bio_records(data_dir: Path) -> List[dict]:
    """디렉터리 내 모든 *.jsonl에서 {"tokens": [...], "labels": [...]} 레코드 로드."""
    records: List[dict] = []
    for jf in sorted(data_dir.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj.get("tokens"), list) and isinstance(obj.get("labels"), list):
                    records.append(obj)
            except Exception:
                continue
    return records


def _sample_by_label(records: List[dict], max_per_label: int) -> List[dict]:
    """라벨(파일 stem 기준 B- 엔티티 타입)별로 max_per_label건 균등 샘플링."""
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        key = "O"
        for lbl in rec.get("labels", []):
            if lbl.startswith("B-"):
                key = lbl[2:]
                break
        by_label[key].append(rec)

    sampled: List[dict] = []
    for key, recs in by_label.items():
        if len(recs) > max_per_label:
            recs = random.sample(recs, max_per_label)
        sampled.extend(recs)
    return sampled


def _token_context(tokens: List[str], i: int, window: int) -> str:
    """토큰 i 기준 window 크기의 문맥을 하나의 문자열로 반환."""
    parts: List[str] = []
    for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
        offset = j - i
        parts.append(f"C{offset:+d}:{tokens[j]}")
    return " ".join(parts)


def _extract_features(
    records: List[dict],
    window: int = _WINDOW,
) -> Tuple[List[str], List[str]]:
    """레코드 → (X_raw: List[str], y: List[str])"""
    X: List[str] = []
    y: List[str] = []
    for rec in records:
        tokens = rec["tokens"]
        labels = rec["labels"]
        for i, (tok, lbl) in enumerate(zip(tokens, labels)):
            X.append(_token_context(tokens, i, window))
            y.append(lbl)
    return X, y


def _build_pipeline(classifier: str):
    """TF-IDF + 분류기 Pipeline 생성."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=_MAX_FEATURES,
        sublinear_tf=True,
    )

    if classifier == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    elif classifier == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="saga",
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"지원하지 않는 classifier: {classifier!r}. 'rf' 또는 'lr' 사용.")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def _aggregate_spans(tokens: List[str], predicted: List[str]) -> Dict[str, List[str]]:
    """BIO 시퀀스에서 엔티티 스팬 추출 → {entity_type: [span_text, ...]}"""
    result: Dict[str, List[str]] = {}
    i = 0
    while i < len(tokens):
        lbl = predicted[i]
        if lbl.startswith("B-"):
            entity_type = lbl[2:]
            span_tokens = [tokens[i]]
            j = i + 1
            while j < len(tokens) and predicted[j] == f"I-{entity_type}":
                span_tokens.append(tokens[j])
                j += 1
            span_text = " ".join(span_tokens)
            result.setdefault(entity_type, []).append(span_text)
            i = j
        else:
            i += 1
    return result
