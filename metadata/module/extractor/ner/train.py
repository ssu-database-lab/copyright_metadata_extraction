"""NER 모델 훈련 로직 (base.py에서만 접근 가능)"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer
from seqeval.metrics import precision_score, recall_score, f1_score

# train.py에서만 ner_crf와 bilstm_crf 접근 가능
from .ner_crf import NER
from .bilstm_crf import BiLSTMCRF
from .config import BIO_LABELS, LABEL_TO_ID, ID_TO_LABEL, MODEL_DIR


def _load_training_data(
    data_dir: Path = Path("data/in/training_csv"), 
    train_ratio: float = 0.8, 
    random_seed: int = 42, 
    dataset_size: Optional[int] = None,
    samples_per_file: Optional[int] = None,
    sample_ratio_per_file: Optional[float] = None
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    """
    training_csv 디렉토리에서 학습 데이터 로드 및 분할
    
    Args:
        data_dir: 학습 데이터 디렉토리
        train_ratio: 학습 데이터 비율 (기본 0.8)
        random_seed: 랜덤 시드
        dataset_size: 전체 데이터셋 크기 제한 (None이면 전체 사용)
        samples_per_file: 각 파일에서 샘플링할 최대 문장 개수 (None이면 전체)
        sample_ratio_per_file: 각 파일에서 샘플링할 비율 (0.0 ~ 1.0, None이면 전체)
                              samples_per_file이 지정되면 무시됨
    
    Returns:
        (train_texts, train_labels, val_texts, val_labels)
    """
    if not data_dir.exists():
        print(f"[경고] 데이터 디렉토리가 없습니다: {data_dir}")
        return [], [], [], []
    
    all_texts: List[List[str]] = []
    all_labels: List[List[str]] = []
    
    # CSV 파일 찾기
    csv_files = list(data_dir.rglob("*.csv"))
    print(f"[데이터 로딩] {len(csv_files)}개의 CSV 파일 발견")
    
    # 파일별 샘플링 설정 확인
    if samples_per_file is not None:
        print(f"[파일별 샘플링] 각 파일당 최대 {samples_per_file}개 문장 샘플링")
    elif sample_ratio_per_file is not None:
        print(f"[파일별 샘플링] 각 파일당 {sample_ratio_per_file*100:.1f}% 샘플링")
    else:
        print(f"[파일별 샘플링] 각 파일 전체 사용")
    
    # 헤더 매핑 로드
    from module.parts.mapping import map_csv_columns, get_label_category
    
    for csv_file in csv_files:
        try:
            # CSV 읽기 (low_memory=False로 경고 제거)
            df = pd.read_csv(csv_file, encoding='utf-8-sig', low_memory=False)
            
            if len(df) == 0:
                continue
            
            # 헤더 찾기: 첫 번째 행이 헤더인지 확인
            first_row_values = df.iloc[0].astype(str).tolist() if len(df) > 0 else []
            if any(keyword in str(first_row_values[0]) for keyword in ['순번', '사이트명', '기관명', '게시판명']):
                # 첫 번째 행이 헤더
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
            elif 'Unnamed' in str(df.columns[0]):
                # Unnamed 컬럼이면 첫 번째 행을 헤더로 사용
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
            
            # 컬럼 매핑 (한국어 → 영문 라벨)
            # 직접 매핑 딕셔너리 (labels.yaml 주석 기반)
            direct_mapping = {
                '순번': 'seq_number',
                '사이트명': 'site_name',
                '기관명': 'agency_name',
                '게시판명': 'board_name',
                '게시판 진입 과정': 'board_path',
                '저작물명': 'work_title',
                '저작권자': 'copyright_holder',
                '공동저작자': 'co_author',
                '저작인접권자': 'neighboring_rights_holder',
            }
            
            # 컬럼 매핑 시도
            column_mapping = map_csv_columns(df.columns.tolist(), mode='all')
            
            # 직접 매핑으로 보완
            for col in df.columns:
                col_str = str(col).strip()
                if col_str in direct_mapping and col_str not in column_mapping:
                    column_mapping[col_str] = direct_mapping[col_str]
            
            # NER 라벨 추출 (ner_labels에 해당하는 필드만)
            ner_columns = {}
            for col, en_label in column_mapping.items():
                category = get_label_category(en_label)
                if category == 'ner':
                    ner_columns[col] = en_label
            
            if not ner_columns:
                # 디버깅: 첫 파일에서만 출력
                if csv_file == csv_files[0]:
                    print(f"[디버깅] {csv_file.name}: NER 컬럼 없음")
                    print(f"  매핑된 컬럼: {list(column_mapping.keys())[:5]}")
                continue
            
            # 파일별 데이터 수집
            file_texts: List[List[str]] = []
            file_labels: List[List[str]] = []
            
            # 각 행 처리
            for idx, row in df.iterrows():
                try:
                    # 각 NER 필드를 별도의 문장으로 처리 (더 나은 학습을 위해)
                    # 방법 1: 각 NER 필드를 독립적인 문장으로
                    for col, en_label in ner_columns.items():
                        if col not in row.index:
                            continue
                        value = row[col]
                        if pd.isna(value):
                            continue
                        value = str(value).strip()
                        if value and value.lower() not in ['nan', 'none', '']:
                            # 간단한 토큰화 (공백 기준)
                            tokens = [t for t in value.split() if t.strip()]
                            if tokens and len(tokens) > 0:
                                sentence_tokens: List[str] = []
                                sentence_labels: List[str] = []
                                
                                # BIO 태깅
                                for i, token in enumerate(tokens):
                                    sentence_tokens.append(token)
                                    if i == 0:
                                        sentence_labels.append(f"B-{en_label}")
                                    else:
                                        sentence_labels.append(f"I-{en_label}")
                                
                                # 문장이 너무 짧으면 스킵 (최소 2개 토큰)
                                if len(sentence_tokens) >= 2:
                                    file_texts.append(sentence_tokens)
                                    file_labels.append(sentence_labels)
                    
                    # 방법 2: 전체 행을 하나의 문장으로 (문맥 포함, 선택적)
                    # 이 방법은 주석 처리하고 필요시 활성화
                    """
                    sentence_tokens: List[str] = []
                    sentence_labels: List[str] = []
                    
                    # NER 라벨이 있는 필드 처리
                    for col, en_label in ner_columns.items():
                        if col not in row.index:
                            continue
                        value = row[col]
                        if pd.isna(value):
                            continue
                        value = str(value).strip()
                        if value and value.lower() not in ['nan', 'none', '']:
                            tokens = [t for t in value.split() if t.strip()]
                            if tokens:
                                for i, token in enumerate(tokens):
                                    sentence_tokens.append(token)
                                    if i == 0:
                                        sentence_labels.append(f"B-{en_label}")
                                    else:
                                        sentence_labels.append(f"I-{en_label}")
                    
                    # 다른 필드들도 추가 (O 태그) - 문맥을 위해
                    for col in df.columns:
                        if col not in ner_columns and col in row.index:
                            value = row[col]
                            if pd.isna(value):
                                continue
                            value = str(value).strip()
                            if value and value.lower() not in ['nan', 'none', '']:
                                tokens = [t for t in value.split() if t.strip()]
                                for token in tokens:
                                    sentence_tokens.append(token)
                                    sentence_labels.append("O")
                    
                    if sentence_tokens and len(sentence_tokens) > 0:
                        all_texts.append(sentence_tokens)
                        all_labels.append(sentence_labels)
                    """
                except Exception as e:
                    # 개별 행 오류는 무시하고 계속
                    continue
            
            # 파일별 샘플링 적용
            if len(file_texts) > 0:
                original_count = len(file_texts)
                
                # samples_per_file 우선 적용
                if samples_per_file is not None and samples_per_file > 0:
                    if len(file_texts) > samples_per_file:
                        # 랜덤 샘플링 (매번 다른 샘플)
                        indices = list(range(len(file_texts)))
                        random.shuffle(indices)
                        selected_indices = indices[:samples_per_file]
                        file_texts = [file_texts[i] for i in selected_indices]
                        file_labels = [file_labels[i] for i in selected_indices]
                
                # sample_ratio_per_file 적용 (samples_per_file이 없을 때만)
                elif sample_ratio_per_file is not None and 0 < sample_ratio_per_file < 1.0:
                    sample_count = max(1, int(len(file_texts) * sample_ratio_per_file))
                    if len(file_texts) > sample_count:
                        # 랜덤 샘플링 (매번 다른 샘플)
                        indices = list(range(len(file_texts)))
                        random.shuffle(indices)
                        selected_indices = indices[:sample_count]
                        file_texts = [file_texts[i] for i in selected_indices]
                        file_labels = [file_labels[i] for i in selected_indices]
                
                # 샘플링된 데이터를 전체 리스트에 추가
                all_texts.extend(file_texts)
                all_labels.extend(file_labels)
                
                if original_count != len(file_texts):
                    print(f"  [{csv_file.name}] {original_count}개 → {len(file_texts)}개 샘플링")
        
        except Exception as e:
            print(f"[경고] 파일 처리 실패 {csv_file.name}: {e}")
            continue
    
    if not all_texts:
        print("[경고] 로드된 데이터가 없습니다.")
        return [], [], [], []
    
    print(f"[데이터 로딩] 총 {len(all_texts)}개 문장 로드 완료")
    
    # 데이터셋 크기 제한 (매번 랜덤하게 선택)
    if dataset_size and dataset_size > 0 and len(all_texts) > dataset_size:
        print(f"[데이터 제한] {len(all_texts)}개 중 {dataset_size}개로 제한")
        # random.seed를 설정하지 않아 매번 다른 샘플 선택
        indices = list(range(len(all_texts)))
        random.shuffle(indices)
        selected_indices = indices[:dataset_size]
        all_texts = [all_texts[i] for i in selected_indices]
        all_labels = [all_labels[i] for i in selected_indices]
        print(f"[데이터 제한] 제한 후: {len(all_texts)}개")
    
    # 학습/검증 분할
    random.seed(random_seed)
    indices = list(range(len(all_texts)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_texts = [all_texts[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_texts = [all_texts[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"[데이터 분할] 학습: {len(train_texts)}개, 검증: {len(val_texts)}개")
    
    return train_texts, train_labels, val_texts, val_labels


def train_model(
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    train_data_path: Optional[str] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    dataset_size: Optional[int] = None,
    samples_per_file: Optional[int] = None,
    sample_ratio_per_file: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """모델 훈련"""
    # 데이터 로드
    if train_data_path:
        train_data_path_obj = Path(train_data_path)
        # 디렉토리인 경우 CSV 파일에서 로드
        if train_data_path_obj.is_dir():
            train_texts, train_labels, val_texts, val_labels = _load_training_data(
                data_dir=train_data_path_obj,
                train_ratio=train_ratio,
                random_seed=random_seed,
                dataset_size=dataset_size,
                samples_per_file=samples_per_file,
                sample_ratio_per_file=sample_ratio_per_file
            )
        # 파일인 경우 JSON으로 읽기
        elif train_data_path_obj.is_file() and train_data_path_obj.suffix == '.json':
            with open(train_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            train_texts = data.get("texts", [])
            train_labels = data.get("labels", [])
            val_texts = data.get("val_texts", [])
            val_labels = data.get("val_labels", [])
        else:
            raise ValueError(f"train_data_path는 디렉토리 또는 JSON 파일이어야 합니다: {train_data_path}")
    else:
        # training_csv에서 로드 및 분할
        train_texts, train_labels, val_texts, val_labels = _load_training_data(
            train_ratio=train_ratio,
            random_seed=random_seed,
            dataset_size=dataset_size
        )
    
    # 데이터 검증
    if not train_texts:
        print("[경고] 학습 데이터가 없어 예제 데이터를 사용합니다.")
        train_texts = [["안녕", "하세요"], ["테스트", "입니다"]]
        train_labels = [["O", "O"], ["O", "O"]]
        val_texts = []
        val_labels = []
    
    # 라벨을 ID로 변환
    train_label_ids = [[LABEL_TO_ID.get(l, 0) for l in label_seq] for label_seq in train_labels]
    val_label_ids = [[LABEL_TO_ID.get(l, 0) for l in label_seq] for label_seq in val_labels] if val_labels else []
    
    # 데이터 통계 출력
    train_entity_count = sum(1 for labels in train_labels if any(l != "O" for l in labels))
    val_entity_count = sum(1 for labels in val_labels if any(l != "O" for l in labels))
    print(f"[데이터 통계] 학습 데이터: 총 {len(train_labels)}개 중 엔티티 포함 {train_entity_count}개")
    print(f"[데이터 통계] 검증 데이터: 총 {len(val_labels)}개 중 엔티티 포함 {val_entity_count}개")
    
    # 라벨 분포 확인
    all_train_labels = [l for labels in train_labels for l in labels]
    label_counts = {}
    for label in all_train_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"[데이터 통계] 학습 라벨 분포 (상위 10개): {dict(list(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]))}")
    
    if model_type == "ner":
        # NER 모델 학습
        print(f"[모델 학습] 모델: {model_name or 'google-bert/bert-base-multilingual-cased'}")
        model = NER(model_name=model_name or "google-bert/bert-base-multilingual-cased",
                   model_path=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name or "google-bert/bert-base-multilingual-cased")
        
        train_dataset = NERDataset(train_texts, train_label_ids, tokenizer=tokenizer)
        val_dataset = NERDataset(val_texts, val_label_ids, tokenizer=tokenizer) if val_label_ids else None
        
        output_dir = str(MODEL_DIR / (model_name or "bert").replace("/", "_"))
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # TrainingArguments 파라미터
        training_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": learning_rate,
            "logging_dir": str(MODEL_DIR / "logs"),
            "save_strategy": "epoch",
            "logging_steps": 10,
            "warmup_steps": int(len(train_texts) / batch_size * 0.1),  # 워밍업 스텝 추가
            "weight_decay": 0.01,  # 정규화 추가
        }
        
        # 검증 데이터셋이 있으면 평가 전략 설정
        if val_dataset:
            training_kwargs["eval_strategy"] = "epoch"
            training_kwargs["load_best_model_at_end"] = True
        
        training_args = TrainingArguments(**training_kwargs)
        
        # 평가 메트릭 계산 함수
        def compute_metrics(eval_pred):
            """평가 메트릭 계산 (F1, Precision, Recall)"""
            predictions, labels = eval_pred
            
            # predictions는 (batch, seq_len, num_labels) 형태
            # argmax로 예측 라벨 ID 추출
            pred_ids = np.argmax(predictions, axis=-1)
            
            # 라벨과 예측을 BIO 태그 문자열로 변환
            true_labels_list = []
            pred_labels_list = []
            
            # 디버깅: 예측 분포 확인
            unique_preds = np.unique(pred_ids)
            unique_labels = np.unique(labels[labels != -100]) if len(labels) > 0 else []
            
            for i in range(len(labels)):
                true_seq = []
                pred_seq = []
                
                for j in range(len(labels[i])):
                    # -100은 무시 (padding 또는 subword)
                    if labels[i][j] != -100:
                        true_label = ID_TO_LABEL.get(int(labels[i][j]), "O")
                        pred_label = ID_TO_LABEL.get(int(pred_ids[i][j]), "O")
                        true_seq.append(true_label)
                        pred_seq.append(pred_label)
                
                # 빈 시퀀스가 아니고, 실제 라벨이 있는 경우만 추가
                if true_seq:
                    # 실제 엔티티가 있는 경우만 평가에 포함
                    if any(label != "O" for label in true_seq):
                        true_labels_list.append(true_seq)
                        pred_labels_list.append(pred_seq)
            
            # seqeval 메트릭 계산
            if len(true_labels_list) > 0:
                try:
                    precision = precision_score(true_labels_list, pred_labels_list, zero_division=0)
                    recall = recall_score(true_labels_list, pred_labels_list, zero_division=0)
                    f1 = f1_score(true_labels_list, pred_labels_list, zero_division=0)
                except Exception as e:
                    print(f"[경고] 메트릭 계산 오류: {e}")
                    precision = recall = f1 = 0.0
            else:
                precision = recall = f1 = 0.0
            
            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        
        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
        )
        
        print(f"[모델 학습] 학습 시작 (epochs: {epochs}, batch_size: {batch_size})")
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # 학습 히스토리 수집 (시각화용)
        history = {
            "epochs": list(range(1, epochs + 1)),
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
        }
        
        # log_history에서 추출 (epoch별로 그룹화)
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            # epoch별로 메트릭 수집
            epoch_metrics = {}
            
            for log in trainer.state.log_history:
                epoch = log.get('epoch', None)
                if epoch is None:
                    continue
                
                if epoch not in epoch_metrics:
                    epoch_metrics[epoch] = {
                        'train_loss': [],
                        'eval_loss': None,
                        'eval_f1': None,
                        'eval_precision': None,
                        'eval_recall': None,
                    }
                
                # 학습 loss (step별)
                if 'loss' in log and 'eval_loss' not in log:
                    epoch_metrics[epoch]['train_loss'].append(log.get('loss', 0.0))
                
                # 평가 메트릭 (epoch별)
                if 'eval_loss' in log:
                    epoch_metrics[epoch]['eval_loss'] = log.get('eval_loss', 0.0)
                if 'eval_f1' in log:
                    epoch_metrics[epoch]['eval_f1'] = log.get('eval_f1', 0.0)
                if 'eval_precision' in log:
                    epoch_metrics[epoch]['eval_precision'] = log.get('eval_precision', 0.0)
                if 'eval_recall' in log:
                    epoch_metrics[epoch]['eval_recall'] = log.get('eval_recall', 0.0)
            
            # epoch 순서대로 정렬하여 히스토리 구성
            sorted_epochs = sorted(epoch_metrics.keys())
            for epoch in sorted_epochs:
                metrics = epoch_metrics[epoch]
                
                # train_loss는 평균값 사용
                if metrics['train_loss']:
                    history["train_loss"].append(np.mean(metrics['train_loss']))
                
                # 평가 메트릭
                if metrics['eval_loss'] is not None:
                    history["val_loss"].append(metrics['eval_loss'])
                if metrics['eval_f1'] is not None:
                    history["val_f1"].append(metrics['eval_f1'])
                if metrics['eval_precision'] is not None:
                    history["val_precision"].append(metrics['eval_precision'])
                if metrics['eval_recall'] is not None:
                    history["val_recall"].append(metrics['eval_recall'])
        
        print(f"[모델 학습] 완료: {output_dir}")
        
        return {
            "status": "success",
            "model_path": output_dir,
            "epochs": epochs,
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "history": history
        }
    
    elif model_type == "bilstm_crf":
        # BiLSTMCRF 모델 학습 (구현 필요)
        raise NotImplementedError("BiLSTMCRF 학습은 아직 구현되지 않았습니다.")
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def validate_model(
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """모델 검증"""
    # 검증 데이터 로드
    if val_data_path:
        with open(val_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = data.get("texts", [])
        true_labels = data.get("labels", [])
    else:
        # training_csv에서 검증 데이터만 로드
        _, _, texts, true_labels = _load_training_data(train_ratio=0.0)  # 검증 데이터만
        
        if not texts:
            print("[경고] 검증 데이터가 없습니다.")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 모델 선택 및 예측
    if model_type == "ner":
        model = NER(model_name=model_name, model_path=model_path)
        pred_labels = model.predict(texts)
    elif model_type == "bilstm_crf":
        model = BiLSTMCRF()
        pred_labels = model.predict(texts)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 메트릭 계산
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    return {"precision": precision, "recall": recall, "f1": f1}


# NERDataset 클래스 정의
class NERDataset(Dataset):
    """NER 학습용 데이터셋"""
    
    def __init__(self, texts: List[List[str]], labels: List[List[int]], 
                 tokenizer=None, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            word_ids = encoded.word_ids(batch_index=0) if hasattr(encoded, 'word_ids') else None
            if word_ids is not None:
                aligned_labels = self._align_labels(word_ids, labels)
            else:
                aligned_labels = labels[:self.max_length] + [-100] * (self.max_length - len(labels))
            return {
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels": torch.tensor(aligned_labels[:self.max_length], dtype=torch.long)
            }
        else:
            return {
                "tokens": text,
                "labels": torch.tensor(labels, dtype=torch.long)
            }
    
    def _align_labels(self, word_ids: List[Optional[int]], labels: List[int]) -> List[int]:
        """subword 토큰에 라벨 정렬"""
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(labels[word_id] if word_id < len(labels) else -100)
            else:
                aligned.append(-100)
            prev_word_id = word_id
        return aligned
