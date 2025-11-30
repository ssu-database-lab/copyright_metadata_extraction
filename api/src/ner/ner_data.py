# ner_data.py
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

try:
    from seqeval.metrics import classification_report, f1_score
except ImportError:
    classification_report, f1_score = None, None


def read_conll(path: str) -> List[Tuple[List[str], List[str]]]:
    """
    CoNLL/BIO 형식 파일을 읽어 (tokens, labels) 리스트로 반환.
    예시 형식:

    안녕  O
    민트  B-NAME
    입니다 I-NAME

    (빈 줄로 문장 구분)
    """
    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    print(f"[read_conll] Reading file: {path}")
    line_num = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.rstrip("\n") # 윈도우/리눅스 개행 문자 모두 처리

            # 빈 줄이면 문장 끝으로 간주
            if not line.strip():
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue

            if line.startswith("#"):
                # 주석 라인은 무시
                continue

            # 탭이나 공백으로 분리
            parts = line.split()
            if not parts:
                continue
                
            if len(parts) == 1:
                # 라벨이 없는 경우 경고 출력 후 O 처리하거나 스킵할 수 있음. 
                # 여기서는 안전하게 O로 처리.
                # print(f"[Warning] Line {line_num}: No label found, assuming 'O'. Content: {line}")
                token, tag = parts[0], "O"
            else:
                # 마지막 컬럼을 라벨로, 나머지를 토큰으로 (공백이 포함된 토큰은 거의 없다고 가정)
                # 만약 토큰 자체에 공백이 있다면 parts[0]만 쓰는게 아니라 :-1까지 합쳐야 함.
                # 하지만 일반적인 CoNLL 포맷은 토큰에 공백을 허용하지 않음.
                token = parts[0] 
                tag = parts[-1]

            if token == "-DOCSTART-":
                continue

            tokens.append(token)
            labels.append(tag)

    if tokens:
        sentences.append((tokens, labels))

    print(f"[read_conll] Read {len(sentences)} sentences from {path}")
    return sentences


def build_label_map(
    sentences: List[Tuple[List[str], List[str]]]
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    학습 데이터로부터 label 리스트 및 매핑 생성.
    항상 'O'가 index 0이 되도록 정렬.
    """
    label_set = set()
    for _, tags in sentences:
        for tag in tags:
            label_set.add(tag)

    if "O" in label_set:
        label_set.remove("O")

    label_list = ["O"] + sorted(label_set)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


class NERDataset(Dataset):
    """
    (tokens, labels) 문장 리스트를 BERT 입력 + 라벨 텐서로 변환하는 Dataset.
    """
    def __init__(
        self,
        sentences: List[Tuple[List[str], List[str]]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.encodings = []
        self.labels = []
        self.label2id = label2id
        self.max_length = max_length
        self.tokenizer = tokenizer

        for i, (tokens, tags) in enumerate(sentences):
            # 토큰 개수와 라벨 개수가 다르면 데이터 오류이므로 경고/스킵
            if len(tokens) != len(tags):
                print(f"[Dataset] Warning: Sentence {i} has {len(tokens)} tokens but {len(tags)} tags. Skipping.")
                continue

            # is_split_into_words=True 필수
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_token_type_ids=False,
            )

            # word_ids()를 사용해 word-level 라벨을 subword 단위로 복제
            word_ids = encoding.word_ids()
            label_ids: List[int] = []
            
            previous_word_idx = None

            for word_id in word_ids:
                if word_id is None:
                    # CLS, SEP, PAD 등은 loss 계산에서 제외 (-100)
                    label_ids.append(-100)
                else:
                    # 모든 subword에 동일한 라벨 부여 (seqeval 평가를 위해)
                    try:
                        tag = tags[word_id]
                        label_ids.append(self.label2id.get(tag, self.label2id.get("O", 0)))
                    except IndexError:
                        # 혹시 모를 인덱스 에러 방지
                        label_ids.append(-100)
                previous_word_idx = word_id
            
            # 최종 길이 검증
            if len(label_ids) != max_length:
                 # truncation 등으로 길이가 다를 수 있으나, padding='max_length' 했으므로 같아야 함.
                 # 다만 word_ids 길이 자체가 max_length임.
                 pass

            self.encodings.append(
                {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                }
            )
            self.labels.append(label_ids)

        print(f"[Dataset] Created {len(self.encodings)} samples.")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.encodings[idx]["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def evaluate_ner(model, data_loader, id2label: Dict[int, str], device) -> Tuple[float, str]:
    """
    주어진 data_loader에 대해 F1 / 리포트 계산.
    seqeval이 설치되어 있으면 sequence-level F1 사용,
    없으면 토큰 단위 정확도를 계산해서 리포트 문자열로 반환.
    """
    model.eval()
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # CRF 모델 체크
            if hasattr(model, "crf"):
                # BertBiLstmCrf - returns List[List[int]]
                pred_paths = model(input_ids=input_ids, attention_mask=attention_mask)
                
                for i in range(len(pred_paths)):
                    t_seq = []
                    p_seq = []
                    
                    # pred_paths[i]는 attention_mask가 1인 토큰에 대한 예측
                    pred_idx = 0
                    
                    for j in range(len(labels[i])):
                        if attention_mask[i][j] == 0:
                            continue
                        
                        if labels[i][j] != -100:
                            t_seq.append(id2label[labels[i][j].item()])
                            if pred_idx < len(pred_paths[i]):
                                p_seq.append(id2label[pred_paths[i][pred_idx]])
                            else:
                                p_seq.append("O")
                        
                        pred_idx += 1
                    
                    all_true.append(t_seq)
                    all_pred.append(p_seq)
                    
            else:
                # Pure Token Classification
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                for i in range(len(labels)):
                    t_seq = []
                    p_seq = []
                    for j, label_id in enumerate(labels[i]):
                        if label_id != -100:
                            t_seq.append(id2label[label_id.item()])
                            p_seq.append(id2label[predictions[i][j].item()])
                    
                    all_true.append(t_seq)
                    all_pred.append(p_seq)

    # seqeval이 있으면 sequence F1
    if classification_report is not None and f1_score is not None:
        score = f1_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, digits=4)
        return float(score), report

    # 없으면 토큰 단위 accuracy
    correct = 0
    total = 0
    for true_seq, pred_seq in zip(all_true, all_pred):
        for t, p in zip(true_seq, pred_seq):
            total += 1
            if t == p:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    report = f"seqeval이 설치되어 있지 않아 토큰 단위 accuracy만 계산했습니다. accuracy = {acc:.4f}"
    return float(acc), report