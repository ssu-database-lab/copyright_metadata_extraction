"""
모델 예측 디버깅 스크립트
"""
import sys
from pathlib import Path

# validation 폴더에서 실행되므로 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# 모델 로드
model_name = "google-bert/bert-base-multilingual-cased"
model_safe = "google-bert-bert-base-multilingual-cased"
model_path = Path("../models/ner") / model_safe

print(f"모델 로드: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForTokenClassification.from_pretrained(str(model_path))
model.eval()

# label_map 로드
label_map_path = model_path / "label_map.json"
with open(label_map_path, 'r', encoding='utf-8') as f:
    label_data = json.load(f)
    id2label = {int(k): v for k, v in label_data['id2label'].items()}

print(f"\n라벨 맵: {id2label}")

# 테스트 문장 (검증 데이터에서)
test_sentence = ['석', '수', '아']
test_labels = ['B-NAME', 'I-NAME', 'I-NAME']

print(f"\n원본 문장: {''.join(test_sentence)}")
print(f"정답 라벨: {test_labels}")

# 예측
text = ''.join(test_sentence)
print(f"\n입력 텍스트: '{text}'")

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True,
    return_offsets_mapping=True
)

print(f"\n토큰화 결과:")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"  토큰: {tokens}")
print(f"  Offset mapping: {inputs['offset_mapping'][0].tolist()}")

offset_mapping = inputs.pop('offset_mapping')[0]

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    pred_label_ids = predictions[0].numpy()

print(f"\n예측 라벨 ID: {pred_label_ids}")
print(f"예측 라벨: {[id2label.get(int(lid), 'O') for lid in pred_label_ids]}")

# Offset mapping을 사용하여 문자 단위 라벨 정렬
char_labels = ['O'] * len(text)

print(f"\n문자 단위 정렬:")
for idx, (start, end) in enumerate(offset_mapping):
    if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
        print(f"  토큰 {idx} ({tokens[idx]}): 특수 토큰, 스킵")
        continue
    
    label_id = int(pred_label_ids[idx])
    label = id2label.get(label_id, 'O')
    
    print(f"  토큰 {idx} ({tokens[idx]}): offset ({start}, {end}), label={label}")
    
    # 해당 offset 범위의 모든 문자에 라벨 할당
    for char_idx in range(start, end):
        if char_idx < len(char_labels):
            char_labels[char_idx] = label

print(f"\n문자별 라벨:")
for i, (char, label) in enumerate(zip(text, char_labels)):
    print(f"  {i}: '{char}' -> {label} (정답: {test_labels[i] if i < len(test_labels) else 'N/A'})")

# 최종 정렬
token_pred_labels = []
for idx, char in enumerate(test_sentence):
    if idx < len(char_labels):
        token_pred_labels.append(char_labels[idx])
    else:
        token_pred_labels.append('O')

print(f"\n최종 예측: {token_pred_labels}")
print(f"정답:      {test_labels}")
print(f"일치 여부: {token_pred_labels == test_labels}")
