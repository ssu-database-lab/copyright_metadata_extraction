"""
복잡한 문장으로 예측 디버깅
"""
import sys
from pathlib import Path

# validation 폴더에서 실행되므로 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# 모델 로드
model_safe = "google-bert-bert-base-multilingual-cased"
model_path = Path("../models/ner") / model_safe

tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForTokenClassification.from_pretrained(str(model_path))
model.eval()

# label_map 로드
label_map_path = model_path / "label_map.json"
with open(label_map_path, 'r', encoding='utf-8') as f:
    label_data = json.load(f)
    id2label = {int(k): v for k, v in label_data['id2label'].items()}

# 검증 데이터에서 실제 문장 (validation.txt 1-100줄)
test_cases = [
    {
        'sentence': ['연', '합', '뉴', '스', ' ', '소', '속', ' ', '직', '원', ' ', '정', '보'],
        'labels': ['B-COMPANY', 'I-COMPANY', 'I-COMPANY', 'I-COMPANY', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    },
    {
        'sentence': ['석', '수', '아'],
        'labels': ['B-NAME', 'I-NAME', 'I-NAME']
    },
    {
        'sentence': ['전', '무', '이', '사'],
        'labels': ['B-POSITION', 'I-POSITION', 'I-POSITION', 'I-POSITION']
    },
    {
        'sentence': ['2', '0', '2', '2', '년', ' ', '1', '2', '월', ' ', '1', '9', '일'],
        'labels': ['B-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE', 'I-DATE']
    }
]

correct = 0
total = 0

for idx, test in enumerate(test_cases, 1):
    sentence = test['sentence']
    true_labels = test['labels']
    
    text = ''.join(sentence)
    
    # 예측
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
        return_offsets_mapping=True
    )
    
    offset_mapping = inputs.pop('offset_mapping')[0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        pred_label_ids = predictions[0].numpy()
    
    # Offset mapping을 사용하여 문자 단위 라벨 정렬
    char_labels = ['O'] * len(text)
    
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        
        label_id = int(pred_label_ids[i])
        label = id2label.get(label_id, 'O')
        
        for char_idx in range(start, end):
            if char_idx < len(char_labels):
                char_labels[char_idx] = label
    
    # 최종 정렬
    token_pred_labels = []
    for i, char in enumerate(sentence):
        if i < len(char_labels):
            token_pred_labels.append(char_labels[i])
        else:
            token_pred_labels.append('O')
    
    # 정확도 계산
    match = sum(1 for p, t in zip(token_pred_labels, true_labels) if p == t)
    total_tokens = len(true_labels)
    accuracy = match / total_tokens * 100
    
    total += len(true_labels)
    correct += match
    
    print(f"\n테스트 케이스 {idx}: {''.join(sentence)}")
    print(f"정답:   {' '.join(true_labels)}")
    print(f"예측:   {' '.join(token_pred_labels)}")
    print(f"정확도: {accuracy:.1f}% ({match}/{total_tokens})")
    
    if token_pred_labels != true_labels:
        print("❌ 불일치 발견:")
        for i, (char, pred, true) in enumerate(zip(sentence, token_pred_labels, true_labels)):
            if pred != true:
                print(f"  위치 {i}: '{char}' -> 예측={pred}, 정답={true}")

print(f"\n전체 정확도: {correct/total*100:.1f}% ({correct}/{total})")
