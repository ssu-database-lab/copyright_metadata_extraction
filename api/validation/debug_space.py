"""
공백 처리 디버깅
"""
import sys
from pathlib import Path

# validation 폴더에서 실행되므로 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

model_safe = "google-bert-bert-base-multilingual-cased"
model_path = Path("../models/ner") / model_safe

tokenizer = AutoTokenizer.from_pretrained(str(model_path))

# 테스트: "2022년 12월"
text = "2022년 12월"
print(f"원본 텍스트: '{text}'")
print(f"텍스트 길이: {len(text)} 문자")
print(f"문자별: {[f'{i}:{c}' for i, c in enumerate(text)]}")

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True,
    return_offsets_mapping=True
)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
offsets = inputs['offset_mapping'][0].tolist()

print(f"\n토큰화 결과:")
for i, (token, offset) in enumerate(zip(tokens, offsets)):
    if offset[0] == 0 and offset[1] == 0:
        print(f"  {i}: '{token}' -> 특수 토큰")
    else:
        substring = text[offset[0]:offset[1]]
        print(f"  {i}: '{token}' -> offset ({offset[0]}, {offset[1]}) = '{substring}'")

# 공백 위치 확인
print(f"\n공백 위치:")
for i, c in enumerate(text):
    if c == ' ':
        print(f"  위치 {i}: 공백")
        # 이 위치가 어떤 토큰에 매핑되는지 확인
        for j, offset in enumerate(offsets):
            if offset[0] <= i < offset[1]:
                print(f"    -> 토큰 {j} (offset {offset})")
                break
