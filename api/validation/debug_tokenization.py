"""
연합뉴스 토큰화 디버깅
"""
import sys
from pathlib import Path

# validation 폴더에서 실행되므로 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

model_safe = "google-bert-bert-base-multilingual-cased"
model_path = Path("../models/ner") / model_safe

tokenizer = AutoTokenizer.from_pretrained(str(model_path))

# 테스트
texts = [
    "연합뉴스",
    "석수아",
    "전무이사"
]

for text in texts:
    print(f"\n원본: '{text}'")
    print(f"문자별: {list(text)}")
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    offsets = inputs['offset_mapping'][0].tolist()
    
    print(f"토큰: {tokens}")
    print(f"Offsets:")
    for i, (token, offset) in enumerate(zip(tokens, offsets)):
        if offset[0] == 0 and offset[1] == 0:
            print(f"  {i}: '{token}' -> 특수 토큰")
        else:
            substring = text[offset[0]:offset[1]]
            print(f"  {i}: '{token}' -> offset ({offset[0]}, {offset[1]}) = '{substring}'")
