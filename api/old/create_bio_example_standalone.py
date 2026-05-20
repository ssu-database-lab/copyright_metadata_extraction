
import random
import os
from pathlib import Path

# --- Generator Logic ---

def generate_random_korean_name():
    surnames = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배조백허유남심노하곽성차주우구신임나전민유진지엄채원천방공현함변염양변여추노도소신석선설마주연방위표명기반왕금옥육인맹제모남궁탁국여진어은편구용"
    syllables = "가강건경고관광구규근기길나남노누다단달담대덕도동두라래로루리마만명무문미민바박백범병보복부비빈사산상서석선설성세소솔수숙순숭슬승시신아안애엄여연영예오옥완요용우원월위유윤율은의이익인일자장재전정제조종주준중지진찬창채천철초춘충치태택판하한해혁현형혜호홍화환회효훈휘희"
    surname = random.choice(surnames)
    name_len = 2 if random.random() < 0.6 else 3
    name = "".join([random.choice(syllables) for _ in range(name_len - 1)])
    return surname + name

def random_phone():
    return f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

def random_date():
    year = random.randint(2000, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}년 {month}월 {day}일"

def random_email():
    return f"user{random.randint(1,999)}@example.com"

ENTITY_GENERATORS = {
    "NAME": generate_random_korean_name,
    "PHONE": random_phone,
    "DATE": random_date,
    "EMAIL": random_email,
    "COMPANY": lambda: f"주식회사 {generate_random_korean_name()}",
    "ADDRESS": lambda: f"서울시 {generate_random_korean_name()}구 {generate_random_korean_name()}로 {random.randint(1,999)}",
    "ID_NUM": lambda: f"{random.randint(0,99):02d}{random.randint(1,12):02d}{random.randint(1,28):02d}-{random.randint(1,4)}******",
    "MONEY": lambda: f"{random.randint(1,9999)}만원",
}

TEMPLATES = [
    "{NAME}입니다.", 
    "{NAME} 님 안녕하세요.", 
    "연락처는 {PHONE}입니다.", 
    "{DATE}에 만나요.", 
    "이메일 {EMAIL}로 보내주세요.",
    "{COMPANY}에서 왔습니다.",
    "주소는 {ADDRESS}입니다.",
    "주민번호: {ID_NUM}",
    "가격은 {MONEY}입니다.",
    "{NAME}의 전화번호는 {PHONE}입니다.",
    "{DATE}까지 {EMAIL}로 제출하세요.",
    "{COMPANY}의 주소는 {ADDRESS}입니다.",
    "{NAME}({ID_NUM}) 확인되었습니다.",
    "{COMPANY}는 {MONEY}를 {DATE}에 지급한다."
]

def generate_samples(num_samples=10):
    samples = []
    for _ in range(num_samples):
        template = random.choice(TEMPLATES)
        text = template
        entities = []
        
        # Find all placeholders
        import re
        placeholders = re.findall(r"\{(\w+)\}", template)
        
        for ph in placeholders:
            if ph in ENTITY_GENERATORS:
                val = ENTITY_GENERATORS[ph]()
                # Replace first occurrence
                # We need to track position to avoid replacing wrong parts if duplicates exist (simplified here)
                # For robust replacement, we should rebuild string or use unique placeholders
                # But for this simple example, replace one by one is okay if values are unique enough
                
                # Better approach: split by placeholder and reconstruct
                # But let's just use replace(..., 1) and track entities
                
                # Actually, we need to know the final position of the entity in the text
                # So we should construct the text and track indices
                pass

        # Re-implement generation with entity tracking
        # Simple parser
        parts = []
        last_pos = 0
        current_text = ""
        current_entities = []
        
        # Regex to find {TAG}
        matches = list(re.finditer(r"\{(\w+)\}", template))
        
        cursor = 0
        for match in matches:
            start, end = match.span()
            tag = match.group(1)
            
            # Add text before tag
            prefix = template[cursor:start]
            current_text += prefix
            
            # Generate value
            if tag in ENTITY_GENERATORS:
                val = ENTITY_GENERATORS[tag]()
                val_start = len(current_text)
                current_text += val
                val_end = len(current_text)
                current_entities.append((val, tag)) # Store text and tag
            else:
                # Unknown tag, keep as is?
                current_text += match.group(0)
            
            cursor = end
            
        current_text += template[cursor:]
        
        samples.append({"text": current_text, "entities": current_entities})
    return samples

def write_bio_word_level(samples, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = sample.get('text', '')
            entities = sample.get('entities', []) or []
            
            tokens = text.split()
            if not tokens:
                continue
                
            labels = ['O'] * len(tokens)
            char_labels = ['O'] * len(text)
            
            # Mark character labels
            # We need to find where the entities are in the final text
            # Since we have the entity text, we can search for it.
            # Warning: Duplicate entity texts might cause issues if we just search.
            # But for this example it's fine.
            
            current_search_pos = 0
            for entity_text, entity_type in entities:
                entity_start = text.find(entity_text, current_search_pos)
                if entity_start == -1:
                    entity_start = text.find(entity_text) # Try from start
                    
                if entity_start != -1:
                    entity_end = entity_start + len(entity_text)
                    
                    char_labels[entity_start] = f"B-{entity_type}"
                    for i in range(entity_start + 1, entity_end):
                        char_labels[i] = f"I-{entity_type}"
                        
                    if entity_start >= current_search_pos:
                        current_search_pos = entity_end
            
            # Map to tokens
            char_pos = 0
            for token_idx, token in enumerate(tokens):
                token_start = text.find(token, char_pos)
                if token_start == -1:
                    char_pos += len(token) + 1
                    continue
                token_end = token_start + len(token)
                
                # Check if token overlaps with any entity label
                # If token has ANY B or I tag, we should label it.
                # Usually we take the label of the first character or majority.
                # Here we check if it contains any part of an entity.
                
                token_labels = char_labels[token_start:token_end]
                
                # If token starts with B, it's B.
                # If token starts with I, it's I (unless it's a new entity starting inside? No, space separated)
                # If token starts with O but contains B/I? (Should not happen if tokenization aligns)
                # Simple logic:
                
                first_char_label = char_labels[token_start]
                if first_char_label != 'O':
                    labels[token_idx] = first_char_label
                else:
                    # Check if any char is B/I
                    for l in token_labels:
                        if l.startswith('B-') or l.startswith('I-'):
                            labels[token_idx] = l # Take the first non-O
                            break
                
                char_pos = token_end
            
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

if __name__ == "__main__":
    samples = generate_samples(10)
    write_bio_word_level(samples, "BIO_TAGGING_EXAMPLE.txt")
    print("Done")
