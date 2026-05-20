import sys
import os
import random
import re
import math

# Entity Types
ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
    "CONTRACT_TYPE", "MONEY", "PERIOD", "ID_NUM", "CONSENT_TYPE", "RIGHT_INFO",
    "PROJECT_NAME", "LAW_REFERENCE", "TITLE", "URL", "DESCRIPTION", "TYPE",
    "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY"
]

# ========== Helper Functions ==========

def extract_entities_from_template(template: str):
    return [match.group(1) for match in re.finditer(r'\{(\w+)\}', template)]

def generate_sample_from_template(template: str, entity_generators):
    entities = {}
    for match in re.finditer(r'\{(\w+)\}', template):
        etype = match.group(1)
        if etype in entity_generators and etype not in entities:
            entities[etype] = entity_generators[etype]()
    
    text = template
    for etype, value in entities.items():
        text = text.replace(f"{{{etype}}}", value)
    
    entity_list = [(value, etype) for etype, value in entities.items() if value in text]
    return text, entity_list, template

def build_template_list(single_templates, dual_templates, multi_templates):
    all_templates = []
    for entity_type, templates_list in single_templates.items():
        for tmpl in templates_list:
            all_templates.append((tmpl, [entity_type]))
    for tmpl in dual_templates + multi_templates:
        entities = extract_entities_from_template(tmpl)
        all_templates.append((tmpl, entities))
    return all_templates

# ========== Generators ==========

def generate_random_korean_name():
    surnames = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전"]
    syllables = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하", "건", "성", "현", "우", "준", "규", "민"]
    surname = random.choice(surnames)
    name_len = 2 if random.random() < 0.6 else 3
    name = "".join([random.choice(syllables) for _ in range(name_len - 1)])
    return surname + name

def random_phone():
    return f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

def random_date():
    return f"{random.randint(2020,2025)}년 {random.randint(1,12)}월 {random.randint(1,28)}일"

def random_email():
    return f"user{random.randint(1,999)}@example.com"

def apply_ocr_noise(text: str, noise_prob: float) -> str:
    if noise_prob <= 0:
        return text
    chars = list(text)
    new_chars = []
    for char in chars:
        if random.random() < noise_prob:
            noise_type = random.choice(['space', 'char'])
            if noise_type == 'space':
                new_chars.append(char + ' ')
            elif noise_type == 'char':
                new_chars.append(char) 
        else:
            new_chars.append(char)
    return "".join(new_chars).replace("  ", " ")

def generate_training_samples(num_samples: int = 3000, balanced: bool = True, noise_level: float = 0.0, dataset_type: str = 'train'):
    # Entity Generators
    entity_generators = {
        "NAME": generate_random_korean_name,
        "PHONE": random_phone,
        "DATE": random_date,
        "EMAIL": random_email,
        "COMPANY": lambda: f"주식회사 {generate_random_korean_name()}",
        "ADDRESS": lambda: f"서울시 강남구 {generate_random_korean_name()}로 {random.randint(1,100)}",
        "ID_NUM": lambda: f"{random.randint(0,99):02d}0101-{random.randint(1,4)}******",
        "MONEY": lambda: f"{random.randint(1,999)}만원",
        "PERIOD": lambda: f"{random.randint(1,12)}개월",
        "CONTRACT_TYPE": lambda: random.choice(["표준계약서", "양도계약서", "이용허락계약서", "비밀유지서약서", "근로계약서"]),
        "POSITION": lambda: random.choice(["팀장", "대표", "사원", "책임", "부장", "이사"]),
        "RIGHT_INFO": lambda: random.choice(["저작재산권", "배포권", "복제권", "전송권", "2차적저작물작성권"]),
        "PROJECT_NAME": lambda: f"프로젝트 {chr(random.randint(65, 90))}{random.randint(1,100)}",
        "LAW_REFERENCE": lambda: f"저작권법 제{random.randint(1,50)}조",
        "TITLE": lambda: f"{generate_random_korean_name()} 관련 합의서",
        "URL": lambda: f"http://www.{generate_random_korean_name()}{random.randint(1,99)}.com",
        "DESCRIPTION": lambda: "본 계약의 상세 내용은 별첨과 같다.",
        "TYPE": lambda: random.choice(["어문저작물", "사진저작물", "영상저작물", "소프트웨어"]),
        "STATUS": lambda: random.choice(["체결 완료", "검토 중", "해지", "갱신"]),
        "DEPARTMENT": lambda: random.choice(["인사팀", "개발팀", "법무팀", "영업팀", "기획팀"]),
        "LANGUAGE": lambda: random.choice(["한국어", "영어", "일본어"]),
        "QUANTITY": lambda: f"{random.randint(1,100)}건",
        "CONSENT_TYPE": lambda: random.choice(["개인정보 수집 이용 동의", "마케팅 수신 동의", "제3자 제공 동의"]),
    }

    # Templates (Simplified for brevity, but enough to show variety)
    train_single_templates = {
        "NAME": ["{NAME}입니다.", "{NAME} 님 안녕하세요.", "작성자: {NAME}"],
        "PHONE": ["연락처는 {PHONE}입니다.", "문의: {PHONE}"],
        "DATE": ["{DATE}에 만나요.", "기한: {DATE}"],
        "EMAIL": ["이메일 {EMAIL}로 보내주세요.", "E-mail: {EMAIL}"],
        "COMPANY": ["{COMPANY}에서 왔습니다.", "소속: {COMPANY}"],
        "ADDRESS": ["주소는 {ADDRESS}입니다.", "위치: {ADDRESS}"],
        "ID_NUM": ["주민번호: {ID_NUM}", "등록번호 {ID_NUM}입니다."],
        "MONEY": ["가격은 {MONEY}입니다.", "비용: {MONEY}"],
    }
    # Fill missing
    for etype in ENTITY_TYPES:
        if etype not in train_single_templates:
            train_single_templates[etype] = [f"{etype}: {{{etype}}}"]

    train_dual_templates = ["{NAME}의 전화번호는 {PHONE}입니다.", "{DATE}까지 {EMAIL}로 제출하세요."]
    train_multi_templates = ["{COMPANY} {DEPARTMENT}의 {NAME} {POSITION}입니다."]

    dev_single_templates = {"NAME": ["Who is {NAME}?"], "PHONE": ["Call {PHONE} now."]}
    # Fill missing
    for etype in ENTITY_TYPES:
        if etype not in dev_single_templates:
            dev_single_templates[etype] = [f"Check {etype}: {{{etype}}}"]
            
    dev_dual_templates = ["Please contact {NAME} at {PHONE}."]
    dev_multi_templates = ["{NAME} ({POSITION}) from {COMPANY} {DEPARTMENT}."]

    if dataset_type == 'train':
        single, dual, multi = train_single_templates, train_dual_templates, train_multi_templates
    else:
        single, dual, multi = dev_single_templates, dev_dual_templates, dev_multi_templates
    
    all_templates = build_template_list(single, dual, multi)
    
    samples = []
    seen_texts = set()
    
    # Generate
    remaining_samples = num_samples
    samples_per_entity = max(1, math.ceil(remaining_samples / len(ENTITY_TYPES)))
    
    for entity_type in ENTITY_TYPES:
        relevant = [(t, e) for t, e in all_templates if entity_type in e]
        if not relevant: relevant = all_templates
        
        count = 0
        random.shuffle(relevant)
        while count < samples_per_entity:
            template, _ = random.choice(relevant)
            text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
            
            if text not in seen_texts:
                if noise_level > 0:
                    text = apply_ocr_noise(text, noise_level)
                samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
                seen_texts.add(text)
                count += 1
                if len(samples) >= num_samples: break
        if len(samples) >= num_samples: break
            
    return samples

def check_data_generation():
    # Define output file path
    output_dir = os.path.join("data", "out")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_samples.txt")
    
    print(f"Generating samples to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        # 1. 학습 데이터 (Train Templates) 확인
        f.write("--- Training Data Samples (Seen Patterns) ---\n")
        train_data = generate_training_samples(num_samples=10, balanced=False, dataset_type='train')
        for sample in train_data:
            f.write(f"{sample['text']}\n")

        # 2. 검증 데이터 (Dev Templates) 확인 - 학습 때 보지 못한 패턴
        f.write("\n--- Validation Data Samples (Unseen Patterns) ---\n")
        dev_data = generate_training_samples(num_samples=10, balanced=False, dataset_type='dev')
        for sample in dev_data:
            f.write(f"{sample['text']}\n")

        # 3. OCR 노이즈 시뮬레이션 확인
        f.write("\n--- OCR Noise Simulation (Noise Prob: 30%) ---\n")
        sample_text = "홍길동의 전화번호는 010-1234-5678이며, 계약일은 2025년 1월 1일입니다."
        f.write(f"Original: {sample_text}\n")
        
        # 노이즈 5번 적용해보기
        for i in range(5):
            noisy_text = apply_ocr_noise(sample_text, noise_prob=0.3)
            f.write(f"Noisy {i+1}: {noisy_text}\n")

    print("Done.")

if __name__ == "__main__":
    check_data_generation()