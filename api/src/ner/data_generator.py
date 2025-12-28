import random
import re
import math
from typing import List, Dict, Tuple, Any, Optional, Union

# Entity Types
ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
    "CONTRACT_TYPE", "MONEY", "PERIOD", "ID_NUM", "CONSENT_TYPE", "RIGHT_INFO",
    "PROJECT_NAME", "LAW_REFERENCE", "TITLE", "URL", "DESCRIPTION", "TYPE",
    "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY"
]

# BIO Labels
BIO_LABELS = ["O"] + [f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ["B", "I"]]
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# ========== 데이터 생성 헬퍼 함수 ==========

def extract_entities_from_template(template: str) -> List[str]:
    return [match.group(1) for match in re.finditer(r'\{(\w+)\}', template)]

def generate_sample_from_template(template: str, entity_generators: Dict) -> Tuple[str, List[Tuple[str, str]], str]:
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

def build_template_list(single_templates: Dict, dual_templates: List, multi_templates: List) -> List[Tuple[str, List[str]]]:
    all_templates = []
    for entity_type, templates_list in single_templates.items():
        for tmpl in templates_list:
            all_templates.append((tmpl, [entity_type]))
    for tmpl in dual_templates + multi_templates:
        entities = extract_entities_from_template(tmpl)
        all_templates.append((tmpl, entities))
    return all_templates

# ========== 생성기 함수들 ==========

def generate_random_korean_name():
    surnames = [
        "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
        "홍", "고", "문", "양", "손", "배", "조", "백", "허", "유", "남", "심", "노", "하", "곽", "성", "차", "주", "우", "구",
        "신", "임", "나", "전", "민", "유", "진", "지", "엄", "채", "원", "천", "방", "공", "강", "현", "함", "변", "염", "양",
        "변", "여", "추", "노", "도", "소", "신", "석", "선", "설", "마", "길", "주", "연", "방", "위", "표", "명", "기", "반",
        "왕", "금", "옥", "육", "인", "맹", "제", "모", "장", "남", "탁", "국", "여", "진", "어", "은", "편", "구", "용",
        "독고", "제갈", "남궁", "황보", "선우", "사공", "서문", "제", "단", "빈", "복", "사", "목", "탄", "온"
    ]
    syllables = [
        "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하", 
        "건", "성", "현", "우", "준", "규", "민", "지", "석", "진", "일", "수", "호", "영", "환", "식", "철", "훈", "원", "동", "창", "상", "재", "종", "근", "광", "명", "승", "세", "대", "두", "만", "병", "보", "비", "빈", "산", "서", "선", "설", "섭", "성", "소", "솔", "수", "순", "시", "신", "실", "안", "애", "엄", "여", "연", "영", "예", "오", "옥", "완", "요", "용", "우", "운", "웅", "원", "월", "위", "유", "윤", "율", "은", "의", "이", "익", "인", "일", "임", "자", "장", "재", "전", "정", "제", "조", "종", "주", "준", "중", "지", "진", "찬", "창", "채", "천", "철", "초", "춘", "충", "치", "태", "택", "판", "하", "학", "한", "해", "혁", "현", "형", "혜", "호", "홍", "화", "환", "회", "효", "훈", "휘", "희",
        "겸", "경", "계", "고", "곡", "공", "관", "광", "교", "구", "국", "군", "궁", "권", "궤", "귀", "규", "균", "극", "근", "금", "급", "긍", "기", "길", "김",
        "란", "랑", "래", "랭", "량", "려", "력", "련", "렬", "렴", "령", "례", "로", "록", "론", "롱", "뢰", "료", "룡", "루", "류", "륙", "륜", "률", "륭", "륵", "름", "릉", "리", "린", "림", "립"
    ]
    surname = random.choice(surnames)
    name_len = 2 if random.random() < 0.6 else 3
    name = "".join([random.choice(syllables) for _ in range(name_len - 1)])
    return surname + name

def random_phone():
    formats = [
        f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
        f"010{random.randint(1000,9999)}{random.randint(1000,9999)}",
        f"02-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"031-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"070-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
        f"+82-10-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    ]
    return random.choice(formats)

def random_date():
    year = random.randint(1990, 2030)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    
    formats = [
        f"{year}년 {month}월 {day}일",
        f"{year}.{month:02d}.{day:02d}",
        f"{year}-{month:02d}-{day:02d}",
        f"{year}/{month:02d}/{day:02d}",
        f"{year}년 {month}월 {day}일 (월)",
        f"{year}. {month}. {day}."
    ]
    return random.choice(formats)

def random_email():
    domains = ["gmail.com", "naver.com", "daum.net", "kakao.com", "outlook.com", "yahoo.com", "icloud.com", "company.co.kr", "univ.ac.kr"]
    user_len = random.randint(4, 12)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789._"
    user = "".join(random.choice(chars) for _ in range(user_len))
    return f"{user}@{random.choice(domains)}"

def random_company():
    suffixes = ['주식회사', '(주)', '유한회사', '사단법인', '재단법인', '합자회사', '협동조합', 'Inc.', 'Corp.', 'Ltd.', 'Co.']
    industries = ['산업', '건설', '무역', '통상', '전자', '시스템', '소프트', '네트웍스', '기획', '디자인', '컨설팅', '홀딩스', '그룹', '물산', '제약', '바이오', '엔터테인먼트', '미디어', '스튜디오', '화학', '에너지', '솔루션', '테크', '로보틱스', 'AI', '데이터', '클라우드', '게임즈', '푸드', '리테일', '로지스틱스']
    
    name_type = random.choice(['korean', 'english_korean', 'acronym'])
    
    if name_type == 'korean':
        base = generate_random_korean_name() + random.choice(industries)
    elif name_type == 'english_korean':
        base = random.choice(['애플', '구글', '아마존', '테슬라', '삼성', '현대', '엘지', '에스케이', '롯데', '한화', '씨제이', '카카오', '네이버', '라인', '쿠팡', '배달의민족', '토스', '당근']) + random.choice(industries)
    else:
        base = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(random.randint(2, 4))) + random.choice(industries)
        
    if random.random() < 0.5:
        return f"{random.choice(suffixes)} {base}"
    else:
        return f"{base} {random.choice(suffixes)}"

def random_address():
    cities = ['서울', '경기', '부산', '인천', '대구', '대전', '광주', '울산', '제주', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '세종']
    districts = ['강남구', '서초구', '송파구', '종로구', '중구', '마포구', '영등포구', '분당구', '일산동구', '해운대구', '수성구', '유성구', '남구', '북구', '동구', '서구']
    roads = ['테헤란로', '강남대로', '세종대로', '종로', '을지로', '퇴계로', '한강대로', '올림픽로', '송파대로', '중앙로', '대학로', '학동로', '도산대로', '압구정로', '삼성로', '영동대로']
    
    city = random.choice(cities)
    if city in ['서울', '부산', '인천', '대구', '대전', '광주', '울산']:
        city += "시"
    elif city in ['경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남']:
        city += "도"
    elif city == '세종':
        city += "특별자치시"
        
    addr = f"{city} {random.choice(districts)} {random.choice(roads)} {random.randint(1,999)}"
    if random.random() < 0.5:
        addr += f"-{random.randint(1,50)}"
    
    if random.random() < 0.3:
        addr += f" ({generate_random_korean_name()}빌딩)"
    elif random.random() < 0.3:
        addr += f" {random.randint(1,30)}층 {random.randint(101, 999)}호"
        
    return addr

def apply_ocr_noise(text: str, noise_prob: float) -> str:
    """
    Simulate OCR errors:
    - Space insertion (common in Korean OCR)
    - Character mutation (rare but possible)
    - Deletion
    """
    if noise_prob <= 0:
        return text
        
    chars = list(text)
    new_chars = []
    
    for char in chars:
        if random.random() < noise_prob:
            noise_type = random.choice(['space', 'char', 'delete'])
            
            if noise_type == 'space':
                new_chars.append(char + ' ')
            elif noise_type == 'char':
                confusions = {'가': '거', '나': '너', '다': '더', '1': 'l', '0': 'O', '이': '아', '은': '을', '의': '이', '한': '하'}
                new_chars.append(confusions.get(char, char))
            elif noise_type == 'delete':
                pass # Skip char
        else:
            new_chars.append(char)
            
    return "".join(new_chars).replace("  ", " ")

def generate_training_samples(num_samples: int = 3000, balanced: bool = True, noise_level: float = 0.0, dataset_type: str = 'train') -> List[Dict]:
    """
    학습 데이터 생성 (다양성 강화 + OCR 노이즈 버전)
    dataset_type: 'train' or 'dev'/'test'. 
    'dev'/'test' will use a different set of templates to evaluate generalization.
    """
    # 엔티티 생성기 매핑
    entity_generators = {
        "NAME": generate_random_korean_name,
        "PHONE": random_phone,
        "DATE": random_date,
        "EMAIL": random_email,
        "COMPANY": lambda: f"{random.choice(['주식회사', '(주)', '유한회사', '사단법인', '재단법인'])} {generate_random_korean_name()}{random.choice(['산업', '건설', '무역', '통상', '전자', '시스템', '소프트', '네트웍스', '기획', '디자인', '컨설팅', '홀딩스', '그룹', '물산', '제약', '바이오', '엔터테인먼트', '미디어', '스튜디오'])}",
        "ADDRESS": lambda: f"{random.choice(['서울시', '경기도', '부산시', '인천시', '대구시', '대전시', '광주시', '울산시', '제주시', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', '경상남도'])} {generate_random_korean_name()}{random.choice(['구', '시', '군'])} {generate_random_korean_name()}{random.choice(['로', '길', '대로'])} {random.randint(1,999)}{random.choice(['', f'-{random.randint(1,50)}'])}",
        "ID_NUM": lambda: f"{random.randint(0,99):02d}{random.randint(1,12):02d}{random.randint(1,28):02d}-{random.randint(1,4)}******",
        "MONEY": lambda: f"{random.randint(1,9999)}{random.choice(['만원', '천원', '억원', '백만원', '천만원'])}",
        "PERIOD": lambda: f"{random.randint(1,36)}개월",
        "CONTRACT_TYPE": lambda: random.choice(["표준계약서", "양도계약서", "이용허락계약서", "비밀유지서약서", "근로계약서", "용역계약서", "도급계약서", "임대차계약서", "매매계약서", "업무협약서", "MOU", "라이선스계약서", "출판권설정계약서", "전속계약서", "가맹계약서"]),
        "POSITION": lambda: random.choice(["팀장", "대표", "사원", "책임", "부장", "이사", "상무", "전무", "사장", "회장", "대리", "과장", "차장", "주임", "연구원", "선임연구원", "수석연구원", "매니저", "파트장", "본부장", "실장", "국장", "센터장"]),
        "RIGHT_INFO": lambda: random.choice(["저작재산권", "배포권", "복제권", "전송권", "2차적저작물작성권", "공연권", "전시권", "대여권", "방송권", "성명표시권", "공표권", "동일성유지권", "출판권", "초상권", "퍼블리시티권"]),
        "PROJECT_NAME": lambda: f"프로젝트 {chr(random.randint(65, 90))}{random.randint(1,100)}",
        "LAW_REFERENCE": lambda: f"저작권법 제{random.randint(1,100)}조 {random.randint(1,10)}항",
        "TITLE": lambda: f"{generate_random_korean_name()} 관련 {random.choice(['합의서', '계약서', '약정서', '동의서', '확약서', '각서', '신청서', '청구서', '명세서', '보고서', '제안서', '기획서', '계획서'])}",
        "URL": lambda: f"http://www.{random.choice(['google', 'naver', 'daum', 'kakao', 'samsung', 'lg', 'sk', 'hyundai', 'coupang', 'woowahan', 'toss', 'carrot', 'line', 'facebook', 'twitter', 'instagram', 'youtube', 'netflix', 'disney', 'apple', 'microsoft', 'amazon'])}{random.randint(1,999)}.com",
        "DESCRIPTION": lambda: random.choice(["본 계약의 상세 내용은 별첨과 같다.", "세부 사항은 첨부 문서를 참조한다.", "기타 사항은 상호 협의하여 결정한다.", "특약 사항은 별도로 정한다.", "본 계약은 신의성실의 원칙에 따라 이행한다.", "분쟁 발생 시 관할 법원의 판결에 따른다."]),
        "TYPE": lambda: random.choice(["어문저작물", "사진저작물", "영상저작물", "소프트웨어", "음악저작물", "미술저작물", "건축저작물", "연극저작물", "도형저작물", "편집저작물", "2차적저작물"]),
        "STATUS": lambda: random.choice(["체결 완료", "검토 중", "해지", "갱신", "만료", "보류", "협의 중", "작성 중", "승인 대기", "반려", "수정 요청"]),
        "DEPARTMENT": lambda: random.choice(["인사팀", "개발팀", "법무팀", "영업팀", "기획팀", "재무팀", "회계팀", "총무팀", "마케팅팀", "홍보팀", "디자인팀", "CS팀", "QA팀", "보안팀", "전략팀", "해외사업팀", "구매팀", "물류팀", "생산팀", "연구소"]),
        "LANGUAGE": lambda: random.choice(["한국어", "영어", "일본어", "중국어", "프랑스어", "독일어", "스페인어", "러시아어", "아랍어"]),
        "QUANTITY": lambda: f"{random.randint(1,1000)}{random.choice(['건', '개', '부', '매', '편', '곡', '점', '회', '시간', '일', '개월', '년'])}",
        "CONSENT_TYPE": lambda: random.choice(["개인정보 수집 이용 동의", "마케팅 수신 동의", "제3자 제공 동의", "위치정보 이용 동의", "서비스 이용 약관 동의", "전자금융거래 약관 동의"]),
    }

    # 1. Train Templates (Common patterns)
    train_single_templates = {
        "NAME": [
            "{NAME}입니다.", "{NAME} 님 안녕하세요.", "작성자: {NAME}", "본인은 {NAME}로서 서명합니다.",
            "{NAME} 귀하에게 알립니다.", "담당자는 {NAME}입니다.", "수신: {NAME}", "발신: {NAME}",
            "성명: {NAME}", "이름: {NAME}", "{NAME} (인)", "{NAME} (서명)", "대리인 {NAME}",
            "계약 당사자: {NAME}", "대표자: {NAME}", "신청인: {NAME}", "청구인: {NAME}", "피청구인: {NAME}",
            "채권자: {NAME}", "채무자: {NAME}", "임대인: {NAME}", "임차인: {NAME}", "매도인: {NAME}", "매수인: {NAME}",
            "근로자: {NAME}", "사용자: {NAME}", "저작권자: {NAME}", "저작인접권자: {NAME}", "배타적발행권자: {NAME}",
            "확인자: {NAME}", "검수자: {NAME}", "승인자: {NAME}", "참조: {NAME}", "문의: {NAME}"
        ],
        "PHONE": [
            "연락처는 {PHONE}입니다.", "문의: {PHONE}", "Tel: {PHONE}", "비상연락망: {PHONE}",
            "{PHONE}으로 전화주세요.", "휴대전화 {PHONE} 기재 요망.", "H.P: {PHONE}", "전화: {PHONE}",
            "팩스: {PHONE}", "대표번호: {PHONE}", "고객센터: {PHONE}", "상담문의: {PHONE}",
            "직통번호: {PHONE}", "내선번호: {PHONE}", "모바일: {PHONE}", "Cell: {PHONE}",
            "Fax: {PHONE}", "Telephone: {PHONE}", "Phone: {PHONE}", "Contact: {PHONE}",
            "긴급 연락처: {PHONE}", "예약 문의: {PHONE}", "가입 문의: {PHONE}", "AS 접수: {PHONE}"
        ],
        "DATE": [
            "{DATE}에 만나요.", "기한: {DATE}", "날짜: {DATE}", "계약일: {DATE}",
            "{DATE}부터 효력이 발생합니다.", "마감일은 {DATE}까지입니다.", "작성일: {DATE}",
            "체결일자: {DATE}", "유효기간: {DATE}", "{DATE} 기준", "{DATE} 현재",
            "발행일: {DATE}", "지급일: {DATE}", "만기일: {DATE}", "개시일: {DATE}", "종료일: {DATE}",
            "납부기한: {DATE}", "제출기한: {DATE}", "승인일: {DATE}", "등록일: {DATE}", "수정일: {DATE}",
            "{DATE} 시행", "{DATE} 공고", "{DATE} 접수", "{DATE} 처리", "{DATE} 완료"
        ],
        "EMAIL": [
            "이메일 {EMAIL}로 보내주세요.", "E-mail: {EMAIL}", "문의 사항은 {EMAIL}로.",
            "회신 주소: {EMAIL}", "{EMAIL} (업무용)", "전자우편: {EMAIL}", "메일: {EMAIL}",
            "contact: {EMAIL}", "support: {EMAIL}", "help: {EMAIL}", "info: {EMAIL}",
            "admin: {EMAIL}", "webmaster: {EMAIL}", "recruit: {EMAIL}", "sales: {EMAIL}",
            "billing: {EMAIL}", "privacy: {EMAIL}", "security: {EMAIL}", "press: {EMAIL}",
            "제출처: {EMAIL}", "접수처: {EMAIL}", "문의처: {EMAIL}", "담당자 이메일: {EMAIL}"
        ],
        "COMPANY": [
            "{COMPANY}에서 왔습니다.", "소속: {COMPANY}", "{COMPANY} 대표이사 귀하",
            "당사자는 {COMPANY}입니다.", "{COMPANY}와의 협력.", "상호: {COMPANY}",
            "법인명: {COMPANY}", "업체명: {COMPANY}", "발주처: {COMPANY}", "수주처: {COMPANY}",
            "시행사: {COMPANY}", "시공사: {COMPANY}", "대행사: {COMPANY}", "주관사: {COMPANY}",
            "후원사: {COMPANY}", "협찬사: {COMPANY}", "제휴사: {COMPANY}", "공급사: {COMPANY}",
            "제조사: {COMPANY}", "판매사: {COMPANY}", "유통사: {COMPANY}", "운영사: {COMPANY}",
            "개발사: {COMPANY}", "투자사: {COMPANY}", "배급사: {COMPANY}", "제작사: {COMPANY}"
        ],
        "ADDRESS": [
            "주소는 {ADDRESS}입니다.", "위치: {ADDRESS}", "사업장 소재지: {ADDRESS}",
            "{ADDRESS}로 배송 바랍니다.", "본점: {ADDRESS}", "거주지: {ADDRESS}",
            "등록기준지: {ADDRESS}", "배달 장소: {ADDRESS}", "설치 장소: {ADDRESS}",
            "납품 장소: {ADDRESS}", "공사 현장: {ADDRESS}", "행사 장소: {ADDRESS}",
            "모임 장소: {ADDRESS}", "방문 장소: {ADDRESS}", "수령 장소: {ADDRESS}",
            "반품 주소: {ADDRESS}", "교환 주소: {ADDRESS}", "우편물 수령지: {ADDRESS}"
        ],
        "ID_NUM": [
            "주민번호: {ID_NUM}", "등록번호 {ID_NUM}입니다.", "사업자번호 {ID_NUM} 기재.",
            "신분증 번호: {ID_NUM}", "주민등록번호: {ID_NUM}", "법인등록번호: {ID_NUM}",
            "외국인등록번호: {ID_NUM}", "여권번호: {ID_NUM}", "운전면허번호: {ID_NUM}",
            "고유번호: {ID_NUM}", "관리번호: {ID_NUM}", "접수번호: {ID_NUM}",
            "승인번호: {ID_NUM}", "허가번호: {ID_NUM}", "인가번호: {ID_NUM}",
            "등록증 번호: {ID_NUM}", "자격증 번호: {ID_NUM}", "회원번호: {ID_NUM}"
        ],
        "MONEY": [
            "가격은 {MONEY}입니다.", "비용: {MONEY}", "계약금 {MONEY}를 지급한다.",
            "총액 {MONEY} (VAT 별도)", "보상금 {MONEY} 산정.", "금액: {MONEY}",
            "일금 {MONEY}정", "합계: {MONEY}", "잔금: {MONEY}", "계약보증금: {MONEY}",
            "중도금: {MONEY}", "선급금: {MONEY}", "기성금: {MONEY}", "유보금: {MONEY}",
            "지체상금: {MONEY}", "위약금: {MONEY}", "손해배상금: {MONEY}", "합의금: {MONEY}",
            "수수료: {MONEY}", "인건비: {MONEY}", "재료비: {MONEY}", "경비: {MONEY}",
            "임대료: {MONEY}", "관리비: {MONEY}", "보증금: {MONEY}", "권리금: {MONEY}"
        ],
    }
    
    # Fill missing types for train
    for etype in ENTITY_TYPES:
        if etype not in train_single_templates:
            train_single_templates[etype] = [
                f"이것은 {etype} 예시인 {{{etype}}}입니다.", 
                f"{etype}: {{{etype}}}",
                f"상세 {etype} 정보: {{{etype}}}",
                f"{{{etype}}}에 관한 내용.",
                f"다음 {etype}을 확인하세요: {{{etype}}}",
                f"입력된 {etype} 값은 {{{etype}}} 입니다.",
                f"필수 입력 항목 ({etype}): {{{etype}}}",
                f"선택 입력 항목 ({etype}): {{{etype}}}",
                f"변경된 {etype}: {{{etype}}}",
                f"기존 {etype}: {{{etype}}}",
                f"추가된 {etype}: {{{etype}}}",
                f"삭제된 {etype}: {{{etype}}}"
            ]

    train_dual_templates = [
        "{NAME}의 전화번호는 {PHONE}입니다.", "{DATE}까지 {EMAIL}로 제출하세요.",
        "{COMPANY}의 주소는 {ADDRESS}입니다.", "{NAME}님({ID_NUM}) 확인되었습니다.",
        "{COMPANY}는 {MONEY}를 {DATE}에 지급한다.", "{NAME} {POSITION}님의 연락처는 {PHONE}입니다.",
        "{CONTRACT_TYPE} 체결일은 {DATE}입니다.", "{PROJECT_NAME} 예산은 {MONEY}입니다.",
        "{DEPARTMENT} 소속 {NAME}입니다.", "{RIGHT_INFO} 양도 대가는 {MONEY}입니다.",
        "{LAW_REFERENCE}에 의거하여 {CONTRACT_TYPE}을 체결합니다.", "{NAME}은 {ADDRESS}에 거주합니다.",
        "{COMPANY} (대표: {NAME})", "{DATE} 자로 {COMPANY}와 계약함",
        "{NAME} ({PHONE})", "{EMAIL} / {PHONE}", "{ADDRESS} ({COMPANY})",
        "{MONEY} ({DATE} 지급)", "{CONTRACT_TYPE} ({DATE})", "{NAME} - {POSITION}",
        "{COMPANY} {DEPARTMENT} {NAME}", "{NAME} ({EMAIL})", "{PHONE} ({NAME})",
        "{DATE} {TITLE}", "{MONEY} ({QUANTITY})", "{TYPE} ({LANGUAGE})",
        "{STATUS}: {PROJECT_NAME}", "{CONSENT_TYPE} ({DATE})", "{URL} ({COMPANY})",
        "{DESCRIPTION} ({DATE})", "{LAW_REFERENCE} ({RIGHT_INFO})", "{ID_NUM} ({NAME})",
        "{ADDRESS} ({PHONE})", "{EMAIL} ({COMPANY})", "{POSITION} {NAME}",
        "{CONTRACT_TYPE} - {STATUS}", "{PROJECT_NAME} - {MONEY}", "{TITLE} - {DATE}",
        "{TYPE} - {QUANTITY}", "{LANGUAGE} - {TYPE}", "{CONSENT_TYPE} - {NAME}",
        "{URL} - {DESCRIPTION}", "{LAW_REFERENCE} - {TITLE}", "{RIGHT_INFO} - {MONEY}"
    ]
    
    train_multi_templates = [
        "{COMPANY} {DEPARTMENT}의 {NAME} {POSITION}입니다.",
        "{DATE}에 {COMPANY}와 {NAME}은 {CONTRACT_TYPE}을 체결했다.",
        "본 {CONTRACT_TYPE}은 {DATE}부터 {PERIOD}간 유효하며 금액은 {MONEY}이다.",
        "{NAME}({ID_NUM})은 {ADDRESS}에 거주하며 {PHONE}을 사용한다.",
        "{PROJECT_NAME} 수행을 위해 {COMPANY}는 {MONEY}를 투자하고 {DATE}에 완료한다.",
        "{TITLE}에 명시된 {RIGHT_INFO}는 {LAW_REFERENCE}에 따라 {COMPANY}에 귀속된다.",
        "{NAME} {POSITION}은 {DATE}에 {CONSENT_TYPE}에 서명하고 {EMAIL}로 제출했다.",
        "{COMPANY}는 {ADDRESS}에 위치하며 대표전화는 {PHONE}, 홈페이지는 {URL}이다.",
        "{TYPE} 저작물 {QUANTITY}에 대한 {RIGHT_INFO}를 {MONEY}에 양도한다.",
        "갑: {COMPANY}, 을: {NAME}, 계약일: {DATE}",
        "{NAME} (주민번호: {ID_NUM}, 주소: {ADDRESS})",
        "1. {NAME} 2. {PHONE} 3. {EMAIL}",
        "상기 {NAME}은 {DATE}에 {COMPANY}에 입사하였음을 증명함.",
        "{COMPANY} 귀중. 참조: {DEPARTMENT} {NAME} {POSITION}",
        "계약금 {MONEY}는 {DATE}에 입금하고 잔금 {MONEY}는 {DATE}에 지급한다.",
        "{NAME}은 {DATE}에 {COMPANY}와 {CONTRACT_TYPE}을 체결하고 {MONEY}를 수령했다.",
        "{PROJECT_NAME} ({DATE} ~ {DATE}): {COMPANY} 주관, {MONEY} 예산.",
        "{TITLE} ({TYPE}, {LANGUAGE}, {QUANTITY})에 대한 {RIGHT_INFO} 양도.",
        "{NAME} ({ID_NUM}) - {ADDRESS}, {PHONE}, {EMAIL}",
        "{COMPANY} ({ID_NUM}) - {ADDRESS}, {PHONE}, {URL}",
        "{LAW_REFERENCE}에 따라 {NAME}은 {COMPANY}에게 {RIGHT_INFO}를 허락한다.",
        "{DATE} {COMPANY} {DEPARTMENT} {NAME} {POSITION} {PHONE} {EMAIL}",
        "{CONTRACT_TYPE}: {TITLE}, {DATE}, {MONEY}, {STATUS}",
        "{CONSENT_TYPE}: {NAME}, {DATE}, {PHONE}, {EMAIL}",
        "{PROJECT_NAME}: {COMPANY}, {DATE}, {MONEY}, {DESCRIPTION}",
        "{TYPE} {QUANTITY} ({LANGUAGE}) - {RIGHT_INFO} {MONEY} ({PERIOD})",
        "{NAME} {POSITION} ({COMPANY}) - {PHONE}, {EMAIL}, {ADDRESS}",
        "{DATE} {TITLE} {CONTRACT_TYPE} {STATUS} {MONEY}",
        "{LAW_REFERENCE} {RIGHT_INFO} {TYPE} {QUANTITY} {MONEY}",
        "{COMPANY} {DEPARTMENT} {PROJECT_NAME} {DATE} {STATUS}"
    ]

    # 2. Dev/Test Templates (Unseen patterns to test generalization)
    dev_single_templates = {
        "NAME": [
            "Who is {NAME}?", "Contact person: {NAME}", "{NAME} signed here.",
            "Approved by {NAME}", "To: {NAME}", "From: {NAME}", "User: {NAME}"
        ],
        "PHONE": [
            "Call {PHONE} now.", "Mobile: {PHONE}", "Phone Number: {PHONE}",
            "Reach me at {PHONE}", "Dial {PHONE}", "SMS: {PHONE}"
        ],
        "DATE": [
            "Due by {DATE}", "Date: {DATE}", "Effective from {DATE}",
            "Signed on {DATE}", "Expires: {DATE}", "Since {DATE}"
        ],
        "EMAIL": [
            "Send to {EMAIL}", "Email address: {EMAIL}", "Reply-To: {EMAIL}",
            "CC: {EMAIL}", "Mail: {EMAIL}"
        ],
        "COMPANY": [
            "Vendor: {COMPANY}", "Client: {COMPANY}", "Organization: {COMPANY}",
            "Made by {COMPANY}", "Copyright {COMPANY}"
        ],
        "ADDRESS": [
            "Located at {ADDRESS}", "Ship to: {ADDRESS}", "Office: {ADDRESS}",
            "Residence: {ADDRESS}", "Site: {ADDRESS}"
        ],
        "ID_NUM": [
            "ID: {ID_NUM}", "SSN: {ID_NUM}", "Reg No: {ID_NUM}",
            "License: {ID_NUM}"
        ],
        "MONEY": [
            "Cost: {MONEY}", "Price: {MONEY}", "Total: {MONEY}",
            "Fee: {MONEY}", "Payment: {MONEY}", "Amount: {MONEY}"
        ],
    }
    
    # Fill missing types for dev
    for etype in ENTITY_TYPES:
        if etype not in dev_single_templates:
            dev_single_templates[etype] = [
                f"Check {etype}: {{{etype}}}", 
                f"Value of {etype} is {{{etype}}}",
                f"Please provide {{{etype}}} for {etype}.",
                f"Missing {etype}: {{{etype}}}"
            ]

    dev_dual_templates = [
        "Please contact {NAME} at {PHONE}.",
        "Submit to {EMAIL} by {DATE}.",
        "{COMPANY} is located at {ADDRESS}.",
        "Identity verified: {NAME}, {ID_NUM}.",
        "Payment of {MONEY} due on {DATE}.",
        "{NAME} ({POSITION}) can be reached at {PHONE}.",
        "Agreement {CONTRACT_TYPE} signed on {DATE}.",
        "Budget for {PROJECT_NAME}: {MONEY}.",
        "{NAME} works at {DEPARTMENT}.",
        "Transfer fee for {RIGHT_INFO} is {MONEY}.",
        "Under {LAW_REFERENCE}, we sign {CONTRACT_TYPE}.",
        "{NAME} lives in {ADDRESS}.",
        "{COMPANY} CEO {NAME}",
        "{DATE}: {COMPANY}",
        "{PHONE} ({NAME})",
        "{EMAIL}, {PHONE}",
        "{COMPANY} - {ADDRESS}",
        "{DATE} / {MONEY}",
        "{DATE} - {CONTRACT_TYPE}",
        "{POSITION}: {NAME}"
    ]
    
    dev_multi_templates = [
        "{NAME} ({POSITION}) from {COMPANY} {DEPARTMENT}.",
        "On {DATE}, {COMPANY} and {NAME} signed {CONTRACT_TYPE}.",
        "This {CONTRACT_TYPE} is valid from {DATE} for {PERIOD}, value {MONEY}.",
        "{NAME} (ID: {ID_NUM}) resides at {ADDRESS}, phone: {PHONE}.",
        "{COMPANY} invests {MONEY} in {PROJECT_NAME}, completion {DATE}.",
        "{RIGHT_INFO} in {TITLE} belongs to {COMPANY} per {LAW_REFERENCE}.",
        "{NAME} ({POSITION}) signed {CONSENT_TYPE} on {DATE} -> {EMAIL}.",
        "{COMPANY} @ {ADDRESS}, Tel: {PHONE}, Web: {URL}.",
        "Transfer {RIGHT_INFO} of {QUANTITY} {TYPE} for {MONEY}.",
        "Party A: {COMPANY}, Party B: {NAME}, Date: {DATE}",
        "Details: {NAME}, {ID_NUM}, {ADDRESS}",
        "Info: 1.{NAME} 2.{PHONE} 3.{EMAIL}",
        "Certificate: {NAME} joined {COMPANY} on {DATE}.",
        "Attn: {NAME} {POSITION}, {DEPARTMENT}, {COMPANY}",
        "Deposit {MONEY} on {DATE}, Balance {MONEY} on {DATE}."
    ]

    # Select templates based on dataset_type
    if dataset_type == 'train':
        single = train_single_templates
        dual = train_dual_templates
        multi = train_multi_templates
    else:
        # For dev/test, use different templates to test generalization
        single = dev_single_templates
        dual = dev_dual_templates
        multi = dev_multi_templates
    
    all_templates = build_template_list(single, dual, multi)
    
    # Add "Negative" samples (sentences with NO entities)
    # This helps precision by teaching the model what is NOT an entity.
    negative_templates = [
        "안녕하세요, 반갑습니다.", "오늘 날씨가 참 좋네요.", "식사는 하셨나요?",
        "회의는 2시에 시작합니다.", "문서를 검토해 주세요.", "확인 부탁드립니다.",
        "감사합니다.", "수고하셨습니다.", "다음에 뵙겠습니다.",
        "이 내용은 중요합니다.", "참고하시기 바랍니다.", "문의사항이 있으시면 연락주세요.",
        "첨부파일을 확인하세요.", "작업이 완료되었습니다.", "오류가 발생했습니다.",
        "시스템 점검 중입니다.", "잠시 후 다시 시도해 주세요.", "로그인이 필요합니다.",
        "회원가입을 환영합니다.", "비밀번호를 변경해 주세요."
    ]
    
    samples = []
    seen_texts = set()
    
    # Add negative samples (approx 10% of data)
    num_negatives = int(num_samples * 0.1)
    for _ in range(num_negatives):
        text = random.choice(negative_templates)
        # Add some random noise to negative samples too
        if random.random() < 0.5:
            text += f" ({random.randint(1,100)})"
        # Use the text itself as the template key for negatives
        samples.append({"text": text, "entities": [], "template": "NEGATIVE_SAMPLE"})
        seen_texts.add(text)
        
    remaining_samples = num_samples - len(samples)
    
    if balanced:
        # 균형 데이터 생성 로직
        print(f"[DataGen] Generating balanced data for {len(ENTITY_TYPES)} entity types ({dataset_type})...")
        samples_per_entity = max(1, math.ceil(remaining_samples / len(ENTITY_TYPES)))
        
        for entity_type in ENTITY_TYPES:
            # 해당 엔티티를 포함하는 템플릿 필터링
            relevant = [(t, e) for t, e in all_templates if entity_type in e]
            if not relevant: relevant = all_templates
            
            count = 0
            attempts = 0
            random.shuffle(relevant)
            
            while count < samples_per_entity and attempts < samples_per_entity * 5:
                attempts += 1
                template, _ = random.choice(relevant)
                text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
                
                # Apply Noise
                if noise_level > 0:
                    pass

                if text not in seen_texts:
                    seen_texts.add(text)
                    samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
                    count += 1
                    
        # 부족하거나 넘치는 경우 처리
        if len(samples) > num_samples:
            random.shuffle(samples)
            samples = samples[:num_samples]
        elif len(samples) < num_samples:
            while len(samples) < num_samples:
                template, _ = random.choice(all_templates)
                text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
                if text not in seen_texts:
                    seen_texts.add(text)
                    samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
    else:
        # 단순 랜덤 생성
        while len(samples) < num_samples:
            template, _ = random.choice(all_templates)
            text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
                
    print(f"[DataGen] Generated {len(samples)} unique samples ({dataset_type}).")
    return samples
