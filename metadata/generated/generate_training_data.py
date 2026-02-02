#!/usr/bin/env python3
"""
고품질 NER 학습 데이터 생성 스크립트
각 라벨별로 500-1000개의 unique하고 다양한 패턴의 데이터 생성
"""
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

def generate_person_names(count=500):
    """한국 인명 데이터 생성"""
    surnames = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍', '전', '고', '문', '양', '손', '배', '백', '허', '남', '심', '노', '하', '곽', '성', '차', '주', '우', '구', '방', '지', '변', '여', '탁', '엄', '나', '채', '라', '마', '석']
    
    given_names = ['민수', '지은', '서연', '동욱', '수진', '현우', '소영', '준호', '혜진', '태영', '민석', '지원', '성호', '도현', '재인', '석열', '재명', '철수', '영희', '영수', '민지', '현정', '은영', '은정', '선영', '미경', '상현', '지훈', '진우', '서준', '유진', '다은', '하은', '예은', '승현', '지후', '민준', '시우', '하준', '도윤', '예준', '건우', '현서', '우진', '지안', '시윤', '서윤', '수빈', '지우', '서현', '지혜', '수연', '예지', '채원', '민서', '유나', '하윤', '서아', '수아', '윤서', '하린', '채은', '지아', '서율', '다인', '수현', '지유', '지민']
    
    contexts = ['대표이사', '대표자', '대표', '사장', '담당자', '작성자', '성명', '이름', '책임자', '담당', '신청자', '등록자', '발신자', '수신자', '관리자', '총괄', '부장', '과장', '팀장', '주임', '직원', '근로자', '피보험자', '수급자', '보호자', '법정대리인', '후견인', '원장', '교장', '교수', '연구원', '의사', '변호사', '회계사', '세무사', '공인중개사']
    
    data = []
    used_combinations = set()
    
    for i in range(count):
        while True:
            surname = random.choice(surnames)
            given = random.choice(given_names)
            context = random.choice(contexts)
            combo = f"{surname}{given}{context}"
            
            if combo not in used_combinations:
                used_combinations.add(combo)
                break
        
        has_colon = random.random() < 0.7
        
        if has_colon:
            if len(given) == 2:
                tokens = [context, ':', surname, given[0], given[1]]
                labels = ['O', 'O', 'B-person_name', 'I-person_name', 'I-person_name']
            else:
                tokens = [context, ':', surname, given]
                labels = ['O', 'O', 'B-person_name', 'I-person_name']
        else:
            if len(given) == 2:
                tokens = [context, surname, given[0], given[1]]
                labels = ['O', 'B-person_name', 'I-person_name', 'I-person_name']
            else:
                tokens = [context, surname, given]
                labels = ['O', 'B-person_name', 'I-person_name']
        
        data.append({
            'id': f'pn_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def generate_companies(count=600):
    """한국 기업/기관명 데이터 생성"""
    companies = [
        '삼성전자', 'LG전자', 'SK하이닉스', '현대자동차', '기아자동차', 'POSCO', 'LG화학', 'SK이노베이션',
        '네이버', '카카오', '쿠팡', '삼성물산', '한국전력공사', '한국철도공사', '한국가스공사', '한국수자원공사',
        '대한상공회의소', '중소기업중앙회', '한국무역협회', '한국관광공사', '한국산업은행', '한국수출입은행',
        '경기도청', '서울특별시청', '부산광역시청', '인천광역시청', '대구광역시청', '광주광역시청', '대전광역시청',
        '한국전자통신연구원', '한국과학기술원', '한국표준과학연구원', '한국생명공학연구원', '한국한의학연구원',
        'GS리테일', 'CJ대한통운', '롯데케미칼', '한화솔루션', '두산중공업', 'LS전선', '효성ITX',
        '우리은행', '신한은행', 'KB국민은행', 'NH농협은행', '하나은행', 'IBK기업은행', 'KDB산업은행',
        '삼성생명', '한화생명', '교보생명', 'KB손해보험', '삼성화재', '현대해상', 'DB손해보험',
        '대우건설', '현대건설', 'GS건설', '삼성엔지니어링', 'SK건설', '롯데건설',
        '아모레퍼시픽', 'LG생활건강', '농심', '오리온', 'CJ제일제당', '롯데제과', '해태제과',
        '넷마블', '엔씨소프트', '넥슨', '크래프톤', '펄어비스', '스마일게이트', '위메이드',
        '대한항공', '아시아나항공', '진에어', '티웨이항공', '제주항공', '에어부산',
        '현대백화점', '롯데백화점', '신세계백화점', '이마트', '홈플러스', '코스트코',
        '배달의민족', '요기요', '쿠팡이츠', '당근마켓', '토스', '카카오뱅크', '케이뱅크',
    ]
    
    prefixes = ['주식회사', '(주)', '㈜', '유한회사', '(유)', '']
    contexts = ['기관명', '회사명', '법인명', '기업명', '단체명', '조합명', '회사', '기업', '법인', '기관', '단체', '조합', '기업체', '사업자', '업체', '계약처', '거래처', '발주처', '수급사업자']
    
    data = []
    for i in range(count):
        company = random.choice(companies)
        prefix = random.choice(prefixes)
        context = random.choice(contexts) if random.random() < 0.6 else None
        has_colon = random.random() < 0.8
        
        tokens = []
        labels = []
        
        if context:
            tokens.append(context)
            labels.append('O')
            if has_colon:
                tokens.append(':')
                labels.append('O')
        
        if prefix:
            tokens.append(prefix)
            labels.append('B-company_name')
            tokens.append(company)
            labels.append('I-company_name')
        else:
            tokens.append(company)
            labels.append('B-company_name')
        
        data.append({
            'id': f'cn_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def generate_addresses(count=600):
    """한국 주소 데이터 생성"""
    regions = {
        '서울특별시': ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구'],
        '부산광역시': ['강서구', '금정구', '남구', '동구', '동래구', '부산진구', '북구', '사상구', '사하구', '서구', '수영구', '연제구', '영도구', '중구', '해운대구'],
        '대구광역시': ['남구', '달서구', '동구', '북구', '서구', '수성구', '중구', '달성군'],
        '인천광역시': ['계양구', '남동구', '동구', '미추홀구', '부평구', '서구', '연수구', '중구', '강화군', '옹진군'],
        '광주광역시': ['광산구', '남구', '동구', '북구', '서구'],
        '대전광역시': ['대덕구', '동구', '서구', '유성구', '중구'],
        '울산광역시': ['남구', '동구', '북구', '중구', '울주군'],
        '경기도': ['수원시', '성남시', '고양시', '용인시', '부천시', '안산시', '안양시', '남양주시', '화성시', '평택시', '의정부시', '시흥시', '파주시', '김포시', '광명시', '광주시', '군포시', '하남시', '오산시', '양주시', '이천시'],
        '강원도': ['춘천시', '원주시', '강릉시', '동해시', '태백시', '속초시', '삼척시'],
        '충청북도': ['청주시', '충주시', '제천시'],
        '충청남도': ['천안시', '공주시', '보령시', '아산시', '서산시', '논산시', '계룡시', '당진시'],
        '전라북도': ['전주시', '군산시', '익산시', '정읍시', '남원시', '김제시'],
        '전라남도': ['목포시', '여수시', '순천시', '나주시', '광양시'],
        '경상북도': ['포항시', '경주시', '김천시', '안동시', '구미시', '영주시', '영천시', '상주시', '문경시', '경산시'],
        '경상남도': ['창원시', '진주시', '통영시', '사천시', '김해시', '밀양시', '거제시', '양산시'],
        '제주특별자치도': ['제주시', '서귀포시']
    }
    
    road_types = ['로', '길', '대로', '중앙로', '역삼로', '테헤란로', '강남대로', '송파대로', '올림픽로', '한강대로', '세종대로', '종로', '을지로', '충무로', '퇴계로', '동작대로', '서초대로', '논현로', '가로수길', '압구정로', '봉은사로', '선릉로', '언주로']
    
    contexts = ['주소', '소재지', '본사', '사무실', '위치', '도로명주소', '지번주소', '주소지', '사업장', '본점', '지점', '공장', '창고', '연구소', '등록지', '영업소']
    
    data = []
    for i in range(count):
        region = random.choice(list(regions.keys()))
        district = random.choice(regions[region])
        road = random.choice(road_types)
        number = random.randint(1, 999)
        
        context = random.choice(contexts) if random.random() < 0.6 else None
        has_colon = random.random() < 0.8
        
        tokens = []
        labels = []
        
        if context:
            tokens.append(context)
            labels.append('O')
            if has_colon:
                tokens.append(':')
                labels.append('O')
        
        tokens.extend([region, district, road, str(number)])
        labels.extend(['B-address', 'I-address', 'I-address', 'I-address'])
        
        data.append({
            'id': f'addr_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def generate_phone_numbers(count=600):
    """전화번호 데이터 생성"""
    area_codes = ['02', '031', '032', '033', '041', '042', '043', '044', '051', '052', '053', '054', '055', '061', '062', '063', '064']
    mobile_prefixes = ['010', '011', '016', '017', '018', '019']
    special_numbers = ['1588', '1577', '1544', '1566', '1599', '1600', '0505', '0507', '070']
    
    contexts = ['전화번호', '연락처', 'TEL', '전화', '연락', '휴대전화', '핸드폰', '팩스', 'FAX', '대표번호', '고객센터']
    
    data = []
    for i in range(count):
        num_type = random.choice(['area', 'mobile', 'special'])
        context = random.choice(contexts)
        has_colon = random.random() < 0.8
        has_dash = random.random() < 0.85
        
        tokens = []
        labels = []
        
        tokens.append(context)
        labels.append('O')
        if has_colon:
            tokens.append(':')
            labels.append('O')
        
        if num_type == 'area':
            prefix = random.choice(area_codes)
            middle = f"{random.randint(100, 9999)}"
            last = f"{random.randint(1000, 9999)}"
            
            if has_dash:
                tokens.extend([prefix, '-', middle, '-', last])
                labels.extend(['B-phone_number', 'I-phone_number', 'I-phone_number', 'I-phone_number', 'I-phone_number'])
            else:
                tokens.extend([prefix, middle, last])
                labels.extend(['B-phone_number', 'I-phone_number', 'I-phone_number'])
        
        elif num_type == 'mobile':
            prefix = random.choice(mobile_prefixes)
            middle = f"{random.randint(1000, 9999)}"
            last = f"{random.randint(1000, 9999)}"
            
            if has_dash:
                tokens.extend([prefix, '-', middle, '-', last])
                labels.extend(['B-phone_number', 'I-phone_number', 'I-phone_number', 'I-phone_number', 'I-phone_number'])
            else:
                tokens.extend([prefix, middle, last])
                labels.extend(['B-phone_number', 'I-phone_number', 'I-phone_number'])
        
        else:  # special
            prefix = random.choice(special_numbers)
            last = f"{random.randint(1000, 9999)}"
            
            if has_dash:
                tokens.extend([prefix, '-', last])
                labels.extend(['B-phone_number', 'I-phone_number', 'I-phone_number'])
            else:
                tokens.extend([prefix + last])
                labels.extend(['B-phone_number'])
        
        data.append({
            'id': f'ph_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def generate_emails(count=600):
    """이메일 데이터 생성"""
    prefixes = ['contact', 'info', 'support', 'admin', 'help', 'sales', 'service', 'mail', 'webmaster', 'hello', 'inquiry', 'pr', 'marketing', 'hr', 'recruit', 'cs', 'manager', 'team', 'official']
    
    domains = ['naver.com', 'gmail.com', 'kakao.com', 'daum.net', 'hanmail.net', 'nate.com', 'outlook.com', 'yahoo.com', 'hotmail.com']
    
    company_domains = ['samsung.com', 'lg.com', 'sk.com', 'hyundai.com', 'kakao.com', 'naver.com', 'coupang.com', 'lotte.com', 'gs.com', 'posco.com']
    
    contexts = ['이메일', '메일', 'E-mail', 'Email', '이메일주소', '메일주소', '연락처', '문의처']
    
    data = []
    for i in range(count):
        prefix = random.choice(prefixes)
        domain = random.choice(domains + company_domains)
        context = random.choice(contexts)
        has_colon = random.random() < 0.85
        
        tokens = []
        labels = []
        
        tokens.append(context)
        labels.append('O')
        if has_colon:
            tokens.append(':')
            labels.append('O')
        
        tokens.extend([prefix, '@', domain])
        labels.extend(['B-email', 'I-email', 'I-email'])
        
        data.append({
            'id': f'em_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def generate_urls(count=600):
    """URL 데이터 생성"""
    protocols = ['https', 'http']
    subdomains = ['www', 'blog', 'shop', 'mall', 'store', 'support', 'help', 'api', 'dev', 'test']
    
    domains = ['naver.com', 'google.com', 'kakao.com', 'daum.net', 'samsung.com', 'lg.com', 'sk.com', 'github.com', 'gitlab.com', 'bitbucket.org']
    
    paths = ['', '/about', '/contact', '/products', '/services', '/support', '/news', '/blog', '/info']
    
    contexts = ['웹사이트', '홈페이지', 'URL', '사이트', '주소', '링크', '바로가기']
    
    data = []
    for i in range(count):
        protocol = random.choice(protocols)
        domain = random.choice(domains)
        subdomain = random.choice(subdomains) if random.random() < 0.6 else None
        path = random.choice(paths) if random.random() < 0.3 else ''
        context = random.choice(contexts)
        has_colon = random.random() < 0.85
        
        tokens = []
        labels = []
        
        tokens.append(context)
        labels.append('O')
        if has_colon:
            tokens.append(':')
            labels.append('O')
        
        tokens.extend([protocol, ':', '//'])
        labels.extend(['B-url', 'I-url', 'I-url'])
        
        if subdomain:
            url_part = f"{subdomain}.{domain}{path}"
        else:
            url_part = f"{domain}{path}"
        
        tokens.append(url_part)
        labels.append('I-url')
        
        data.append({
            'id': f'url_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def generate_dates(count=600):
    """날짜 데이터 생성"""
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    contexts = ['등록일', '작성일', '날짜', '등록일자', '작성일시', '생년월일', '발급일', '계약일', '유효기간', '시작일', '종료일', '신청일', '접수일', '처리일', '완료일']
    
    formats = ['년월일', '하이픈', '점', '슬래시', '년월일한글']
    
    data = []
    for i in range(count):
        days_offset = random.randint(0, (end_date - start_date).days)
        random_date = start_date + timedelta(days=days_offset)
        
        context = random.choice(contexts)
        date_format = random.choice(formats)
        has_colon = random.random() < 0.85
        
        tokens = []
        labels = []
        
        tokens.append(context)
        labels.append('O')
        if has_colon:
            tokens.append(':')
            labels.append('O')
        
        year = str(random_date.year)
        month = f"{random_date.month:02d}"
        day = f"{random_date.day:02d}"
        
        if date_format == '년월일':
            tokens.extend([year, '년', month, '월', day, '일'])
            labels.extend(['B-date', 'I-date', 'I-date', 'I-date', 'I-date', 'I-date'])
        elif date_format == '하이픈':
            tokens.extend([year, '-', month, '-', day])
            labels.extend(['B-date', 'I-date', 'I-date', 'I-date', 'I-date'])
        elif date_format == '점':
            tokens.extend([year, '.', month, '.', day])
            labels.extend(['B-date', 'I-date', 'I-date', 'I-date', 'I-date'])
        elif date_format == '슬래시':
            tokens.extend([year, '/', month, '/', day])
            labels.extend(['B-date', 'I-date', 'I-date', 'I-date', 'I-date'])
        else:  # 년월일한글
            tokens.extend([f"{year}년", f"{random_date.month}월", f"{random_date.day}일"])
            labels.extend(['B-date', 'I-date', 'I-date'])
        
        data.append({
            'id': f'dt_{i+1:04d}',
            'tokens': tokens,
            'labels': labels
        })
    
    return data

def main():
    output_dir = Path('configs/training/ner_labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 라벨별 데이터 생성
    datasets = {
        'person_name': generate_person_names(500),
        'company_name': generate_companies(600),
        'address': generate_addresses(600),
        'phone_number': generate_phone_numbers(600),
        'email': generate_emails(600),
        'url': generate_urls(600),
        'date': generate_dates(600),
    }
    
    # 파일 저장
    for label, data in datasets.items():
        output_file = output_dir / f'{label}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Generated {len(data)} samples for {label}')
    
    print(f'\n총 {sum(len(d) for d in datasets.values())}개의 학습 데이터 생성 완료!')
    print(f'저장 위치: {output_dir.absolute()}')

if __name__ == '__main__':
    main()
