import re

def extract_consent_entities(text):
    """Extract entities from consent document text using regex patterns"""
    entities = []
    
    # Extract entities using simple patterns
    patterns = [
        (r'성\s*명\s*[:：]?\s*([^\s,]+)', '양수인 성명'),
        (r'주\s*소\s*[:：]?\s*([^\n]+)', '양도인 주소'),
        (r'전화\s*번호\s*[:：]?\s*([\d\-]+)', '양도인 전화번호'),
        (r'기관\s*명\s*[:：]?\s*([^\n,]+)', '양수인 기관명'),
        (r'대표자\s*명\s*[:：]?\s*([^\n,]+)', '양수인 대표자명'),
        (r'대표자\s*주소\s*[:：]?\s*([^\n]+)', '양수인 대표자 주소'),
        (r'대표자\s*연락처\s*[:：]?\s*([\d\-]+)', '양수인 대표자 연락처'),
        (r'동의\s*여부\s*[:：]?\s*([^\n,]+)', '동의여부'),
        (r'날짜\s*[:：]?\s*([^\n]+)', '동의날짜')
    ]
    
    for pattern, label in patterns:
        matches = re.findall(pattern, text)
        if matches:
            entities.append({
                'text': matches[0].strip(),
                'label': label
            })
    
    return entities

def extract_contract_entities(text):
    """Extract entities from contract document text using regex patterns"""
    entities = []
    
    # Extract entities using simple patterns
    patterns = [
        (r'저작물\s*명\s*[:：]?\s*([^\n,]+)', '저작물명'),
        (r'저작물\s*상세\s*정보\s*[:：]?\s*([^\n]+)', '대상 저작물 상세정보'),
        (r'양수자\s*기관\s*명\s*[:：]?\s*([^\n,]+)', '양수자 기관명'),
        (r'양수자\s*주소\s*[:：]?\s*([^\n]+)', '양수자 주소'),
        (r'양도자\s*기관\s*명\s*[:：]?\s*([^\n,]+)', '양도자 기관(개인)명'),
        (r'양도자\s*소속\s*[:：]?\s*([^\n,]+)', '양도자 소속'),
        (r'양도자\s*주소\s*[:：]?\s*([^\n]+)', '양도자 대표주소'),
        (r'양도자\s*연락처\s*[:：]?\s*([\d\-]+)', '양도자 연락처'),
        (r'동의\s*여부\s*[:：]?\s*([^\n,]+)', '동의여부'),
        (r'날짜\s*[:：]?\s*([^\n]+)', '날짜')
    ]
    
    for pattern, label in patterns:
        matches = re.findall(pattern, text)
        if matches:
            entities.append({
                'text': matches[0].strip(),
                'label': label
            })
    
    return entities
