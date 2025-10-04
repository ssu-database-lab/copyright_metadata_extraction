# OCR 설정 가이드

## 설정 파일 설정 방법

1. `ocr_config_example.json` 파일을 `ocr_config.json`으로 복사
2. 각 OCR 서비스별로 API 키와 설정값을 입력

## 각 OCR 서비스 설정 방법

### 1. Naver CLOVA OCR

**API 신청:**
1. [Naver Cloud Platform](https://www.ncloud.com/) 가입
2. AI·Application Service > CLOVA OCR 선택
3. 이용 신청 후 API URL과 Secret Key 발급

**설정:**
```json
"naver": {
  "api_url": "https://naveropenapi.apigw.ntruss.com/vision/v1/ocr",
  "secret_key": "실제_시크릿_키_입력",
  "template_ids": []  // 일반 OCR의 경우 빈 배열
}
```

**특징:**
- 한국어 문서에 특화
- 월 1,000건까지 무료
- 지원 형식: JPG, PNG, PDF, TIFF (최대 20MB)

### 2. Google Cloud Vision

**API 신청:**
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. Cloud Vision API 활성화
4. 서비스 계정 생성 및 JSON 키 파일 다운로드

**설정:**
```json
"google": {
  "credentials_path": "C:/path/to/service-account-key.json",
  "use_document_detection": true  // 문서용: true, 일반 이미지: false
}
```

**특징:**
- 높은 정확도
- 다양한 언어 지원
- 월 1,000건까지 무료

### 3. Mistral AI Vision

**API 신청:**
1. [Mistral AI Console](https://console.mistral.ai/) 가입
2. API 키 발급

**설정:**
```json
"mistral": {
  "api_key": "실제_API_키_입력", 
  "model_name": "pixtral-12b-2409",
  "prompt": "이 이미지에서 모든 텍스트를 정확히 추출해주세요. 줄바꿈과 형식을 유지해주세요.",
  "max_tokens": 2000
}
```

**특징:**
- 최신 AI 기술 활용
- 컨텍스트 이해 능력 우수
- 지원 형식: JPG, PNG

## 사용 예시

```python
from module.ocr_system import ocr_naver, ocr_google, ocr_mistral, ocr_complete

# 단일 OCR 엔진 사용
result = ocr_naver("images/", "output/")
result = ocr_google("document.jpg", "output/")
result = ocr_mistral("photo.png", "output/")

# 모든 OCR 엔진 동시 사용
result = ocr_complete("images/", "output/")
```

## 주의사항

- API 키는 절대 공개하지 마세요
- `ocr_config.json` 파일을 git에 올리지 마세요
- 각 서비스의 사용량 제한을 확인하세요
- Google Cloud의 경우 서비스 계정 JSON 파일의 경로를 정확히 입력하세요