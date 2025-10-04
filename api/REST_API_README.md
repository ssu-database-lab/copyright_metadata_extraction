# REST API Server - PDF → OCR → NER Pipeline

PDF 문서를 업로드하면 자동으로 OCR 처리 후 NER(개체명 인식) 결과를 반환하는 REST API 서버입니다.

## 🚀 기능

1. **PDF 업로드** - 최대 50MB PDF 파일 수신
2. **자동 OCR** - Google Vision API를 통한 텍스트 추출
3. **NER 분석** - RoBERTa 기반 한국어 개체명 인식
4. **결과 반환** - JSON 형식의 엔티티 정보
5. **요청 로그** - 시간순 로그 자동 기록 (`log/` 디렉토리)

## 📋 요구사항

```bash
pip install Flask werkzeug
```

또는:

```bash
pip install -r requirements.txt
```

## 🏃 서버 실행

```bash
python call.py
```

서버가 실행되면:
- URL: `http://localhost:5000`
- 로그 디렉토리: `log/`
- 임시 파일: `temp/` (자동 정리)

## 📡 API 엔드포인트

### 1. POST /process - PDF 처리

PDF 파일을 업로드하여 NER 결과를 받습니다.

**요청:**
```bash
curl -X POST http://localhost:5000/process \
  -F "file=@document.pdf"
```

**응답 (성공):**
```json
{
  "success": true,
  "request_id": "20250103_153045_123456",
  "filename": "document.pdf",
  "entities": [
    ["홍길동", "NAME"],
    ["서울특별시 강남구", "ADDRESS"],
    ["010-1234-5678", "PHONE"],
    ["한국문화정보원", "COMPANY"]
  ],
  "entity_count": 4,
  "steps": {
    "pdf_to_image": {
      "success": true,
      "images_created": 3
    },
    "ocr": {
      "success": true,
      "files_processed": 3
    },
    "ner": {
      "success": true,
      "files_processed": 1,
      "total_entities": 4
    }
  },
  "processing_time_seconds": 12.5,
  "log_file": "20250103_153045_123456.json"
}
```

**응답 (실패):**
```json
{
  "success": false,
  "error": "OCR processing failed",
  "steps": {...}
}
```

### 2. GET /health - 서버 상태 확인

서버가 정상 작동 중인지 확인합니다.

**요청:**
```bash
curl http://localhost:5000/health
```

**응답:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-03T15:30:45.123456",
  "version": "1.0.0"
}
```

### 3. GET /logs - 요청 로그 조회

최근 처리된 요청 로그를 조회합니다.

**요청:**
```bash
curl "http://localhost:5000/logs?limit=10"
```

**응답:**
```json
{
  "total": 5,
  "logs": [
    {
      "timestamp": "2025-10-03T15:30:45.123456",
      "request_id": "20250103_153045_123456",
      "filename": "document.pdf",
      "success": true,
      "entity_count": 4,
      "processing_time_seconds": 12.5,
      "file_size_mb": 2.3,
      "client_ip": "127.0.0.1"
    }
  ]
}
```

## 🧪 테스트 클라이언트

Python 테스트 클라이언트가 포함되어 있습니다:

```bash
# 서버 상태 확인
python test_client.py

# PDF 파일 처리
python test_client.py document/example.pdf
```

테스트 클라이언트는 자동으로:
- 서버 상태 확인
- PDF 업로드
- 처리 결과 출력
- 결과를 `result/` 디렉토리에 JSON으로 저장
- 최근 로그 조회

## 📊 처리 흐름

```
PDF 업로드
    ↓
[1단계] PDF → 이미지 변환
    ↓
[2단계] Google OCR (텍스트 추출)
    ↓
[3단계] NER 모델 (엔티티 추출)
    ↓
JSON 결과 반환
```

## 📁 디렉토리 구조

```
api/
├── call.py              # REST API 서버 (메인)
├── test_client.py       # 테스트 클라이언트
├── api.py               # 코어 API 함수들
├── module/              # 각 기능 모듈
│   ├── pdf/             # PDF 처리
│   ├── ocr/             # OCR 처리
│   └── ner/             # NER 처리
├── log/                 # 요청 로그 (자동 생성)
│   ├── server_20250103.log
│   ├── 20250103_153045_123456.json
│   └── ...
├── temp/                # 임시 파일 (자동 정리)
└── result/              # 테스트 결과 (클라이언트)
```

## 🔍 로그 형식

각 요청은 `log/` 디렉토리에 JSON 파일로 기록됩니다:

**파일명:** `YYYYMMDD_HHMMSS_microseconds.json`

**내용:**
```json
{
  "timestamp": "2025-10-03T15:30:45.123456",
  "endpoint": "/process",
  "method": "POST",
  "request_id": "20250103_153045_123456",
  "filename": "document.pdf",
  "file_size_bytes": 2415360,
  "file_size_mb": 2.3,
  "client_ip": "127.0.0.1",
  "user_agent": "python-requests/2.31.0",
  "success": true,
  "error": null,
  "entity_count": 4,
  "steps": {...},
  "processing_time_seconds": 12.5,
  "start_time": "2025-10-03T15:30:32.623456",
  "end_time": "2025-10-03T15:30:45.123456"
}
```

## ⚙️ 설정

### 파일 크기 제한

`call.py`의 설정을 변경하여 최대 파일 크기를 조정할 수 있습니다:

```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
```

### 서버 포트

기본 포트는 5000입니다. 변경하려면 `call.py`의 마지막 부분을 수정하세요:

```python
app.run(
    host='0.0.0.0',
    port=5000,  # 원하는 포트로 변경
    debug=False
)
```

## 🐛 에러 처리

### 400 Bad Request
- 파일이 제공되지 않음
- 파일명이 비어있음
- PDF 파일이 아님

### 413 Request Entity Too Large
- 파일 크기가 50MB 초과

### 500 Internal Server Error
- PDF 처리 실패
- OCR 처리 실패
- NER 처리 실패

모든 에러는 로그 파일에 기록되며, 응답에도 포함됩니다.

## 🔐 보안 고려사항

**현재 버전은 개발/테스트용입니다.** 프로덕션 환경에서는:

1. **인증/인가** 추가 (API 키, JWT 등)
2. **HTTPS** 적용
3. **Rate Limiting** 구현
4. **파일 검증** 강화 (바이러스 스캔 등)
5. **WSGI 서버** 사용 (Gunicorn, uWSGI 등)

## 📝 예제 코드

### Python에서 API 호출

```python
import requests

# PDF 파일 업로드
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/process', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"추출된 엔티티: {result['entity_count']}개")
        
        for entity, entity_type in result['entities']:
            print(f"[{entity_type}] {entity}")
```

### JavaScript에서 API 호출

```javascript
const formData = new FormData();
formData.append('file', pdfFile);

fetch('http://localhost:5000/process', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('추출된 엔티티:', data.entity_count);
    console.log('결과:', data.entities);
});
```

### cURL로 API 호출

```bash
# PDF 처리
curl -X POST http://localhost:5000/process \
  -F "file=@document.pdf" \
  -o result.json

# 로그 조회
curl "http://localhost:5000/logs?limit=5" | jq .
```

## 🎯 성능

- **평균 처리 시간**: 10-15초 (페이지 수에 따라 변동)
- **동시 요청**: Multi-threaded Flask 서버
- **메모리**: GPU 사용 시 ~2GB, CPU만 사용 시 ~1GB

## 📞 문의

문제가 발생하면 `log/` 디렉토리의 로그 파일을 확인하세요.

## 📄 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.
