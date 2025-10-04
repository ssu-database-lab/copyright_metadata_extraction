# REST API 서버 사용 가이드

## 🚀 빠른 시작

### 1. 서버 실행

**방법 1: PowerShell에서 직접 실행**
```powershell
cd C:\Users\peppermint\Desktop\Project\api
python call.py
```

서버가 시작되면 다음과 같이 표시됩니다:
```
============================================================
REST API Server - PDF → OCR → NER Pipeline  
============================================================
Log directory: C:\Users\peppermint\Desktop\Project\api\log
Temp directory: C:\Users\peppermint\Desktop\Project\api\temp

Available endpoints:
  POST /process - Process PDF file
  GET  /health  - Server health check
  GET  /logs    - View request logs

Starting server on http://0.0.0.0:5000
============================================================

 * Running on http://127.0.0.1:5000
```

서버를 종료하려면 `Ctrl+C`를 누르세요.

---

### 2. 다른 터미널에서 테스트

서버를 실행한 상태에서 **새 PowerShell 창**을 열고:

**서버 상태 확인:**
```powershell
cd C:\Users\peppermint\Desktop\Project\api
python test_client.py
```

**PDF 파일 처리:**
```powershell
cd C:\Users\peppermint\Desktop\Project\api
python test_client.py "document\7.저작물양도계약서.pdf"
```

---

### 3. Python 코드로 API 호출

```python
import requests

# 서버 상태 확인
response = requests.get('http://localhost:5000/health')
print(response.json())

# PDF 파일 업로드
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/process', files=files)
    
    result = response.json()
    print(f"성공: {result['success']}")
    print(f"추출된 엔티티: {result['entity_count']}개")
```

---

### 4. cURL로 API 호출

**Windows PowerShell:**
```powershell
# 서버 상태 확인
curl http://localhost:5000/health

# PDF 처리
curl -X POST http://localhost:5000/process -F "file=@document.pdf"

# 로그 조회
curl "http://localhost:5000/logs?limit=5"
```

---

## 📊 API 응답 예시

### POST /process

**성공 응답:**
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

---

## 📁 로그 확인

모든 요청은 `log/` 디렉토리에 JSON 파일로 저장됩니다:

```
api/
└── log/
    ├── server_20250103.log          # 서버 로그
    ├── 20250103_153045_123456.json  # 요청 1
    ├── 20250103_154523_789012.json  # 요청 2
    └── ...
```

각 로그 파일에는:
- 요청 시간
- 파일명 및 크기
- 처리 결과
- 추출된 엔티티 개수
- 처리 시간
- 클라이언트 IP

등이 기록됩니다.

---

## 🔧 문제 해결

### 포트 5000이 이미 사용 중인 경우

`call.py` 마지막 부분을 수정:
```python
app.run(
    host='0.0.0.0',
    port=8080,  # 다른 포트로 변경
    debug=False
)
```

### 서버가 응답하지 않는 경우

1. 서버가 실행 중인지 확인
2. 방화벽 설정 확인
3. `log/server_YYYYMMDD.log` 파일 확인

---

## 📝 실전 사용 예시

### 시나리오: 계약서 PDF 파일 처리

1. **서버 시작** (터미널 1)
   ```powershell
   python call.py
   ```

2. **PDF 업로드** (터미널 2)
   ```powershell
   python test_client.py "contracts/계약서.pdf"
   ```

3. **결과 확인**
   - 터미널 출력에서 실시간 확인
   - `result/` 디렉토리의 JSON 파일
   - `log/` 디렉토리의 로그 파일

4. **로그 조회**
   ```powershell
   curl "http://localhost:5000/logs?limit=10"
   ```

---

## 🎯 다음 단계

- [ ] 프로덕션 환경에서는 Gunicorn 또는 uWSGI 사용
- [ ] HTTPS 적용
- [ ] API 키 인증 추가
- [ ] Rate limiting 구현
- [ ] 대용량 파일 처리 최적화

---

## 📞 참고 문서

- **API 상세 문서**: `REST_API_README.md`
- **코어 API**: `README.md`
- **사용법**: `USAGE.txt`
