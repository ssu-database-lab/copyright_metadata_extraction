# PaddleOCR-VL 설치 가이드 (WSL + .venv)

## 사전 요구사항
- WSL (Windows Subsystem for Linux)
- Python 3.12.9 (프로젝트 테스트 환경)
- 가상환경 (.venv)

## 설치 단계

### 1. 가상환경 활성화
```bash
# 프로젝트 루트 디렉토리에서
source .venv/bin/activate
```

### 2. PaddlePaddle 설치 확인
현재 requirements.txt에 CPU 버전이 포함되어 있습니다.

**CPU 버전 사용 (기본):**
```bash
pip install paddlepaddle>=3.2.0
```

**GPU 버전 사용 (CUDA가 설치된 경우):**
```bash
# requirements.txt의 paddlepaddle 라인을 주석 처리하고
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

### 3. PaddleOCR 및 의존성 설치
```bash
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install paddleocr>=3.0.0 paddlex>=3.0.0
```

### 4. 설치 확인
```bash
python3 -c "from paddleocr import PaddleOCRVL; print('PaddleOCR-VL 설치 완료!')"
```

## 문제 해결

### CUDA 관련 오류 (GPU 사용 시)
WSL에서 CUDA를 사용하려면:
1. Windows에 NVIDIA 드라이버 설치
2. WSL에 CUDA Toolkit 설치
3. `nvidia-smi` 명령어로 확인

### 메모리 부족 오류
PaddleOCR-VL은 VLM 모델이므로 상당한 메모리가 필요합니다:
- 최소 8GB RAM 권장
- GPU 사용 시 VRAM 4GB 이상 권장

### 설치 속도가 느린 경우
중국 미러를 사용하여 설치 속도 향상:
```bash
pip install paddlepaddle>=3.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 사용 예시
코드에서 이미 `PaddleOCRVL`을 사용하고 있습니다:
- `module/parts/ocr.py`에서 `PaddleOCRVL` 사용
- `module/api.py`의 `ocr_extract()` 함수에서 호출

## 참고 자료
- [PaddleOCR-VL 공식 문서](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddlePaddle 설치 가이드](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

