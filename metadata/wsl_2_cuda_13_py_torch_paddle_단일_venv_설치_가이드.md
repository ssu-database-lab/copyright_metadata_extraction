# WSL2 + CUDA 13 + PyTorch + PaddleOCR 단일 venv 재설치 가이드

이 문서는 **WSL2 환경에서 RTX 5070 + CUDA 13.0**을 기준으로,
**PaddleOCR(PaddlePaddle-GPU) → PyTorch(Transformers)** 를 **하나의 Python 가상환경(.venv)** 에서
안정적으로 다시 구성하기 위한 **재설치 순서 요약본**입니다.

> ⚠️ 중요 전제
> - Ubuntu 24.04 (WSL2)
> - Python 3.12.x
> - CUDA 13.0
> - torch와 paddle을 **같은 프로세스에서 연속 사용**
> - `pip check` 경고보다 **실제 런타임 동작을 우선**

---

## 0. WSL & 드라이버 확인 (Windows 쪽)

```powershell
wsl --install -d Ubuntu
wsl --shutdown
```

WSL 재시작 후:
```bash
nvidia-smi
```
GPU 정보가 나오면 OK.

---

## 1. 시스템 패키지 & Python 준비

```bash
sudo apt update
sudo apt install -y \
  python3.12 python3.12-venv python3.12-dev \
  python3-pip zlib1g-dev
```

---

## 2. 가상환경(.venv) 생성 (⚠️ 반드시 Linux FS)

```bash
cd ~
python3 -m venv .venv
```

프로젝트 디렉토리에서 심볼릭 링크:
```bash
cd /mnt/c/Users/<USER>/Desktop/<PROJECT>
ln -s ~/.venv .venv
source .venv/bin/activate
```

확인:
```bash
which python
```
→ `/home/<user>/.venv/bin/python`

---

## 3. pip 업그레이드

```bash
python -m pip install -U pip
```

---

## 4. PyTorch (CUDA 13.0) 설치 — **가장 먼저**

```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
```

설치 확인:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## 5. PyTorch 기준 NVIDIA 라이브러리 고정

```bash
pip install \
  nvidia-cublas==13.1.0.3 \
  nvidia-cuda-runtime==13.0.96 \
  nvidia-cudnn-cu13==9.15.1.9 \
  nvidia-cusparselt-cu13==0.8.0 \
  nvidia-nccl-cu13==2.28.9 \
  nvidia-cuda-cccl==13.0.85
```

---

## 6. PaddlePaddle-GPU 설치 (⚠️ 의존성 덮어쓰기 방지)

```bash
pip install paddlepaddle-gpu==3.2.2 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

확인:
```bash
python -c "import paddle; print(paddle.is_compiled_with_cuda(), paddle.get_device())"
```

---

## 7. CUDA 라이브러리 경로 설정 (필수)

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)/nvidia/cu13/lib:$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)/paddle/libs
```

(권장) `.venv/bin/activate`에 추가해서 자동 적용.

---

## 8. Python 패키지 설치 (requirements)

### requirements.txt 예시
```text
numpy==1.26.4
PyYAML==6.0.2

transformers[torch]==4.51.3
gliner2>=0.2.0
huggingface_hub[hf_xet]>=0.16.4
blingfire>=0.1.1
kiwipiepy>=0.10.0

paddleocr==3.3.0
paddlex[ocr]==3.3.0
```

설치:
```bash
pip install -r requirements.txt
```

---

## 9. 최종 통합 테스트 (가장 중요)

```bash
python - <<'PY'
import torch, paddle
print('torch:', torch.cuda.is_available(), torch.version.cuda)
print('paddle:', paddle.is_compiled_with_cuda(), paddle.get_device())

x = torch.randn(1024, 1024, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('torch ok:', y.mean().item())

paddle.set_device('gpu:0')
a = paddle.randn([1024, 1024])
b = paddle.matmul(a, a)
print('paddle ok:', float(paddle.mean(b)))

print('BOTH OK')
PY
```

`BOTH OK` 출력되면 설치 완료.

---

## 10. 상태 고정 (강력 권장)

```bash
pip freeze > requirements.lock.txt
```

---

## ⚠️ 주의사항 (매우 중요)

❌ 아래 명령은 **환경을 깨뜨릴 수 있음**
- `pip install -U torch`
- `pip install -U paddlepaddle-gpu`
- `pip install -U nvidia-*`

패키지 변경은 항상 **새 venv에서 테스트 후** 진행.

---

## 요약

1. torch cu130 먼저 설치
2. torch 기준 nvidia-* 고정
3. paddle은 `--no-deps`
4. LD_LIBRARY_PATH 설정
5. **실제 GPU 연산 테스트로만 검증**

이 순서만 지키면 재설치해도 동일한 안정 상태를 재현할 수 있습니다.

