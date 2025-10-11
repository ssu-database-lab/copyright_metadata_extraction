"""간단한 경로 테스트"""
from pathlib import Path
import subprocess, sys, os

# 현재 위치
print(f"Current dir: {Path.cwd()}")

# 경로 설정
TRAINING_DIR = Path("../module/ner/training")
model_safe = "google-bert-bert-base-multilingual-cased"

# subprocess 테스트
env = os.environ.copy()
env["NER_MODEL_NAME"] = "google-bert/bert-base-multilingual-cased"
env["GENERATE_DATA_ONLY"] = "1"

api_dir = Path(__file__).parent.parent
print(f"API dir: {api_dir}")
print(f"API dir absolute: {api_dir.resolve()}")

print(f"\nRunning subprocess in: {api_dir}")
process = subprocess.run(
    [sys.executable, "module/ner/ner_train.py"], 
    env=env, 
    capture_output=True,
    cwd=str(api_dir),
    text=True
)

print(f"Return code: {process.returncode}")
print(f"Last 5 lines of output:")
lines = process.stdout.strip().split('\n')
for line in lines[-5:]:
    print(f"  {line}")

# 파일 확인
source_dir = TRAINING_DIR / model_safe
validation_file = source_dir / "validation.txt"

print(f"\nChecking files:")
print(f"  source_dir: {source_dir}")
print(f"  Absolute: {source_dir.resolve()}")
print(f"  validation_file: {validation_file}")
print(f"  Absolute: {validation_file.resolve()}")
print(f"  Exists: {validation_file.exists()}")

if validation_file.exists():
    print("✓ File found!")
else:
    print("✗ File NOT found!")
    # 실제로 어디에 있는지 찾기
    import glob
    matches = glob.glob(str(api_dir / "module/ner/training/*/validation.txt"))
    print(f"\nFound validation.txt files:")
    for m in matches:
        print(f"  {m}")
