from pathlib import Path
import shutil, subprocess, sys, time, os
from datetime import datetime

# validation 폴더에서 실행되므로 부모 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import *

LOG_FILE = Path("../data/out/debug/model_evaluation_log.txt")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("NER Model Evaluation\n")
    f.write("=" * 80 + "\n")
    f.write(f"Start: {datetime.now()}\n")
    f.write("=" * 80 + "\n\n")

print("=" * 80); print("NER Model Evaluation"); print("=" * 80)

MODELS = [
    ("google-bert/bert-base-multilingual-cased", "google-bert-bert-base-multilingual-cased"),
    ("klue/roberta-large", "klue-roberta-large"),
    ("FacebookAI/xlm-roberta-large", "FacebookAI-xlm-roberta-large")
]
# 부모 디렉토리(api/) 기준 경로
DOWNLOADED_DIR = Path("../model_downloaded")
MODELS_DIR = Path("../models/ner")
TRAINING_DIR = Path("../module/ner/training")

def is_model_complete(path):
    if not path.exists(): return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    has_tokenizer = (path / "tokenizer_config.json").exists()
    return has_config and has_weights and has_tokenizer

# Step 1: 모델 다운로드
print("\nStep 1: Check and download models"); print("-" * 80)

for idx, (model_name, model_safe) in enumerate(MODELS, 1):
    print(f"\n[{idx}/{len(MODELS)}] {model_name}")
    downloaded_path = DOWNLOADED_DIR / model_safe
    if is_model_complete(downloaded_path):
        print("  Already downloaded")
    else:
        print("  Downloading...")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        downloaded_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(downloaded_path))
        model.save_pretrained(str(downloaded_path))
        print("  Download complete")

# Step 2: 모델 복사
print("\nStep 2: Copy to models/ner"); print("-" * 80)

for idx, (model_name, model_safe) in enumerate(MODELS, 1):
    print(f"\n[{idx}/{len(MODELS)}] {model_name}")
    downloaded_path = DOWNLOADED_DIR / model_safe
    model_path = MODELS_DIR / model_safe
    if not is_model_complete(downloaded_path):
        print("  Error: Downloaded model incomplete"); continue
    print("  Copying...")
    if model_path.exists(): shutil.rmtree(model_path)
    shutil.copytree(downloaded_path, model_path)
    print("  Copy complete")

# Step 3: 평가 데이터 생성 (전처리)
print("\nStep 3: Generate evaluation data (Preprocessing)"); print("-" * 80)

model_name, model_safe = MODELS[0]
print(f"\nGenerating evaluation data using {model_name}...")

env = os.environ.copy()
env["NER_MODEL_NAME"] = model_name
env["GENERATE_DATA_ONLY"] = "1"

# 부모 디렉토리(api/)에서 실행
api_dir = Path(__file__).parent.parent
process = subprocess.run(
    [sys.executable, "module/ner/ner_train.py"], 
    env=env, 
    capture_output=False,
    cwd=str(api_dir)
)

if process.returncode == 0:
    source_dir = TRAINING_DIR / model_safe
    validation_file = source_dir / "validation.txt"
    
    print(f"\n[DEBUG] Checking validation file:")
    print(f"  TRAINING_DIR: {TRAINING_DIR}")
    print(f"  source_dir: {source_dir}")
    print(f"  validation_file: {validation_file}")
    print(f"  Absolute path: {validation_file.resolve()}")
    print(f"  Exists: {validation_file.exists()}")
    
    if validation_file.exists():
        print(f" Evaluation data generated: {validation_file}")
        
        for _, target_safe in MODELS:
            target_dir = TRAINING_DIR / target_safe
            
            # 자기 자신에게 복사하는 것 방지
            if source_dir.resolve() == target_dir.resolve():
                print(f"  Skipped {target_safe} (source == target)")
                continue
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for file in ["train.txt", "validation.txt", "test.txt"]:
                src = source_dir / file
                dst = target_dir / file
                if src.exists():
                    # 파일 복사 재시도 (PermissionError 방지)
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            time.sleep(0.5)  # 잠시 대기
                            shutil.copy2(src, dst)
                            print(f"  Copied {file} to {target_safe}")
                            break
                        except PermissionError:
                            if attempt < max_retries - 1:
                                print(f"  Retry copying {file} to {target_safe}...")
                                time.sleep(1)
                            else:
                                print(f"  Warning: Could not copy {file} to {target_safe}")
        
        print("\n✓ Evaluation data copied to all model folders")
    else:
        print(" Error: validation.txt not generated")
        sys.exit(1)
else:
    print(f" Data generation failed (code: {process.returncode})")
    sys.exit(1)

# Step 4: 훈련 전 평가 (1~3번)
print("\nStep 4: Evaluate before training (1~3)"); print("-" * 80)

for idx, (model_name, model_safe) in enumerate(MODELS, 1):
    print(f"\n[{idx}/{len(MODELS)}] {model_name}")
    validation_path = TRAINING_DIR / model_safe / "validation.txt"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{idx}. {model_name} (Before Training)\n")
        f.write("=" * 80 + "\n")
        f.flush()
    
    if not validation_path.exists():
        print(f"   Error: validation.txt not found")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("Error: validation.txt not found\n")
            f.flush()
        continue
    
    print("  Evaluating...")
    start_time = time.time()
    eval_result = ner_evaluate(model_name=model_name, test_data_path=str(validation_path), max_samples=500, verbose=False, debug=False)
    elapsed = time.time() - start_time
    
    if eval_result.get("success"):
        overall = eval_result["overall"]
        print(f"   Evaluation complete ({elapsed:.1f}s)")
        print(f"    Precision: {overall['precision']:.2f}%")
        print(f"    Recall: {overall['recall']:.2f}%")
        print(f"    F1 Score: {overall['f1_score']:.2f}%")
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"Time: {elapsed:.1f}s\n")
            f.write(f"Samples: 500\n")
            f.write(f"Precision: {overall['precision']:.2f}%\n")
            f.write(f"Recall: {overall['recall']:.2f}%\n")
            f.write(f"F1 Score: {overall['f1_score']:.2f}%\n")
            f.flush()
    else:
        print(f"   Evaluation error: {eval_result.get('error', 'Unknown')}")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"Error: {eval_result.get('error', 'Unknown')}\n")
            f.flush()

# Step 5: 훈련 및 즉시 평가
print("\nStep 5: Fine-tuning and Evaluation"); print("-" * 80)

for idx, (model_name, model_safe) in enumerate(MODELS, 1):
    print(f"\n[{idx}/{len(MODELS)}] {model_name}")
    
    downloaded_path = DOWNLOADED_DIR / model_safe
    model_path = MODELS_DIR / model_safe
    
    if is_model_complete(downloaded_path):
        print("  Restoring original model...")
        if model_path.exists(): shutil.rmtree(model_path)
        shutil.copytree(downloaded_path, model_path)
    
    print("  Training...")
    train_start = time.time()
    env = os.environ.copy()
    env["NER_MODEL_NAME"] = model_name
    
    # 부모 디렉토리(api/)에서 실행
    api_dir = Path(__file__).parent.parent
    process = subprocess.run(
        [sys.executable, "module/ner/ner_train.py"], 
        env=env, 
        capture_output=False,
        cwd=str(api_dir)
    )
    
    train_elapsed = time.time() - train_start
    train_minutes = train_elapsed / 60
    
    if process.returncode == 0:
        print(f"  ✓ Training complete ({train_minutes:.1f}min)")
        
        # 훈련 완료 즉시 평가 (4~6번)
        print(f"  Evaluating after training...")
        validation_path = TRAINING_DIR / model_safe / "validation.txt"
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{idx+len(MODELS)}. {model_name} (After Training)\n")
            f.write("=" * 80 + "\n")
            f.flush()
        
        if validation_path.exists():
            eval_start = time.time()
            eval_result = ner_evaluate(
                model_name=model_name,
                test_data_path=str(validation_path),
                max_samples=2000,
                verbose=False,
                debug=False
            )
            eval_elapsed = time.time() - eval_start
            
            if eval_result.get("success"):
                overall = eval_result["overall"]
                print(f"  ✓ Evaluation complete ({eval_elapsed:.1f}s)")
                print(f"    Precision: {overall['precision']:.2f}%")
                print(f"    Recall: {overall['recall']:.2f}%")
                print(f"    F1 Score: {overall['f1_score']:.2f}%")
                
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"Training time: {train_minutes:.1f}min\n")
                    f.write(f"Evaluation time: {eval_elapsed:.1f}s\n")
                    f.write(f"Samples: 2,000\n")
                    f.write(f"Precision: {overall['precision']:.2f}%\n")
                    f.write(f"Recall: {overall['recall']:.2f}%\n")
                    f.write(f"F1 Score: {overall['f1_score']:.2f}%\n")
                    f.flush()
            else:
                print(f"  ✗ Evaluation error: {eval_result.get('error', 'Unknown')}")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"Training time: {train_minutes:.1f}min\n")
                    f.write(f"Evaluation error: {eval_result.get('error', 'Unknown')}\n")
                    f.flush()
        else:
            print(f"  ✗ Error: validation.txt not found")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"Training time: {train_minutes:.1f}min\n")
                f.write("Error: validation.txt not found\n")
                f.flush()
    else:
        print(f"  ✗ Training failed (code: {process.returncode})")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{idx+len(MODELS)}. {model_name} (After Training)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training failed (code: {process.returncode})\n")
            f.flush()

print("\n" + "=" * 80)
print("All evaluations complete!")
print("=" * 80)
print(f"Log file: {LOG_FILE}")
print("=" * 80)

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"\nComplete: {datetime.now()}\n")
    f.write("=" * 80 + "\n")
    f.flush()
