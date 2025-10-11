from api import *
from pathlib import Path
import time

print("=" * 80)
print("간단한 평가 테스트")
print("=" * 80)

validation_path = Path("module/ner/training/google-bert-bert-base-multilingual-cased/validation.txt")
print(f"\nvalidation.txt 존재: {validation_path.exists()}")

if validation_path.exists():
    size = validation_path.stat().st_size
    print(f"파일 크기: {size:,} bytes")
    
    print("\n평가 시작 (500 샘플)...")
    start = time.time()
    
    result = ner_evaluate(
        model_name="google-bert/bert-base-multilingual-cased",
        test_data_path=str(validation_path),
        max_samples=500,
        verbose=True,
        debug=False
    )
    
    elapsed = time.time() - start
    print(f"\n평가 완료 ({elapsed:.1f}초)")
    
    if result.get("success"):
        overall = result["overall"]
        print(f"Precision: {overall['precision']:.2f}%")
        print(f"Recall: {overall['recall']:.2f}%")
        print(f"F1 Score: {overall['f1_score']:.2f}%")
    else:
        print(f"오류: {result.get('error', 'Unknown')}")
else:
    print("validation.txt 없음!")

print("\n" + "=" * 80)
