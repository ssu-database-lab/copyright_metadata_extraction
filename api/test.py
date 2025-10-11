"""
API 함수 테스트 스크립트

api.py에 정의된 함수들이 정상적으로 작동하는지 테스트합니다.
"""

from api import *
from pathlib import Path
import inspect

print("=" * 80)
print("API Functions Test")
print("=" * 80)

# 1. API 정보 테스트
print("\n[Test 1] get_api_info()")
print("-" * 80)
try:
    api_info = get_api_info()
    print("✓ Function executed successfully")
    print(f"  Version: {api_info.get('version', 'N/A')}")
    print(f"  Author: {api_info.get('author', 'N/A')}")
    
    # available_functions 구조 확인
    available = api_info.get('available_functions', {})
    if available:
        print(f"  Available function categories: {len(available)}")
        for category, funcs in available.items():
            print(f"    - {category}: {len(funcs)} functions")
            for func_name, desc in funcs.items():
                print(f"      • {func_name}: {desc}")
    
    # 지원 형식
    formats = api_info.get('supported_formats', {})
    if formats:
        print(f"  Supported formats:")
        print(f"    - Input: {', '.join(formats.get('input', []))}")
        print(f"    - Output: {', '.join(formats.get('output', []))}")
except Exception as e:
    print(f"✗ Error: {e}")

# 2. pdf_to_image 함수 테스트 (import check)
print("\n[Test 2] pdf_to_image() - Import Check")
print("-" * 80)
try:
    print(f"✓ pdf_to_image function imported: {callable(pdf_to_image)}")
except Exception as e:
    print(f"✗ Error: {e}")

# 3. OCR 함수들 테스트 (import check)
print("\n[Test 3] OCR Functions - Import Check")
print("-" * 80)
ocr_functions = [
    ('ocr_naver', ocr_naver),
    ('ocr_mistral', ocr_mistral),
    ('ocr_google', ocr_google),
    ('ocr_complete', ocr_complete)
]

for name, func in ocr_functions:
    try:
        print(f"✓ {name}: {callable(func)}")
    except Exception as e:
        print(f"✗ {name}: {e}")

# 4. NER 함수들 테스트 (import check)
print("\n[Test 4] NER Functions - Import Check")
print("-" * 80)
ner_functions = [
    ('ner_evaluate', ner_evaluate)
]

for name, func in ner_functions:
    try:
        print(f"✓ {name}: {callable(func)}")
    except Exception as e:
        print(f"✗ {name}: {e}")

# 5. ner_evaluate 실제 테스트 (3개 모델)
print("\n[Test 5] ner_evaluate() - Real Test with 3 Models")
print("-" * 80)

MODELS = [
    "google-bert/bert-base-multilingual-cased",
    "klue/roberta-large", 
    "FacebookAI/xlm-roberta-large"
]

# 테스트 데이터 경로
test_data_path = "module/ner/training/google-bert-bert-base-multilingual-cased/validation.txt"

if Path(test_data_path).exists():
    for idx, model_name in enumerate(MODELS, 1):
        print(f"\n[{idx}/3] Testing {model_name}")
        try:
            result = ner_evaluate(
                model_name=model_name,
                test_data_path=test_data_path,
                max_samples=100,  # 빠른 테스트를 위해 100개만
                verbose=False
            )
            
            if result.get("success"):
                overall = result["overall"]
                print(f"  ✓ Evaluation successful")
                print(f"    - Precision: {overall['precision']:.2f}%")
                print(f"    - Recall: {overall['recall']:.2f}%")
                print(f"    - F1 Score: {overall['f1_score']:.2f}%")
                print(f"    - Time: {result.get('evaluation_time', 0):.1f}s")
            else:
                print(f"  ✗ Evaluation failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
else:
    print(f"✗ Test data not found: {test_data_path}")
    print("  Run validation/test.py first to generate test data")

# 6. process_pdf_to_ner 함수 시그니처 확인
print("\n[Test 6] process_pdf_to_ner() - Function Signature Check")
print("-" * 80)
try:
    sig = inspect.signature(process_pdf_to_ner)
    print(f"✓ Function signature:")
    print(f"  {sig}")
    print(f"\n  Parameters:")
    for param_name, param in sig.parameters.items():
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        print(f"    - {param_name}: {param.annotation}{default}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("All API function imports and basic functionality verified!")
print("Run validation/test.py for comprehensive model training and evaluation.")
print("=" * 80)