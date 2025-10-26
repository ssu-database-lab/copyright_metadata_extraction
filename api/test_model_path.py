"""모델 경로 확인 테스트"""
from module.ner.ner_system import get_model_path

models = [
    "klue/roberta-large",
    "FacebookAI/xlm-roberta-large",
    "google-bert/bert-base-multilingual-cased"
]

print("=" * 80)
print("모델 경로 확인 테스트")
print("=" * 80)

for model_name in models:
    print(f"\n📦 {model_name}")
    path = get_model_path(model_name)
    print(f"  경로: {path}")
    print(f"  존재: {path.exists()}")
    
    if path.exists():
        config_exists = (path / "config.json").exists()
        training_info_exists = (path / "training_info.json").exists()
        print(f"  config.json: {config_exists}")
        print(f"  training_info.json: {training_info_exists}")
        
        # 파일 목록
        files = [f.name for f in path.iterdir() if f.is_file()]
        print(f"  파일 수: {len(files)}")
        if files:
            print(f"  주요 파일: {', '.join(files[:5])}")

print("\n" + "=" * 80)
