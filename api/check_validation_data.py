"""검증 데이터 구성 확인 스크립트"""
from pathlib import Path

# 훈련 데이터 경로
training_dir = Path("module/ner/training/klue-roberta-large")

# 파일 정보
files = {
    "train.txt": training_dir / "train.txt",
    "validation.txt": training_dir / "validation.txt",
    "test.txt": training_dir / "test.txt"
}

print("=" * 80)
print("NER 훈련/검증 데이터 구성")
print("=" * 80)

for name, path in files.items():
    if path.exists():
        # 파일 크기
        size_kb = path.stat().st_size / 1024
        
        # 문장 수 카운트 (빈 줄로 구분)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            sentences = [s.strip() for s in content.split('\n\n') if s.strip()]
            num_sentences = len(sentences)
        
        # 토큰 수 카운트
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and '\t' in line]
            num_tokens = len(lines)
        
        # 엔티티 수 카운트 (B- 태그)
        num_entities = 0
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1].startswith('B-'):
                num_entities += 1
        
        print(f"\n📄 {name}")
        print(f"  - 경로: {path}")
        print(f"  - 크기: {size_kb:.1f} KB")
        print(f"  - 문장 수: {num_sentences:,}개")
        print(f"  - 토큰 수: {num_tokens:,}개")
        print(f"  - 엔티티 수: {num_entities:,}개")
        
        # 샘플 출력
        print(f"\n  📋 샘플 (첫 3개 문장):")
        for i, sentence in enumerate(sentences[:3], 1):
            lines = sentence.split('\n')[:5]  # 첫 5개 토큰만
            print(f"    문장 {i}:")
            for line in lines:
                if '\t' in line:
                    token, label = line.split('\t')
                    print(f"      {token:10} → {label}")
            if len(sentence.split('\n')) > 5:
                print(f"      ... (총 {len(sentence.split('\n'))}개 토큰)")
    else:
        print(f"\n❌ {name}: 파일 없음")

print("\n" + "=" * 80)
print("📊 데이터 분할 비율")
print("=" * 80)

train_path = files["train.txt"]
val_path = files["validation.txt"]

if train_path.exists() and val_path.exists():
    with open(train_path, 'r', encoding='utf-8') as f:
        train_sentences = len([s for s in f.read().split('\n\n') if s.strip()])
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_sentences = len([s for s in f.read().split('\n\n') if s.strip()])
    
    total = train_sentences + val_sentences
    train_ratio = (train_sentences / total) * 100
    val_ratio = (val_sentences / total) * 100
    
    print(f"\n전체 샘플: {total:,}개")
    print(f"  - 훈련 데이터: {train_sentences:,}개 ({train_ratio:.1f}%)")
    print(f"  - 검증 데이터: {val_sentences:,}개 ({val_ratio:.1f}%)")
    
    if 75 <= train_ratio <= 85:
        print(f"\n✅ 80/20 분할 기준 충족!")
    else:
        print(f"\n⚠️  80/20 분할 기준 미달 (목표: 80/20)")

print("\n" + "=" * 80)
