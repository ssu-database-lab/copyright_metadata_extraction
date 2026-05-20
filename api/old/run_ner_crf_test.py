import sys
import os
import json
from pathlib import Path

# Add api to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.ner.ner_system import load_model_for_inference, predict_texts
except ImportError as e:
    print(f"[Error] Could not import ner_system: {e}")
    sys.exit(1)

# Configuration
MODELS = [
    # (Tag, Path)
    # Assuming models are in models/ner_benchmark as per ner_test.py
    ("crf_bert", Path("models/ner_benchmark/crf_bert")),
    ("crf_roberta", Path("models/ner_benchmark/crf_roberta")),
    ("crf_xlm", Path("models/ner_benchmark/crf_xlm")),
]

INPUT_DIR = Path("data/in")
OUTPUT_ROOT = Path("ner/251211")

def save_labels(id2label, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# NER Labels\n\n")
        f.write("| ID | Label |\n")
        f.write("|----|-------|\n")
        for idx in sorted(id2label.keys()):
            f.write(f"| {idx} | {id2label[idx]} |\n")
    print(f"[Info] Saved labels to {output_path}")

def run_inference(model_tag, model_path, input_dir, output_dir):
    print(f"\n>>> Processing Model: {model_tag} <<<")
    
    if not model_path.exists():
        print(f"[Error] Model path not found: {model_path}")
        # Try to find in subfolder (e.g. fold_1)
        candidates = list(model_path.glob("fold_*"))
        if candidates:
            model_path = candidates[0]
            print(f"[Info] Found fold model: {model_path}")
        else:
            print(f"[Skip] Skipping {model_tag}")
            return None

    try:
        print(f"[Info] Loading model from {model_path}...")
        model, tokenizer, id2label, config, device = load_model_for_inference(str(model_path))
    except Exception as e:
        print(f"[Error] Failed to load model {model_tag}: {e}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labels if not already saved (using the first successful model)
    label_file = OUTPUT_ROOT / "labels.md"
    if not label_file.exists():
        save_labels(id2label, label_file)

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"[Warning] No .txt files found in {input_dir}")
        return

    print(f"[Info] Found {len(txt_files)} files. Running inference...")
    
    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8")
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not lines:
                continue
                
            results = predict_texts(model, tokenizer, id2label, config, lines, device)
            
            # Aggregate entities
            all_entities = []
            for res in results:
                tokens_labels = res['tokens']
                
                current_entity = None
                for token, label in tokens_labels:
                    if label == 'O':
                        if current_entity:
                            all_entities.append(current_entity)
                            current_entity = None
                        continue
                        
                    if label.startswith('B-'):
                        if current_entity:
                            all_entities.append(current_entity)
                        
                        entity_type = label[2:]
                        clean_token = token.replace("##", "")
                        current_entity = {
                            "label": entity_type,
                            "text": clean_token
                        }
                    
                    elif label.startswith('I-'):
                        if current_entity and current_entity['label'] == label[2:]:
                            clean_token = token.replace("##", "")
                            current_entity['text'] += clean_token
                        else:
                            if current_entity:
                                all_entities.append(current_entity)
                            
                            entity_type = label[2:]
                            clean_token = token.replace("##", "")
                            current_entity = {
                                "label": entity_type,
                                "text": clean_token
                            }
                if current_entity:
                    all_entities.append(current_entity)

            # Save result
            out_file = output_dir / f"{txt_file.stem}_metadata.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({
                    "source": str(txt_file),
                    "model": model_tag,
                    "entities": all_entities
                }, f, ensure_ascii=False, indent=2)
            print(f"[Info] Saved {out_file.name}")
            
        except Exception as e:
            print(f"[Error] Failed to process {txt_file.name}: {e}")

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    for tag, path in MODELS:
        out_dir = OUTPUT_ROOT / tag
        run_inference(tag, path, INPUT_DIR, out_dir)

if __name__ == "__main__":
    main()
