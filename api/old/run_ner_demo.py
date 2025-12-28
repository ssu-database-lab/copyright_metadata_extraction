import os
import sys
import json
import torch
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

try:
    from src.ner.ner_system import load_model_for_inference, predict_texts
except ImportError:
    # Fallback if running from api/
    sys.path.append(os.path.join(os.getcwd(), "src"))
    from ner.ner_system import load_model_for_inference, predict_texts

def aggregate_entities(tokens_with_labels):
    """
    Aggregate BIO tags into entities.
    tokens_with_labels: [(token, label), ...]
    """
    entities = []
    current_entity = None
    
    for token, label in tokens_with_labels:
        # Skip special tokens if they leaked through (though predict_texts handles them)
        if label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]
            # Clean up token (remove ## for subwords)
            clean_token = token.replace("##", "")
            current_entity = {
                "label": entity_type,
                "text": clean_token,
                "tokens": [token]
            }
        
        elif label.startswith('I-'):
            if current_entity and current_entity['label'] == label[2:]:
                clean_token = token.replace("##", "")
                current_entity['text'] += clean_token
                current_entity['tokens'].append(token)
            else:
                # Start new entity if I- tag appears without B- (shouldn't happen often)
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                clean_token = token.replace("##", "")
                current_entity = {
                    "label": entity_type,
                    "text": clean_token,
                    "tokens": [token]
                }

    if current_entity:
        entities.append(current_entity)
        
    return entities

def main():
    # Configuration
    ocr_file = r"out\asdf\consent_form_3_ocr.txt"
    model_dir = r"..\metadata\models\ner\bert-base-multilingual-cased"
    output_file = r"out\asdf\consent_form_3_ner_result.json"
    
    print(f"[Info] Loading OCR text from: {ocr_file}")
    if not os.path.exists(ocr_file):
        print(f"[Error] File not found: {ocr_file}")
        return

    with open(ocr_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"[Info] Loading NER model from: {model_dir}")
    try:
        model, tokenizer, id2label, config, device = load_model_for_inference(model_dir)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    print("[Info] Running prediction...")
    # Split text into lines or chunks if too long, but predict_texts handles truncation.
    # For better results on long docs, we might want to split by lines.
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    all_entities = []
    
    # Predict line by line to avoid max_length issues and keep context somewhat local
    results = predict_texts(model, tokenizer, id2label, config, lines, device)
    
    for res in results:
        line_text = res['text']
        tokens_labels = res['tokens'] # [(token, label), ...]
        
        entities = aggregate_entities(tokens_labels)
        all_entities.extend(entities)

    # Post-process to merge adjacent entities of same type if needed? 
    # For now, just list them.
    
    # Format output
    output_data = {
        "source_file": ocr_file,
        "full_text": text,
        "extracted_entities": all_entities
    }
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"[Success] NER result saved to: {output_file}")
    
    # Print summary
    print("\n--- Extracted Entities ---")
    for ent in all_entities:
        print(f"[{ent['label']}] {ent['text']}")

if __name__ == "__main__":
    main()
