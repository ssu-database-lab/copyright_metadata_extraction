
import sys
import os
import random
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

try:
    from src.ner.ner_train import generate_training_samples
except ImportError as e:
    print(f"Error importing generate_training_samples: {e}")
    sys.exit(1)

def apply_token_noise(token: str) -> str:
    """Apply simple OCR-like noise to a token."""
    if len(token) < 2 or random.random() > 0.3: 
        return token
    
    noise_type = random.choice(['jamo', 'typo', 'space'])
    if noise_type == 'space':
        split_idx = random.randint(1, len(token)-1)
        return token[:split_idx] + " " + token[split_idx:]
    elif noise_type == 'typo':
        idx = random.randint(0, len(token)-1)
        return token[:idx] + "?" + token[idx+1:]
    elif noise_type == 'jamo':
        hangul_indices = [i for i, c in enumerate(token) if 0xAC00 <= ord(c) <= 0xD7A3]
        if not hangul_indices:
            return token
        idx = random.choice(hangul_indices)
        char = token[idx]
        code = ord(char) - 0xAC00
        jong = code % 28
        jung = ((code - jong) // 28) % 21
        cho = ((code - jong) // 28) // 21
        cho_char = chr(0x1100 + cho)
        jung_char = chr(0x1161 + jung)
        jong_char = chr(0x11A7 + jong) if jong > 0 else ''
        return token[:idx] + f"{cho_char}{jung_char}{jong_char}" + token[idx+1:]
    return token

def write_bio_word_level(samples, filepath, apply_noise=False):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = sample.get('text', '')
            entities = sample.get('entities', []) or []
            
            tokens = text.split()
            if not tokens:
                continue
                
            labels = ['O'] * len(tokens)
            char_labels = ['O'] * len(text)
            
            entity_spans = []
            current_search_pos = 0
            for entity_text, entity_type in entities:
                entity_start = text.find(entity_text, current_search_pos)
                if entity_start == -1:
                    entity_start = text.find(entity_text)
                    
                if entity_start != -1:
                    entity_end = entity_start + len(entity_text)
                    entity_spans.append((entity_start, entity_end, entity_type))
                    if entity_start >= current_search_pos:
                        current_search_pos = entity_end
            
            for start, end, etype in entity_spans:
                char_labels[start] = f"B-{etype}"
                for i in range(start + 1, end):
                    char_labels[i] = f"I-{etype}"
            
            char_pos = 0
            for token_idx, token in enumerate(tokens):
                token_start = text.find(token, char_pos)
                if token_start == -1:
                    char_pos += len(token) + 1
                    continue
                token_end = token_start + len(token)
                
                for i in range(token_start, token_end):
                    if i < len(char_labels) and char_labels[i] != 'O':
                        labels[token_idx] = char_labels[i]
                        break
                char_pos = token_end
            
            for token, label in zip(tokens, labels):
                final_token = token
                if apply_noise:
                    final_token = apply_token_noise(token)
                f.write(f"{final_token}\t{label}\n")
            f.write("\n")

if __name__ == "__main__":
    print("Generating samples...")
    samples = generate_training_samples(10, balanced=True, dataset_type='train')
    output_file = "BIO_TAGGING_EXAMPLE.txt"
    write_bio_word_level(samples, output_file, apply_noise=False)
    print(f"Generated {output_file}")
