#!/usr/bin/env python3
"""BIO 태깅 검증"""
from typing import List, Dict, Tuple

def simple_word_tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.strip()
    tokens = []
    current_token = ""
    for char in text:
        if char == ' ':
            if current_token:
                tokens.append(current_token)
                current_token = ""
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def test_bio():
    """Test 홍길동 tagging"""
    # Input: "담당자: 홍 길 동"
    template = "담당자: {NAME}"
    entity_map = {"NAME": "홍길동"}
    
    text = template
    for ent_type, ent_text in entity_map.items():
        key = "{" + ent_type + "}"
        char_separated = " ".join(ent_text)
        text = text.replace(key, char_separated)
    
    words = simple_word_tokenize(text)
    labels = ["O"] * len(words)
    
    print(f"Template: {template}")
    print(f"Entity map: {entity_map}")
    print(f"Text after rendering: {text}")
    print(f"Words: {words}")
    print(f"Initial labels: {labels}")
    
    # Match "홍길동" (3 consecutive chars: '홍', '길', '동')
    ent_text = "홍길동"
    ent_chars = list(ent_text)
    
    for start_pos in range(len(words)):
        if start_pos + len(ent_chars) > len(words):
            break
        
        can_match = True
        for i, char in enumerate(ent_chars):
            word = words[start_pos + i]
            if not word.startswith(char):
                can_match = False
                break
        
        if can_match:
            print(f"\nMatched at position {start_pos}: {words[start_pos:start_pos+len(ent_chars)]}")
            labels[start_pos] = "B-NAME"
            for i in range(1, len(ent_chars)):
                labels[start_pos + i] = "I-NAME"
            break
    
    print(f"\nFinal result:")
    for w, l in zip(words, labels):
        print(f"  {w:10s} → {l}")
    
    expected = ["O", "B-NAME", "I-NAME", "I-NAME"]
    if labels == expected:
        print(f"\n✓ PASS")
    else:
        print(f"\n✗ FAIL (expected: {expected})")

if __name__ == "__main__":
    test_bio()
