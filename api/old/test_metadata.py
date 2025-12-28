import json
import time
import re
from pathlib import Path

# Configuration
INPUT_FILE = r"c:\Users\peppermint\Desktop\copyright_metadata_extraction\api\data\in\샘플_저작물-20250812T232645Z-1-001_7.저작물양도계약서_extracted_text.txt"

def mask_nouns(text):
    """
    Masks nouns in the text using a heuristic approach since a morphological analyzer is not available.
    Identifies words ending with common Korean particles and masks the stem.
    """
    words = text.split()
    masked_words = []
    noun_count = 0
    
    # Common particles indicating a noun might precede them
    particles = ['은', '는', '이', '가', '을', '를', '의', '에', '로', '으로', '와', '과', '도', '에서', '에게', '한테', '께']
    # Common verb/adjective endings to exclude
    verb_endings = ['다', '요', '죠', '까', '는', 'ㄴ', 'ㄹ', '고', '며', '지']

    for word in words:
        is_noun = False
        masked_word = word
        
        # Heuristic 1: Check for particles
        for p in particles:
            if word.endswith(p) and len(word) > len(p):
                stem = word[:-len(p)]
                # Check if stem looks like a verb
                if not any(stem.endswith(v) for v in verb_endings):
                    is_noun = True
                    # Mask the stem, keep the particle? Or mask the whole word?
                    # User said "mask nouns". Usually, we mask the sensitive info (the noun).
                    # Let's mask the stem and keep the particle for readability of structure, 
                    # or mask the whole word to be safe. 
                    # Let's mask the stem.
                    masked_word = "*" * len(stem) + p
                    break
        
        # Heuristic 2: Words that are likely proper nouns (no particles, not verbs)
        if not is_noun and len(word) > 1:
            # If it doesn't end in a verb ending, assume it might be a noun (e.g. in a list)
            if not any(word.endswith(v) for v in verb_endings):
                # Check if it contains Hangul
                if re.search(r'[가-힣]', word):
                    is_noun = True
                    masked_word = "*" * len(word)

        if is_noun:
            noun_count += 1
            masked_words.append(masked_word)
        else:
            masked_words.append(word)
            
    return " ".join(masked_words), noun_count

def main():
    file_path = Path(INPUT_FILE)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Processing file: {file_path.name}")
    start_time = time.time()
    
    content = file_path.read_text(encoding='utf-8')
    
    # Mask nouns
    masked_content, noun_count = mask_nouns(content)
    
    end_time = time.time()
    processing_time = end_time - start_time

    # Show a snippet of masked text
    print("\n--- Masked Text Snippet (First 200 chars) ---")
    print(masked_content[:200])
    print("---------------------------------------------\n")

    # Generate metadata based on summary.json structure
    # {
    #   "total_files_processed": 1,
    #   "total_entities_found": 15,
    #   "unique_entities": 15,
    #   "entity_types_count": {
    #     "ADDRESS": 7,
    #     "COMPANY": 7,
    #     "PHONE": 1
    #   },
    #   "processing_time": 4.172943353652954,
    #   "timestamp": "20251110_140306"
    # }
    
    metadata = {
        "total_files_processed": 1,
        "total_entities_found": noun_count,
        "unique_entities": noun_count, # Simplified assumption
        "entity_types_count": {
            "NOUN": noun_count # We are treating all masked nouns as one type
        },
        "processing_time": processing_time,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }

    print("Generated Metadata (referencing summary.json structure):")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
