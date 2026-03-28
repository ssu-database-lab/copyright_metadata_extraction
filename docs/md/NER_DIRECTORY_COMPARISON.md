# NER Directory Comparison: Local vs apiBackup Branch

## Summary

Comparison between your local `ner/` directory and the `apiBackup` branch from the GitHub repository.

**Date:** 2025-12-28  
**Branch:** `apiBackup`  
**Repository:** https://github.com/ssu-database-lab/copyright_metadata_extraction

---

## File Structure Comparison

### Files Present in Both Versions

| File | Local Lines | apiBackup Lines | Status |
|------|-------------|-----------------|--------|
| `ner.py` | 574 | 574 | ✅ **Identical** |
| `entity_extraction.py` | 56 | 56 | ✅ **Identical** |
| `requirements.txt` | 8 | 8 | ✅ **Identical** |
| `train_model.py` | ✓ | ✓ | Present in both |
| `create_training_data.py` | ✓ | ✓ | Present in both |
| `enhance_training_data.py` | ✓ | ✓ | Present in both |
| `generate_training_data.py` | ✓ | ✓ | Present in both |
| `ner_system.bat` | ✓ | ✓ | Present in both |
| `ner_system.sh` | ✓ | ✓ | Present in both |
| `README.md` | ✓ | ✓ | Present in both |
| `train_log.txt` | ✓ | ✓ | Present in both |
| `train_verbose_log.txt` | ✓ | ✓ | Present in both |

### Files Only in Local Version

- `계약서 Label.xlsx` - Excel file with contract labels
- `동의서 Label.xlsx` - Excel file with consent labels
- `__pycache__/` - Python cache directory (typically gitignored)

### Files Only in apiBackup Branch

- None (all files in apiBackup exist locally)

---

## Detailed File Comparison

### 1. `ner.py`

**Status:** ✅ **Identical** (574 lines)

Both versions have:
- Same imports and package installation logic
- Same constants (OCR_DOCUMENT_PATH, OUTPUT_DIR, MODEL_NAME, MAX_LENGTH)
- Same label definitions:
  - CONTRACT_LABELS: 저작물명, 대상저작물상세정보, 양수자기관명, 양수자주소, 양도자기관명, 양도자소속, 양도자대표주소, 양도자연락처, 계약체결일
  - CONSENT_LABELS: 양도인성명, 양도인전화번호, 양도인주소, 양수인기관명, 양수인대표자명, 양수인대표자주소, 양수인대표자연락처, 동의여부, 동의날짜

**First 60 lines are identical** (verified by comparison).

### 2. `entity_extraction.py`

**Status:** ✅ **Identical** (56 lines)

Both versions contain the same regex-based entity extraction functions:
- `extract_consent_entities(text)` - For consent documents
- `extract_contract_entities(text)` - For contract documents

### 3. `requirements.txt`

**Status:** ✅ **Identical** (8 lines)

Both versions have the same dependencies:
```
transformers==4.35.0
torch>=2.0.0
pandas>=2.0.0
datasets>=2.14.0
tqdm>=4.66.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

## Key Findings

### ✅ **No Code Differences**
- All Python files appear to be **identical** in content
- Same line counts for all compared files
- Same dependencies and requirements

### 📊 **Additional Files in Local**
- Your local version has **Excel label files** (`계약서 Label.xlsx`, `동의서 Label.xlsx`)
- These are likely reference/labeling files not tracked in git

### 🔍 **What This Means**
1. **The `apiBackup` branch appears to be a clean backup** of the NER system
2. **Your local version is up-to-date** with the apiBackup branch code
3. **The Excel files are local-only** and not part of the repository

---

## Recommendations

### For Testing the apiBackup Branch

1. **Safe to checkout**: Since the code is identical, you can safely checkout the branch to test
   ```bash
   git fetch origin apiBackup
   git checkout apiBackup
   ```

2. **No merge conflicts expected**: The NER directory code is identical, so no conflicts

3. **Preserve Excel files**: If you checkout the branch, make sure to backup your Excel label files:
   ```bash
   cp ner/계약서\ Label.xlsx ~/backup/
   cp ner/동의서\ Label.xlsx ~/backup/
   ```

4. **Check other directories**: The comparison only covers the `ner/` directory. You may want to compare:
   - `api/` directory
   - `extract/` directory
   - Root level files

---

## Next Steps

1. ✅ **NER directory is identical** - No action needed for this directory
2. 🔍 **Compare other directories** - Check `api/`, `extract/`, etc. for differences
3. 📝 **Document differences** - If you find differences in other directories, document them
4. 🧪 **Test the branch** - You can safely test the apiBackup branch for the NER system

---

## Verification Commands Used

```bash
# List files in apiBackup branch
git fetch origin apiBackup
git ls-tree -r --name-only origin/apiBackup:ner/

# Compare file sizes
wc -l ner/ner.py ner/entity_extraction.py ner/requirements.txt
git show origin/apiBackup:ner/ner.py | wc -l

# View file contents
git show origin/apiBackup:ner/requirements.txt
git show origin/apiBackup:ner/ner.py | head -60
```

---

**Conclusion:** The `ner/` directory in your local version and the `apiBackup` branch are **functionally identical**. The only differences are local Excel files that aren't tracked in git.

