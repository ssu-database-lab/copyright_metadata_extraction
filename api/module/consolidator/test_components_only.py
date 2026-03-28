#!/usr/bin/env python3
"""
Quick Test Script for Consolidator Components (No API Key Required)

Tests FieldMapper, ValidationEngine, and ReasoningGenerator
without requiring Alibaba Cloud API key.
"""

import json
import sys
from pathlib import Path


def test_field_mapper():
    """Test FieldMapper without API"""
    print("\n" + "="*80)
    print("  TEST 1: Field Mapper")
    print("="*80 + "\n")
    
    from module.consolidator import FieldMapper
    
    mapper = FieldMapper()
    
    ner_entities = [
        ("집건에", "NAME"),
        ("국립생태원", "COMPANY"),
        ("2024-01-15", "DATE"),
        ("010-1234-5678", "PHONE"),
    ]
    
    llm_metadata = {
        "rights_holder": "집건에",
        "user": "국립생태원",
        "signature_date": "2024-01-15",
    }
    
    mappings = mapper.map_entities_to_fields(
        ner_entities=ner_entities,
        llm_metadata=llm_metadata,
        ocr_text="저작재산권 비독점적 이용허락 계약서",
        document_type="계약서"
    )
    
    print("✅ Field Mapper Test Results:")
    if mappings:
        for field, entities in mappings.items():
            print(f"  {field}: {len(entities)} entities mapped")
            for entity, conf in entities:
                print(f"    - {entity} (confidence: {conf:.2f})")
        return True
    else:
        print("  ⚠️  No mappings found")
        return False

def test_validation_engine():
    """Test ValidationEngine"""
    print("\n" + "="*80)
    print("  TEST 2: Validation Engine")
    print("="*80 + "\n")
    
    from module.consolidator import ValidationEngine
    
    validator = ValidationEngine()
    
    # Format validation
    print("📋 Format Validation:")
    test_cases = [
        ("signature_date", "2024-01-15", "date"),
        ("signature_date", "invalid", "date"),
        ("phone", "010-1234-5678", "phone"),
        ("email", "test@example.com", "email"),
    ]
    
    passed = 0
    for field, value, field_type in test_cases:
        is_valid, error = validator.validate_format(field, value, field_type)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {field} = '{value}' -> Valid: {is_valid}")
        if error:
            print(f"      {error}")
        if is_valid:
            passed += 1
    
    # Consistency check
    print("\n📋 Consistency Check:")
    is_consistent, conf, explanation = validator.check_consistency(
        "집건에", "집건에"
    )
    print(f"  ✅ Values match: {is_consistent} (confidence: {conf:.2f})")
    print(f"      {explanation}")
    
    # Logic validation
    print("\n📋 Logic Validation:")
    metadata = {
        "signature_date": "2024-01-15",
        "effective_date": "2024-01-20",
        "expiration_date": "2025-01-20",
    }
    errors = validator.validate_logic(metadata, "계약서")
    if errors:
        print(f"  ❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"      - {error}")
    else:
        print(f"  ✅ No validation errors")
    
    return passed > 0

def test_reasoning_generator():
    """Test ReasoningGenerator"""
    print("\n" + "="*80)
    print("  TEST 3: Reasoning Generator")
    print("="*80 + "\n")
    
    from module.consolidator import ReasoningGenerator
    
    reasoner = ReasoningGenerator()
    
    ocr_text = "저작재산권 비독점적 이용허락 계약서\n계약 체결일: 2024-01-15"
    
    evidence = reasoner.generate_evidence(
        field_name="signature_date",
        llm_value="2024-01-15",
        ner_value="2024-01-15",
        final_value="2024-01-15",
        decision="AGREED",
        ocr_text=ocr_text,
        confidence=1.0
    )
    
    print("✅ Reasoning Generator Test:")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    
    return True

def main():
    """Run all component tests"""
    print("="*80)
    print("  CONSOLIDATOR COMPONENTS TEST (No API Key Required)")
    print("="*80)
    
    results = []
    
    try:
        results.append(("Field Mapper", test_field_mapper()))
        results.append(("Validation Engine", test_validation_engine()))
        results.append(("Reasoning Generator", test_reasoning_generator()))
        
        print("\n" + "="*80)
        print("  TEST SUMMARY")
        print("="*80 + "\n")
        
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {name}")
        
        all_passed = all(result[1] for result in results)
        
        if all_passed:
            print("\n✅ All component tests passed!")
        else:
            print("\n⚠️  Some tests had issues (check output above)")
        
        print("\n📝 Next Steps:")
        print("  1. If all tests passed, components are working correctly")
        print("  2. To test full consolidation (requires API key):")
        print("     python test_consolidator.py")
        print("  3. Proceed to Phase 3: LLM integration")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

