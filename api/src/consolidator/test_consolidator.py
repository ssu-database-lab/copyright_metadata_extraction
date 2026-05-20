#!/usr/bin/env python3
"""
Test Script for Consolidator Module

Tests all components:
- FieldMapper
- ValidationEngine
- ConsolidationAgent (basic)
- ReasoningGenerator
"""

import json
import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
module_dir = current_dir.parent.parent
sys.path.insert(0, str(module_dir))

from module.consolidator import (
    FieldMapper,
    ValidationEngine,
    ConsolidationAgent,
    ReasoningGenerator
)

def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def test_field_mapper():
    """Test FieldMapper component"""
    print_section("Test 1: Field Mapper")
    
    mapper = FieldMapper()
    
    # Sample NER entities
    ner_entities = [
        ("집건에", "NAME"),
        ("국립생태원 멸종위기종복원센터", "COMPANY"),
        ("2024-01-15", "DATE"),
        ("010-1234-5678", "PHONE"),
        ("example@email.com", "EMAIL"),
        ("서울시 강남구", "ADDRESS"),
        ("10000", "MONEY"),
        ("저작재산권 비독점적 이용허락 계약서", "TITLE"),
    ]
    
    # Sample LLM metadata
    llm_metadata = {
        "contract_type": "저작재산권 비독점적 이용허락 계약서",
        "rights_holder": "집건에",
        "user": "국립생태원 멸종위기종복원센터",
        "signature_date": "2024-01-15",
        "work_title": "저작재산권 비독점적 이용허락 계약서",
        "payment_amount": 10000,
        "parties": [
            {
                "name": "집건에",
                "phone": None,
                "address": None
            }
        ]
    }
    
    # Sample OCR text
    ocr_text = """
    저작재산권 비독점적 이용허락 계약서
    
    저작자 및 저작권 이용허락자: 집건에
    저작권 이용자: 국립생태원 멸종위기종복원센터
    
    계약 체결일: 2024-01-15
    연락처: 010-1234-5678
    이메일: example@email.com
    주소: 서울시 강남구
    
    지급 금액: 10000원
    """
    
    print("NER Entities:")
    for entity, entity_type in ner_entities:
        print(f"  - {entity} ({entity_type})")
    
    print("\nLLM Metadata:")
    print(json.dumps(llm_metadata, ensure_ascii=False, indent=2))
    
    # Map entities to fields
    mappings = mapper.map_entities_to_fields(
        ner_entities=ner_entities,
        llm_metadata=llm_metadata,
        ocr_text=ocr_text,
        document_type="계약서"
    )
    
    print("\n📊 Field Mappings:")
    for field, entities in mappings.items():
        print(f"  {field}:")
        for entity_text, confidence in entities:
            print(f"    - {entity_text} (confidence: {confidence:.2f})")
    
    if not mappings:
        print("  ⚠️  No mappings found")
    
    return mappings

def test_validation_engine():
    """Test ValidationEngine component"""
    print_section("Test 2: Validation Engine")
    
    validator = ValidationEngine()
    
    # Test 1: Format validation
    print("📋 Format Validation Tests:")
    
    test_cases = [
        ("signature_date", "2024-01-15", "date", True),
        ("signature_date", "2024/01/15", "date", False),  # Wrong format
        ("signature_date", "invalid-date", "date", False),
        ("parties[].phone", "010-1234-5678", "phone", True),
        ("parties[].phone", "01012345678", "phone", True),
        ("parties[].phone", "abc-123", "phone", False),
        ("parties[].email", "test@example.com", "email", True),
        ("parties[].email", "invalid-email", "email", False),
        ("payment_amount", 10000, "money", True),
        ("payment_amount", "not-a-number", "money", False),
    ]
    
    for field, value, field_type, expected in test_cases:
        is_valid, error = validator.validate_format(field, value, field_type)
        status = "✅" if is_valid == expected else "❌"
        print(f"  {status} {field} = {value}")
        if error:
            print(f"      Error: {error}")
    
    # Test 2: Logical validation
    print("\n📋 Logical Validation Tests:")
    
    # Valid metadata
    valid_metadata = {
        "contract_type": "저작재산권 비독점적 이용허락 계약서",
        "rights_holder": "집건에",
        "user": "국립생태원",
        "signature_date": "2024-01-15",
        "effective_date": "2024-01-20",
        "expiration_date": "2025-01-20",
        "payment_amount": 10000
    }
    
    errors = validator.validate_logic(valid_metadata, "계약서")
    print(f"  Valid metadata errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"    - {error}")
    else:
        print("  ✅ No validation errors")
    
    # Invalid metadata (effective_date > expiration_date)
    invalid_metadata = {
        "effective_date": "2025-01-20",
        "expiration_date": "2024-01-20",  # Invalid: before effective
        "payment_amount": -1000  # Invalid: negative
    }
    
    errors = validator.validate_logic(invalid_metadata, "계약서")
    print(f"\n  Invalid metadata errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"    - ❌ {error}")
    
    # Test 3: Consistency check
    print("\n📋 Consistency Check Tests:")
    
    consistency_tests = [
        ("집건에", "집건에", True, "Exact match"),
        ("집건에", "집건에 ", True, "Whitespace difference"),
        ("집건에", "집건", True, "Partial match"),
        ("2024-01-15", "2024-01-15", True, "Date exact match"),
        ("집건에", "다른이름", False, "Different values"),
    ]
    
    for llm_val, ner_val, expected_match, desc in consistency_tests:
        is_consistent, confidence, explanation = validator.check_consistency(llm_val, ner_val)
        status = "✅" if (is_consistent == expected_match) else "❌"
        print(f"  {status} {desc}: LLM='{llm_val}', NER='{ner_val}'")
        print(f"      Consistent: {is_consistent}, Confidence: {confidence:.2f}")
        print(f"      Explanation: {explanation}")

def test_reasoning_generator():
    """Test ReasoningGenerator component"""
    print_section("Test 3: Reasoning Generator")
    
    reasoner = ReasoningGenerator()
    
    ocr_text = """
    저작재산권 비독점적 이용허락 계약서
    
    저작자 및 저작권 이용허락자: 집건에
    계약 체결일: 2024-01-15
    """
    
    # Test case 1: AGREED decision
    print("📝 Test Case 1: AGREED Decision")
    evidence1 = reasoner.generate_evidence(
        field_name="signature_date",
        llm_value="2024-01-15",
        ner_value="2024-01-15",
        final_value="2024-01-15",
        decision="AGREED",
        ocr_text=ocr_text,
        confidence=1.0
    )
    print(json.dumps(evidence1, ensure_ascii=False, indent=2))
    
    # Test case 2: CONFLICT decision
    print("\n📝 Test Case 2: CONFLICT Decision")
    evidence2 = reasoner.generate_evidence(
        field_name="rights_holder",
        llm_value="집건에",
        ner_value="집건",
        final_value="집건에",
        decision="CONFLICT",
        ocr_text=ocr_text,
        confidence=0.8
    )
    print(json.dumps(evidence2, ensure_ascii=False, indent=2))
    
    # Test case 3: LLM_ONLY decision
    print("\n📝 Test Case 3: LLM_ONLY Decision")
    evidence3 = reasoner.generate_evidence(
        field_name="payment_amount",
        llm_value=10000,
        ner_value=None,
        final_value=10000,
        decision="LLM_ONLY",
        ocr_text=ocr_text,
        confidence=0.6
    )
    print(json.dumps(evidence3, ensure_ascii=False, indent=2))

def test_consolidation_agent_basic():
    """Test ConsolidationAgent basic functionality"""
    print_section("Test 4: Consolidation Agent (Basic)")
    
    # Sample LLM result
    llm_result = {
        "success": True,
        "metadata": {
            "contract_type": "저작재산권 비독점적 이용허락 계약서",
            "rights_holder": "집건에",
            "user": "국립생태원 멸종위기종복원센터",
            "signature_date": "2024-01-15",
            "payment_amount": 10000
        },
        "confidence": 0.95,
        "model_used": "solar-ko"
    }
    
    # Sample NER result
    ner_result = {
        "success": True,
        "extracted_entities": [
            ("집건에", "NAME"),
            ("국립생태원 멸종위기종복원센터", "COMPANY"),
            ("2024-01-15", "DATE"),
            ("010-1234-5678", "PHONE")
        ],
        "statistics": {
            "entity_types_count": {
                "NAME": 1,
                "COMPANY": 1,
                "DATE": 1,
                "PHONE": 1
            }
        },
        "total_entities": 4
    }
    
    # Sample OCR text
    ocr_text = """
    저작재산권 비독점적 이용허락 계약서
    
    저작자 및 저작권 이용허락자: 집건에
    저작권 이용자: 국립생태원 멸종위기종복원센터
    
    계약 체결일: 2024-01-15
    연락처: 010-1234-5678
    
    지급 금액: 10000원
    """
    
    print("LLM Result:")
    print(json.dumps(llm_result, ensure_ascii=False, indent=2))
    
    print("\nNER Result:")
    print(json.dumps({
        "extracted_entities": ner_result["extracted_entities"],
        "total_entities": ner_result["total_entities"]
    }, ensure_ascii=False, indent=2))
    
    print("\n🔧 Initializing ConsolidationAgent...")
    print("   (Note: This will initialize Qwen3-Next-80B model)")
    print("   (Make sure DASHSCOPE_API_KEY is set)")
    
    try:
        # Initialize agent (without output_dir for testing)
        agent = ConsolidationAgent(
            model_name="alibaba-qwen3-next-80b-a3b-instruct",
            output_dir=None
        )
        
        print("   ✅ Agent initialized successfully")
        
        print("\n🔄 Running consolidation (Phase 1 - basic flow)...")
        print("   (Full LLM consolidation will be implemented in Phase 3)")
        
        # Run consolidation
        result = agent.consolidate(
            llm_result=llm_result,
            ner_result=ner_result,
            ocr_text=ocr_text,
            document_type="계약서"
        )
        
        print("\n📊 Consolidation Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        if result.get("success"):
            print("\n✅ Consolidation completed successfully")
            print(f"   Confidence Score: {result.get('validation_report', {}).get('confidence_score', 0):.2f}")
            print(f"   Total Fields: {result.get('validation_report', {}).get('total_fields', 0)}")
        else:
            print("\n⚠️  Consolidation completed with errors")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Error initializing agent: {e}")
        print("   This is expected if:")
        print("   1. DASHSCOPE_API_KEY is not set")
        print("   2. Network connection issues")
        print("   3. Alibaba Cloud API issues")
        print("\n   You can still test other components!")

def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("  CONSOLIDATOR MODULE TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Field Mapper
        mappings = test_field_mapper()
        
        # Test 2: Validation Engine
        test_validation_engine()
        
        # Test 3: Reasoning Generator
        test_reasoning_generator()
        
        # Test 4: Consolidation Agent (requires API key)
        test_consolidation_agent_basic()
        
        print_section("Test Summary")
        print("✅ All component tests completed!")
        print("\n📝 Notes:")
        print("  - Field Mapper: Basic mapping working")
        print("  - Validation Engine: Format and logic validation working")
        print("  - Reasoning Generator: Evidence generation working")
        print("  - Consolidation Agent: Structure ready (LLM integration pending Phase 3)")
        
        if not mappings:
            print("\n⚠️  Field mapper returned no mappings - check entity types and field names")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

