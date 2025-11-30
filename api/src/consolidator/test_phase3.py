#!/usr/bin/env python3
"""
Test Phase 3: LLM Integration with Qwen3-Next-80B

Tests the full consolidation flow with actual LLM calls.
Requires DASHSCOPE_API_KEY environment variable.
"""

import json
import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
module_dir = current_dir.parent.parent
sys.path.insert(0, str(module_dir))

def test_phase3_consolidation():
    """Test Phase 3 consolidation with Qwen3-Next-80B"""
    print("=" * 80)
    print("  PHASE 3 TEST: LLM Consolidation with Qwen3-Next-80B")
    print("=" * 80)
    
    # Check API key
    api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
    if not api_key:
        print("\n❌ DASHSCOPE_API_KEY not set!")
        print("   Set it with: export DASHSCOPE_API_KEY='your_key_here'")
        print("\n   Skipping Phase 3 test (requires API key)")
        return False
    
    print(f"\n✅ API Key found: {api_key[:10]}...")
    
    try:
        from module.consolidator import ConsolidationAgent
        
        # Sample data
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
        
        ner_result = {
            "success": True,
            "extracted_entities": [
                ("집건에", "NAME"),
                ("국립생태원", "COMPANY"),
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
        
        ocr_text = """
        저작재산권 비독점적 이용허락 계약서
        
        저작자 및 저작권 이용허락자: 집건에
        저작권 이용자: 국립생태원 멸종위기종복원센터
        
        계약 체결일: 2024-01-15
        연락처: 010-1234-5678
        
        지급 금액: 10000원
        """
        
        print("\n📋 Sample Data:")
        print(f"  LLM Metadata: {len(llm_result['metadata'])} fields")
        print(f"  NER Entities: {len(ner_result['extracted_entities'])} entities")
        print(f"  OCR Text: {len(ocr_text)} characters")
        
        print("\n🔄 Initializing ConsolidationAgent...")
        agent = ConsolidationAgent(
            model_name="alibaba-qwen3-next-80b-a3b-instruct",
            output_dir=None
        )
        
        print("✅ Agent initialized")
        
        print("\n🚀 Running consolidation...")
        print("   (This will call Qwen3-Next-80B API)")
        
        result = agent.consolidate(
            llm_result=llm_result,
            ner_result=ner_result,
            ocr_text=ocr_text,
            document_type="계약서"
        )
        
        print("\n📊 Consolidation Result:")
        print(f"  Success: {result.get('success')}")
        print(f"  Status: {result.get('status')}")
        
        if result.get('success'):
            report = result.get('validation_report', {})
            print(f"\n  📈 Statistics:")
            print(f"    Total Fields: {report.get('total_fields', 0)}")
            print(f"    Agreed: {report.get('agreed_fields', 0)}")
            print(f"    Conflicted: {report.get('conflicted_fields', 0)}")
            print(f"    LLM Only: {report.get('llm_only_fields', 0)}")
            print(f"    NER Only: {report.get('ner_only_fields', 0)}")
            print(f"    Confidence: {report.get('confidence_score', 0):.2f}")
            
            decisions = report.get('decisions', [])
            if decisions:
                print(f"\n  🔍 Sample Decisions ({len(decisions)} total):")
                for i, decision in enumerate(decisions[:3], 1):  # Show first 3
                    print(f"    {i}. {decision.get('field')}: {decision.get('decision')} (confidence: {decision.get('confidence', 0):.2f})")
            
            print(f"\n  📝 Consolidated Metadata:")
            metadata = result.get('consolidated_metadata', {})
            for key, value in list(metadata.items())[:5]:
                print(f"    - {key}: {value}")
            
            print("\n✅ Phase 3 test completed successfully!")
            return True
        else:
            print(f"\n❌ Consolidation failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase3_consolidation()
    sys.exit(0 if success else 1)

