#!/usr/bin/env python3
"""
Universal Checkbox Extractor for Korean Document Metadata Extraction
Handles multiple checkbox patterns and document types with flexible detection
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas.document_schemas import CheckboxPattern

class UniversalCheckboxExtractor:
    """Universal checkbox extractor for all Korean document types"""
    
    def __init__(self):
        # Comprehensive checkbox patterns
        self.checkbox_patterns = {
            CheckboxPattern.PATTERN_A: {
                "checked": ["📧", "☑", "✓", "■", "●", "◼", "◉"],
                "unchecked": ["☐", "□", "○", "◯", "◻", "◦"]
            },
            CheckboxPattern.PATTERN_B: {
                "checked": ["☑", "✓", "■", "●", "◼", "◉", "📧"],
                "unchecked": ["□", "☐", "○", "◯", "◻", "◦"]
            },
            CheckboxPattern.PATTERN_C: {
                "checked": ["✓", "☑", "■", "●", "◼", "◉"],
                "unchecked": ["○", "◯", "☐", "□", "◻", "◦"]
            },
            CheckboxPattern.PATTERN_D: {
                "checked": ["■", "●", "◼", "◉", "☑", "✓"],
                "unchecked": ["□", "☐", "○", "◯", "◻", "◦"]
            }
        }
        
        # Document-specific checkbox mappings
        self.document_checkbox_mappings = {
            "contract": {
                "payment_terms": {
                    "prepaid": ["선불", "사전결제", "prepaid"],
                    "postpaid": ["후불", "사후결제", "postpaid"],
                    "installment": ["할부", "분할결제", "installment"]
                },
                "rights_granted": {
                    "reproduction": ["복제권", "목제권", "reproduction"],
                    "performance": ["공연권", "공면권", "performance"],
                    "broadcasting": ["공중송신권", "broadcasting"],
                    "exhibition": ["전시권", "exhibition"],
                    "distribution": ["배포권", "distribution"],
                    "rental": ["대여권", "rental"],
                    "derivative": ["2차적저작물작성권", "2차적저작물", "derivative"]
                },
                "contract_type": {
                    "exclusive": ["독점적", "exclusive"],
                    "non_exclusive": ["비독점적", "non-exclusive"]
                },
                "renewal": {
                    "auto_renewal": ["자동갱신", "auto renewal"],
                    "manual_renewal": ["수동갱신", "manual renewal"]
                }
            },
            "consent": {
                "data_purposes": {
                    "service_provision": ["서비스제공", "service provision"],
                    "marketing": ["마케팅", "marketing"],
                    "research": ["연구", "research"],
                    "statistics": ["통계", "statistics"]
                },
                "third_party_sharing": {
                    "allowed": ["제3자제공동의", "third party sharing"],
                    "not_allowed": ["제3자제공거부", "no third party sharing"]
                },
                "marketing_consent": {
                    "email": ["이메일마케팅", "email marketing"],
                    "sms": ["SMS마케팅", "SMS marketing"],
                    "phone": ["전화마케팅", "phone marketing"]
                }
            },
            "public_copyright_consent": {
                "work_details": {
                    "stage": ["무대", "stage"],
                    "lighting": ["장치", "lighting", "조명"],
                    "costume": ["의상", "costume"],
                    "accessories": ["장신구", "accessories"],
                    "props": ["소품", "props"],
                    "meditation": ["명상", "meditation"],
                    "sound": ["음향", "sound"],
                    "video": ["영상", "video"],
                    "lighting_equipment": ["조명", "lighting equipment"]
                },
                "granted_rights": {
                    "reproduction": ["복제권", "목제권", "reproduction"],
                    "performance": ["공연권", "공면권", "performance"],
                    "broadcasting": ["공중송신권", "broadcasting"],
                    "exhibition": ["전시권", "exhibition"],
                    "distribution": ["배포권", "distribution"],
                    "rental": ["대여권", "rental"],
                    "derivative": ["2차적저작물작성권", "2차적저작물", "derivative"]
                },
                "nuri_types": {
                    "type_1": ["제1유형", "type 1"],
                    "type_2": ["제2유형", "type 2"],
                    "type_3": ["제3유형", "type 3"],
                    "type_4": ["제4유형", "type 4"]
                }
            },
            "copyright_transfer": {
                "transfer_scope": {
                    "reproduction": ["복제권", "reproduction"],
                    "performance": ["공연권", "performance"],
                    "broadcasting": ["공중송신권", "broadcasting"],
                    "exhibition": ["전시권", "exhibition"],
                    "distribution": ["배포권", "distribution"],
                    "rental": ["대여권", "rental"],
                    "derivative": ["2차적저작물작성권", "2차적저작물", "derivative"],
                    "moral_rights": ["인격권", "moral rights"]
                },
                "transfer_type": {
                    "full_transfer": ["전체양도", "full transfer"],
                    "partial_transfer": ["부분양도", "partial transfer"],
                    "license": ["이용허락", "license"]
                },
                "nuri_conditions": {
                    "attribution": ["저작자표시", "attribution"],
                    "commercial_use": ["상업적이용", "commercial use"],
                    "modification": ["변경허용", "modification"],
                    "share_alike": ["동일조건변경허락", "share alike"]
                }
            }
        }
    
    def detect_checkbox_pattern(self, text: str) -> CheckboxPattern:
        """Detect which checkbox pattern is used in the text"""
        for pattern, symbols in self.checkbox_patterns.items():
            for checked_symbol in symbols["checked"]:
                if checked_symbol in text:
                    return pattern
        return CheckboxPattern.PATTERN_A  # default
    
    def extract_checkbox_state(self, text: str, item_name: str, pattern: CheckboxPattern = None) -> bool:
        """Extract checkbox state for a specific item"""
        if pattern is None:
            pattern = self.detect_checkbox_pattern(text)
        
        checked_symbols = self.checkbox_patterns[pattern]["checked"]
        unchecked_symbols = self.checkbox_patterns[pattern]["unchecked"]
        
        # Create regex pattern to find the item with checkbox
        for checked_symbol in checked_symbols:
            if re.search(rf"{re.escape(checked_symbol)}\s*{re.escape(item_name)}", text):
                return True
        
        for unchecked_symbol in unchecked_symbols:
            if re.search(rf"{re.escape(unchecked_symbol)}\s*{re.escape(item_name)}", text):
                return False
        
        return False  # Default to False if not found
    
    def extract_document_checkboxes(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract all checkboxes for a specific document type"""
        if document_type not in self.document_checkbox_mappings:
            return {}
        
        pattern = self.detect_checkbox_pattern(text)
        checkbox_data = {}
        
        for category, items in self.document_checkbox_mappings[document_type].items():
            checkbox_data[category] = {}
            for key, possible_names in items.items():
                for name in possible_names:
                    if self.extract_checkbox_state(text, name, pattern):
                        checkbox_data[category][key] = True
                        break
                else:
                    checkbox_data[category][key] = False
        
        return checkbox_data
    
    def extract_work_display_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced work display extraction with flexible checkbox detection"""
        work_display = {}
        pattern = self.detect_checkbox_pattern(text)
        
        # Extract work names - handle different formats
        work_names_match = re.search(r"저작물명\s*:\s*(.*?)(?=○|$)", text, re.DOTALL)
        if work_names_match:
            work_names_text = work_names_match.group(1).strip()
            
            # Handle different work name formats
            if "<" in work_names_text and ">" in work_names_text:
                # Format: 국립극단 <공하늘> 외(...)
                work_match = re.search(r"([^<]+)<([^>]+)>", work_names_text)
                if work_match:
                    institution = work_match.group(1).strip()
                    work_title = work_match.group(2).strip()
                    work_display["work_names"] = [f"{institution}<{work_title}>"]
                    work_display["institution"] = institution
            else:
                # Format: 2000년 <수궁가>와 <우루왕>, 2001년 <춘향전>과 <우루왕>
                work_names = re.findall(r"\d{4}년\s*<[^>]+>", work_names_text)
                work_display["work_names"] = work_names
        
        # Extract work category
        if "우대미술" in text:
            work_display["work_category"] = "우대미술"
        elif "의상" in text and "디자인" in text:
            work_display["work_category"] = "의상디자인"
        
        # Extract work details with flexible checkbox detection
        work_details = {}
        theater_mapping = self.document_checkbox_mappings["public_copyright_consent"]["work_details"]
        
        for key, possible_names in theater_mapping.items():
            for name in possible_names:
                if self.extract_checkbox_state(text, name, pattern):
                    work_details[key] = True
                    break
            else:
                work_details[key] = False
        
        work_display["work_details"] = work_details
        
        # Extract detailed info
        detail_match = re.search(r"상세정보\s*:\s*(.*?)(?=□|$)", text, re.DOTALL)
        if detail_match:
            work_display["detailed_info"] = detail_match.group(1).strip()
        
        return work_display
    
    def extract_copyright_license_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced copyright license extraction with flexible checkbox detection"""
        license_info = {}
        pattern = self.detect_checkbox_pattern(text)
        
        # Extract license purpose
        purpose_match = re.search(r"저작물의 개방을 통해 이용자가 자유롭게 이용할 수 있도록", text)
        if purpose_match:
            license_info["license_purpose"] = purpose_match.group(0)
        
        # Extract licensing institution
        institution_match = re.search(r"국립극장", text)
        if institution_match:
            license_info["licensing_institution"] = institution_match.group(0)
        
        # Extract granted rights with flexible checkbox detection
        granted_rights = {}
        rights_mapping = self.document_checkbox_mappings["public_copyright_consent"]["granted_rights"]
        
        for key, possible_names in rights_mapping.items():
            for name in possible_names:
                if self.extract_checkbox_state(text, name, pattern):
                    granted_rights[key] = True
                    break
            else:
                granted_rights[key] = False
        
        license_info["granted_rights"] = granted_rights
        license_info["license_type"] = "비독점적 이용허락"
        
        return license_info
    
    def extract_public_nuri_license_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced public Nuri license extraction with flexible checkbox detection"""
        nuri_info = {}
        pattern = self.detect_checkbox_pattern(text)
        
        # Extract license purpose
        purpose_match = re.search(r"공공저작물의 자유이용 활성화", text)
        if purpose_match:
            nuri_info["license_purpose"] = purpose_match.group(0)
        
        # Extract Nuri type with flexible checkbox detection
        nuri_mapping = self.document_checkbox_mappings["public_copyright_consent"]["nuri_types"]
        for key, possible_names in nuri_mapping.items():
            for name in possible_names:
                if self.extract_checkbox_state(text, name, pattern):
                    nuri_info["nuri_type"] = name
                    break
        
        # Extract available types
        available_types = re.findall(r"제[1-4]유형", text)
        nuri_info["available_types"] = list(set(available_types))
        
        # Extract modification rights
        modification_rights = {}
        modification_rights["integrity_right_waiver"] = bool(re.search(r"동일성유지권.*동의", text))
        modification_rights["modification_allowed"] = bool(re.search(r"변경.*가능", text))
        
        # Extract conditions
        conditions_match = re.search(r"연구.*결과.*명예.*심각한.*훼손.*특별한.*사정.*없는.*한", text)
        if conditions_match:
            modification_rights["conditions"] = conditions_match.group(0)
        
        nuri_info["modification_rights"] = modification_rights
        
        return nuri_info
    
    def calculate_confidence(self, checkbox_data: Dict) -> float:
        """Calculate extraction confidence based on checkbox data"""
        total_fields = sum(len(category) for category in checkbox_data.values())
        if total_fields == 0:
            return 0.0
        
        # Simple confidence calculation based on data completeness
        return min(1.0, total_fields / 10.0)  # Normalize to 0-1
    
    def get_checkbox_fields(self, checkbox_data: Dict) -> List[str]:
        """Get list of all checkbox fields found"""
        fields = []
        for category, items in checkbox_data.items():
            for field_name in items.keys():
                fields.append(f"{category}.{field_name}")
        return fields

# Example usage and testing
if __name__ == "__main__":
    # Test the checkbox extractor
    extractor = UniversalCheckboxExtractor()
    
    # Test text with different checkbox patterns
    test_text_a = "저작물명: 테스트 (📧복제권, ☐공연권, 📧공중송신권)"
    test_text_b = "저작물명: 테스트 (☑복제권, □공연권, ☑공중송신권)"
    
    print("Testing Checkbox Extractor:")
    print(f"Pattern A detected: {extractor.detect_checkbox_pattern(test_text_a)}")
    print(f"Pattern B detected: {extractor.detect_checkbox_pattern(test_text_b)}")
    
    # Test checkbox state extraction
    print(f"복제권 in text A: {extractor.extract_checkbox_state(test_text_a, '복제권')}")
    print(f"공연권 in text A: {extractor.extract_checkbox_state(test_text_a, '공연권')}")
    print(f"복제권 in text B: {extractor.extract_checkbox_state(test_text_b, '복제권')}")
    print(f"공연권 in text B: {extractor.extract_checkbox_state(test_text_b, '공연권')}")
    
    # Test document-specific extraction
    checkbox_data = extractor.extract_document_checkboxes(test_text_a, "public_copyright_consent")
    print(f"Extracted checkbox data: {checkbox_data}")
    print(f"Confidence: {extractor.calculate_confidence(checkbox_data)}")
    print(f"Fields found: {extractor.get_checkbox_fields(checkbox_data)}")
