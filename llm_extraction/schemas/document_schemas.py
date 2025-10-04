#!/usr/bin/env python3
"""
JSON Schema Templates for Korean Document Metadata Extraction
Defines structured schemas for contracts and consent forms
"""

from typing import Dict, Any

class DocumentSchemas:
    """Collection of JSON schemas for different document types"""
    
    @staticmethod
    def get_contract_schema() -> Dict[str, Any]:
        """Schema for contract documents (계약서)"""
        return {
            "type": "object",
            "properties": {
                "contract_type": {
                    "type": "string",
                    "description": "계약서 유형 (예: 저작재산권 비독점적 이용허락 계약서)"
                },
                "rights_holder": {
                    "type": "string", 
                    "description": "권리자 (저작자 및 저작권 이용허락자)"
                },
                "user": {
                    "type": "string",
                    "description": "이용자 (저작권 이용자)"
                },
                "work_title": {
                    "type": "string",
                    "description": "저작물 제목"
                },
                "work_category": {
                    "type": "string",
                    "description": "저작물 종별 (어문저작물, 음악저작물, 미술저작물 등)"
                },
                "granted_rights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "허락된 권리 (복제권, 공연권, 공중송신권 등)"
                },
                "contract_purpose": {
                    "type": "string",
                    "description": "계약의 목적"
                },
                "contract_duration": {
                    "type": "string",
                    "description": "계약 기간"
                },
                "payment_amount": {
                    "type": "number",
                    "description": "지급 금액 (숫자만)"
                },
                "payment_currency": {
                    "type": "string",
                    "description": "통화 (원, 달러 등)"
                },
                "signature_date": {
                    "type": "string",
                    "format": "date",
                    "description": "계약 체결일 (YYYY-MM-DD)"
                },
                "effective_date": {
                    "type": "string",
                    "format": "date", 
                    "description": "계약 효력 발생일 (YYYY-MM-DD)"
                },
                "expiration_date": {
                    "type": "string",
                    "format": "date",
                    "description": "계약 만료일 (YYYY-MM-DD)"
                },
                "special_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "특별 약정 사항"
                },
                "termination_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "계약 해지 조건"
                },
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "당사자 이름 또는 회사명"
                            },
                            "phone": {
                                "type": "string",
                                "description": "전화번호 (숫자와 하이픈만)"
                            },
                            "address": {
                                "type": "string",
                                "description": "주소"
                            },
                            "registration_no": {
                                "type": "string",
                                "description": "사업자등록번호 또는 주민등록번호 (숫자와 하이픈만)"
                            },
                            "role": {
                                "type": "string",
                                "description": "계약에서의 역할 (권리자, 이용자, 증인 등)"
                            }
                        },
                        "required": ["name"]
                    },
                    "description": "계약 당사자들의 정보"
                }
            },
            "required": ["contract_type", "rights_holder", "user"]
        }
    
    @staticmethod
    def get_consent_schema() -> Dict[str, Any]:
        """Schema for consent forms (동의서)"""
        return {
            "type": "object",
            "properties": {
                "consent_type": {
                    "type": "string",
                    "description": "동의서 유형 (개인정보 수집 및 이용 동의서 등)"
                },
                "data_controller": {
                    "type": "string",
                    "description": "개인정보 처리자 (기관명)"
                },
                "data_subject": {
                    "type": "string",
                    "description": "정보주체 (동의자)"
                },
                "collection_purpose": {
                    "type": "string",
                    "description": "개인정보 수집 및 이용 목적"
                },
                "collected_data_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "수집하는 개인정보 항목 (성명, 전화번호, 주소 등)"
                },
                "retention_period": {
                    "type": "string",
                    "description": "개인정보 보유 및 이용 기간"
                },
                "third_party_sharing": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string", "description": "제공받는 자"},
                        "purpose": {"type": "string", "description": "이용 목적"},
                        "data_types": {"type": "array", "items": {"type": "string"}, "description": "제공하는 개인정보 항목"}
                    },
                    "description": "제3자 제공 정보"
                },
                "consent_status": {
                    "type": "string",
                    "enum": ["동의함", "동의하지 않음", "null"],
                    "description": "동의 여부"
                },
                "consent_date": {
                    "type": "string",
                    "format": "date",
                    "description": "동의일 (YYYY-MM-DD)"
                },
                "signature": {
                    "type": "string",
                    "description": "서명자 정보"
                },
                "contact_info": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string", "description": "연락처"},
                        "address": {"type": "string", "description": "주소"},
                        "email": {"type": "string", "description": "이메일"}
                    },
                    "description": "연락처 정보"
                },
                "withdrawal_rights": {
                    "type": "string",
                    "description": "동의 철회 권리에 대한 안내"
                },
                "consequences_of_refusal": {
                    "type": "string",
                    "description": "동의 거부 시 불이익"
                },
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "당사자 이름 또는 회사명"
                            },
                            "phone": {
                                "type": "string",
                                "description": "전화번호 (숫자와 하이픈만)"
                            },
                            "address": {
                                "type": "string",
                                "description": "주소"
                            },
                            "registration_no": {
                                "type": "string",
                                "description": "사업자등록번호 또는 주민등록번호 (숫자와 하이픈만)"
                            },
                            "role": {
                                "type": "string",
                                "description": "동의서에서의 역할 (정보주체, 처리자, 대리인 등)"
                            }
                        },
                        "required": ["name"]
                    },
                    "description": "동의서 관련 당사자들의 정보"
                }
            },
            "required": ["consent_type", "data_controller", "consent_status"]
        }
    
    @staticmethod
    def get_general_document_schema() -> Dict[str, Any]:
        """General schema for any document type"""
        return {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "문서 유형"
                },
                "title": {
                    "type": "string",
                    "description": "문서 제목"
                },
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "당사자 이름 또는 회사명"
                            },
                            "phone": {
                                "type": "string",
                                "description": "전화번호 (숫자와 하이픈만)"
                            },
                            "address": {
                                "type": "string",
                                "description": "주소"
                            },
                            "registration_no": {
                                "type": "string",
                                "description": "사업자등록번호 또는 주민등록번호 (숫자와 하이픈만)"
                            },
                            "role": {
                                "type": "string",
                                "description": "문서에서의 역할 (발신자, 수신자, 증인, 대리인 등)"
                            }
                        },
                        "required": ["name"]
                    },
                    "description": "문서 관련 당사자들의 정보"
                },
                "key_dates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date"},
                            "description": {"type": "string"}
                        }
                    },
                    "description": "중요한 날짜들"
                },
                "key_amounts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number"},
                            "currency": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    },
                    "description": "중요한 금액들"
                },
                "main_content": {
                    "type": "string",
                    "description": "문서의 주요 내용 요약"
                },
                "important_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "중요한 조항이나 조건들"
                }
            },
            "required": ["document_type", "title"]
        }
    
    @staticmethod
    def get_schema_by_document_type(document_type: str) -> Dict[str, Any]:
        """Get appropriate schema based on document type"""
        if "계약서" in document_type or "contract" in document_type.lower():
            return DocumentSchemas.get_contract_schema()
        elif "동의서" in document_type or "consent" in document_type.lower():
            return DocumentSchemas.get_consent_schema()
        else:
            return DocumentSchemas.get_general_document_schema()

# Example usage and testing
if __name__ == "__main__":
    # Test schema creation
    contract_schema = DocumentSchemas.get_contract_schema()
    consent_schema = DocumentSchemas.get_consent_schema()
    general_schema = DocumentSchemas.get_general_document_schema()
    
    print("Contract Schema:")
    print(contract_schema)
    
    print("\nConsent Schema:")
    print(consent_schema)
    
    print("\nGeneral Schema:")
    print(general_schema)
    
    # Test schema selection
    test_types = ["계약서", "동의서", "기타문서"]
    for doc_type in test_types:
        schema = DocumentSchemas.get_schema_by_document_type(doc_type)
        print(f"\nSchema for '{doc_type}': {schema['properties'].keys()}")
