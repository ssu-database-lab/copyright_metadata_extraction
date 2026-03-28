#!/usr/bin/env python3
"""
JSON Schema Templates for Korean Document Metadata Extraction
Defines structured schemas for contracts and consent forms with universal checkbox support
"""

from typing import Dict, Any, List
from enum import Enum

class CheckboxPattern(Enum):
    PATTERN_A = "pattern_a"  # 📧/☐
    PATTERN_B = "pattern_b"  # ☑/□
    PATTERN_C = "pattern_c"  # ✓/○
    PATTERN_D = "pattern_d"  # ■/□

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
    def get_contract_schema_enhanced() -> Dict[str, Any]:
        """Enhanced contract schema with universal checkbox support"""
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
                
                # Enhanced rights section with checkbox support
                "granted_rights": {
                    "type": "object",
                    "properties": {
                        "reproduction_right": {"type": "boolean", "description": "복제권 (체크박스 상태)"},
                        "performance_right": {"type": "boolean", "description": "공연권 (체크박스 상태)"},
                        "broadcasting_right": {"type": "boolean", "description": "공중송신권 (체크박스 상태)"},
                        "exhibition_right": {"type": "boolean", "description": "전시권 (체크박스 상태)"},
                        "distribution_right": {"type": "boolean", "description": "배포권 (체크박스 상태)"},
                        "rental_right": {"type": "boolean", "description": "대여권 (체크박스 상태)"},
                        "derivative_work_right": {"type": "boolean", "description": "2차적저작물작성권 (체크박스 상태)"}
                    },
                    "description": "허락된 권리 (체크박스 기반 추출)"
                },
                
                # Contract terms with checkbox support
                "contract_terms": {
                    "type": "object",
                    "properties": {
                        "contract_type_selection": {
                            "type": "object",
                            "properties": {
                                "exclusive": {"type": "boolean", "description": "독점적 계약 (체크박스)"},
                                "non_exclusive": {"type": "boolean", "description": "비독점적 계약 (체크박스)"}
                            }
                        },
                        "payment_terms": {
                            "type": "object",
                            "properties": {
                                "prepaid": {"type": "boolean", "description": "선불 (체크박스)"},
                                "postpaid": {"type": "boolean", "description": "후불 (체크박스)"},
                                "installment": {"type": "boolean", "description": "할부 (체크박스)"}
                            }
                        },
                        "renewal_options": {
                            "type": "object",
                            "properties": {
                                "auto_renewal": {"type": "boolean", "description": "자동갱신 (체크박스)"},
                                "manual_renewal": {"type": "boolean", "description": "수동갱신 (체크박스)"}
                            }
                        }
                    }
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
                
                # Enhanced parties section
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
                },
                
                # Checkbox processing information
                "checkbox_info": {
                    "type": "object",
                    "properties": {
                        "pattern_detected": {
                            "type": "string",
                            "enum": ["pattern_a", "pattern_b", "pattern_c", "pattern_d"],
                            "description": "감지된 체크박스 패턴"
                        },
                        "extraction_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "체크박스 추출 신뢰도"
                        },
                        "checkbox_fields_found": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "발견된 체크박스 필드들"
                        }
                    }
                }
            },
            "required": ["contract_type", "rights_holder", "user", "granted_rights"]
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
    def get_consent_schema_enhanced() -> Dict[str, Any]:
        """Enhanced consent schema with universal checkbox support"""
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
                },
                
                # Checkbox processing information
                "checkbox_info": {
                    "type": "object",
                    "properties": {
                        "pattern_detected": {
                            "type": "string",
                            "enum": ["pattern_a", "pattern_b", "pattern_c", "pattern_d"],
                            "description": "감지된 체크박스 패턴"
                        },
                        "extraction_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "체크박스 추출 신뢰도"
                        },
                        "checkbox_fields_found": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "발견된 체크박스 필드들"
                        }
                    }
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
    def get_general_document_schema_enhanced() -> Dict[str, Any]:
        """Enhanced general document schema with universal checkbox support"""
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
                
                # Universal checkbox section
                "checkbox_data": {
                    "type": "object",
                    "properties": {
                        "status_indicators": {
                            "type": "object",
                            "description": "상태 표시 (승인, 대기, 완료 등)"
                        },
                        "priority_levels": {
                            "type": "object",
                            "description": "우선순위 (높음, 보통, 낮음 등)"
                        },
                        "category_selections": {
                            "type": "object",
                            "description": "카테고리 선택 (유형별 분류)"
                        },
                        "approval_states": {
                            "type": "object",
                            "description": "승인 상태 (승인, 거부, 검토중 등)"
                        },
                        "service_options": {
                            "type": "object",
                            "description": "서비스 옵션 (기본, 프리미엄, 기업 등)"
                        },
                        "contact_preferences": {
                            "type": "object",
                            "description": "연락처 선호도 (이메일, 전화, SMS 등)"
                        }
                    },
                    "description": "문서 내 체크박스 데이터"
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
                },
                
                # Checkbox processing information
                "checkbox_info": {
                    "type": "object",
                    "properties": {
                        "pattern_detected": {
                            "type": "string",
                            "enum": ["pattern_a", "pattern_b", "pattern_c", "pattern_d"],
                            "description": "감지된 체크박스 패턴"
                        },
                        "extraction_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "체크박스 추출 신뢰도"
                        },
                        "checkbox_fields_found": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "발견된 체크박스 필드들"
                        }
                    }
                }
            },
            "required": ["document_type", "title"]
        }
    
    @staticmethod
    def get_copyright_transfer_consent_schema() -> Dict[str, Any]:
        """Schema for copyright transfer consent forms (저작재산권 양도동의서)"""
        return {
            "type": "object",
            "properties": {
                # Basic document information
                "document_type": {
                    "type": "string",
                    "description": "문서 유형 (저작재산권 양도동의서)"
                },
                "document_title": {
                    "type": "string",
                    "description": "문서 제목"
                },
                "work_category": {
                    "type": "string",
                    "description": "작품 카테고리 (출판도서, 음악, 미술 등)"
                },
                
                # Work information
                "work_info": {
                    "type": "object",
                    "properties": {
                        "work_title": {
                            "type": "string",
                            "description": "저작물 제목"
                        },
                        "work_subtitle": {
                            "type": "string",
                            "description": "저작물 부제목"
                        },
                        "work_series": {
                            "type": "string",
                            "description": "작품 시리즈 (세계속담, 우리옛이야기 등)"
                        },
                        "publication_year": {
                            "type": "string",
                            "description": "출판년도"
                        },
                        "work_type": {
                            "type": "string",
                            "enum": ["도서", "음악", "미술", "영상", "기타"],
                            "description": "저작물 유형"
                        }
                    },
                    "required": ["work_title", "work_type"]
                },
                
                # Copyright transfer information
                "copyright_transfer": {
                    "type": "object",
                    "properties": {
                        "transfer_type": {
                            "type": "string",
                            "enum": ["전체양도", "부분양도", "이용허락"],
                            "description": "양도 유형"
                        },
                        "transfer_scope": {
                            "type": "object",
                            "properties": {
                                "reproduction_right": {"type": "boolean", "description": "복제권"},
                                "performance_right": {"type": "boolean", "description": "공연권"},
                                "broadcasting_right": {"type": "boolean", "description": "공중송신권"},
                                "exhibition_right": {"type": "boolean", "description": "전시권"},
                                "distribution_right": {"type": "boolean", "description": "배포권"},
                                "rental_right": {"type": "boolean", "description": "대여권"},
                                "derivative_work_right": {"type": "boolean", "description": "2차적저작물작성권"},
                                "moral_rights": {"type": "boolean", "description": "인격권"}
                            },
                            "description": "양도 범위 (체크박스 기반)"
                        },
                        "transfer_conditions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "양도 조건"
                        },
                        "compensation": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number", "description": "보상 금액"},
                                "currency": {"type": "string", "description": "통화"},
                                "payment_method": {"type": "string", "description": "지급 방법"},
                                "payment_schedule": {"type": "string", "description": "지급 일정"}
                            }
                        }
                    },
                    "required": ["transfer_type", "transfer_scope"]
                },
                
                # Public Nuri License integration
                "public_nuri_license": {
                    "type": "object",
                    "properties": {
                        "nuri_type": {
                            "type": "string",
                            "enum": ["제1유형", "제2유형", "제3유형", "제4유형"],
                            "description": "공공누리 유형"
                        },
                        "license_conditions": {
                            "type": "object",
                            "properties": {
                                "attribution_required": {"type": "boolean", "description": "저작자표시"},
                                "commercial_use": {"type": "boolean", "description": "상업적이용"},
                                "modification_allowed": {"type": "boolean", "description": "변경허용"},
                                "share_alike": {"type": "boolean", "description": "동일조건변경허락"}
                            }
                        },
                        "license_duration": {
                            "type": "string",
                            "description": "라이선스 기간"
                        }
                    }
                },
                
                # Consent information
                "consent_info": {
                    "type": "object",
                    "properties": {
                        "consent_status": {
                            "type": "string",
                            "enum": ["동의함", "동의하지 않음", "조건부동의"],
                            "description": "동의 상태"
                        },
                        "consent_date": {
                            "type": "string",
                            "format": "date",
                            "description": "동의일 (YYYY-MM-DD)"
                        },
                        "consent_scope": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "동의 범위"
                        },
                        "withdrawal_conditions": {
                            "type": "string",
                            "description": "동의 철회 조건"
                        }
                    },
                    "required": ["consent_status", "consent_date"]
                },
                
                # Parties information
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "당사자 이름"},
                            "role": {
                                "type": "string",
                                "enum": ["저작자", "출판사", "기관", "대리인", "증인"],
                                "description": "역할"
                            },
                            "organization": {"type": "string", "description": "소속 기관"},
                            "phone": {"type": "string", "description": "전화번호"},
                            "address": {"type": "string", "description": "주소"},
                            "email": {"type": "string", "description": "이메일"},
                            "registration_no": {"type": "string", "description": "사업자등록번호"}
                        },
                        "required": ["name", "role"]
                    },
                    "description": "관련 당사자들"
                },
                
                # Contract terms
                "contract_terms": {
                    "type": "object",
                    "properties": {
                        "effective_date": {
                            "type": "string",
                            "format": "date",
                            "description": "계약 효력 발생일"
                        },
                        "expiration_date": {
                            "type": "string",
                            "format": "date",
                            "description": "계약 만료일"
                        },
                        "territory": {
                            "type": "string",
                            "description": "적용 지역"
                        },
                        "language": {
                            "type": "string",
                            "description": "적용 언어"
                        },
                        "special_conditions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "특별 조건"
                        },
                        "termination_conditions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "계약 해지 조건"
                        }
                    }
                },
                
                # Checkbox processing information
                "checkbox_info": {
                    "type": "object",
                    "properties": {
                        "pattern_detected": {
                            "type": "string",
                            "enum": ["pattern_a", "pattern_b", "pattern_c", "pattern_d"],
                            "description": "감지된 체크박스 패턴"
                        },
                        "extraction_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "체크박스 추출 신뢰도"
                        },
                        "checkbox_fields_found": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "발견된 체크박스 필드들"
                        }
                    }
                }
            },
            "required": ["document_type", "work_info", "copyright_transfer", "consent_info", "parties"]
        }
    
    @staticmethod
    def get_public_copyright_consent_schema_enhanced() -> Dict[str, Any]:
        """Enhanced schema with flexible checkbox support for public copyright consent forms"""
        return {
            "type": "object",
            "properties": {
                # Basic consent information
                "consent_type": {
                    "type": "string",
                    "description": "동의서 유형 (공공저작물 자유이용허락 동의서)"
                },
                "data_controller": {
                    "type": "string",
                    "description": "개인정보 처리자 (기관명)"
                },
                "data_subject": {
                    "type": "string",
                    "description": "정보주체 (동의자)"
                },
                
                # Work Display Section (저작물 표시) - Enhanced
                "work_display": {
                    "type": "object",
                    "properties": {
                        "work_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "저작물명 목록"
                        },
                        "institution": {
                            "type": "string",
                            "description": "기관명 (국립극단 등)"
                        },
                        "work_category": {
                            "type": "string",
                            "description": "저작물 종별 (우대미술, 의상디자인 등)"
                        },
                        "work_details": {
                            "type": "object",
                            "properties": {
                                "stage": {"type": "boolean", "description": "무대"},
                                "lighting": {"type": "boolean", "description": "장치"},
                                "costume": {"type": "boolean", "description": "의상"},
                                "accessories": {"type": "boolean", "description": "장신구"},
                                "props": {"type": "boolean", "description": "소품"},
                                "meditation": {"type": "boolean", "description": "명상"},
                                "sound": {"type": "boolean", "description": "음향"},
                                "video": {"type": "boolean", "description": "영상"},
                                "lighting_equipment": {"type": "boolean", "description": "조명"}
                            },
                            "description": "저작물 상세 구성 요소 (체크박스 상태)"
                        },
                        "detailed_info": {
                            "type": "string",
                            "description": "상세정보 (별지 저작물 목록 등)"
                        }
                    },
                    "required": ["work_names", "work_category"]
                },
                
                # Copyright License Section (저작재산권 이용허락 동의) - Enhanced
                "copyright_license": {
                    "type": "object",
                    "properties": {
                        "license_purpose": {
                            "type": "string",
                            "description": "이용허락 목적"
                        },
                        "licensing_institution": {
                            "type": "string",
                            "description": "이용허락 기관 (국립극장 등)"
                        },
                        "granted_rights": {
                            "type": "object",
                            "properties": {
                                "reproduction_right": {"type": "boolean", "description": "복제권 (목제권 포함)"},
                                "performance_right": {"type": "boolean", "description": "공연권 (공면권 포함)"},
                                "broadcasting_right": {"type": "boolean", "description": "공중송신권"},
                                "exhibition_right": {"type": "boolean", "description": "전시권"},
                                "distribution_right": {"type": "boolean", "description": "배포권"},
                                "rental_right": {"type": "boolean", "description": "대여권"},
                                "derivative_work_right": {"type": "boolean", "description": "2차적저작물작성권"}
                            },
                            "description": "허락된 저작재산권 (체크박스 상태)"
                        },
                        "license_type": {
                            "type": "string",
                            "description": "이용허락 유형 (독점적/비독점적)"
                        }
                    },
                    "required": ["license_purpose", "licensing_institution", "granted_rights"]
                },
                
                # Public Nuri License Section (공공누리 적용 동의) - Enhanced
                "public_nuri_license": {
                    "type": "object",
                    "properties": {
                        "license_purpose": {
                            "type": "string",
                            "description": "공공누리 적용 목적"
                        },
                        "nuri_type": {
                            "type": "string",
                            "enum": ["제1유형", "제2유형", "제3유형", "제4유형"],
                            "description": "선택된 공공누리 유형 (체크박스 상태)"
                        },
                        "available_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "사용 가능한 공공누리 유형들"
                        },
                        "modification_rights": {
                            "type": "object",
                            "properties": {
                                "integrity_right_waiver": {"type": "boolean", "description": "동일성유지권 행사 제안 동의"},
                                "modification_allowed": {"type": "boolean", "description": "변경 이용 가능 여부"},
                                "conditions": {"type": "string", "description": "변경 이용 조건"}
                            },
                            "description": "저작물 변경 권리"
                        }
                    },
                    "required": ["nuri_type", "license_purpose"]
                },
                
                # Personal Information Section (개인정보 제공 동의)
                "personal_info_consent": {
                    "type": "object",
                    "properties": {
                        "collected_data_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "수집하는 개인정보 항목"
                        },
                        "collection_purpose": {
                            "type": "string",
                            "description": "개인정보 수집 이용목적"
                        },
                        "retention_period": {
                            "type": "string",
                            "description": "개인정보 보유, 이용기간"
                        }
                    },
                    "required": ["collected_data_types", "collection_purpose", "retention_period"]
                },
                
                # Dates and Signatures
                "consent_date": {
                    "type": "string",
                    "format": "date",
                    "description": "동의일 (YYYY-MM-DD)"
                },
                "signature": {
                    "type": "string",
                    "description": "서명자 정보"
                },
                "utilizing_institution": {
                    "type": "string",
                    "description": "활용기관"
                },
                
                # Parties information
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "당사자 이름"},
                            "phone": {"type": "string", "description": "전화번호"},
                            "address": {"type": "string", "description": "주소"},
                            "role": {"type": "string", "description": "역할 (저작자, 활용기관 등)"}
                        },
                        "required": ["name", "role"]
                    },
                    "description": "관련 당사자들"
                },
                
                # Processing Information
                "processing_info": {
                    "type": "object",
                    "properties": {
                        "checkbox_pattern_detected": {
                            "type": "string",
                            "enum": ["pattern_a", "pattern_b", "pattern_c", "pattern_d"],
                            "description": "감지된 체크박스 패턴"
                        },
                        "extraction_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "추출 신뢰도"
                        }
                    }
                }
            },
            "required": ["consent_type", "work_display", "copyright_license", "public_nuri_license"]
        }
    
    @staticmethod
    def get_digital_content_schema() -> Dict[str, Any]:
        """Legacy alias — returns unified schema"""
        return DocumentSchemas.get_unified_schema()

    @staticmethod
    def get_unified_schema() -> Dict[str, Any]:
        """Unified schema for ALL document types (계약서, 동의서, 양도동의서, 공공저작물, 기타문서).

        Combines the 20 mandatory metadata fields with contract/consent-specific
        fields and digital content fields. The LLM returns null for fields not
        present in the document — no harm, no confusion.
        """
        return {
            "type": "object",
            "properties": {
                # ============================================
                # 저작물 정보 (8 mandatory fields)
                # ============================================
                "work_title": {
                    "type": ["string", "null"],
                    "description": "저작물명 (작품 제목, 계약서 제목, 문서 제목)"
                },
                "work_type": {
                    "type": ["string", "null"],
                    "description": "유형 (저작물 유형: 어문저작물, 음악저작물, 미술저작물, 영상저작물, 사진저작물 등)"
                },
                "digital_format": {
                    "type": ["string", "null"],
                    "description": "디지털화 형태 (디지털 파일 형식: PDF, JPG, MP4, HWP 등)"
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "설명 (콘텐츠 설명, 계약 목적, 동의 내용 요약)"
                },
                "keyword": {
                    "type": ["string", "array", "null"],
                    "description": "주제어 (키워드 또는 태그 목록)"
                },
                "language": {
                    "type": ["string", "null"],
                    "description": "언어 (콘텐츠 언어: 한국어, 영어 등)"
                },
                "created_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "제작일/작성일 (YYYY-MM-DD 형식)"
                },
                "production_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "제작일 (YYYY-MM-DD 형식, created_date와 같을 수 있음)"
                },

                # ============================================
                # 저작자 정보 (3 mandatory fields)
                # ============================================
                "copyright_holder": {
                    "type": ["string", "array", "null"],
                    "description": "저작권자 (저작권 보유자, 권리자, 양수기관 등)"
                },
                "co_author": {
                    "type": ["string", "array", "null"],
                    "description": "공동저작자 (공동 저작자 목록)"
                },
                "neighboring_rights_holder": {
                    "type": ["string", "array", "null"],
                    "description": "저작인접권자 (실연자, 음반제작자, 방송사업자 등)"
                },

                # ============================================
                # 권리 정보 (9 mandatory fields)
                # ============================================
                "disclosure_type": {
                    "type": ["string", "null"],
                    "description": "공개유형 (공개, 비공개, 제한공개 등)"
                },
                "copyrightability": {
                    "type": ["string", "null"],
                    "description": "저작물성 (저작물로 인정되는지 여부: 인정, 미인정, 불명)"
                },
                "unprotected_work": {
                    "type": ["string", "null"],
                    "description": "비보호저작물 (저작권 보호 대상이 아닌 저작물 여부: 해당, 비해당)"
                },
                "work_for_hire": {
                    "type": ["string", "null"],
                    "description": "업무상저작물 (업무상 작성된 저작물 여부: 해당, 비해당)"
                },
                "commercial_use": {
                    "type": ["string", "null"],
                    "description": "상업적 이용허락 (상업적 이용 허용 여부 또는 조건)"
                },
                "economic_rights": {
                    "type": ["string", "array", "null"],
                    "description": "저작재산권 (복제권, 공연권, 공중송신권, 배포권, 대여권, 2차적저작물작성권 등)"
                },
                "co_author_consent": {
                    "type": ["string", "null"],
                    "description": "공동저작자 동의 (공동저작자 동의 여부: 동의, 미동의)"
                },
                "valid_period": {
                    "type": ["string", "null"],
                    "description": "유효기간 (계약 기간, 보존 기간, 이용허락 기간 등)"
                },
                "portrait_rights": {
                    "type": ["string", "null"],
                    "description": "초상권 (초상권 관련 정보: 해당, 비해당)"
                },

                # ============================================
                # 계약서 전용 필드 (from contract schemas)
                # ============================================
                "contract_type": {
                    "type": ["string", "null"],
                    "description": "계약서/문서 유형 (저작재산권 이용허락 계약서, 개인정보 수집 동의서, 양도동의서 등)"
                },
                "granted_rights": {
                    "type": ["object", "array", "null"],
                    "description": "허락된 권리 (복제권, 공연권, 공중송신권, 전시권, 배포권, 대여권, 2차적저작물작성권의 체크박스 상태 또는 목록)"
                },
                "contract_duration": {
                    "type": ["string", "null"],
                    "description": "계약 기간 (시작일~종료일 또는 기간 문자열)"
                },
                "payment_amount": {
                    "type": ["number", "string", "null"],
                    "description": "지급 금액"
                },
                "payment_currency": {
                    "type": ["string", "null"],
                    "description": "통화 (원, 달러 등)"
                },
                "signature_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "서명일/계약 체결일 (YYYY-MM-DD)"
                },
                "effective_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "계약 효력 발생일 (YYYY-MM-DD)"
                },
                "expiration_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "계약 만료일 (YYYY-MM-DD)"
                },
                "special_terms": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "특별 약정 사항"
                },
                "termination_conditions": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "계약 해지 조건"
                },
                "contract_terms": {
                    "type": ["object", "null"],
                    "description": "계약 조건 (독점/비독점, 결제 방식, 갱신 옵션 등의 체크박스 상태)"
                },

                # ============================================
                # 동의서 전용 필드 (from consent schemas)
                # ============================================
                "consent_type": {
                    "type": ["string", "null"],
                    "description": "동의서 유형 (개인정보 수집 동의서, 저작재산권 양도 동의서 등)"
                },
                "consent_status": {
                    "type": ["string", "null"],
                    "description": "동의 여부 (동의함, 미동의 등)"
                },
                "consent_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "동의 날짜 (YYYY-MM-DD)"
                },
                "data_controller": {
                    "type": ["string", "null"],
                    "description": "개인정보 처리자/수집기관 (회사명 또는 기관명)"
                },
                "data_subject": {
                    "type": ["string", "null"],
                    "description": "정보주체/동의자 (개인 이름)"
                },
                "collection_purpose": {
                    "type": ["string", "null"],
                    "description": "수집 목적"
                },
                "collected_data_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "수집 항목 (성명, 전화번호, 주소 등)"
                },
                "retention_period": {
                    "type": ["string", "null"],
                    "description": "보존 기간"
                },
                "third_party_sharing": {
                    "type": ["object", "null"],
                    "description": "제3자 제공 (수령자, 목적, 제공 항목)"
                },
                "withdrawal_rights": {
                    "type": ["string", "null"],
                    "description": "동의 철회 방법"
                },
                "consequences_of_refusal": {
                    "type": ["string", "null"],
                    "description": "동의 거부 시 불이익"
                },
                "signature": {
                    "type": ["string", "null"],
                    "description": "서명 (서명자 이름)"
                },

                # ============================================
                # 당사자 정보 (shared across all types)
                # ============================================
                "parties": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "당사자 이름 또는 회사명"},
                            "phone": {"type": "string", "description": "전화번호"},
                            "address": {"type": "string", "description": "주소"},
                            "registration_no": {"type": "string", "description": "사업자등록번호 또는 주민등록번호"},
                            "role": {"type": "string", "description": "역할 (권리자, 이용자, 양도인, 양수인, 동의자 등)"}
                        },
                        "required": ["name"]
                    },
                    "description": "당사자들의 정보 (계약 당사자, 양도인/양수인, 동의자/수집자 등)"
                },
                "contact_info": {
                    "type": ["object", "null"],
                    "description": "연락처 정보 (전화번호, 주소, 이메일)"
                },

                # ============================================
                # 공공누리/디지털콘텐츠 필드
                # ============================================
                "kogl_type": {
                    "type": ["string", "null"],
                    "description": "공공누리유형 (KOGL 라이선스 유형: 제1유형, 제2유형, 제3유형, 제4유형)"
                },
                "third_party_rights": {
                    "type": ["string", "array", "null"],
                    "description": "제3자 권리 (제3자에게 귀속된 권리 정보)"
                },
                "personal_info": {
                    "type": ["string", "array", "null"],
                    "description": "개인정보 (개인정보 포함 여부 또는 항목)"
                },

                # ============================================
                # 디지털콘텐츠 관리 필드 (supplementary)
                # ============================================
                "seq_number": {
                    "type": ["integer", "string", "null"],
                    "description": "순번"
                },
                "site_name": {
                    "type": ["string", "null"],
                    "description": "사이트명"
                },
                "agency_name": {
                    "type": ["string", "null"],
                    "description": "기관명"
                },
                "board_name": {
                    "type": ["string", "null"],
                    "description": "게시판명"
                },
                "board_path": {
                    "type": ["string", "null"],
                    "description": "게시판 진입 과정"
                },
                "category": {
                    "type": ["string", "null"],
                    "description": "카테고리"
                },
                "url": {
                    "type": ["string", "null"],
                    "format": "uri",
                    "description": "URL"
                },
                "registration_date": {
                    "type": ["string", "null"],
                    "format": "date",
                    "description": "등록일 (YYYY-MM-DD)"
                },
                "attachment": {
                    "type": ["string", "array", "null"],
                    "description": "첨부파일"
                },
                "video_count": {"type": ["integer", "null"], "description": "영상 개수"},
                "photo_count": {"type": ["integer", "null"], "description": "사진 개수"},
                "document_count": {"type": ["integer", "null"], "description": "문서 개수"},
                "quantity": {"type": ["integer", "string", "null"], "description": "수량"},
                "view_count": {"type": ["integer", "null"], "description": "조회수"},
                "contract": {
                    "type": ["string", "object", "null"],
                    "description": "계약서 정보"
                },
                "review_impossible": {
                    "type": ["string", "null"],
                    "description": "검토불가 사유"
                },
                "phone": {
                    "type": ["string", "null"],
                    "description": "전화번호"
                },
                "memo": {
                    "type": ["string", "null"],
                    "description": "비고"
                },

                # ============================================
                # 체크박스 정보 (shared)
                # ============================================
                "checkbox_info": {
                    "type": ["object", "null"],
                    "properties": {
                        "pattern_detected": {"type": "string", "description": "감지된 체크박스 패턴"},
                        "extraction_confidence": {"type": "number", "description": "체크박스 추출 신뢰도"},
                        "checkbox_fields_found": {"type": "array", "items": {"type": "string"}, "description": "발견된 체크박스 필드들"}
                    },
                    "description": "체크박스 추출 정보"
                }
            },
            "required": ["work_title"]
        }
    
    @staticmethod
    def get_schema_by_document_type(document_type: str) -> Dict[str, Any]:
        """Get schema for any document type.

        Returns the unified schema for all document types. The unified schema
        contains all fields from all document-type-specific schemas. The LLM
        simply returns null for fields not present in the document.

        The document_type parameter is still accepted for backward compatibility
        and is passed to the LLM prompt to guide extraction focus, but the
        schema structure is the same regardless.
        """
        return DocumentSchemas.get_unified_schema()
    
    @staticmethod
    def detect_document_type_from_title(title: str) -> str:
        """Detect document type from document title"""
        
        # Copyright Transfer Consent
        if any(keyword in title for keyword in [
            "저작재산권 양도동의서", "양도동의서", "copyright transfer", "transfer consent"
        ]):
            return "저작재산권 양도동의서"
        
        # Public Copyright Consent
        elif any(keyword in title for keyword in [
            "공공저작물 자유이용허락 동의서", "공공누리", "public copyright"
        ]):
            return "공공저작물 자유이용허락 동의서"
        
        # Regular Contracts
        elif any(keyword in title for keyword in [
            "계약서", "contract", "agreement"
        ]):
            return "계약서"
        
        # Regular Consent Forms
        elif any(keyword in title for keyword in [
            "동의서", "consent", "agreement"
        ]):
            return "동의서"
        
        # Digital Content Management
        elif any(keyword in title for keyword in [
            "디지털 콘텐츠", "공공저작물", "콘텐츠 관리", "아카이브", "게시판", "사이트",
            "digital content", "public content", "kogl", "공공누리"
        ]):
            return "디지털 콘텐츠"
        
        # Default
        else:
            return "기타문서"

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
    
    # Test enhanced schemas
    print("\nEnhanced Schemas:")
    enhanced_contract = DocumentSchemas.get_contract_schema_enhanced()
    enhanced_consent = DocumentSchemas.get_consent_schema_enhanced()
    copyright_transfer = DocumentSchemas.get_copyright_transfer_consent_schema()
    public_copyright = DocumentSchemas.get_public_copyright_consent_schema_enhanced()
    
    print(f"Enhanced Contract Schema: {len(enhanced_contract['properties'])} properties")
    print(f"Enhanced Consent Schema: {len(enhanced_consent['properties'])} properties")
    print(f"Copyright Transfer Schema: {len(copyright_transfer['properties'])} properties")
    print(f"Public Copyright Schema: {len(public_copyright['properties'])} properties")
    
    # Test schema selection with enhanced detection
    test_types = [
        "계약서", 
        "동의서", 
        "기타문서",
        "저작재산권 양도동의서",
        "공공저작물 자유이용허락 동의서",
        "[05 출판도서] 공공누리 저작권 계약서(세계속담. 우리옛이야기) - 지노랩"
    ]
    
    for doc_type in test_types:
        schema = DocumentSchemas.get_schema_by_document_type(doc_type)
        detected_type = DocumentSchemas.detect_document_type_from_title(doc_type)
        print(f"\nDocument: '{doc_type}'")
        print(f"Detected Type: '{detected_type}'")
        print(f"Schema Properties: {len(schema['properties'])} properties")
        print(f"Required Fields: {schema.get('required', [])}")
