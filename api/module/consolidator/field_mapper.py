#!/usr/bin/env python3
"""
Field Mapper - Maps NER entities to LLM metadata fields

This module provides intelligent mapping between NER entity types
and LLM metadata field names.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class FieldMapper:
    """Maps NER entities to LLM metadata fields"""
    
    def __init__(self):
        """Initialize field mapper with mapping rules"""
        self.entity_to_field_map = self._initialize_mappings()
        self.field_priority = self._initialize_priorities()
    
    def _initialize_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize entity type to LLM field mappings
        
        Returns:
            Dictionary mapping NER entity types to possible LLM field names
        """
        return {
            # Name entities
            'NAME': [
                'rights_holder',
                'user',
                'data_controller',
                'data_subject',
                'parties[].name',
                # Digital content schema fields
                'copyright_holder',
                'co_author'
            ],
            
            # Date entities
            'DATE': [
                'signature_date',
                'effective_date',
                'expiration_date',
                'contract_duration',
                'consent_date',
                'retention_period',
                # Digital content schema fields
                'created_date',
                'registration_date',
                'production_date',
                'valid_period'
            ],
            
            # Phone entities
            'PHONE': [
                'parties[].phone',
                'contact_info.phone',
                'phone',  # Digital content schema field
                'telephone'
            ],
            
            # Email entities
            'EMAIL': [
                'parties[].email',
                'contact_info.email',
                'email'
            ],
            
            # Address entities
            'ADDRESS': [
                'parties[].address',
                'contact_info.address',
                'address',
                'registered_address'
            ],
            
            # Company/Organization entities
            'COMPANY': [
                'user',
                'rights_holder',
                'data_controller',
                'parties[].company',
                'organization',
                # Digital content schema fields
                'agency_name',
                'site_name',
                'board_name'
            ],
            
            # Money entities
            'MONEY': [
                'payment_amount',
                'price',
                'fee',
                'compensation'
            ],
            
            # Position/Role entities
            'POSITION': [
                'parties[].role',
                'position',
                'title'
            ],
            
            # ID Number entities
            'ID_NUM': [
                'parties[].registration_no',
                'parties[].id_number',
                'registration_no',
                'business_registration_no'
            ],
            
            # Title entities
            'TITLE': [
                'work_title',
                'document_title',
                'title'
            ],
            
            # URL entities
            'URL': [
                'website',
                'url',  # Digital content schema field
                'homepage'
            ],
            
            # Description entities
            'DESCRIPTION': [
                'contract_purpose',
                'collection_purpose',
                'description',  # Digital content schema field
                'remarks',
                'memo'  # Digital content schema field (비고)
            ],
            
            # Contract/Consent types
            'CONTRACT_TYPE': ['contract_type'],
            'CONSENT_TYPE': ['consent_type'],
            
            # Period entities (기간)
            'PERIOD': [
                'contract_duration',
                'valid_period',  # Digital content schema field (유효기간)
                'retention_period'
            ],
            
            # Right/Authority information (권리 정보)
            'RIGHT_INFO': [
                'granted_rights',
                'economic_rights',  # Digital content schema field (저작재산권)
                'third_party_rights',  # Digital content schema field (제3자 권리)
                'portrait_rights',  # Digital content schema field (초상권)
                'neighboring_rights_holder'  # Digital content schema field (저작인접권자)
            ],
            
            # Project/Business name (프로젝트명, 사업명)
            'PROJECT_NAME': [
                'work_title',  # Can be project/work name
                'title',
                'document_title'
            ],
            
            # Law/Regulation reference (법령, 조항)
            'LAW_REFERENCE': [
                'special_terms',  # May contain law references
                'important_terms',
                'termination_conditions'
            ],
            
            # Type/Category (유형, 종류)
            'TYPE': [
                'work_type',  # Digital content schema field (유형)
                'work_category',
                'category',  # Digital content schema field (카테고리)
                'document_type',
                'contract_type',
                'consent_type'
            ],
            
            # Status (상태, 진행현황)
            'STATUS': [
                'review_impossible',  # Digital content schema field (검토불가)
                'disclosure_type'  # Digital content schema field (공개유형)
            ],
            
            # Department (담당부서)
            'DEPARTMENT': [
                'agency_name',  # Digital content schema field (기관명)
                'data_controller',
                'organization'
            ],
            
            # Language (언어)
            'LANGUAGE': [
                'language'  # Digital content schema field (언어)
            ],
            
            # Quantity (수량, 분량)
            'QUANTITY': [
                'quantity',  # Digital content schema field (수량)
                'video_count',  # Digital content schema field (영상)
                'photo_count',  # Digital content schema field (사진)
                'document_count',  # Digital content schema field (문서)
                'view_count'  # Digital content schema field (조회수)
            ],
        }
    
    def _initialize_priorities(self) -> Dict[str, Dict[str, int]]:
        """
        Initialize priority scores for field mappings
        Higher priority = more likely match
        
        Returns:
            Dictionary of priority scores
        """
        return {
            'NAME': {
                'rights_holder': 10,
                'user': 9,
                'data_controller': 10,
                'data_subject': 9,
                'parties[].name': 5
            },
            'DATE': {
                'signature_date': 10,
                'effective_date': 8,
                'expiration_date': 7,
                'consent_date': 10,
                'contract_duration': 5,
                # Digital content schema priorities
                'created_date': 9,
                'registration_date': 8,
                'production_date': 7,
                'valid_period': 6
            },
            'PHONE': {
                'parties[].phone': 10,
                'contact_info.phone': 9,
                'phone': 8
            },
            'COMPANY': {
                'user': 10,
                'rights_holder': 9,
                'data_controller': 10,
                'parties[].company': 5,
                # Digital content schema priorities
                'agency_name': 10,
                'site_name': 9,
                'board_name': 8
            },
            'NAME': {
                'rights_holder': 10,
                'user': 9,
                'data_controller': 10,
                'data_subject': 9,
                'parties[].name': 5,
                # Digital content schema priorities
                'copyright_holder': 10,
                'co_author': 8
            },
            'DESCRIPTION': {
                'description': 10,  # Digital content schema field
                'contract_purpose': 9,
                'collection_purpose': 9,
                'memo': 8  # Digital content schema field
            },
            'PERIOD': {
                'valid_period': 10,  # Digital content schema field
                'contract_duration': 9,
                'retention_period': 8
            },
            'RIGHT_INFO': {
                'economic_rights': 10,  # Digital content schema field
                'granted_rights': 9,
                'third_party_rights': 8,  # Digital content schema field
                'portrait_rights': 7,  # Digital content schema field
                'neighboring_rights_holder': 6  # Digital content schema field
            },
            'TYPE': {
                'work_type': 10,  # Digital content schema field
                'category': 9,  # Digital content schema field
                'work_category': 8,
                'document_type': 7
            },
            'STATUS': {
                'disclosure_type': 10,  # Digital content schema field
                'review_impossible': 8  # Digital content schema field
            },
            'DEPARTMENT': {
                'agency_name': 10,  # Digital content schema field
                'data_controller': 9,
                'organization': 7
            },
            'LANGUAGE': {
                'language': 10  # Digital content schema field
            },
            'QUANTITY': {
                'quantity': 10,  # Digital content schema field
                'video_count': 9,  # Digital content schema field
                'photo_count': 9,  # Digital content schema field
                'document_count': 9,  # Digital content schema field
                'view_count': 8  # Digital content schema field
            }
        }
    
    def map_entities_to_fields(
        self, 
        ner_entities: List[Tuple[str, str]], 
        llm_metadata: Dict[str, Any],
        ocr_text: str = "",
        document_type: str = ""
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Map NER entities to LLM metadata fields
        
        Args:
            ner_entities: List of (entity_text, entity_type) tuples
            llm_metadata: LLM extracted metadata dictionary
            ocr_text: Original OCR text for context
            document_type: Type of document (계약서, 동의서, etc.)
        
        Returns:
            Dictionary mapping LLM field names to list of (entity_text, confidence) tuples
        """
        field_mappings = defaultdict(list)
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity_text, entity_type in ner_entities:
            entities_by_type[entity_type].append(entity_text)
        
        # Map each entity type to possible fields
        for entity_type, entities in entities_by_type.items():
            if entity_type not in self.entity_to_field_map:
                logger.warning(f"Unknown entity type: {entity_type}")
                continue
            
            # Get possible field names for this entity type
            possible_fields = self.entity_to_field_map[entity_type]
            
            # For each entity, find best matching field
            for entity_text in entities:
                best_field, confidence = self._find_best_field_match(
                    entity_text,
                    entity_type,
                    possible_fields,
                    llm_metadata,
                    ocr_text,
                    document_type
                )
                
                if best_field:
                    field_mappings[best_field].append((entity_text, confidence))
        
        return dict(field_mappings)
    
    def _find_best_field_match(
        self,
        entity_text: str,
        entity_type: str,
        possible_fields: List[str],
        llm_metadata: Dict[str, Any],
        ocr_text: str,
        document_type: str
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching field for an entity
        
        Args:
            entity_text: The entity text value
            entity_type: Type of entity (NAME, DATE, etc.)
            possible_fields: List of possible field names
            llm_metadata: LLM metadata to check against
            ocr_text: OCR text for context
            document_type: Document type
        
        Returns:
            Tuple of (best_field_name, confidence_score)
        """
        best_field = None
        best_confidence = 0.0
        
        for field in possible_fields:
            confidence = self._calculate_field_confidence(
                entity_text,
                entity_type,
                field,
                llm_metadata,
                ocr_text,
                document_type
            )
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_field = field
        
        # Only return if confidence is above threshold
        if best_confidence >= 0.3:
            return best_field, best_confidence
        else:
            return None, 0.0
    
    def _calculate_field_confidence(
        self,
        entity_text: str,
        entity_type: str,
        field_name: str,
        llm_metadata: Dict[str, Any],
        ocr_text: str,
        document_type: str
    ) -> float:
        """
        Calculate confidence score for a field match
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Base priority score
        if entity_type in self.field_priority:
            priorities = self.field_priority[entity_type]
            if field_name in priorities:
                confidence += priorities[field_name] / 10.0
        
        # Check if LLM metadata already has a value in this field
        llm_value = self._get_nested_value(llm_metadata, field_name)
        if llm_value:
            # Check if values match (exact or fuzzy)
            if self._values_match(entity_text, llm_value):
                confidence += 0.4  # High boost for exact match
            else:
                confidence += 0.1  # Small boost for field existing
        
        # Context-based scoring (simplified for Phase 1)
        # Phase 2 will add more sophisticated context matching
        
        return min(confidence, 1.0)
    
    def _get_nested_value(self, metadata: Dict[str, Any], field_path: str) -> Optional[Any]:
        """
        Get value from nested dictionary using dot notation or array notation
        
        Examples:
            "rights_holder" -> metadata["rights_holder"]
            "parties[].name" -> checks all items in parties array
        """
        if '[]' in field_path:
            # Array notation - check all items
            base_field, sub_field = field_path.split('[].')
            if base_field in metadata and isinstance(metadata[base_field], list):
                values = []
                for item in metadata[base_field]:
                    if isinstance(item, dict) and sub_field in item:
                        values.append(item[sub_field])
                return values[0] if values else None
        
        # Simple dot notation
        parts = field_path.split('.')
        value = metadata
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    def _values_match(self, value1: str, value2: Any) -> bool:
        """
        Check if two values match (exact or fuzzy)
        
        Phase 1: Simple exact/partial matching
        Phase 2: Will add fuzzy matching with OCR error tolerance
        """
        if value1 is None or value2 is None:
            return False
        
        # Convert to string
        str1 = str(value1).strip().lower()
        str2 = str(value2).strip().lower()
        
        # Exact match
        if str1 == str2:
            return True
        
        # Partial match (one contains the other)
        if str1 in str2 or str2 in str1:
            return True
        
        # For dates, try normalized comparison
        if re.match(r'\d{4}-\d{2}-\d{2}', str1) and re.match(r'\d{4}-\d{2}-\d{2}', str2):
            return str1 == str2
        
        return False

