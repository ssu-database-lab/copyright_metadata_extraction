#!/usr/bin/env python3
"""
Validation Engine - Validates metadata formats and logical consistency

This module provides validation for consolidated metadata including
format validation, logical consistency checks, and cross-validation.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationEngine:
    """Validates metadata formats and logical consistency"""
    
    def __init__(self):
        """Initialize validation engine with validation rules"""
        self.date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        self.phone_pattern = re.compile(r'^[0-9\-]+$')
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    def validate_format(self, field_name: str, value: Any, field_type: str = None) -> Tuple[bool, Optional[str]]:
        """
        Validate field format
        
        Args:
            field_name: Name of the field
            value: Value to validate
            field_type: Type of field (auto-detected if None)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            return True, None  # null values are valid
        
        # Auto-detect field type from field name
        if field_type is None:
            field_type = self._detect_field_type(field_name)
        
        # Convert to string for validation
        str_value = str(value).strip()
        
        if field_type == 'date':
            return self._validate_date(str_value)
        elif field_type == 'phone':
            return self._validate_phone(str_value)
        elif field_type == 'email':
            return self._validate_email(str_value)
        elif field_type == 'money':
            return self._validate_money(value)
        elif field_type == 'string':
            return self._validate_string(str_value)
        
        return True, None
    
    def validate_logic(self, metadata: Dict[str, Any], document_type: str = "") -> List[str]:
        """
        Validate logical consistency of metadata
        
        Args:
            metadata: Metadata dictionary to validate
            document_type: Type of document
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Date range validation
        date_errors = self._validate_date_ranges(metadata)
        errors.extend(date_errors)
        
        # Required field validation (basic)
        required_errors = self._validate_required_fields(metadata, document_type)
        errors.extend(required_errors)
        
        # Logical consistency checks
        logic_errors = self._validate_logical_consistency(metadata, document_type)
        errors.extend(logic_errors)
        
        return errors
    
    def check_consistency(
        self, 
        llm_value: Any, 
        ner_value: Any
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check consistency between LLM and NER values
        
        Args:
            llm_value: Value from LLM extraction
            ner_value: Value from NER extraction
        
        Returns:
            Tuple of (is_consistent, confidence, explanation)
        """
        if llm_value is None and ner_value is None:
            return True, 1.0, "Both values are null"
        
        if llm_value is None or ner_value is None:
            return False, 0.5, "One value is missing"
        
        # Convert to strings for comparison
        llm_str = str(llm_value).strip()
        ner_str = str(ner_value).strip()
        
        # Exact match
        if llm_str == ner_str:
            return True, 1.0, "Values match exactly"
        
        # Normalized comparison (lowercase)
        if llm_str.lower() == ner_str.lower():
            return True, 0.95, "Values match (case-insensitive)"
        
        # Partial match (one contains the other)
        if llm_str in ner_str or ner_str in llm_str:
            return True, 0.8, "Values partially match"
        
        # Date format normalization
        if self._normalize_date(llm_str) == self._normalize_date(ner_str):
            return True, 0.9, "Dates match after normalization"
        
        # Mismatch
        return False, 0.0, f"Values differ: LLM='{llm_str}', NER='{ner_str}'"
    
    def _detect_field_type(self, field_name: str) -> str:
        """Auto-detect field type from field name"""
        field_lower = field_name.lower()
        
        if 'date' in field_lower or '날짜' in field_name:
            return 'date'
        elif 'phone' in field_lower or '전화' in field_name:
            return 'phone'
        elif 'email' in field_lower or '이메일' in field_name:
            return 'email'
        elif 'amount' in field_lower or 'payment' in field_lower or '금액' in field_name:
            return 'money'
        else:
            return 'string'
    
    def _validate_date(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate date format (YYYY-MM-DD)"""
        if not self.date_pattern.match(value):
            return False, f"Invalid date format: {value}. Expected YYYY-MM-DD"
        
        try:
            datetime.strptime(value, '%Y-%m-%d')
            return True, None
        except ValueError:
            return False, f"Invalid date: {value}"
    
    def _validate_phone(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate phone number format"""
        if not self.phone_pattern.match(value):
            return False, f"Invalid phone format: {value}. Expected digits and hyphens only"
        
        # Check length (Korean phone numbers: 10-15 digits including hyphens)
        digits_only = re.sub(r'[^0-9]', '', value)
        if len(digits_only) < 10 or len(digits_only) > 15:
            return False, f"Invalid phone length: {value}"
        
        return True, None
    
    def _validate_email(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate email format"""
        if not self.email_pattern.match(value):
            return False, f"Invalid email format: {value}"
        return True, None
    
    def _validate_money(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate money/numeric value"""
        try:
            float(value)
            return True, None
        except (ValueError, TypeError):
            return False, f"Invalid money value: {value}. Expected numeric"
    
    def _validate_string(self, value: str) -> Tuple[bool, Optional[str]]:
        """Validate string value (basic checks)"""
        if len(value) < 1:
            return False, "Empty string value"
        if len(value) > 10000:  # Arbitrary limit
            return False, "String value too long"
        return True, None
    
    def _validate_date_ranges(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate date range logic (e.g., start < end)"""
        errors = []
        
        # Check effective_date < expiration_date
        if 'effective_date' in metadata and 'expiration_date' in metadata:
            effective = metadata.get('effective_date')
            expiration = metadata.get('expiration_date')
            
            if effective and expiration:
                try:
                    eff_date = datetime.strptime(effective, '%Y-%m-%d')
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                    if eff_date >= exp_date:
                        errors.append(f"Effective date ({effective}) must be before expiration date ({expiration})")
                except (ValueError, TypeError):
                    pass  # Format errors handled elsewhere
        
        # Check signature_date <= effective_date (if both exist)
        if 'signature_date' in metadata and 'effective_date' in metadata:
            signature = metadata.get('signature_date')
            effective = metadata.get('effective_date')
            
            if signature and effective:
                try:
                    sig_date = datetime.strptime(signature, '%Y-%m-%d')
                    eff_date = datetime.strptime(effective, '%Y-%m-%d')
                    if sig_date > eff_date:
                        errors.append(f"Signature date ({signature}) should not be after effective date ({effective})")
                except (ValueError, TypeError):
                    pass
        
        return errors
    
    def _validate_required_fields(self, metadata: Dict[str, Any], document_type: str) -> List[str]:
        """Validate required fields based on document type"""
        errors = []
        
        # Basic required fields for contracts
        if '계약서' in document_type or 'contract' in document_type.lower():
            required = ['contract_type', 'rights_holder', 'user']
            for field in required:
                if field not in metadata or metadata[field] is None:
                    errors.append(f"Required field missing: {field}")
        
        # Basic required fields for consent forms
        elif '동의서' in document_type or 'consent' in document_type.lower():
            required = ['consent_type', 'data_controller']
            for field in required:
                if field not in metadata or metadata[field] is None:
                    errors.append(f"Required field missing: {field}")
        
        return errors
    
    def _validate_logical_consistency(self, metadata: Dict[str, Any], document_type: str) -> List[str]:
        """Validate logical consistency rules"""
        errors = []
        
        # Check payment amount is positive (if exists)
        if 'payment_amount' in metadata:
            amount = metadata['payment_amount']
            if amount is not None:
                try:
                    if float(amount) < 0:
                        errors.append("Payment amount cannot be negative")
                except (ValueError, TypeError):
                    pass
        
        # Additional logical checks can be added here
        
        return errors
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date string to YYYY-MM-DD format"""
        if not date_str:
            return None
        
        # Try to parse and normalize various date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y.%m.%d',
            '%d/%m/%Y',
            '%d-%m-%Y',
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None

