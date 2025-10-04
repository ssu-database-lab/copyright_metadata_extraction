#!/usr/bin/env python3
"""
Document-Specific Metadata Extractors
Handles different document types with specialized extraction logic
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from models.base_extractor import BaseLLMExtractor, ExtractionResult
from schemas.document_schemas import DocumentSchemas

logger = logging.getLogger(__name__)

class ContractExtractor:
    """Specialized extractor for contract documents (계약서)"""
    
    def __init__(self, llm_extractor: BaseLLMExtractor):
        self.llm_extractor = llm_extractor
        self.schema = DocumentSchemas.get_contract_schema()
    
    def extract_contract_metadata(self, text: str, document_name: str = "") -> ExtractionResult:
        """Extract metadata from contract documents"""
        logger.info(f"Extracting contract metadata from: {document_name}")
        
        # Preprocess text for better extraction
        processed_text = self._preprocess_contract_text(text)
        
        # Extract metadata using LLM
        result = self.llm_extractor.extract_metadata(
            processed_text, 
            self.schema, 
            "계약서"
        )
        
        # Post-process results
        result.metadata = self._postprocess_contract_metadata(result.metadata)
        
        return result
    
    def _preprocess_contract_text(self, text: str) -> str:
        """Preprocess contract text for better extraction"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Add section markers for better parsing
        text = text.replace("제1조", "\n제1조")
        text = text.replace("제2조", "\n제2조")
        text = text.replace("제3조", "\n제3조")
        text = text.replace("제4조", "\n제4조")
        text = text.replace("제5조", "\n제5조")
        
        return text
    
    def _postprocess_contract_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extracted contract metadata"""
        # Clean up extracted data
        if "rights_holder" in metadata and metadata["rights_holder"]:
            metadata["rights_holder"] = metadata["rights_holder"].strip()
        
        if "user" in metadata and metadata["user"]:
            metadata["user"] = metadata["user"].strip()
        
        if "work_title" in metadata and metadata["work_title"]:
            metadata["work_title"] = metadata["work_title"].strip()
        
        # Convert payment amount to number if possible
        if "payment_amount" in metadata and isinstance(metadata["payment_amount"], str):
            try:
                # Extract numbers from string
                import re
                numbers = re.findall(r'\d+', metadata["payment_amount"])
                if numbers:
                    metadata["payment_amount"] = int(numbers[0])
            except:
                pass
        
        return metadata

class ConsentExtractor:
    """Specialized extractor for consent forms (동의서)"""
    
    def __init__(self, llm_extractor: BaseLLMExtractor):
        self.llm_extractor = llm_extractor
        self.schema = DocumentSchemas.get_consent_schema()
    
    def extract_consent_metadata(self, text: str, document_name: str = "") -> ExtractionResult:
        """Extract metadata from consent forms"""
        logger.info(f"Extracting consent metadata from: {document_name}")
        
        # Preprocess text for better extraction
        processed_text = self._preprocess_consent_text(text)
        
        # Extract metadata using LLM
        result = self.llm_extractor.extract_metadata(
            processed_text, 
            self.schema, 
            "동의서"
        )
        
        # Post-process results
        result.metadata = self._postprocess_consent_metadata(result.metadata)
        
        return result
    
    def _preprocess_consent_text(self, text: str) -> str:
        """Preprocess consent text for better extraction"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Add section markers
        text = text.replace("○", "\n○")
        text = text.replace("동의함", "\n동의함")
        text = text.replace("동의하지 않음", "\n동의하지 않음")
        
        return text
    
    def _postprocess_consent_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process extracted consent metadata"""
        # Clean up extracted data
        if "data_controller" in metadata and metadata["data_controller"]:
            metadata["data_controller"] = metadata["data_controller"].strip()
        
        if "consent_status" in metadata and metadata["consent_status"]:
            status = metadata["consent_status"].strip()
            if "동의함" in status or "✓" in status:
                metadata["consent_status"] = "동의함"
            elif "동의하지 않음" in status or "✗" in status:
                metadata["consent_status"] = "동의하지 않음"
            else:
                metadata["consent_status"] = "null"
        
        return metadata

class DocumentMetadataExtractor:
    """Main extractor that handles different document types"""
    
    def __init__(self, llm_extractor: BaseLLMExtractor):
        self.llm_extractor = llm_extractor
        self.contract_extractor = ContractExtractor(llm_extractor)
        self.consent_extractor = ConsentExtractor(llm_extractor)
    
    def extract_metadata(self, text: str, document_type: str, document_name: str = "") -> ExtractionResult:
        """Extract metadata based on document type"""
        logger.info(f"Extracting metadata for {document_type}: {document_name}")
        
        if "계약서" in document_type or "contract" in document_type.lower():
            return self.contract_extractor.extract_contract_metadata(text, document_name)
        elif "동의서" in document_type or "consent" in document_type.lower():
            return self.consent_extractor.extract_consent_metadata(text, document_name)
        else:
            # Use general schema for unknown document types
            schema = DocumentSchemas.get_general_document_schema()
            return self.llm_extractor.extract_metadata(text, schema, document_type)
    
    def batch_extract_from_ocr_results(self, ocr_results_dir: str, output_dir: str) -> List[ExtractionResult]:
        """Extract metadata from all OCR results in a directory"""
        ocr_path = Path(ocr_results_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process each provider directory
        for provider_dir in ocr_path.iterdir():
            if not provider_dir.is_dir():
                continue
            
            provider_name = provider_dir.name
            logger.info(f"Processing OCR results from: {provider_name}")
            
            # Process each category directory
            for category_dir in provider_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                category = category_dir.name
                logger.info(f"Processing category: {category}")
                
                # Process each document directory
                for doc_dir in category_dir.iterdir():
                    if not doc_dir.is_dir():
                        continue
                    
                    doc_name = doc_dir.name
                    logger.info(f"Processing document: {doc_name}")
                    
                    # Find extracted text file
                    text_file = doc_dir / f"{doc_name}_extracted_text.txt"
                    
                    if text_file.exists():
                        try:
                            # Read OCR text
                            with open(text_file, 'r', encoding='utf-8') as f:
                                ocr_text = f.read()
                            
                            # Extract metadata
                            result = self.extract_metadata(ocr_text, category, doc_name)
                            
                            # Save result
                            result_file = output_path / f"{provider_name}_{category}_{doc_name}_metadata.json"
                            with open(result_file, 'w', encoding='utf-8') as f:
                                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
                            
                            results.append(result)
                            logger.info(f"✅ Extracted metadata for: {doc_name}")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to extract metadata for {doc_name}: {e}")
                    else:
                        logger.warning(f"⚠️ No extracted text file found for: {doc_name}")
        
        return results

# Example usage
if __name__ == "__main__":
    from models.base_extractor import create_extractor
    
    # Create LLM extractor
    llm_extractor = create_extractor("solar-ko")
    
    # Create document extractor
    doc_extractor = DocumentMetadataExtractor(llm_extractor)
    
    # Test with sample text
    sample_contract_text = """
    저작재산권 비독점적 이용허락 계약서
    
    저작자 및 저작권 이용허락자 집건에 (이하 "권리자" 이라 함)와 
    저작권 이용자 국립생태원 멸종위기종복원센터(이하 "이용자" 이라 함)는 
    아래 저작물 멸종위기 야생생물 대국민 온라인 홍보물 제작에 관한 
    저작재산권 이용허락과 관련하여 다음과 같이 계약을 체결한다.
    """
    
    result = doc_extractor.extract_metadata(sample_contract_text, "계약서", "test_contract")
    print(f"Extraction Result: {result}")
