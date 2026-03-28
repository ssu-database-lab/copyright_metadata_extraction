#!/usr/bin/env python3
"""
LLM Extraction Processor for API Integration
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models.base_extractor import create_extractor, ExtractionResult
from .extractors.document_extractors import DocumentMetadataExtractor
from .schemas.document_schemas import DocumentSchemas

logger = logging.getLogger(__name__)

class LLMExtractionProcessor:
    """LLM-based metadata extraction processor for API integration"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("llm_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available models
        self.available_models = {
            # Qwen3.5 (current generation — recommended)
            "alibaba-qwen3.5-122b-a10b": "Qwen3.5-122B 10B active (Cloud, recommended)",
            "alibaba-qwen3.5-plus": "Qwen3.5-Plus 397B (Cloud, flagship)",
            "alibaba-qwen3.5-flash": "Qwen3.5-Flash 35B (Cloud, cost-effective)",
            # Qwen3 (previous generation)
            "alibaba-qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B (Cloud)",
            "alibaba-qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B (Cloud Vision)",
            "alibaba-qwen3-max": "Qwen3-Max (Cloud)",
        }

        # Model configurations
        self.model_configs = {
            "alibaba-qwen3.5-122b-a10b": {"name": "Qwen3.5-122B", "description": "Best value 122B model (recommended)", "type": "cloud"},
            "alibaba-qwen3.5-plus": {"name": "Qwen3.5-Plus", "description": "Flagship 397B model", "type": "cloud"},
            "alibaba-qwen3.5-flash": {"name": "Qwen3.5-Flash", "description": "Cost-effective 35B model", "type": "cloud"},
            "alibaba-qwen3-next-80b-a3b-instruct": {"name": "Qwen3-Next-80B", "description": "Previous generation", "type": "cloud"},
            "alibaba-qwen3-vl-235b-a22b-instruct": {"name": "Qwen3-VL-235B", "description": "Vision-language model", "type": "cloud"},
            "alibaba-qwen3-max": {"name": "Qwen3-Max", "description": "Previous flagship", "type": "cloud"},
        }
        
        self.extractor = None
        self.doc_extractor = None
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models"""
        return self.model_configs.copy()
    
    def initialize_model(self, model_name: str) -> bool:
        """Initialize the specified model"""
        try:
            logger.info(f"Initializing LLM model: {model_name}")
            
            # Create extractor
            self.extractor = create_extractor(model_name, "config/model_config.yaml")
            
            # Create document extractor
            self.doc_extractor = DocumentMetadataExtractor(self.extractor)
            
            logger.info(f"✅ LLM model {model_name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM model {model_name}: {e}")
            return False
    
    def extract_metadata_from_text(self, text: str, document_type: str = "기타문서", 
                                 document_name: str = "", model_name: str = "alibaba-qwen3.5-122b-a10b") -> Dict[str, Any]:
        """Extract metadata from text using LLM"""
        try:
            # Initialize model if not already done or different model requested
            if not self.extractor or self.current_model != model_name:
                if not self.initialize_model(model_name):
                    return {
                        "success": False,
                        "error": f"Failed to initialize model: {model_name}",
                        "metadata": {},
                        "confidence": 0.0,
                        "extraction_time": 0.0,
                        "model_used": model_name
                    }
                self.current_model = model_name
            
            # Extract metadata
            result = self.doc_extractor.extract_metadata(text, document_type, document_name)
            
            return {
                "success": True,
                "metadata": result.metadata,
                "confidence": result.confidence,
                "extraction_time": result.extraction_time,
                "model_used": result.model_used,
                "document_type": result.document_type,
                "raw_response": result.raw_response,
                "error": result.error
            }
            
        except Exception as e:
            logger.error(f"Error during LLM metadata extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {},
                "confidence": 0.0,
                "extraction_time": 0.0,
                "model_used": model_name
            }
    
    def extract_metadata_from_file(self, file_path: str, model_name: str = "alibaba-qwen3.5-122b-a10b") -> Dict[str, Any]:
        """Extract metadata from a text file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "metadata": {},
                    "confidence": 0.0,
                    "extraction_time": 0.0,
                    "model_used": model_name
                }
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Detect document type from filename
            document_name = file_path.stem
            document_type = self._detect_document_type(document_name, text)
            
            # Extract metadata
            result = self.extract_metadata_from_text(text, document_type, document_name, model_name)
            
            # Save result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.output_dir / f"{model_name}_{timestamp}_{document_name}_metadata.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            result["result_file"] = str(result_file)
            return result
            
        except Exception as e:
            logger.error(f"Error extracting metadata from file: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {},
                "confidence": 0.0,
                "extraction_time": 0.0,
                "model_used": model_name
            }
    
    def batch_extract_from_directory(self, input_dir: str, model_name: str = "alibaba-qwen3.5-122b-a10b") -> Dict[str, Any]:
        """Extract metadata from all text files in a directory"""
        try:
            input_path = Path(input_dir)
            
            if not input_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {input_dir}",
                    "results": [],
                    "total_files": 0,
                    "successful_extractions": 0,
                    "failed_extractions": 0
                }
            
            # Find all text files
            text_files = list(input_path.glob("**/*.txt"))
            
            if not text_files:
                return {
                    "success": False,
                    "error": "No text files found in directory",
                    "results": [],
                    "total_files": 0,
                    "successful_extractions": 0,
                    "failed_extractions": 0
                }
            
            results = []
            successful = 0
            failed = 0
            
            # Process each file
            for file_path in text_files:
                try:
                    result = self.extract_metadata_from_file(str(file_path), model_name)
                    results.append(result)
                    
                    if result["success"]:
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "file_path": str(file_path),
                        "metadata": {},
                        "confidence": 0.0,
                        "extraction_time": 0.0,
                        "model_used": model_name
                    })
                    failed += 1
            
            # Save batch results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_result_file = self.output_dir / f"{model_name}_{timestamp}_batch_results.json"
            
            batch_summary = {
                "model_used": model_name,
                "extraction_time": datetime.now().isoformat(),
                "total_files": len(text_files),
                "successful_extractions": successful,
                "failed_extractions": failed,
                "results": results
            }
            
            with open(batch_result_file, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "results": results,
                "total_files": len(text_files),
                "successful_extractions": successful,
                "failed_extractions": failed,
                "batch_result_file": str(batch_result_file)
            }
            
        except Exception as e:
            logger.error(f"Error in batch extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_files": 0,
                "successful_extractions": 0,
                "failed_extractions": 0
            }
    
    def _detect_document_type(self, filename: str, content: str) -> str:
        """Detect document type from filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Check filename for document type indicators
        if any(keyword in filename_lower for keyword in ['계약서', 'contract']):
            return "계약서"
        elif any(keyword in filename_lower for keyword in ['동의서', 'consent']):
            return "동의서"
        elif any(keyword in filename_lower for keyword in ['양도', 'transfer']):
            return "저작재산권 양도동의서"
        elif any(keyword in filename_lower for keyword in ['공공', 'public']):
            return "공공저작물 자유이용허락 동의서"
        
        # Check content for document type indicators
        if any(keyword in content_lower for keyword in ['계약서', '계약', 'contract']):
            return "계약서"
        elif any(keyword in content_lower for keyword in ['동의서', '동의', 'consent']):
            return "동의서"
        elif any(keyword in content_lower for keyword in ['양도', 'transfer']):
            return "저작재산권 양도동의서"
        elif any(keyword in content_lower for keyword in ['공공저작물', '공공', 'public']):
            return "공공저작물 자유이용허락 동의서"
        
        # Default to general document
        return "기타문서"
    
    def get_schema_for_document_type(self, document_type: str) -> Dict[str, Any]:
        """Get JSON schema for a specific document type"""
        return DocumentSchemas.get_schema_by_document_type(document_type)
    
    def test_extraction(self, model_name: str = "alibaba-qwen3.5-122b-a10b") -> Dict[str, Any]:
        """Test the extraction pipeline with sample data"""
        sample_text = """
        저작재산권 비독점적 이용허락 계약서
        
        저작자 및 저작권 이용허락자 집건에 (이하 "권리자" 이라 함)와 
        저작권 이용자 국립생태원 멸종위기종복원센터(이하 "이용자" 이라 함)는 
        아래 저작물 멸종위기 야생생물 대국민 온라인 홍보물 제작에 관한 
        저작재산권 이용허락과 관련하여 다음과 같이 계약을 체결한다.
        
        제1조 (계약의 목적)
        본 계약은 저작재산권 이용허락과 관련하여 권리자와 이용자 사이의 권리관계를 명확히 하는 것을 목적으로 한다.
        
        제2조 (계약의 대상)
        본 계약의 이용허락 대상이 되는 권리는 아래의 저작물에 대한 저작재산권 중 당사자가 합의한 권리로 한다.
        
        제목: 멸종위기 야생생물 대국민 온라인 홍보물 제작
        종별: 어문저작물, 사진저작물
        권리: 복제권, 공중송신권
        """
        
        return self.extract_metadata_from_text(sample_text, "계약서", "test_contract", model_name)
