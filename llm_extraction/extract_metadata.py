#!/usr/bin/env python3
"""
Main Script for Korean Document Metadata Extraction
Processes OCR results and extracts structured metadata using open-source LLMs
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.base_extractor import create_extractor
from extractors.document_extractors import DocumentMetadataExtractor
from schemas.document_schemas import DocumentSchemas
from models.cloud_extractor import load_env_file

# Load environment variables from .env file
load_env_file()

# Configure logging
logger = logging.getLogger(__name__)

class MetadataExtractionPipeline:
    """Main pipeline for metadata extraction"""
    
    def __init__(self, model_name: str = "solar-ko", config_path: str = "config/model_config.yaml"):
        self.model_name = model_name
        self.config_path = config_path
        self.extractor = None
        self.doc_extractor = None
        
    def initialize(self):
        """Initialize the extraction pipeline"""
        logger.info(f"Initializing metadata extraction pipeline with model: {self.model_name}")
        
        try:
            # Create LLM extractor
            self.extractor = create_extractor(self.model_name, self.config_path)
            
            # Create document extractor
            self.doc_extractor = DocumentMetadataExtractor(self.extractor)
            
            logger.info("✅ Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    def extract_single_document(self, text: str, document_type: str, document_name: str = "") -> Dict[str, Any]:
        """Extract metadata from a single document"""
        logger.info(f"Extracting metadata for single document: {document_name}")
        
        try:
            result = self.doc_extractor.extract_metadata(text, document_type, document_name)
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return {
                "document_type": document_type,
                "metadata": {},
                "confidence": 0.0,
                "extraction_time": 0.0,
                "model_used": self.model_name,
                "error": str(e)
            }
    
    def batch_extract_from_ocr_results(self, ocr_results_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Extract metadata from all OCR results"""
        logger.info(f"Starting batch extraction from: {ocr_results_dir}")
        
        # Create timestamped model-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = Path(output_dir) / f"{self.model_name}_{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            results = self.doc_extractor.batch_extract_from_ocr_results(ocr_results_dir, str(model_output_dir))
            
            # Convert to dictionaries
            result_dicts = [result.model_dump() for result in results]
            
            # Save summary
            summary_file = model_output_dir / "extraction_summary.json"
            summary = {
                "total_documents": len(result_dicts),
                "successful_extractions": len([r for r in result_dicts if not r.get('error')]),
                "failed_extractions": len([r for r in result_dicts if r.get('error')]),
                "average_confidence": sum(r.get('confidence', 0) for r in result_dicts) / len(result_dicts) if result_dicts else 0,
                "model_used": self.model_name,
                "extraction_time": datetime.now().isoformat(),
                "results": result_dicts
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Batch extraction completed. Results saved to: {model_output_dir}")
            logger.info(f"📊 Summary: {summary['successful_extractions']}/{summary['total_documents']} successful")
            
            return result_dicts
            
        except Exception as e:
            logger.error(f"❌ Batch extraction failed: {e}")
            raise
    
    def extract_from_single_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Extract metadata from a single OCR text file"""
        file_path = Path(file_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.txt':
            raise ValueError(f"File must be a .txt file: {file_path}")
        
        # Create timestamped model-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = output_path / f"{self.model_name}_{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Read OCR text
            with open(file_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
            
            # Determine document type from filename or content
            document_name = file_path.stem
            document_type = self._detect_document_type(document_name, ocr_text)
            
            logger.info(f"Processing single file: {file_path.name}")
            logger.info(f"Detected document type: {document_type}")
            
            # Extract metadata
            result = self.doc_extractor.extract_metadata(ocr_text, document_type, document_name)
            
            # Save result
            result_file = model_output_dir / f"{document_name}_metadata.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
            
            # Save summary
            summary_file = model_output_dir / "extraction_summary.json"
            summary = {
                "total_documents": 1,
                "successful_extractions": 1 if not result.error else 0,
                "failed_extractions": 1 if result.error else 0,
                "average_confidence": result.confidence,
                "model_used": self.model_name,
                "extraction_time": datetime.now().isoformat(),
                "results": [result.model_dump()]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Single file extraction completed. Results saved to: {model_output_dir}")
            logger.info(f"📊 Summary: {summary['successful_extractions']}/{summary['total_documents']} successful")
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"❌ Single file extraction failed: {e}")
            raise
    
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
    
    def test_extraction(self):
        """Test the extraction pipeline with sample data"""
        logger.info("Testing extraction pipeline...")
        
        # Sample contract text
        sample_contract = """
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
        
        # Sample consent text
        sample_consent = """
        개인정보 수집 및 이용 동의서
        
        ○ 개인정보 수집 및 이용 목적: 저작인접권의 저작재산권 양도 의사표시 확인 및 초상 공개·사용 등 의사표시 확인
        ○ 수집하는 개인정보 항목: 성명, 전화번호(휴대전화), 주소
        ○ 개인정보 보유 및 이용 기간: 동의 시부터 저작인접권 보호기간 만료일까지
        ○ 개인정보 수집 및 이용에 동의하지 않을 권리가 있으며, 본 동의를 거절하실 경우에는 저작물의 제작이 불가함을 알려드립니다.
        
        양도자 본인은 개인정보 수집 및 이용에 동의합니다. 동의함 ✓ 동의하지 않음 □
        """
        
        try:
            # Test contract extraction
            logger.info("Testing contract extraction...")
            contract_result = self.extract_single_document(sample_contract, "계약서", "test_contract")
            logger.info(f"Contract extraction result: {contract_result}")
            
            # Test consent extraction
            logger.info("Testing consent extraction...")
            consent_result = self.extract_single_document(sample_consent, "동의서", "test_consent")
            logger.info(f"Consent extraction result: {consent_result}")
            
            logger.info("✅ Test extraction completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Test extraction failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Korean Document Metadata Extraction using Open-Source LLMs")
    
    parser.add_argument(
        "--model", 
        default="alibaba-qwen-plus",
        choices=["solar-ko", "qwen", "lightweight", "llama", "qwen72b", "qwenvl", "qwen3", "qwen3-next", "qwen3-30b", "qwen3-235b", "gemma3", "mixtral",
                "alibaba-qwen-plus", "alibaba-qwen-max", "alibaba-qwen-turbo", "alibaba-qwen-vl-plus",
                "alibaba-qwen3-next-80b-a3b-instruct", "alibaba-qwen3-vl-235b-a22b-instruct", "alibaba-qwen3-235b-a22b-instruct-2507"],
        help="LLM model to use (default: qwen3-235b). Alibaba Cloud models require DASHSCOPE_API_KEY environment variable."
    )
    
    parser.add_argument(
        "--ocr-results-dir",
        type=str,
        help="Directory containing OCR results"
    )
    
    parser.add_argument(
        "--single-file",
        type=str,
        help="Single OCR text file to process"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metadata_results",
        help="Output directory for extracted metadata (default: metadata_results)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test extraction with sample data"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/metadata_extraction_{args.model}_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration of logging
    )
    
    logger.info(f"Logging configured. Log file: {log_file}")
    
    # Initialize pipeline
    pipeline = MetadataExtractionPipeline(args.model, args.config)
    
    try:
        pipeline.initialize()
        
        if args.test:
            # Run test extraction
            pipeline.test_extraction()
        
        elif args.ocr_results_dir:
            # Run batch extraction
            if not Path(args.ocr_results_dir).exists():
                logger.error(f"OCR results directory not found: {args.ocr_results_dir}")
                sys.exit(1)
            
            pipeline.batch_extract_from_ocr_results(args.ocr_results_dir, args.output_dir)
        
        elif args.single_file:
            # Run single file extraction
            if not Path(args.single_file).exists():
                logger.error(f"Single file not found: {args.single_file}")
                sys.exit(1)
            
            pipeline.extract_from_single_file(args.single_file, args.output_dir)
        
        else:
            logger.info("No action specified. Use --test for testing, --ocr-results-dir for batch extraction, or --single-file for single file processing.")
            logger.info("Available options:")
            logger.info("  --test: Run test extraction")
            logger.info("  --ocr-results-dir: Process OCR results directory")
            logger.info("  --single-file: Process single OCR text file")
            logger.info("  --model: Choose model:")
            logger.info("    Local models: solar-ko, qwen, lightweight, llama, qwen72b, qwenvl, qwen3, qwen3-next, qwen3-30b, qwen3-235b, gemma3, mixtral")
            logger.info("    Alibaba Cloud: alibaba-qwen-plus, alibaba-qwen-max, alibaba-qwen-turbo, alibaba-qwen-vl-plus")
            logger.info("    Alibaba Qwen3: alibaba-qwen3-next-80b-a3b-instruct, alibaba-qwen3-vl-235b-a22b-instruct, alibaba-qwen3-235b-a22b-instruct-2507")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
