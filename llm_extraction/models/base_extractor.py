#!/usr/bin/env python3
"""
Base LLM Extractor for Korean Document Metadata Extraction
Supports multiple open-source LLM models with JSON schema-based extraction
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from pydantic import BaseModel, Field
import time

# Import model cache manager
from .model_cache import ModelCacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionResult(BaseModel):
    """Standardized extraction result format"""
    document_type: str
    metadata: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_time: float
    model_used: str
    raw_response: Optional[str] = None
    error: Optional[str] = None

class BaseLLMExtractor(ABC):
    """Abstract base class for LLM-based metadata extraction"""
    
    def __init__(self, model_config: Dict[str, Any], device: str = "auto", config_path: str = "config/model_config.yaml"):
        self.model_config = model_config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.cache_manager = ModelCacheManager(config_path)
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the specific model implementation"""
        pass
    
    @abstractmethod
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using the loaded model"""
        pass
    
    def _create_prompt(self, text: str, schema: Dict[str, Any], document_type: str) -> str:
        """Create a structured prompt for metadata extraction"""
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt = f"""당신은 한국어 문서(계약서, 동의서, 기타)에서 정보를 추출하는 도우미입니다.
반드시 유효한 JSON만 출력하세요. 설명·주석·마크다운·코드블록 금지.

다음은 {document_type} 문서의 OCR 텍스트입니다. 주어진 JSON 스키마에 따라 메타데이터를 추출해주세요.

문서 텍스트:
{text}

추출할 메타데이터 스키마:
{schema_str}

지시사항:
1. 텍스트에서 각 필드에 해당하는 정보를 정확히 찾아 추출하세요
2. 정보가 명시적으로 존재하지 않거나 불분명한 경우 반드시 null을 사용하세요 (추측 금지)
3. 날짜는 YYYY-MM-DD 형식으로 변환하세요
4. 금액은 숫자만 추출하세요 (단위 제외)
5. 전화번호는 숫자와 하이픈(-)만 포함하세요
6. 주소는 전체 주소를 정확히 추출하세요
7. 사업자등록번호는 숫자와 하이픈(-)만 포함하세요
8. 체크박스 정보 처리:
   - 체크박스가 체크된 상태(📧, ☑, ✓, ■, ●, ◼, ◉)인 경우 true로 설정
   - 체크박스가 체크되지 않은 상태(☐, □, ○, ◯, ◻, ◦)인 경우 false로 설정
   - 체크박스 패턴을 자동으로 감지하여 일관성 있게 처리
   - OCR 오류 고려: "목제권"은 "복제권"으로 해석
9. 반드시 유효한 JSON 형식으로 응답하세요
10. 추가 정보나 설명은 포함하지 마세요
11. ```json이나 ``` 같은 마크다운 문법 사용 금지

응답 (JSON만):

---
You are a helper that extracts information from Korean documents (contracts, consent forms, other documents).
Output only valid JSON. No explanations, comments, markdown, or code blocks allowed.

The following is OCR text from a {document_type} document. Extract metadata according to the given JSON schema.

Document text:
{text}

Metadata schema to extract:
{schema_str}

Instructions:
1. Find and extract information corresponding to each field in the text
2. Use null if information is not explicitly present or unclear (no guessing policy)
3. Convert dates to YYYY-MM-DD format
4. Extract only numbers for amounts (exclude units)
5. Include only numbers and hyphens(-) for phone numbers
6. Extract complete addresses accurately
7. Include only numbers and hyphens(-) for registration numbers
8. Checkbox information processing:
   - Set true for checked checkboxes (📧, ☑, ✓, ■, ●, ◼, ◉)
   - Set false for unchecked checkboxes (☐, □, ○, ◯, ◻, ◦)
   - Automatically detect checkbox patterns for consistent processing
   - Consider OCR errors: "목제권" should be interpreted as "복제권"
9. Respond only with valid JSON format
10. Do not include additional information or explanations
11. Do not use ```json or ``` markdown syntax

Response (JSON only):"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON"""
        try:
            # Clean up the response - remove markdown formatting
            cleaned_response = response.strip()
            
            # Remove markdown code blocks
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            # Split by ```json to get the first complete JSON block
            json_blocks = cleaned_response.split('```json')
            if len(json_blocks) > 1:
                # Take the first complete JSON block
                first_block = json_blocks[1].split('```')[0]
                cleaned_response = first_block.strip()
            
            # Try to find JSON in the response
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = cleaned_response[start_idx:end_idx]
                parsed_json = json.loads(json_str)
                
                # Validate that we got meaningful data
                if parsed_json and len(parsed_json) > 0:
                    return parsed_json
                else:
                    logger.warning("Empty JSON object found in response")
                    return {}
            else:
                logger.warning("No JSON found in response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response[:500]}...")  # Log first 500 chars
            return {}
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            logger.error(f"Response: {response[:500]}...")
            return {}
    
    def _calculate_confidence(self, metadata: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate extraction confidence based on completeness"""
        if not metadata:
            return 0.0
        
        total_fields = len(schema.get('properties', {}))
        filled_fields = sum(1 for v in metadata.values() if v is not None and v != "")
        
        if total_fields == 0:
            return 1.0
        
        return min(filled_fields / total_fields, 1.0)

class SOLARKoExtractor(BaseLLMExtractor):
    """SOLAR-Ko model extractor"""
    
    def _load_model(self):
        """Load SOLAR-Ko model using cache manager"""
        model_id = self.model_config['model_id']
        max_length = self.model_config.get('max_length', 4096)
        
        logger.info(f"Loading SOLAR-Ko model: {model_id}")
        
        try:
            # Get cached model path
            model_path = self.cache_manager.get_model_path('primary')
            
            logger.info(f"Using model from cache: {model_path}")
            
            # Load tokenizer from cached path
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model from cached path
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=max_length,
                temperature=self.model_config.get('temperature', 0.1),
                top_p=self.model_config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("SOLAR-Ko model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SOLAR-Ko model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using SOLAR-Ko"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                num_return_sequences=1,
                truncation=True
            )[0]['generated_text']
            
            # Extract the generated part (remove prompt)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(metadata, schema)
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Qwen2.5-7B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Qwen2.5-7B",
                error=str(e)
            )

class QwenExtractor(BaseLLMExtractor):
    """Qwen2.5 model extractor"""
    
    def _load_model(self):
        """Load Qwen2.5 model using cache manager"""
        model_id = self.model_config['model_id']
        max_length = self.model_config.get('max_length', 4096)
        
        logger.info(f"Loading Qwen2.5 model: {model_id}")
        
        try:
            # Get cached model path
            model_path = self.cache_manager.get_model_path('secondary')
            
            logger.info(f"Using model from cache: {model_path}")
            
            # Load tokenizer from cached path
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model from cached path
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=max_length,
                temperature=self.model_config.get('temperature', 0.1),
                top_p=self.model_config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Qwen2.5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5 model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Qwen2.5"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                num_return_sequences=1,
                truncation=True
            )[0]['generated_text']
            
            # Extract the generated part
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(metadata, schema)
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Qwen2.5-7B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Qwen2.5-7B",
                error=str(e)
            )

class LlamaExtractor(BaseLLMExtractor):
    """Llama 3.1-70B model extractor"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self, model_id: str, local_path: str = None):
        """Load Llama 3.1-70B model using cache manager"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Llama 3.1-70B model: {model_id}")
            
            # Use cache manager if available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                model_path = self.cache_manager.get_model_path(model_id)
            else:
                model_path = model_id
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Device set to use {self.device}")
            
            logger.info("Llama 3.1-70B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama 3.1-70B model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Llama 3.1-70B"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.model_config.get('max_length', 8192),
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.model_config.get('temperature', 0.1),
                    top_p=self.model_config.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            confidence = self._calculate_confidence(metadata, schema)
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Llama-3.1-70B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Llama-3.1-70B",
                error=str(e)
            )

class Qwen72BExtractor(BaseLLMExtractor):
    """Qwen 2.5-72B model extractor"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.pipeline = None
        
    def _load_model(self):
        """Load Qwen 2.5-72B model"""
        self.load_model('qwen72b')
        
    def load_model(self, model_id: str, local_path: str = None):
        """Load Qwen 2.5-72B model using cache manager"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Qwen 2.5-72B model: {model_id}")
            
            # Use cache manager if available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                model_path = self.cache_manager.get_model_path(model_id)
            else:
                model_path = model_id
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Device set to use {self.device}")
            
            # Create pipeline for consistency with other extractors
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.model_config.get('max_length', 8192),
                temperature=self.model_config.get('temperature', 0.1),
                top_p=self.model_config.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Qwen 2.5-72B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen 2.5-72B model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Qwen 2.5-72B"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Generate response using pipeline
            response = self.pipeline(
                prompt,
                max_new_tokens=1024,
                num_return_sequences=1,
                truncation=True,
                temperature=self.model_config.get('temperature', 0.1),
                top_p=self.model_config.get('top_p', 0.9),
                do_sample=True
            )[0]['generated_text']
            
            # Extract the generated part (remove prompt)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(metadata, schema)
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Qwen2.5-72B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Qwen2.5-72B",
                error=str(e)
            )

class QwenVLExtractor(BaseLLMExtractor):
    """Qwen 2.5-VL-72B model extractor"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        """Load Qwen 2.5-VL-72B model"""
        self.load_model('qwenvl')
        
    def load_model(self, model_id: str, local_path: str = None):
        """Load Qwen 2.5-VL-72B model using cache manager"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Qwen 2.5-VL-72B model: {model_id}")
            
            # Use cache manager if available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                model_path = self.cache_manager.get_model_path(model_id)
            else:
                model_path = model_id
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Device set to use {self.device}")
            
            logger.info("Qwen 2.5-VL-72B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen 2.5-VL-72B model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Qwen 2.5-VL-72B"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.model_config.get('max_length', 8192),
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.model_config.get('temperature', 0.1),
                    top_p=self.model_config.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            confidence = self._calculate_confidence(metadata, schema)
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Qwen2.5-VL-72B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Qwen2.5-VL-72B",
                error=str(e)
            )

class Qwen3Extractor(BaseLLMExtractor):
    """Qwen3-4B model extractor"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        """Load Qwen3-4B model"""
        self.load_model('qwen3')
        
    def load_model(self, model_id: str, local_path: str = None):
        """Load Qwen3-4B model using cache manager"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Qwen3-4B model: {model_id}")
            
            # Use cache manager if available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                model_path = self.cache_manager.get_model_path(model_id)
            else:
                model_path = model_id
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Device set to use {self.device}")
            
            logger.info("Qwen3-4B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3-4B model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Qwen3-4B"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.model_config.get('max_length', 4096),
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.model_config.get('temperature', 0.1),
                    top_p=self.model_config.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            confidence = self._calculate_confidence(metadata, schema)
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Qwen3-4B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Qwen3-4B",
                error=str(e)
            )

class Gemma3Extractor(BaseLLMExtractor):
    """Gemma 3 12B model extractor"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        """Load Gemma 3 12B model"""
        self.load_model('gemma3_12b')
        
    def load_model(self, model_id: str, local_path: str = None):
        """Load Gemma 3 12B model using cache manager"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Gemma 3 12B model: {model_id}")
            
            # Use cache manager if available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                model_path = self.cache_manager.get_model_path(model_id)
            else:
                model_path = model_id
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Device set to use {self.device}")
            
            logger.info("Gemma 3 12B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma 3 12B model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Gemma 3 12B"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.model_config.get('max_length', 131072),
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.model_config.get('temperature', 0.1),
                    top_p=self.model_config.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            confidence = self._calculate_confidence(metadata, schema)
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Gemma3-12B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Gemma3-12B",
                error=str(e)
            )

class MixtralExtractor(BaseLLMExtractor):
    """Mixtral 8x7B model extractor"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        """Load Mixtral 8x7B model"""
        self.load_model('mixtral_8x7b')
        
    def load_model(self, model_id: str, local_path: str = None):
        """Load Mixtral 8x7B model using cache manager"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading Mixtral 8x7B model: {model_id}")
            
            # Use cache manager if available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                model_path = self.cache_manager.get_model_path(model_id)
            else:
                model_path = model_id
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Device set to use {self.device}")
            
            logger.info("Mixtral 8x7B model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Mixtral 8x7B model: {e}")
            raise
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using Mixtral 8x7B"""
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self._create_prompt(text, schema, document_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.model_config.get('max_length', 32768),
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=self.model_config.get('temperature', 0.1),
                    top_p=self.model_config.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # Parse response
            metadata = self._parse_response(generated_text)
            confidence = self._calculate_confidence(metadata, schema)
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=extraction_time,
                model_used="Mixtral-8x7B",
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error during metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used="Mixtral-8x7B",
                error=str(e)
            )

def create_extractor(model_name: str, config_path: str = "config/model_config.yaml") -> BaseLLMExtractor:
    """Factory function to create appropriate extractor"""
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if model_name.lower() == "solar-ko" or model_name.lower() == "solar":
        return SOLARKoExtractor(config['models']['primary'])
    elif model_name.lower() == "qwen" or model_name.lower() == "qwen2.5":
        return QwenExtractor(config['models']['secondary'])
    elif model_name.lower() == "lightweight" or model_name.lower() == "solar-ko-1.7b":
        return SOLARKoExtractor(config['models']['lightweight'])
    elif model_name.lower() == "llama" or model_name.lower() == "llama3.1":
        return LlamaExtractor(config['models']['llama'])
    elif model_name.lower() == "qwen72b" or model_name.lower() == "qwen2.5-72b":
        return Qwen72BExtractor(config['models']['qwen72b'])
    elif model_name.lower() == "qwenvl" or model_name.lower() == "qwen2.5-vl":
        return QwenVLExtractor(config['models']['qwenvl'])
    elif model_name.lower() == "qwen3" or model_name.lower() == "qwen3-4b":
        return Qwen3Extractor(config['models']['qwen3'])
    elif model_name.lower() == "qwen3-next" or model_name.lower() == "qwen3-next-80b":
        return Qwen3Extractor(config['models']['qwen3_next_80b'])
    elif model_name.lower() == "qwen3-30b":
        return Qwen3Extractor(config['models']['qwen3_30b'])
    elif model_name.lower() == "qwen3-235b":
        return Qwen3Extractor(config['models']['qwen3_235b'])
    elif model_name.lower() == "gemma3" or model_name.lower() == "gemma3-12b":
        return Gemma3Extractor(config['models']['gemma3_12b'])
    elif model_name.lower() == "mixtral" or model_name.lower() == "mixtral-8x7b":
        return MixtralExtractor(config['models']['mixtral_8x7b'])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    # Test the extractor
    extractor = create_extractor("solar-ko")
    
    test_text = """
    저작재산권 비독점적 이용허락 계약서
    
    저작자 및 저작권 이용허락자 집건에 (이하 "권리자" 이라 함)와 
    저작권 이용자 국립생태원 멸종위기종복원센터(이하 "이용자" 이라 함)는 
    아래 저작물 멸종위기 야생생물 대국민 온라인 홍보물 제작에 관한 
    저작재산권 이용허락과 관련하여 다음과 같이 계약을 체결한다.
    """
    
    test_schema = {
        "type": "object",
        "properties": {
            "contract_type": {"type": "string", "description": "계약서 유형"},
            "rights_holder": {"type": "string", "description": "권리자"},
            "user": {"type": "string", "description": "이용자"},
            "work_title": {"type": "string", "description": "저작물 제목"}
        }
    }
    
    result = extractor.extract_metadata(test_text, test_schema, "계약서")
    print(f"Extraction Result: {result}")
