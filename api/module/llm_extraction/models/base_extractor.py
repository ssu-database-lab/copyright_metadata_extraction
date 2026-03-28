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

# Import cloud extractors
from .cloud_extractor import create_cloud_extractor, create_extraction_prompt, load_env_file

# Load environment variables from .env file
load_env_file()

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

    # Default config path relative to this file: models/ → ../config/model_config.yaml
    _DEFAULT_CONFIG = str(Path(__file__).parent.parent / "config" / "model_config.yaml")

    def __init__(self, model_config: Dict[str, Any], device: str = "auto", config_path: str = None):
        self.model_config = model_config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.cache_manager = ModelCacheManager(config_path or self._DEFAULT_CONFIG)
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
        """Create a structured prompt for metadata extraction."""
        return create_extraction_prompt(text, schema, document_type)
    
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

class LocalModelExtractor(BaseLLMExtractor):
    """Unified local model extractor — replaces all model-specific subclasses.

    All local HuggingFace models share the same load/extract pattern.
    Differences (model name, cache key, max_new_tokens) are driven by
    ``model_config`` and the ``model_display_name`` / ``cache_key`` params.
    """

    def __init__(self, model_config: Dict[str, Any], device: str = "auto",
                 config_path: str = None,
                 model_display_name: str = None, cache_key: str = None):
        self.model_display_name = model_display_name or model_config.get('model_id', 'local-model')
        self.cache_key = cache_key or model_config.get('model_id', 'primary')
        super().__init__(model_config, device, config_path)

    def _load_model(self):
        """Load any HuggingFace causal-LM model using cache manager."""
        model_id = self.model_config['model_id']
        max_length = self.model_config.get('max_length', 4096)

        logger.info(f"Loading model: {self.model_display_name} ({model_id})")

        try:
            model_path = self.cache_manager.get_model_path(self.cache_key)
            logger.info(f"Using model from cache: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

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

            logger.info(f"{self.model_display_name} model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load {self.model_display_name} model: {e}")
            raise

    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using the loaded model."""
        start_time = time.time()
        max_new_tokens = self.model_config.get('max_new_tokens', 1024)

        try:
            prompt = self._create_prompt(text, schema, document_type)

            response = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                truncation=True
            )[0]['generated_text']

            generated_text = response[len(prompt):].strip()
            metadata = self._parse_response(generated_text)
            confidence = self._calculate_confidence(metadata, schema)

            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=time.time() - start_time,
                model_used=self.model_display_name,
                raw_response=generated_text
            )

        except Exception as e:
            logger.error(f"Error during metadata extraction with {self.model_display_name}: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used=self.model_display_name,
                error=str(e)
            )


# Legacy aliases — kept for backward compatibility if imported directly
QwenExtractor = LocalModelExtractor
Qwen3Extractor = LocalModelExtractor

class CloudExtractorWrapper(BaseLLMExtractor):
    """Wrapper class to make cloud extractors compatible with BaseLLMExtractor interface."""
    
    def __init__(self, cloud_extractor, model_name: str):
        self.cloud_extractor = cloud_extractor
        self.model_name = model_name
        self.model_config = {}  # Empty config for cloud models
    
    def _load_model(self):
        """Load model - not needed for cloud extractors."""
        pass
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using cloud extractor."""
        start_time = time.time()
        
        try:
            # Call cloud extractor
            result = self.cloud_extractor.extract_metadata(text, schema, document_type)
            
            if "error" in result:
                return ExtractionResult(
                    document_type=document_type,
                    metadata={},
                    confidence=0.0,
                    extraction_time=time.time() - start_time,
                    model_used=self.model_name,
                    error=result["error"]
                )
            
            # Extract metadata and calculate confidence
            metadata = result.get("metadata", {})
            confidence = self._calculate_confidence(metadata, schema)
            
            return ExtractionResult(
                document_type=document_type,
                metadata=metadata,
                confidence=confidence,
                extraction_time=time.time() - start_time,
                model_used=self.model_name,
                raw_response=str(result)
            )
            
        except Exception as e:
            logger.error(f"Error during cloud metadata extraction: {e}")
            return ExtractionResult(
                document_type=document_type,
                metadata={},
                confidence=0.0,
                extraction_time=time.time() - start_time,
                model_used=self.model_name,
                error=str(e)
            )
    
    def _calculate_confidence(self, metadata: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted metadata."""
        if not metadata:
            return 0.0
        
        # Count non-null fields
        total_fields = 0
        filled_fields = 0
        
        def count_fields(obj, schema_obj):
            nonlocal total_fields, filled_fields
            
            if isinstance(schema_obj, dict) and "properties" in schema_obj:
                for key, prop_schema in schema_obj["properties"].items():
                    total_fields += 1
                    if key in obj and obj[key] is not None and obj[key] != "":
                        filled_fields += 1
                    elif isinstance(obj.get(key), dict) and isinstance(prop_schema, dict) and "properties" in prop_schema:
                        count_fields(obj[key], prop_schema)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        count_fields(value, {})
                    else:
                        total_fields += 1
                        if value is not None and value != "":
                            filled_fields += 1
        
        count_fields(metadata, schema)
        
        if total_fields == 0:
            return 0.0
        
        return min(filled_fields / total_fields, 1.0)


def create_extractor(model_name: str, config_path: str = None) -> BaseLLMExtractor:
    """Factory function to create appropriate extractor"""
    
    # Check for cloud-based models first
    if model_name.lower().startswith("alibaba-"):
        # Extract model ID from alibaba- prefix
        alibaba_model = model_name.lower().replace("alibaba-", "")
        api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY or ALIBABA_API_KEY environment variable not set")
        
        # Map model names to Alibaba Cloud model IDs
        model_mapping = {
            "qwen3.5-122b-a10b": "qwen3.5-122b-a10b",
            "qwen3.5-plus": "qwen3.5-plus",
            "qwen3.5-flash": "qwen3.5-flash",
            "qwen3-max": "qwen3-max",
            "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct",
            "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b-a22b-instruct",
            "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-instruct-2507",
            "qwen-plus": "qwen-plus",
            "qwen-max": "qwen-max",
            "qwen-turbo": "qwen-turbo",
            "qwen-vl-plus": "qwen-vl-plus",
        }
        
        alibaba_model_id = model_mapping.get(alibaba_model, alibaba_model)
        
        # Create cloud extractor wrapper
        cloud_extractor = create_cloud_extractor(
            "alibaba", 
            api_key, 
            alibaba_model_id,
            region="singapore",
            temperature=1.0,
            top_p=0.8
        )
        
        # Return a wrapper that implements BaseLLMExtractor interface
        return CloudExtractorWrapper(cloud_extractor, f"Alibaba-{alibaba_model_id}")
    
    # Local model registry: maps user-facing names → (config_key, cache_key, display_name)
    # Only models that are realistically usable on deployment hardware are kept.
    LOCAL_MODEL_REGISTRY = {
        "qwen":           ("secondary",  "secondary",  "Qwen2.5-7B"),
        "qwen2.5":        ("secondary",  "secondary",  "Qwen2.5-7B"),
        "qwen3":          ("qwen3",      "qwen3",      "Qwen3-4B"),
        "qwen3-4b":       ("qwen3",      "qwen3",      "Qwen3-4B"),
    }

    key = model_name.lower()
    if key not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}. Available: {', '.join(sorted(set(LOCAL_MODEL_REGISTRY.keys())))}")

    config_key, cache_key, display_name = LOCAL_MODEL_REGISTRY[key]

    resolved_config_path = config_path or BaseLLMExtractor._DEFAULT_CONFIG
    with open(resolved_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return LocalModelExtractor(
        config['models'][config_key],
        config_path=config_path,
        model_display_name=display_name,
        cache_key=cache_key
    )

if __name__ == "__main__":
    # Test the extractor
    extractor = create_extractor("alibaba-qwen3-next-80b-a3b-instruct")
    
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
