"""
Hybrid Model Extractor for LLM Metadata Extraction

This module provides a hybrid approach that can use both local models and cloud APIs
based on availability, cost, and performance requirements.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .base_extractor import BaseLLMExtractor, ExtractionResult
from .cloud_extractor import create_cloud_extractor, CloudExtractor
from .model_cache import ModelCacheManager

logger = logging.getLogger(__name__)


class HybridModelExtractor(BaseLLMExtractor):
    """Hybrid extractor that can use local models or cloud APIs."""
    
    def __init__(self, config_path: str = "config/model_config.yaml", 
                 cloud_config_path: str = "config/cloud_model_config.yaml"):
        """Initialize hybrid extractor with local and cloud configurations."""
        self.config_path = config_path
        self.cloud_config_path = cloud_config_path
        
        # Load configurations
        self.local_config = self._load_config(config_path)
        self.cloud_config = self._load_config(cloud_config_path)
        
        # Initialize local cache manager
        self.cache_manager = ModelCacheManager(config_path)
        
        # Initialize cloud extractors
        self.cloud_extractors = self._initialize_cloud_extractors()
        
        # Current strategy (local, cloud, hybrid)
        self.strategy = os.getenv("EXTRACTION_STRATEGY", "hybrid")
        
        logger.info(f"Hybrid extractor initialized with strategy: {self.strategy}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_path}: {e}")
            return {}
    
    def _initialize_cloud_extractors(self) -> Dict[str, CloudExtractor]:
        """Initialize cloud extractors based on configuration."""
        extractors = {}
        
        for provider_name, provider_config in self.cloud_config.get("providers", {}).items():
            if provider_config.get("enabled", False):
                api_key = os.getenv(provider_config.get("api_key_env", ""))
                if api_key:
                    try:
                        # Get default model for this provider
                        default_model = self._get_default_model_for_provider(provider_name)
                        if default_model:
                            extractor = create_cloud_extractor(provider_name, api_key, default_model)
                            extractors[provider_name] = extractor
                            logger.info(f"Initialized {provider_name} cloud extractor")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {provider_name} extractor: {e}")
                else:
                    logger.warning(f"No API key found for {provider_name}")
        
        return extractors
    
    def _get_default_model_for_provider(self, provider: str) -> Optional[str]:
        """Get default model ID for a provider."""
        provider_models = {
            "huggingface": "upstage/SOLAR-10.7B-Instruct-v1.0",
            "openai": "gpt-4o-mini",
            "together": "meta-llama/Llama-2-7b-chat-hf"
        }
        return provider_models.get(provider)
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using hybrid approach."""
        if self.strategy == "local":
            return self._extract_local(text, schema, document_type)
        elif self.strategy == "cloud":
            return self._extract_cloud(text, schema, document_type)
        else:  # hybrid
            return self._extract_hybrid(text, schema, document_type)
    
    def _extract_local(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using local models only."""
        try:
            # Use the existing local extractor logic
            from .base_extractor import create_extractor
            
            # Determine which local model to use
            model_name = self._select_local_model(document_type)
            local_extractor = create_extractor(model_name, self.config_path)
            
            return local_extractor.extract_metadata(text, schema, document_type)
            
        except Exception as e:
            logger.error(f"Local extraction failed: {e}")
            return ExtractionResult(
                metadata={},
                model="local_failed",
                provider="local",
                processing_time=None,
                confidence=None,
                error=str(e)
            )
    
    def _extract_cloud(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using cloud APIs only."""
        # Try providers in order of preference
        preferred_providers = self.cloud_config.get("cost_optimization", {}).get("preferred_providers", [])
        
        for provider in preferred_providers:
            if provider in self.cloud_extractors:
                try:
                    result = self.cloud_extractors[provider].extract_metadata(text, schema, document_type)
                    if "error" not in result:
                        return ExtractionResult(
                            metadata=result.get("metadata", {}),
                            model=result.get("model", provider),
                            provider=result.get("provider", provider),
                            processing_time=result.get("processing_time"),
                            confidence=result.get("confidence"),
                            error=None
                        )
                except Exception as e:
                    logger.warning(f"Cloud extraction failed with {provider}: {e}")
                    continue
        
        # All cloud providers failed
        return ExtractionResult(
            metadata={},
            model="cloud_failed",
            provider="cloud",
            processing_time=None,
            confidence=None,
            error="All cloud providers failed"
        )
    
    def _extract_hybrid(self, text: str, schema: Dict[str, Any], document_type: str) -> ExtractionResult:
        """Extract metadata using hybrid approach (local first, cloud fallback)."""
        # Try local first
        local_result = self._extract_local(text, schema, document_type)
        if local_result.error is None:
            logger.info("Using local model for extraction")
            return local_result
        
        # Fallback to cloud
        logger.info("Local model failed, falling back to cloud")
        cloud_result = self._extract_cloud(text, schema, document_type)
        if cloud_result.error is None:
            logger.info("Using cloud API for extraction")
            return cloud_result
        
        # Both failed
        logger.error("Both local and cloud extraction failed")
        return ExtractionResult(
            metadata={},
            model="hybrid_failed",
            provider="hybrid",
            processing_time=None,
            confidence=None,
            error=f"Local error: {local_result.error}, Cloud error: {cloud_result.error}"
        )
    
    def _select_local_model(self, document_type: str) -> str:
        """Select appropriate local model based on document type."""
        # Simple heuristic: use primary model for contracts, secondary for others
        if "계약서" in document_type or "contract" in document_type.lower():
            return "solar-ko"  # Primary model
        else:
            return "qwen"  # Secondary model
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models (local and cloud)."""
        models = {}
        
        # Local models
        for model_name, model_config in self.local_config.get("models", {}).items():
            models[f"local_{model_name}"] = {
                "name": model_config.get("name", model_name),
                "provider": "local",
                "model_id": model_config.get("model_id"),
                "available": self.cache_manager.has_model(model_name),
                "type": "local"
            }
        
        # Cloud models
        for model_name, model_config in self.cloud_config.get("models", {}).items():
            provider = model_config.get("provider")
            if provider in self.cloud_extractors:
                models[f"cloud_{model_name}"] = {
                    "name": model_config.get("name", model_name),
                    "provider": provider,
                    "model_id": model_config.get("model_id"),
                    "available": True,
                    "type": "cloud",
                    "cost_per_token": model_config.get("cost_per_token", 0)
                }
        
        return models
    
    def estimate_cost(self, text: str, model_name: str) -> float:
        """Estimate cost for extraction using a specific model."""
        # Rough estimation: 1 token ≈ 4 characters for Korean text
        estimated_tokens = len(text) / 4
        
        if model_name.startswith("cloud_"):
            model_config = self.cloud_config.get("models", {}).get(model_name.replace("cloud_", ""), {})
            cost_per_token = model_config.get("cost_per_token", 0)
            return estimated_tokens * cost_per_token / 1000  # Convert to USD
        else:
            return 0.0  # Local models are free
    
    def switch_strategy(self, strategy: str):
        """Switch extraction strategy."""
        if strategy in ["local", "cloud", "hybrid"]:
            self.strategy = strategy
            logger.info(f"Switched to {strategy} strategy")
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


# Factory function for hybrid extractor
def create_hybrid_extractor(config_path: str = "config/model_config.yaml",
                           cloud_config_path: str = "config/cloud_model_config.yaml") -> HybridModelExtractor:
    """Create a hybrid model extractor."""
    return HybridModelExtractor(config_path, cloud_config_path)


# Example usage
if __name__ == "__main__":
    # Initialize hybrid extractor
    extractor = create_hybrid_extractor()
    
    # Check available models
    models = extractor.get_available_models()
    print("Available models:")
    for model_id, model_info in models.items():
        status = "✅" if model_info["available"] else "❌"
        cost_info = f" (${model_info.get('cost_per_token', 0):.4f}/1K tokens)" if model_info["type"] == "cloud" else " (Free)"
        print(f"  {status} {model_id}: {model_info['name']}{cost_info}")
    
    # Test extraction
    test_text = "계약서 내용..."
    test_schema = {"title": "string", "date": "string"}
    
    # Try different strategies
    for strategy in ["local", "cloud", "hybrid"]:
        print(f"\n--- Testing {strategy} strategy ---")
        extractor.switch_strategy(strategy)
        result = extractor.extract_metadata(test_text, test_schema, "계약서")
        print(f"Result: {result.model} ({result.provider})")
        if result.error:
            print(f"Error: {result.error}")
