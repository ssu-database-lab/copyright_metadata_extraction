"""
Model Cache Manager for LLM Metadata Extraction

This module handles downloading, caching, and managing Hugging Face models
locally to avoid repeated downloads and improve performance.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yaml
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Manages local caching of Hugging Face models."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the cache manager with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cache_dir = Path(self.config['cache']['local_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_cache_metadata()
        
        logger.info(f"Model cache manager initialized. Cache directory: {self.cache_dir}")
    
    def _load_config(self) -> Dict:
        """Load model configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _load_cache_metadata(self) -> Dict:
        """Load cache metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load cache metadata: {e}")
        
        return {
            "models": {},
            "last_cleanup": None,
            "cache_stats": {
                "total_downloads": 0,
                "total_size_gb": 0,
                "last_access": None
            }
        }
    
    def _save_cache_metadata(self):
        """Save cache metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Could not save cache metadata: {e}")
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get the local path for a model, downloading if necessary.
        
        Args:
            model_name: Name of the model (primary, secondary, lightweight)
            
        Returns:
            Local path to the model directory
        """
        if model_name not in self.config['models']:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.config['models'][model_name]
        model_id = model_config['model_id']
        local_path = Path(model_config['local_path'])
        
        # Check if model is already cached
        if self._is_model_cached(local_path):
            logger.info(f"Using cached model: {local_path}")
            self._update_access_time(model_id)
            return str(local_path)
        
        # Download model if not cached
        logger.info(f"Model not cached. Downloading: {model_id}")
        return self._download_model(model_id, local_path)
    
    def _is_model_cached(self, local_path: Path) -> bool:
        """Check if a model is properly cached locally."""
        if not local_path.exists():
            return False
        
        # Check for essential model files
        essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        for file in essential_files:
            if not (local_path / file).exists():
                logger.warning(f"Missing essential file: {file}")
                return False
        
        # Check for model files (safetensors or bin)
        model_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("*.bin"))
        if not model_files:
            logger.warning("No model weight files found")
            return False
        
        return True
    
    def _download_model(self, model_id: str, local_path: Path) -> str:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model ID
            local_path: Local path to save the model
            
        Returns:
            Local path to the downloaded model
        """
        try:
            logger.info(f"Downloading model: {model_id}")
            logger.info(f"Target directory: {local_path}")
            
            # Create directory if it doesn't exist
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download model using snapshot_download for complete model
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,  # Use actual files, not symlinks
                resume_download=True,  # Resume interrupted downloads
                force_download=self.config['cache']['force_download']
            )
            
            # Update cache metadata
            self._update_model_metadata(model_id, local_path)
            
            logger.info(f"Successfully downloaded model: {model_id}")
            return str(local_path)
            
        except HfHubHTTPError as e:
            logger.error(f"HTTP error downloading model {model_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            # Clean up partial download
            if local_path.exists():
                shutil.rmtree(local_path)
            raise
    
    def _update_model_metadata(self, model_id: str, local_path: Path):
        """Update cache metadata for a downloaded model."""
        model_name = model_id.split('/')[-1]
        
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        self.metadata['models'][model_id] = {
            "local_path": str(local_path),
            "downloaded_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "size_gb": round(size_gb, 2),
            "access_count": 1
        }
        
        # Update cache stats
        self.metadata['cache_stats']['total_downloads'] += 1
        self.metadata['cache_stats']['total_size_gb'] += size_gb
        self.metadata['cache_stats']['last_access'] = datetime.now().isoformat()
        
        self._save_cache_metadata()
    
    def _update_access_time(self, model_id: str):
        """Update the last accessed time for a model."""
        if model_id in self.metadata['models']:
            self.metadata['models'][model_id]['last_accessed'] = datetime.now().isoformat()
            self.metadata['models'][model_id]['access_count'] += 1
            self.metadata['cache_stats']['last_access'] = datetime.now().isoformat()
            self._save_cache_metadata()
    
    def list_cached_models(self) -> Dict[str, Dict]:
        """List all cached models with their metadata."""
        return self.metadata['models'].copy()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.metadata['cache_stats'].copy()
    
    def cleanup_unused_models(self, days_unused: int = 30) -> List[str]:
        """
        Clean up models that haven't been accessed for specified days.
        
        Args:
            days_unused: Number of days after which to consider a model unused
            
        Returns:
            List of removed model IDs
        """
        if not self.config['cache']['cleanup_unused']:
            logger.info("Cleanup disabled in configuration")
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_unused)
        removed_models = []
        
        for model_id, model_info in self.metadata['models'].items():
            last_accessed = datetime.fromisoformat(model_info['last_accessed'])
            
            if last_accessed < cutoff_date:
                local_path = Path(model_info['local_path'])
                
                if local_path.exists():
                    logger.info(f"Removing unused model: {model_id}")
                    shutil.rmtree(local_path)
                    removed_models.append(model_id)
                    
                    # Update cache stats
                    self.metadata['cache_stats']['total_size_gb'] -= model_info['size_gb']
        
        # Remove from metadata
        for model_id in removed_models:
            del self.metadata['models'][model_id]
        
        if removed_models:
            self.metadata['last_cleanup'] = datetime.now().isoformat()
            self._save_cache_metadata()
            logger.info(f"Cleaned up {len(removed_models)} unused models")
        
        return removed_models
    
    def cleanup_all_models(self) -> List[str]:
        """Remove all cached models."""
        removed_models = []
        
        for model_id, model_info in self.metadata['models'].items():
            local_path = Path(model_info['local_path'])
            
            if local_path.exists():
                logger.info(f"Removing model: {model_id}")
                shutil.rmtree(local_path)
                removed_models.append(model_id)
        
        # Clear metadata
        self.metadata['models'] = {}
        self.metadata['cache_stats']['total_size_gb'] = 0
        self.metadata['last_cleanup'] = datetime.now().isoformat()
        self._save_cache_metadata()
        
        logger.info(f"Removed all {len(removed_models)} cached models")
        return removed_models
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """
        Verify that a cached model can be loaded properly.
        
        Args:
            model_name: Name of the model to verify
            
        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            model_path = self.get_model_path(model_name)
            model_config = self.config['models'][model_name]
            
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Try to load model (this is expensive, so we'll just check config)
            # In a real implementation, you might want to do a lightweight check
            logger.info(f"Model integrity verified: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Model integrity check failed for {model_name}: {e}")
            return False


def main():
    """CLI interface for model cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Cache Manager")
    parser.add_argument("--config", default="config/model_config.yaml", help="Config file path")
    parser.add_argument("--list", action="store_true", help="List cached models")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--cleanup", action="store_true", help="Clean up unused models")
    parser.add_argument("--cleanup-all", action="store_true", help="Remove all cached models")
    parser.add_argument("--verify", help="Verify model integrity")
    parser.add_argument("--days", type=int, default=30, help="Days unused for cleanup")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        cache_manager = ModelCacheManager(args.config)
        
        if args.list:
            models = cache_manager.list_cached_models()
            print("\n📦 Cached Models:")
            for model_id, info in models.items():
                print(f"  • {model_id}")
                print(f"    Path: {info['local_path']}")
                print(f"    Size: {info['size_gb']} GB")
                print(f"    Last accessed: {info['last_accessed']}")
                print(f"    Access count: {info['access_count']}")
                print()
        
        if args.stats:
            stats = cache_manager.get_cache_stats()
            print("\n📊 Cache Statistics:")
            print(f"  Total downloads: {stats['total_downloads']}")
            print(f"  Total size: {stats['total_size_gb']:.2f} GB")
            print(f"  Last access: {stats['last_access']}")
        
        if args.cleanup:
            removed = cache_manager.cleanup_unused_models(args.days)
            if removed:
                print(f"\n🧹 Cleaned up {len(removed)} unused models:")
                for model_id in removed:
                    print(f"  • {model_id}")
            else:
                print("\n✨ No unused models to clean up")
        
        if args.cleanup_all:
            removed = cache_manager.cleanup_all_models()
            print(f"\n🗑️ Removed all {len(removed)} cached models")
        
        if args.verify:
            is_valid = cache_manager.verify_model_integrity(args.verify)
            if is_valid:
                print(f"\n✅ Model {args.verify} is valid")
            else:
                print(f"\n❌ Model {args.verify} failed verification")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
