#!/usr/bin/env python3
"""
Model Cache Cleanup Command

This script provides command-line interface for managing the model cache,
including cleanup of unused models and cache statistics.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.model_cache import ModelCacheManager

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def list_models(cache_manager: ModelCacheManager):
    """List all cached models."""
    models = cache_manager.list_cached_models()
    
    if not models:
        print("📦 No models cached")
        return
    
    print("\n📦 Cached Models:")
    print("=" * 80)
    
    total_size = 0
    for model_id, info in models.items():
        print(f"Model: {model_id}")
        print(f"  Path: {info['local_path']}")
        print(f"  Size: {info['size_gb']:.2f} GB")
        print(f"  Downloaded: {info['downloaded_at']}")
        print(f"  Last accessed: {info['last_accessed']}")
        print(f"  Access count: {info['access_count']}")
        print("-" * 40)
        total_size += info['size_gb']
    
    print(f"\nTotal cached size: {total_size:.2f} GB")

def show_stats(cache_manager: ModelCacheManager):
    """Show cache statistics."""
    stats = cache_manager.get_cache_stats()
    
    print("\n📊 Cache Statistics:")
    print("=" * 50)
    print(f"Total downloads: {stats['total_downloads']}")
    print(f"Total size: {stats['total_size_gb']:.2f} GB")
    print(f"Last access: {stats['last_access']}")
    
    # Show disk usage
    cache_dir = Path(cache_manager.cache_dir)
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        actual_size_gb = total_size / (1024**3)
        print(f"Actual disk usage: {actual_size_gb:.2f} GB")

def cleanup_unused(cache_manager: ModelCacheManager, days: int):
    """Clean up unused models."""
    print(f"\n🧹 Cleaning up models unused for {days} days...")
    
    removed = cache_manager.cleanup_unused_models(days)
    
    if removed:
        print(f"✅ Removed {len(removed)} unused models:")
        for model_id in removed:
            print(f"  • {model_id}")
    else:
        print("✨ No unused models to clean up")

def cleanup_all(cache_manager: ModelCacheManager, confirm: bool = False):
    """Remove all cached models."""
    models = cache_manager.list_cached_models()
    
    if not models:
        print("📦 No models to remove")
        return
    
    if not confirm:
        print(f"\n⚠️  This will remove {len(models)} cached models:")
        for model_id in models.keys():
            print(f"  • {model_id}")
        
        response = input("\nAre you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("❌ Operation cancelled")
            return
    
    print(f"\n🗑️  Removing all {len(models)} cached models...")
    removed = cache_manager.cleanup_all_models()
    
    print(f"✅ Removed all cached models")

def verify_model(cache_manager: ModelCacheManager, model_name: str):
    """Verify model integrity."""
    print(f"\n🔍 Verifying model: {model_name}")
    
    try:
        is_valid = cache_manager.verify_model_integrity(model_name)
        
        if is_valid:
            print(f"✅ Model {model_name} is valid and can be loaded")
        else:
            print(f"❌ Model {model_name} failed verification")
            print("   Consider re-downloading the model")
    
    except Exception as e:
        print(f"❌ Error verifying model {model_name}: {e}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Model Cache Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    # List cached models
  %(prog)s --stats                   # Show cache statistics
  %(prog)s --cleanup --days 30       # Clean up models unused for 30 days
  %(prog)s --cleanup-all --yes       # Remove all cached models
  %(prog)s --verify primary          # Verify primary model integrity
        """
    )
    
    parser.add_argument(
        "--config", 
        default="config/model_config.yaml",
        help="Path to model configuration file (default: config/model_config.yaml)"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all cached models"
    )
    
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Show cache statistics"
    )
    
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="Clean up unused models"
    )
    
    parser.add_argument(
        "--cleanup-all", 
        action="store_true",
        help="Remove all cached models"
    )
    
    parser.add_argument(
        "--verify", 
        metavar="MODEL_NAME",
        help="Verify model integrity (primary, secondary, lightweight)"
    )
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=30,
        help="Days unused for cleanup (default: 30)"
    )
    
    parser.add_argument(
        "--yes", 
        action="store_true",
        help="Skip confirmation for cleanup-all"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize cache manager
        cache_manager = ModelCacheManager(args.config)
        
        # Execute requested actions
        if args.list:
            list_models(cache_manager)
        
        if args.stats:
            show_stats(cache_manager)
        
        if args.cleanup:
            cleanup_unused(cache_manager, args.days)
        
        if args.cleanup_all:
            cleanup_all(cache_manager, args.yes)
        
        if args.verify:
            verify_model(cache_manager, args.verify)
        
        # If no action specified, show help
        if not any([args.list, args.stats, args.cleanup, args.cleanup_all, args.verify]):
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
