"""
Dataset caching system for unified RLHF training.

This module provides efficient caching of datasets and tokenized data
using hash-based keys to avoid reprocessing identical datasets.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datasets import Dataset
try:
    from datasets import load_from_disk, save_to_disk
except ImportError:
    # Fallback for older versions
    from datasets import Dataset
    def load_from_disk(path):
        return Dataset.load_from_disk(path)
    def save_to_disk(dataset, path):
        dataset.save_to_disk(path)

logger = logging.getLogger(__name__)


class DatasetCache:
    """Dataset caching system with hash-based keys."""
    
    def __init__(self, cache_root: Union[str, Path] = "./cache"):
        """Initialize dataset cache."""
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized dataset cache at {self.cache_root}")
    
    def get_cache_key(
        self,
        dataset_name: str,
        split: str,
        tokenizer_name: str,
        max_length: int,
        template: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        Generate cache key for dataset configuration.
        
        Args:
            dataset_name: Name or path of the dataset
            split: Dataset split (train, validation, etc.)
            tokenizer_name: Name of the tokenizer
            max_length: Maximum sequence length
            template: Chat template used
            column_mapping: Column mapping configuration
            **kwargs: Additional parameters that affect the dataset
            
        Returns:
            Hash-based cache key
        """
        # Create a dictionary of all parameters that affect the dataset
        cache_params = {
            "dataset_name": dataset_name,
            "split": split,
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "template": template,
            "column_mapping": column_mapping or {},
            **kwargs
        }
        
        # Sort keys for consistent hashing
        cache_params_str = json.dumps(cache_params, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.sha256(cache_params_str.encode()).hexdigest()[:16]
        
        return cache_key
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache directory path for a given key."""
        return self.cache_root / f"dataset_{cache_key}"
    
    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists for the given key."""
        cache_path = self.get_cache_path(cache_key)
        return cache_path.exists() and (cache_path / "dataset_info.json").exists()
    
    def save_to_cache(self, dataset: Dataset, cache_key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save dataset to cache.
        
        Args:
            dataset: Dataset to cache
            cache_key: Cache key for the dataset
            metadata: Additional metadata to store
        """
        cache_path = self.get_cache_path(cache_key)
        
        # Remove existing cache if it exists
        if cache_path.exists():
            shutil.rmtree(cache_path)
        
        # Create cache directory
        cache_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save dataset
            save_to_disk(dataset, str(cache_path / "dataset"))
            
            # Save metadata
            metadata = metadata or {}
            metadata.update({
                "cache_key": cache_key,
                "num_rows": len(dataset),
                "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else [],
                "created_at": str(Path().cwd())
            })
            
            with open(cache_path / "dataset_info.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Dataset cached: {cache_key} ({len(dataset)} rows)")
            
        except Exception as e:
            # Cleanup on failure
            if cache_path.exists():
                shutil.rmtree(cache_path)
            raise RuntimeError(f"Failed to cache dataset {cache_key}: {e}")
    
    def load_from_cache(self, cache_key: str) -> Dataset:
        """
        Load dataset from cache.
        
        Args:
            cache_key: Cache key for the dataset
            
        Returns:
            Cached dataset
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not self.cache_exists(cache_key):
            raise FileNotFoundError(f"Cache not found for key: {cache_key}")
        
        try:
            # Load dataset
            dataset = load_from_disk(str(cache_path / "dataset"))
            
            # Load metadata
            metadata_path = cache_path / "dataset_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded cached dataset: {cache_key} ({metadata.get('num_rows', 'unknown')} rows)")
            
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load cached dataset {cache_key}: {e}")
    
    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for cached dataset."""
        cache_path = self.get_cache_path(cache_key)
        metadata_path = cache_path / "dataset_info.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata for {cache_key}: {e}")
            return None
    
    def list_cached_datasets(self) -> List[Dict[str, Any]]:
        """List all cached datasets with their metadata."""
        cached_datasets = []
        
        for cache_dir in self.cache_root.glob("dataset_*"):
            if cache_dir.is_dir():
                cache_key = cache_dir.name.replace("dataset_", "")
                metadata = self.get_cache_info(cache_key)
                
                if metadata:
                    cached_datasets.append({
                        "cache_key": cache_key,
                        "cache_path": str(cache_dir),
                        **metadata
                    })
        
        return cached_datasets
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        if cache_key is not None:
            # Clear specific cache
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache: {cache_key}")
        else:
            # Clear all caches
            for cache_dir in self.cache_root.glob("dataset_*"):
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
            logger.info("Cleared all caches")
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        
        for cache_dir in self.cache_root.glob("dataset_*"):
            if cache_dir.is_dir():
                for file_path in cache_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        return total_size
    
    def cleanup_old_caches(self, max_age_days: int = 30) -> None:
        """
        Clean up old cache entries.
        
        Args:
            max_age_days: Maximum age in days for cache entries
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        cleaned_count = 0
        
        for cache_dir in self.cache_root.glob("dataset_*"):
            if cache_dir.is_dir():
                # Check creation time
                creation_time = cache_dir.stat().st_ctime
                age_seconds = current_time - creation_time
                
                if age_seconds > max_age_seconds:
                    shutil.rmtree(cache_dir)
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old cache entries")
    
    def validate_cache(self, cache_key: str) -> bool:
        """
        Validate cache integrity.
        
        Args:
            cache_key: Cache key to validate
            
        Returns:
            True if cache is valid, False otherwise
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not self.cache_exists(cache_key):
            return False
        
        try:
            # Try to load the dataset
            dataset = self.load_from_cache(cache_key)
            
            # Basic validation
            if len(dataset) == 0:
                logger.warning(f"Cache {cache_key} contains empty dataset")
                return False
            
            # Check metadata consistency
            metadata = self.get_cache_info(cache_key)
            if metadata and metadata.get("num_rows") != len(dataset):
                logger.warning(f"Cache {cache_key} metadata inconsistent")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache {cache_key} validation failed: {e}")
            return False
    
    def repair_cache(self, cache_key: str) -> bool:
        """
        Attempt to repair corrupted cache.
        
        Args:
            cache_key: Cache key to repair
            
        Returns:
            True if repair was successful, False otherwise
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            return False
        
        try:
            # Try to load and re-save the dataset
            dataset = self.load_from_cache(cache_key)
            metadata = self.get_cache_info(cache_key)
            
            # Clear and re-save
            shutil.rmtree(cache_path)
            self.save_to_cache(dataset, cache_key, metadata)
            
            logger.info(f"Repaired cache: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to repair cache {cache_key}: {e}")
            # Remove corrupted cache
            if cache_path.exists():
                shutil.rmtree(cache_path)
            return False
