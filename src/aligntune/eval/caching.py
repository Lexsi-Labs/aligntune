"""
Simple caching mechanism for evaluation results.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class EvaluationCache:
    def __init__(self, cache_dir: str = ".eval_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, model_name: str, dataset_name: str, config: Dict[str, Any]) -> str:
        """Generate a unique hash for the evaluation run."""
        # Sort keys to ensure consistent hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        unique_str = f"{model_name}_{dataset_name}_{config_str}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def get(self, model_name: str, dataset_name: str, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        key = self._generate_key(model_name, dataset_name, config)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save(self, model_name: str, dataset_name: str, config: Dict[str, Any], results: Dict[str, float]):
        key = self._generate_key(model_name, dataset_name, config)
        cache_file = self.cache_dir / f"{key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)