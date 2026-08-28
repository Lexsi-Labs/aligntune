"""
ES Trainer Base Class - Abstract interface for ES training.

All ES trainer implementations inherit from this.
Defines the interface and common utilities.
"""

import logging
import os
import tempfile
from abc import abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from aligntune.core.trainer_base import UnifiedTrainerBase

logger = logging.getLogger(__name__)


class ESTrainerBase(UnifiedTrainerBase):
    """
    Abstract base class for Evolution Strategies training.

    Subclasses (e.g., ESTrainerBackend) implement the actual training logic.
    """

    def __init__(self, config: Any, callbacks: Optional[List] = None):
        super().__init__(config, callbacks)

        # Will be set by subclass
        self.rollout_backend = None
        self.reward_fn = None
        self.task = None

        # Tracking
        self.fitness_history = []

    def _setup_adapter_cache_dir(self) -> Path:
        """
        Setup adapter cache directory with intelligent fallback strategy.

        Strategy:
        =========
        1. Try /dev/shm (RAM disk) first
           - Super fast (in CPU RAM, not disk)
           - Adapters stay in RAM between iterations
           - vLLM loads/unloads from RAM (10x faster than disk)

        2. Fall back to /tmp (disk storage)
           - Slower than RAM but still functional
           - Typical on Colab where /dev/shm is limited
           - vLLM still manages caching in VRAM automatically

        Why this works with vLLM:
        - vLLM has a LoRA cache pool (typically 4-8 adapters in VRAM at once)
        - When adapter i is needed, vLLM loads from cache_dir (RAM or disk)
        - If already in VRAM cache, reuse it (no reload)
        - If not in VRAM, evict LRU adapter and load new one
        - Result: Frequently used adapters stay in VRAM, others loaded on demand
        """
        shm_path = Path("/dev/shm") / "es_adapters"

        try:
            # Try to create /dev/shm directory
            shm_path.mkdir(parents=True, exist_ok=True)

            # Check available space in /dev/shm
            stat_info = os.statvfs(str(shm_path))
            available_bytes = stat_info.f_bavail * stat_info.f_frsize
            available_gb = available_bytes / (1024**3)

            # Estimate adapter size: ~500MB each for typical model
            adapters_size_gb = (self.config.es.population_size * 0.5)

            if available_gb >= adapters_size_gb * 1.5:
                # Plenty of space in RAM
                logger.info(f"Using /dev/shm for adapter cache (RAM disk)")
                logger.info(f"  Available: {available_gb:.1f}GB, Needed: ~{adapters_size_gb:.1f}GB")
                logger.info(f"  Adapters will be cached in CPU RAM for maximum speed")
                return shm_path
            else:
                # Not enough space in /dev/shm
                logger.warning(f"/dev/shm space insufficient: {available_gb:.1f}GB available, need ~{adapters_size_gb:.1f}GB")
                logger.warning(f"Falling back to disk storage. vLLM will still cache adapters in VRAM automatically")

        except (OSError, AttributeError) as e:
            # /dev/shm doesn't exist or not accessible (typical on Colab)
            logger.warning(f"/dev/shm not available: {e}")
            logger.info("Falling back to disk storage. vLLM will cache adapters in VRAM automatically")

        # Fallback to system temp directory
        temp_path = Path(tempfile.gettempdir()) / "es_adapters"
        temp_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temp directory for adapter cache: {temp_path}")
        logger.info(f"  vLLM will manage LoRA caching in VRAM (typical 4-8 adapters at a time)")

        return temp_path

    @abstractmethod
    def setup_model(self) -> None:
        """Load model with PEFT using build_model()."""
        pass

    @abstractmethod
    def setup_data(self) -> None:
        """Load dataset using DataManager."""
        pass

    @abstractmethod
    def setup_rollout_backend(self) -> None:
        """Initialize vLLM rollout backend."""
        pass

    @abstractmethod
    def setup_reward_function(self) -> None:
        """Initialize reward function."""
        pass

    @abstractmethod
    def generate_local_adapters(self, iteration: int) -> List[Dict[str, Any]]:
        """Generate population of perturbed LoRA weights."""
        pass

    @abstractmethod
    def generate_and_score(self, adapters: List[Dict[str, Any]]) -> Any:
        """Evaluate population: generate solutions and compute fitness."""
        pass

    @abstractmethod
    def apply_es_update(self, adapters: List[Dict[str, Any]], fitness_scores: Any) -> Any:
        """Compute ES gradient and update LoRA weights."""
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Evaluation (optional)."""
        pass
