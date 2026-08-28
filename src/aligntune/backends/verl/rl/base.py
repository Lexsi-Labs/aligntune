"""
Base class for veRL backends - FSDP/Megatron support and data format bridging.

This module provides VerlBackendBase which extends TrainerBase with:
- Lazy import of verl with helpful error messages
- FSDP/Megatron worker initialization helpers
- Bridge from AlignTune's HuggingFace Dataset inputs to veRL's Parquet format
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC

from aligntune.core.rl.trainer_base import TrainerBase
from aligntune.core.rl.config import UnifiedConfig

logger = logging.getLogger(__name__)


def _ensure_verl_available():
    """Check if veRL is available, raise helpful error if not."""
    try:
        import verl
        return True
    except ImportError as e:
        raise ImportError(
            "veRL not installed. Install with:\n"
            "  pip install verl\n"
            "Or visit: https://github.com/volcengine/verl\n"
            "Note: veRL is optional - use TRL backend if veRL is not needed."
        ) from e


class VerlBackendBase(TrainerBase, ABC):
    """
    Base class for veRL trainers with FSDP/Megatron and data format support.

    Extends TrainerBase with:
    - Lazy verl imports with helpful error messages
    - FSDP/Megatron worker initialization
    - HF Dataset → Parquet conversion for veRL compatibility
    """

    def __init__(self, config: UnifiedConfig):
        """Initialize with veRL availability check."""
        _ensure_verl_available()

        super().__init__(config)

        self.verl_trainer = None
        self.verl_config = None
        self.parquet_dir = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if veRL is available."""
        try:
            import verl
            return True
        except ImportError:
            return False

    def _get_verl_trainer_class(self, trainer_type: str):
        """Lazy import veRL trainer class based on type."""
        try:
            if trainer_type == "ppo":
                from verl.trainer.ppo.ray_trainer import RayPPOTrainer
                return RayPPOTrainer
            elif trainer_type == "grpo":
                from verl.trainer.ppo.ray_trainer import RayPPOTrainer
                return RayPPOTrainer
            else:
                raise ValueError(f"Unknown trainer type: {trainer_type}")
        except ImportError as e:
            raise ImportError(
                f"Failed to import veRL {trainer_type} trainer. "
                "Ensure veRL is properly installed: pip install verl"
            ) from e

    def _setup_fsdp_config(self) -> Dict[str, Any]:
        """Setup FSDP configuration for distributed training."""
        fsdp_config = {}

        if hasattr(self.config, 'train') and hasattr(self.config.train, 'verl_fsdp_config'):
            fsdp_config = self.config.train.verl_fsdp_config or {}

        if 'sharding_strategy' not in fsdp_config:
            fsdp_config['sharding_strategy'] = 'FULL_SHARD'
        if 'cpu_offload' not in fsdp_config:
            fsdp_config['cpu_offload'] = False
        if 'backward_prefetch' not in fsdp_config:
            fsdp_config['backward_prefetch'] = 'BACKWARD_PRE'

        return fsdp_config

    def _get_verl_micro_batch_size(self) -> int:
        """Get veRL micro batch size from config."""
        if hasattr(self.config, 'train') and hasattr(self.config.train, 'verl_micro_batch_size'):
            return self.config.train.verl_micro_batch_size

        batch_size = getattr(self.config.train, 'per_device_batch_size', 32)
        return max(1, batch_size // 2)

    def _get_verl_rollout_n(self) -> int:
        """Get veRL number of rollouts from config."""
        if hasattr(self.config, 'train') and hasattr(self.config.train, 'verl_rollout_n'):
            return self.config.train.verl_rollout_n

        return 4

    def _get_verl_n_gpus_per_node(self) -> int:
        """Get number of GPUs per node."""
        if hasattr(self.config, 'train') and hasattr(self.config.train, 'verl_n_gpus_per_node'):
            return self.config.train.verl_n_gpus_per_node

        return 8

    def _get_verl_nnodes(self) -> int:
        """Get number of nodes for distributed training."""
        if hasattr(self.config, 'train') and hasattr(self.config.train, 'verl_nnodes'):
            return self.config.train.verl_nnodes

        return 1

    def _create_reward_wrapper(self, reward_fn) -> callable:
        """
        Wrap CompositeReward.batch_compute() for veRL compatibility.

        veRL expects: reward_fn(data_source, solution_str, ground_truth, extra_info) -> List[float]
        AlignTune provides: CompositeReward.batch_compute(texts, references) -> List[float]

        Args:
            reward_fn: CompositeReward or callable with batch_compute method

        Returns:
            Callable compatible with veRL's reward interface
        """
        def verl_reward_fn(data_source, solution_str, ground_truth=None, extra_info=None):
            """veRL-compatible reward function wrapper."""
            if isinstance(solution_str, str):
                texts = [solution_str]
            else:
                texts = solution_str

            references = None
            if ground_truth is not None:
                if isinstance(ground_truth, str):
                    references = [ground_truth] * len(texts)
                else:
                    references = ground_truth

            if hasattr(reward_fn, 'batch_compute'):
                rewards = reward_fn.batch_compute(texts, references=references, extra_info=extra_info)
            else:
                rewards = [reward_fn(t, ref) for t, ref in zip(texts, references or [None] * len(texts))]

            if isinstance(solution_str, str):
                return rewards[0] if rewards else 0.0
            return rewards

        return verl_reward_fn

    def cleanup(self):
        """Clean up temporary files (e.g., Parquet conversions)."""
        if self.parquet_dir and Path(self.parquet_dir).exists():
            try:
                shutil.rmtree(self.parquet_dir)
                logger.info(f"Cleaned up temporary data directory: {self.parquet_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {self.parquet_dir}: {e}")
