"""
Rollout backend abstraction for efficient generation during RL training.

This module provides a pluggable system for different generation backends:
- HF: Standard HuggingFace transformers (baseline)
- vLLM: High-throughput inference engine (5-10x faster)
- SGLang: Structured generation (future)

vLLM Rollout Backend
"""

from .base import BaseRolloutBackend
from .hf_rollout import HFRolloutBackend

try:
    from .vllm_rollout import VLLMRolloutBackend
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLMRolloutBackend = None

__all__ = [
    "BaseRolloutBackend",
    "HFRolloutBackend",
    "VLLMRolloutBackend",
    "VLLM_AVAILABLE",
]
