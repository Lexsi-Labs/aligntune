"""
Base class for rollout backends providing unified interface for generation.

vLLM Rollout Backend
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseRolloutBackend(ABC):
    """
    Abstract base class for rollout/generation backends.

    All backends must implement these methods:
    - initialize(): Set up the backend and load model
    - generate(): Generate completions for prompts
    - cleanup(): Release resources
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Initialize rollout backend.

        Args:
            model_name_or_path: Model name or path for loading
            **kwargs: Backend-specific configuration
        """
        self.model_name_or_path = model_name_or_path
        self.config = kwargs
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize and load the model/engine.

        This method should handle:
        - Model loading
        - Device placement
        - Memory allocation
        - Configuration of generation parameters
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[List[str]]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate per prompt
            **kwargs: Backend-specific generation parameters

        Returns:
            List of lists of generated text. Shape: [len(prompts), num_return_sequences]
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources held by the backend.

        This should handle:
        - Model unloading
        - Memory cleanup
        - GPU cache clearing
        - Thread/process shutdown
        """
        pass

    def __del__(self):
        """Ensure cleanup on object destruction."""
        try:
            if self.initialized:
                self.cleanup()
        except BaseException:
            pass
