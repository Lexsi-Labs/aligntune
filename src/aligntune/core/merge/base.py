"""
Abstract base merger class for model merging pipeline.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

SUPPORTED_HF_PREFIXES = ("hf://", "huggingface.co/", "https://huggingface.co/")


def _looks_like_hf_id(model: str) -> bool:
    """Return True if the string looks like a HuggingFace model ID (org/name or legacy single name)."""
    if any(model.startswith(p) for p in SUPPORTED_HF_PREFIXES):
        return True
    # org/model-name with no leading / or .
    parts = model.split("/")
    if len(parts) == 2 and not model.startswith((".", "/", "\\")):
        return True
    # Single-name models (legacy format like "gpt2", "bert-base-uncased")
    # Accept if it doesn't look like a file path
    if len(parts) == 1 and not model.startswith((".", "/", "\\")):
        return True
    return False


class BaseMerger(ABC):
    """
    Abstract base class for model mergers.

    All mergers should inherit from this class and implement the abstract methods.
    Concrete mergers wrap third-party libraries (mergekit, peft) rather than
    re-implementing merging algorithms from scratch.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the merger.

        Args:
            output_dir: Base output directory for merged models.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

    def validate_models(self, models: list[str]) -> bool:
        """
        Validate that each model path exists on disk or is a valid HuggingFace ID.

        Args:
            models: List of local paths or HuggingFace model IDs.

        Returns:
            True if all models are valid; raises ValueError otherwise.

        Raises:
            ValueError: If any model path does not exist and is not a valid HF ID.
        """
        if not models:
            raise ValueError("At least one model must be provided.")

        for model in models:
            path = Path(model)
            if path.exists():
                logger.debug("Model path exists on disk: %s", model)
                continue
            if _looks_like_hf_id(model):
                logger.debug("Treating as HuggingFace model ID: %s", model)
                continue
            raise ValueError(
                f"Model '{model}' does not exist as a local path and does not "
                "look like a valid HuggingFace model ID (expected 'org/model-name')."
            )

        return True

    @abstractmethod
    def supports_method(self) -> list[str]:
        """
        Return the list of merge methods this merger supports.

        Returns:
            List of method name strings (e.g. ['linear', 'task_arithmetic', 'ram']).
        """
        ...

    @abstractmethod
    def merge(self, models: list[str], output_path: str, **kwargs) -> str:
        """
        Merge *models* and write the result to *output_path*.

        Args:
            models: List of local paths or HuggingFace model IDs to merge.
            output_path: Directory where the merged model will be saved.
            **kwargs: Merger-specific parameters (method, weights, density, etc.).

        Returns:
            Absolute path to the merged model directory.
        """
        ...
