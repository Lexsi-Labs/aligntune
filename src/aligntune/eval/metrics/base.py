"""
Base interface for all evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        """
        Compute the metric.
        
        Args:
            predictions: List of model predictions (text, logits, etc.)
            references: List of ground truth values
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metric names and values.
        """
        pass

    def safe_compute(self, *args, **kwargs) -> Dict[str, float]:
        """
        Wrapper to prevent metric computation from crashing the evaluation.
        """
        try:
            return self.compute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to compute metric {self.name}: {e}")
            return {f"{self.name}_error": 0.0}

    @property
    def requires_generation(self) -> bool:
        """Override to True if this metric requires generated text (vs logits)."""
        return True