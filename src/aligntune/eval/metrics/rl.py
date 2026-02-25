"""
Reinforcement Learning specific metrics.
"""

import numpy as np
from typing import List, Dict, Any
from .base import Metric

class KLDivergenceMetric(Metric):
    """
    Computes KL Divergence aggregation.
    """
    
    def __init__(self):
        super().__init__("kl_divergence")

    @property
    def requires_generation(self) -> bool:
        return False

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        if predictions and isinstance(predictions[0], (float, int)):
            return {"kl_divergence": float(np.mean(predictions))}
        return {"kl_divergence": 0.0}

class RewardAccuracyMetric(Metric):
    """
    Computes accuracy of the reward model on chosen vs rejected pairs.
    """
    
    def __init__(self):
        super().__init__("reward_accuracy")

    @property
    def requires_generation(self) -> bool:
        return False

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        if not predictions:
            return {"reward_accuracy": 0.0}
            
        if isinstance(predictions[0], (tuple, list)) and len(predictions[0]) == 2:
            correct = sum(1 for c, r in predictions if c > r)
            return {"reward_accuracy": float(correct / len(predictions))}
            
        return {"reward_accuracy": 0.0}

class PolicyEntropyMetric(Metric):
    """Computes the entropy of the policy."""
    
    def __init__(self):
        super().__init__("policy_entropy")

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        if predictions and isinstance(predictions[0], (float, int)):
            return {"policy_entropy": float(np.mean(predictions))}
        return {"policy_entropy": 0.0}