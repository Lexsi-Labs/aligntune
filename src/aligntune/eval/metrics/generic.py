"""
Generic metrics for SFT and general model evaluation.
"""

import numpy as np
from typing import List, Dict, Any
from .base import Metric

# https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
class PerplexityMetric(Metric):
    """Computes Perplexity (PPL) using the model's loss."""
    
    def __init__(self):
        super().__init__("perplexity")

    @property
    def requires_generation(self) -> bool:
        return False  # Operates on logits/loss

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Dict[str, float]:
        """
        Expects 'predictions' to contain loss values for each sample.
        """
        
        losses = [p for p in predictions if isinstance(p, (float, int))]
        if not losses:
            return {"perplexity": float('nan')}
        
        mean_loss = np.mean(losses)
        # perplexity = np.exp(mean_loss) Not Needed Since doen already when computed at loss end 
        perplexity = mean_loss
        
        if np.isinf(perplexity):
            perplexity = 1e9
            
        return {"perplexity": float(perplexity)}

class AccuracyMetric(Metric):
    """Computes exact match accuracy."""
    
    def __init__(self):
        super().__init__("accuracy")

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        if not predictions or not references:
            return {"accuracy": 0.0}
            
        matches = [
            1.0 if str(p).strip() == str(r).strip() else 0.0 
            for p, r in zip(predictions, references)
        ]
        return {"accuracy": float(np.mean(matches))}