"""
Generic metrics for SFT and general model evaluation.
"""

import numpy as np
import torch
from typing import List, Dict, Any
from .base import Metric


def completion_loss_totals(
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[float, int]:
    """Return masked completion NLL and completion-token count for PPL."""
    return (
        (per_token_loss * completion_mask).sum().item(),
        int(completion_mask.sum().item()),
    )


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
        Expects 'predictions' to contain the global mean token loss.
        """
        
        losses = [p for p in predictions if isinstance(p, (float, int))]
        if not losses:
            return {"perplexity": float('nan')}

        mean_loss = np.mean(losses)
        perplexity = np.exp(mean_loss)  # Correct: exp(mean(losses))

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
