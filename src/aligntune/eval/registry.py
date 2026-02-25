"""
Merged Registry.
Maps string keys to the NEW Metric classes while maintaining backward compatibility.
"""

from typing import Dict, Callable, List
import logging

# Import New Metrics
from .metrics.generic import PerplexityMetric, AccuracyMetric
from .metrics.text import RougeMetric, BleuMetric
from .metrics.math import MathAccuracyMetric
from .metrics.code import PassAtKMetric

logger = logging.getLogger(__name__)

class EvalRegistry:
    """
    Central registry.
    Adapts legacy function calls to new Class-based Metrics.
    """
    
    # Cache for instantiated metric objects
    _metric_instances = {}

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        """
        Returns a callable that mimics the old API: func(predictions, targets)
        But executes the new Class logic.
        """
        if name not in cls._metric_instances:
            # Instantiate the new class
            if name == "accuracy": cls._metric_instances[name] = AccuracyMetric()
            elif name == "perplexity": cls._metric_instances[name] = PerplexityMetric()
            elif name == "rouge": cls._metric_instances[name] = RougeMetric()
            elif name == "bleu": cls._metric_instances[name] = BleuMetric()
            elif name == "math_accuracy": cls._metric_instances[name] = MathAccuracyMetric()
            elif name == "pass_at_k": cls._metric_instances[name] = PassAtKMetric()
            else:
                raise ValueError(f"Unknown metric: {name}")
        
        metric_obj = cls._metric_instances[name]
        
        # Return a wrapper function to match old signature
        def wrapper(predictions, targets, **kwargs):
            result = metric_obj.safe_compute(predictions, targets, **kwargs)
            # Old API expected a single float, new API returns dict
            # We return the first value found
            return list(result.values())[0] if result else 0.0
            
        return wrapper

    @classmethod
    def list_metrics(cls) -> List[str]:
        return ["accuracy", "perplexity", "rouge", "bleu", "math_accuracy", "pass_at_k"]