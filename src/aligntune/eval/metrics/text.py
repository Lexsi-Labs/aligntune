"""
Text generation metrics (BLEU, ROUGE).
"""

import logging
from typing import List, Dict, Any
from .base import Metric
import numpy as np

logger = logging.getLogger(__name__)

class RougeMetric(Metric):
    """Computes ROUGE scores."""
    
    def __init__(self):
        super().__init__("rouge")
        self.available = False
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.available = True
        except ImportError:
            logger.warning("rouge_score not installed. ROUGE metric will return 0.")
        except Exception as e:
            logger.warning(f"Failed to initialize ROUGE scorer: {e}")

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        if len(predictions) != len(references):
            logger.warning(f"[ROUGE] Length mismatch: Preds {len(predictions)} != Refs {len(references)}")
        
        if not self.available:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
        if not predictions or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Safe string conversion
            pred_str = str(pred) if pred is not None else ""
            ref_str = str(ref) if ref is not None else ""
            
            if not pred_str.strip():
                for k in scores: scores[k].append(0.0)
                continue
                
            try:
                score = self.scorer.score(ref_str, pred_str)
                scores["rouge1"].append(score["rouge1"].fmeasure)
                scores["rouge2"].append(score["rouge2"].fmeasure)
                scores["rougeL"].append(score["rougeL"].fmeasure)
            except Exception as e:
                logger.debug(f"[ROUGE] Scoring failed for item {i}: {e}")
                for k in scores: scores[k].append(0.0)

        # Safe averaging with NaN check
        result = {}
        for k, v in scores.items():
            if not v:
                result[k] = 0.0
            else:
                # Filter out NaNs/Nones to prevent propagation
                valid_scores = [s for s in v if s is not None and not np.isnan(s)]
                if not valid_scores:
                    result[k] = 0.0
                else:
                    result[k] = float(np.mean(valid_scores))
        
        # Add snake_case aliases for backward compatibility
        result["rouge_1"] = result.get("rouge1", 0.0)
        result["rouge_2"] = result.get("rouge2", 0.0)
        result["rouge_l"] = result.get("rougeL", 0.0)
        result["rouge"] = result.get("rougeL", 0.0)  # Default "rouge" to rougeL
                
        return result

class BleuMetric(Metric):
    """Computes BLEU scores using NLTK."""
    
    def __init__(self):
        super().__init__("bleu")
        self.available = False
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.bleu_func = sentence_bleu
            self.smoothing = SmoothingFunction().method1
            self.available = True
        except ImportError:
            logger.warning("nltk not installed. BLEU metric will return 0.")
        except Exception as e:
            logger.warning(f"Failed to initialize BLEU: {e}")

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        if not self.available:
            return {"bleu": 0.0}
            
        if not predictions or not references:
            return {"bleu": 0.0}

        scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Safe string conversion
            pred_str = str(pred) if pred is not None else ""
            ref_str = str(ref) if ref is not None else ""

            # NLTK expects tokenized lists
            ref_tokens = [ref_str.split()]
            pred_tokens = pred_str.split()
            
            if not pred_tokens:
                scores.append(0.0)
                continue
                
            try:
                score = self.bleu_func(ref_tokens, pred_tokens, smoothing_function=self.smoothing)
                scores.append(score)
            except Exception as e:
                logger.debug(f"[BLEU] Scoring failed for item {i}: {e}")
                scores.append(0.0)

        # Filter NaNs
        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
        
        if not valid_scores:
            return {"bleu": 0.0}
            
        return {"bleu": float(np.mean(valid_scores))}