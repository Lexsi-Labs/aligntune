"""
DPO (Direct Preference Optimization) specific metrics.

Metrics for evaluating preference-based RL models:
- Win Rate: How often the model prefers the chosen response over rejected
- Reward Margin: Average difference between chosen and rejected rewards
- Preference Accuracy: Binary accuracy of preference predictions
- Log Ratio: Average log probability ratio between chosen and rejected
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import logging
from .base import Metric

logger = logging.getLogger(__name__)


class WinRateMetric(Metric):
    """
    Computes Win Rate for preference-based models.
    
    Win Rate = Probability that the model assigns higher reward/likelihood 
               to the chosen response vs rejected response.
    
    This is the core metric for DPO evaluation.
    """
    
    def __init__(self):
        super().__init__("win_rate")
    
    @property
    def requires_generation(self) -> bool:
        return False  # Uses preference pairs
    
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any], 
        **kwargs
    ) -> Dict[str, float]:
        """
        Args:
            predictions: List of (chosen_score, rejected_score) tuples
            references: Not used for this metric
        
        Returns:
            {"win_rate": float between 0 and 1}
        """
        if not predictions:
            return {"win_rate": 0.0}
        
        # Predictions should be list of (chosen, rejected) tuples
        wins = 0
        total = 0
        
        for pair in predictions:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                chosen_score, rejected_score = pair
                if chosen_score > rejected_score:
                    wins += 1
                total += 1
        
        if total == 0:
            return {"win_rate": 0.0}
        
        win_rate = wins / total
        
        return {
            "win_rate": float(win_rate),
            "win_count": wins,
            "total_pairs": total
        }


class RewardMarginMetric(Metric):
    """
    Computes the average margin between chosen and rejected rewards.
    
    Margin = mean(chosen_score - rejected_score)
    
    Higher margin indicates stronger preference distinction.
    """
    
    def __init__(self):
        super().__init__("reward_margin")
    
    @property
    def requires_generation(self) -> bool:
        return False
    
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any], 
        **kwargs
    ) -> Dict[str, float]:
        """
        Args:
            predictions: List of (chosen_score, rejected_score) tuples
        
        Returns:
            {"reward_margin": float, "reward_margin_std": float}
        """
        if not predictions:
            return {"reward_margin": 0.0}
        
        margins = []
        
        for pair in predictions:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                chosen_score, rejected_score = pair
                margin = chosen_score - rejected_score
                margins.append(margin)
        
        if not margins:
            return {"reward_margin": 0.0}
        
        return {
            "reward_margin": float(np.mean(margins)),
            "reward_margin_std": float(np.std(margins)),
            "reward_margin_min": float(np.min(margins)),
            "reward_margin_max": float(np.max(margins))
        }


class PreferenceAccuracyMetric(Metric):
    """
    Binary classification accuracy for preference prediction.
    
    Similar to win rate but can handle soft labels or probabilities.
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__("preference_accuracy")
        self.threshold = threshold
    
    @property
    def requires_generation(self) -> bool:
        return False
    
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any], 
        **kwargs
    ) -> Dict[str, float]:
        """
        Args:
            predictions: List of (chosen_score, rejected_score) tuples
            references: Optional ground truth labels (1 if chosen is better)
        
        Returns:
            {"preference_accuracy": float}
        """
        if not predictions:
            return {"preference_accuracy": 0.0}
        
        correct = 0
        total = 0
        
        for pair in predictions:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                chosen_score, rejected_score = pair
                
                # Predict: 1 if chosen > rejected, else 0
                predicted = 1 if chosen_score > rejected_score else 0
                
                # Ground truth is always 1 (chosen should be preferred)
                correct += (predicted == 1)
                total += 1
        
        if total == 0:
            return {"preference_accuracy": 0.0}
        
        return {"preference_accuracy": float(correct / total)}


class LogRatioMetric(Metric):
    """
    Computes log probability ratio between chosen and rejected responses.
    
    This measures how much more likely the model finds the chosen response
    compared to the rejected one, in log space.
    """
    
    def __init__(self):
        super().__init__("log_ratio")
    
    @property
    def requires_generation(self) -> bool:
        return False
    
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any], 
        **kwargs
    ) -> Dict[str, float]:
        """
        Args:
            predictions: List of (chosen_logprob, rejected_logprob) tuples
        
        Returns:
            {"log_ratio": float}
        """
        if not predictions:
            return {"log_ratio": 0.0}
        
        log_ratios = []
        
        for pair in predictions:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                chosen_log, rejected_log = pair
                log_ratio = chosen_log - rejected_log
                log_ratios.append(log_ratio)
        
        if not log_ratios:
            return {"log_ratio": 0.0}
        
        return {
            "log_ratio": float(np.mean(log_ratios)),
            "log_ratio_std": float(np.std(log_ratios))
        }


class ImplicitRewardMetric(Metric):
    """
    Estimates implicit reward from DPO model's preferences.
    
    For DPO, the implicit reward is:
    r(x,y) = β * log(π(y|x) / π_ref(y|x))
    
    where β is the DPO temperature parameter.
    """
    
    def __init__(self, beta: float = 0.1):
        super().__init__("implicit_reward")
        self.beta = beta
    
    @property
    def requires_generation(self) -> bool:
        return False
    
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any], 
        **kwargs
    ) -> Dict[str, float]:
        """
        Args:
            predictions: List of (policy_logprob, ref_logprob) tuples
            references: Not used
        
        Returns:
            {"implicit_reward_chosen": float, "implicit_reward_rejected": float}
        """
        if not predictions:
            return {"implicit_reward": 0.0}
        
        # For DPO, predictions should contain:
        # ((policy_chosen_logp, ref_chosen_logp), (policy_rej_logp, ref_rej_logp))
        
        chosen_rewards = []
        rejected_rewards = []
        
        for item in predictions:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                # Unpack chosen and rejected
                chosen_pair = item[0]
                rejected_pair = item[1]
                
                if len(chosen_pair) == 2 and len(rejected_pair) == 2:
                    policy_chosen, ref_chosen = chosen_pair
                    policy_rej, ref_rej = rejected_pair
                    
                    # Implicit reward = beta * log(policy/ref)
                    r_chosen = self.beta * (policy_chosen - ref_chosen)
                    r_rejected = self.beta * (policy_rej - ref_rej)
                    
                    chosen_rewards.append(r_chosen)
                    rejected_rewards.append(r_rejected)
        
        if not chosen_rewards:
            return {"implicit_reward": 0.0}
        
        return {
            "implicit_reward_chosen": float(np.mean(chosen_rewards)),
            "implicit_reward_rejected": float(np.mean(rejected_rewards)),
            "implicit_reward_margin": float(np.mean(chosen_rewards) - np.mean(rejected_rewards))
        }


class CalibrationMetric(Metric):
    """
    Measures how well the model's confidence aligns with actual accuracy.
    
    Good calibration means that when the model is 70% confident in a preference,
    it should be correct 70% of the time.
    """
    
    def __init__(self, num_bins: int = 10):
        super().__init__("calibration")
        self.num_bins = num_bins
    
    @property
    def requires_generation(self) -> bool:
        return False
    
    def compute(
        self, 
        predictions: List[Any], 
        references: List[Any], 
        **kwargs
    ) -> Dict[str, float]:
        """
        Args:
            predictions: List of (chosen_score, rejected_score) tuples
        
        Returns:
            {"calibration_error": float, "avg_confidence": float}
        """
        if not predictions:
            return {"calibration_error": 0.0}
        
        # Convert score differences to confidences
        confidences = []
        correctness = []
        
        for pair in predictions:
            if isinstance(pair, (tuple, list)) and len(pair) == 2:
                chosen_score, rejected_score = pair
                
                # Confidence = softmax probability
                score_diff = chosen_score - rejected_score
                confidence = 1 / (1 + np.exp(-score_diff))
                
                # Correct if chosen > rejected
                correct = 1.0 if chosen_score > rejected_score else 0.0
                
                confidences.append(confidence)
                correctness.append(correct)
        
        if not confidences:
            return {"calibration_error": 0.0}
        
        # Expected Calibration Error (ECE)
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        ece = 0.0
        for i in range(self.num_bins):
            bin_lower = i / self.num_bins
            bin_upper = (i + 1) / self.num_bins
            
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(correctness[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * np.sum(in_bin)
        
        ece /= len(confidences)
        
        return {
            "calibration_error": float(ece),
            "avg_confidence": float(np.mean(confidences)),
            "avg_accuracy": float(np.mean(correctness))
        }