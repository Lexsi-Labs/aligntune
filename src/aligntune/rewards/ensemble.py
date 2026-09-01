"""
Reward Model Ensembles for preventing reward hacking.

This module implements ensemble-based reward aggregation strategies that prevent
overoptimization to any single learned reward model by combining multiple
independent reward models.

Key Ideas:
- Load N reward models from Hugging Face
- Compute scores using multiple aggregation modes:
  - mean: Simple average (baseline)
  - worst_case: Minimum score (conservative, prevents reward hacking)
  - uncertainty_weighted: Penalize divergence between models
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from .core import RewardFunction, RewardConfig, RewardType

logger = logging.getLogger(__name__)


class RewardModelEnsemble(RewardFunction):
    """Ensemble of reward models that aggregates scores to prevent reward hacking.

    This class loads multiple reward models from Hugging Face and computes aggregated
    reward scores using different aggregation strategies:
    - mean: Weighted average of all models (default)
    - worst_case: Minimum score across models (conservative)
    - uncertainty_weighted: Score penalized by model disagreement

    Attributes:
        model_ids: List of Hugging Face model IDs to load
        ensemble_mode: Aggregation strategy ("mean", "worst_case", "uncertainty_weighted")
        models: Dictionary of loaded models
        tokenizers: Dictionary of loaded tokenizers
    """

    def __init__(self, config: RewardConfig):
        """Initialize reward model ensemble.

        Args:
            config: RewardConfig with params containing:
                - model_ids (List[str]): List of HF reward model IDs
                - ensemble_mode (str): Aggregation mode (default: "mean")
                - cache_dir (str, optional): Cache directory for models
        """
        super().__init__(config)

        # Extract ensemble-specific params
        self.model_ids: List[str] = config.params.get(
            "model_ids",
            [
                "OpenAssistant/reward-model-deberta-v3-large-v2",
                "gpt2",  # Fallback for testing
            ],
        )
        self.ensemble_mode: str = config.params.get("ensemble_mode", "mean")
        self.cache_dir: Optional[str] = config.cache_dir or config.params.get(
            "cache_dir", None
        )

        # Validate ensemble mode
        if self.ensemble_mode not in ["mean", "worst_case", "uncertainty_weighted"]:
            raise ValueError(
                f"ensemble_mode must be 'mean', 'worst_case', or 'uncertainty_weighted', "
                f"got {self.ensemble_mode}"
            )

        # Cache for loaded models to avoid reloading
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}

        logger.info(
            f"Initializing RewardModelEnsemble with {len(self.model_ids)} models "
            f"using {self.ensemble_mode} mode"
        )

        # Lazy load models on first use
        self._models_loaded = False

    def _load_models(self) -> None:
        """Load all reward models from Hugging Face.

        Models are loaded lazily on first use to avoid unnecessary initialization.
        """
        if self._models_loaded:
            return

        for model_id in self.model_ids:
            try:
                logger.info(f"Loading reward model: {model_id}")

                # Try to load as a text-classification pipeline first
                # (most reward models are sequence classification models)
                try:
                    pipe = pipeline(
                        "text-classification",
                        model=model_id,
                        device=0 if self.device == "cuda" else -1,
                        cache_dir=self.cache_dir,
                    )
                    self.pipelines[model_id] = pipe

                    # Extract tokenizer if available
                    if hasattr(pipe, "tokenizer"):
                        self.tokenizers[model_id] = pipe.tokenizer
                    else:
                        try:
                            self.tokenizers[model_id] = AutoTokenizer.from_pretrained(
                                model_id, cache_dir=self.cache_dir
                            )
                        except Exception:
                            self.tokenizers[model_id] = None

                    self.models[model_id] = pipe.model
                    logger.info(f"Successfully loaded {model_id} as text-classification")

                except Exception as pipe_error:
                    logger.warning(
                        f"Could not load {model_id} as pipeline: {pipe_error}. "
                        f"Trying direct model loading..."
                    )

                    # Fallback: load model and tokenizer directly
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, cache_dir=self.cache_dir
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_id, cache_dir=self.cache_dir
                    )
                    model = model.to(self.device)
                    model.eval()

                    self.tokenizers[model_id] = tokenizer
                    self.models[model_id] = model
                    logger.info(f"Successfully loaded {model_id} via direct loading")

            except Exception as e:
                logger.error(f"Failed to load reward model {model_id}: {e}")
                # Continue with remaining models

        self._models_loaded = True
        logger.info(
            f"Loaded {len(self.models)} out of {len(self.model_ids)} reward models"
        )

        if not self.models:
            raise RuntimeError(
                f"Failed to load any reward models from {self.model_ids}"
            )

    def _compute_score_pipeline(self, model_id: str, text: str) -> float:
        """Compute reward score using pipeline.

        Args:
            model_id: Model identifier
            text: Input text

        Returns:
            Reward score in [0, 1]
        """
        pipe = self.pipelines[model_id]

        try:
            # Truncate text if needed
            max_length = 512
            if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
                max_length = getattr(pipe.tokenizer, "model_max_length", 512)

            # Truncate to max_length tokens
            tokens = pipe.tokenizer.encode(text, truncation=True, max_length=max_length)
            text_truncated = pipe.tokenizer.decode(tokens, skip_special_tokens=True)

            # Get prediction
            result = pipe(text_truncated, truncation=True, max_length=max_length)

            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            # Extract score
            if isinstance(result, dict):
                score = result.get("score", 0.0)
                label = result.get("label", "")

                # Normalize score to [0, 1]
                score = float(score)
                if score > 1.0:
                    score = 1.0 / (1.0 + np.exp(-score))  # Sigmoid normalization
                elif score < 0.0:
                    score = 0.0

                # Handle label polarity (e.g., NEGATIVE should give low score)
                if label and "neg" in label.lower():
                    score = 1.0 - score

                return score
            else:
                logger.warning(f"Unexpected pipeline output format: {result}")
                return 0.0

        except Exception as e:
            logger.warning(f"Error computing score with {model_id}: {e}")
            return 0.0

    def _compute_score_direct(self, model_id: str, text: str) -> float:
        """Compute reward score using direct model inference.

        Args:
            model_id: Model identifier
            text: Input text

        Returns:
            Reward score in [0, 1]
        """
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]

        if tokenizer is None or model is None:
            return 0.0

        try:
            # Truncate and tokenize
            max_length = getattr(tokenizer, "model_max_length", 512)
            inputs = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract logits
            logits = outputs.logits

            if logits.shape[-1] == 2:
                # Binary classification - use positive class probability
                probs = torch.softmax(logits, dim=-1)
                score = probs[0, 1].item()
            elif logits.shape[-1] == 1:
                # Regression - apply sigmoid
                score = torch.sigmoid(logits[0, 0]).item()
            else:
                # Multi-class - use max probability
                probs = torch.softmax(logits, dim=-1)
                score = probs[0].max().item()

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, float(score)))
            return score

        except Exception as e:
            logger.warning(f"Error computing score with {model_id}: {e}")
            return 0.0

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Compute ensemble reward score.

        Args:
            text: Input text to score
            reference: Optional reference text (unused but kept for compatibility)
            **kwargs: Additional arguments

        Returns:
            Aggregated reward score in [0, 1]
        """
        # Lazy load models on first use
        self._load_models()

        # Compute scores from all models
        scores = []
        for model_id in self.model_ids:
            if model_id not in self.models:
                # Model failed to load
                continue

            if model_id in self.pipelines:
                score = self._compute_score_pipeline(model_id, text)
            else:
                score = self._compute_score_direct(model_id, text)

            scores.append(score)

        if not scores:
            return 0.0

        # Aggregate using specified mode
        if self.ensemble_mode == "mean":
            return float(np.mean(scores))

        elif self.ensemble_mode == "worst_case":
            # Return minimum score (conservative, prevents reward hacking)
            return float(np.min(scores))

        elif self.ensemble_mode == "uncertainty_weighted":
            # Weight by inverse of standard deviation
            mean_score = np.mean(scores)
            std_dev = np.std(scores)

            # If all models agree (low std), use mean
            # If models disagree (high std), penalize by reducing score
            if std_dev < 1e-6:
                return float(mean_score)

            # Uncertainty weight: (1 - normalized_std)
            # Since scores are [0,1], max std is 0.5
            normalized_std = std_dev / 0.5
            uncertainty_weight = max(0.0, 1.0 - normalized_std)

            # Return mean score weighted by confidence
            return float(mean_score * uncertainty_weight)

        return 0.0

    def batch_compute(
        self,
        texts: List[str],
        references: Optional[List[str]] = None,
        return_breakdown: bool = False,
        **kwargs
    ) -> Union[List[float], Tuple[List[float], List[Dict[str, float]]]]:
        """Compute ensemble rewards for a batch of texts.

        Args:
            texts: List of texts to score
            references: Optional list of reference texts
            return_breakdown: If True, return per-model scores
            **kwargs: Additional arguments

        Returns:
            List of aggregated scores, or tuple with breakdown if requested
        """
        if references is None:
            references = [None] * len(texts)

        scores = []
        breakdowns = [] if return_breakdown else None

        for text, ref in zip(texts, references):
            # Lazy load models on first use
            self._load_models()

            # Compute scores from all models
            model_scores = {}
            for model_id in self.model_ids:
                if model_id not in self.models:
                    continue

                if model_id in self.pipelines:
                    score = self._compute_score_pipeline(model_id, text)
                else:
                    score = self._compute_score_direct(model_id, text)

                model_scores[model_id] = score

            if not model_scores:
                scores.append(0.0)
                if return_breakdown:
                    breakdowns.append({})
                continue

            # Aggregate
            model_score_list = list(model_scores.values())

            if self.ensemble_mode == "mean":
                aggregated = float(np.mean(model_score_list))
            elif self.ensemble_mode == "worst_case":
                aggregated = float(np.min(model_score_list))
            elif self.ensemble_mode == "uncertainty_weighted":
                mean_score = np.mean(model_score_list)
                std_dev = np.std(model_score_list)

                if std_dev < 1e-6:
                    aggregated = float(mean_score)
                else:
                    normalized_std = std_dev / 0.5
                    uncertainty_weight = max(0.0, 1.0 - normalized_std)
                    aggregated = float(mean_score * uncertainty_weight)
            else:
                aggregated = 0.0

            scores.append(aggregated)

            if return_breakdown:
                breakdowns.append(model_scores)

        if return_breakdown:
            return scores, breakdowns
        return scores

    def get_ensemble_mode(self) -> str:
        """Get current ensemble aggregation mode."""
        return self.ensemble_mode

    def set_ensemble_mode(self, mode: str) -> None:
        """Change ensemble aggregation mode.

        Args:
            mode: New mode ("mean", "worst_case", or "uncertainty_weighted")
        """
        if mode not in ["mean", "worst_case", "uncertainty_weighted"]:
            raise ValueError(
                f"ensemble_mode must be 'mean', 'worst_case', or 'uncertainty_weighted', "
                f"got {mode}"
            )
        self.ensemble_mode = mode
        logger.info(f"Changed ensemble mode to {mode}")

    def get_model_ids(self) -> List[str]:
        """Get list of loaded model IDs."""
        return list(self.models.keys())

    def get_ensemble_stats(self, text: str) -> Dict[str, Any]:
        """Get detailed ensemble statistics for a text.

        Returns per-model scores, mean, std, min, max.

        Args:
            text: Input text

        Returns:
            Dictionary with ensemble statistics
        """
        self._load_models()

        scores = {}
        for model_id in self.model_ids:
            if model_id not in self.models:
                continue

            if model_id in self.pipelines:
                score = self._compute_score_pipeline(model_id, text)
            else:
                score = self._compute_score_direct(model_id, text)

            scores[model_id] = score

        if not scores:
            return {}

        score_list = list(scores.values())
        return {
            "per_model_scores": scores,
            "mean": float(np.mean(score_list)),
            "std_dev": float(np.std(score_list)),
            "min": float(np.min(score_list)),
            "max": float(np.max(score_list)),
            "num_models": len(scores),
        }
