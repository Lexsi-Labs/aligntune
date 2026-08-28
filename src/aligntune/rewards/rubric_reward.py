"""
Rubric-anchored reward function using LLM-as-judge scoring.

This module provides a bridge between open-domain RL tasks and scalable evaluation
through LLM-scored rubrics. Unlike learned reward models, rubric-anchored rewards:

- Provide interpretable, human-aligned scoring criteria
- Avoid reward model training instability (RLVR collapse)
- Support open-ended generation tasks without reference answers
- Cache scores for identical (rubric, text) pairs

Rubric-Anchored RL for AlignTune
"""

import logging
from typing import List, Optional, Union
import numpy as np

from aligntune.eval.llm_judge import LLMJudge
from .core import RewardFunction, RewardConfig, RewardType

logger = logging.getLogger(__name__)


class RubricReward(RewardFunction):
    """
    Reward function that scores completions against LLM-evaluated rubrics.

    This class wraps an LLMJudge instance to provide stable, interpretable rewards
    for open-ended generation tasks. The underlying judge handles caching, so
    identical (rubric, text) pairs are only scored once.

    Attributes:
        rubric: The evaluation rubric string describing scoring criteria
        judge: LLMJudge instance used for scoring
        _cache_size: Maximum number of cached scores

    Example:
        >>> from aligntune.eval.llm_judge import OpenAIJudge
        >>> judge = OpenAIJudge(model="gpt-4o-mini")
        >>> rubric = "Is this response helpful? Score 1.0 = very helpful, 0.0 = not helpful"
        >>> reward = RubricReward(rubric=rubric, judge=judge)
        >>> score = reward.compute("This is a great answer")
        >>> scores = reward.batch_compute(["Answer 1", "Answer 2"])
    """

    def __init__(
        self,
        rubric: str,
        judge: Optional[LLMJudge] = None,
        cache_size: int = 10000,
        **kwargs
    ):
        """
        Initialize RubricReward.

        Args:
            rubric: Evaluation rubric string (e.g., HELPFULNESS_RUBRIC)
            judge: LLMJudge instance for scoring
            cache_size: Maximum cache size (not directly used here; judge has its own cache)
            **kwargs: Additional arguments passed to parent RewardFunction

        Raises:
            ValueError: If rubric is empty or judge is None
            TypeError: If judge is not an LLMJudge instance
        """
        if not rubric or not isinstance(rubric, str):
            raise ValueError("Rubric must be a non-empty string")
        if judge is None:
            raise ValueError("Judge cannot be None")
        if not isinstance(judge, LLMJudge):
            raise TypeError(f"Judge must be an LLMJudge instance, got {type(judge)}")

        # Create a minimal config if not provided
        if 'config' not in kwargs:
            kwargs['config'] = RewardConfig(
                reward_type=RewardType.RUBRIC if hasattr(RewardType, 'RUBRIC')
                else RewardType.HELPFULNESS,
                weight=1.0,
                params={"rubric": rubric}
            )

        super().__init__(kwargs['config'])

        self.rubric = rubric
        self.judge = judge
        self._cache_size = cache_size

        logger.debug(f"Initialized RubricReward with rubric: {rubric[:50]}...")

    def compute(
        self,
        completion: str,
        reference: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Score a single completion against the rubric.

        Args:
            completion: The text to score
            reference: Unused (for interface compatibility)
            prompt: Optional original prompt (if available, improves judge context)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            A float in [0.0, 1.0] representing the score

        Note:
            - If prompt is not provided, a placeholder is used
            - Scores are cached by the judge, so repeated calls with identical
              (prompt, completion, rubric) are instant on subsequent calls
        """
        if not completion or not isinstance(completion, str):
            logger.warning(f"Invalid completion: {completion!r}")
            return 0.0

        # Use provided prompt or fallback to generic placeholder
        eval_prompt = prompt or "[Original prompt unavailable]"

        try:
            score = self.judge.score(eval_prompt, completion, self.rubric)
            logger.debug(f"Scored completion: {score:.3f}")
            return score
        except Exception as e:
            logger.error(f"Error scoring completion: {e}", exc_info=True)
            return 0.5  # Fallback to neutral score on error

    def batch_compute(
        self,
        completions: Union[List[str], np.ndarray],
        reference: Optional[Union[List[str], np.ndarray]] = None,
        prompts: Optional[Union[List[str], np.ndarray]] = None,
        **kwargs
    ) -> List[float]:
        """
        Score multiple completions against the rubric.

        Args:
            completions: List of texts to score
            reference: Unused (for interface compatibility)
            prompts: Optional list of original prompts (same length as completions)
            **kwargs: Additional keyword arguments passed to compute()

        Returns:
            List of floats in [0.0, 1.0], one per completion

        Note:
            If prompts is not provided, all completions are scored with a placeholder prompt.
            This method benefits from judge caching: identical texts score instantly on
            subsequent batches.
        """
        # Convert numpy arrays to lists first -- `if not completions` on a
        # numpy array with more than one element raises "truth value of an
        # array... is ambiguous", so the emptiness check must come after this.
        if isinstance(completions, np.ndarray):
            completions = completions.tolist()
        if prompts is not None and isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()

        if not completions:
            return []

        # Default prompts if not provided
        if prompts is None:
            prompts = ["[Original prompt unavailable]"] * len(completions)
        elif len(prompts) != len(completions):
            logger.warning(
                f"Length mismatch: {len(completions)} completions but {len(prompts)} prompts. "
                f"Using placeholder for missing prompts."
            )
            prompts = list(prompts) + ["[Original prompt unavailable]"] * (
                len(completions) - len(prompts)
            )

        scores = []
        for completion, prompt in zip(completions, prompts):
            score = self.compute(completion, prompt=prompt, **kwargs)
            scores.append(score)

        return scores

    def get_cache_info(self) -> dict:
        """
        Get cache statistics from the underlying judge.

        Returns:
            Dictionary with cache statistics (e.g., size, hits)
        """
        if not hasattr(self.judge, '_cache'):
            return {"error": "Judge does not expose cache info"}

        return {
            "cache_size": len(self.judge._cache),
            "max_cache_size": self._cache_size
        }

    def clear_cache(self) -> None:
        """Clear the judge's internal cache."""
        if hasattr(self.judge, '_cache'):
            self.judge._cache.clear()
            logger.info("Cleared RubricReward judge cache")
        else:
            logger.warning("Judge does not have a clearable cache")
