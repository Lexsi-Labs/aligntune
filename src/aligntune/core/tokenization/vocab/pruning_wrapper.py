"""
Vocabulary pruning wrapper using tokenizer-extension library.

Provides clean interface for pruning tokenizer vocabularies using
structure-aware methods that avoid creating unreachable tokens.

Credit: Based on tokenizer-extension by Purason et al. (2025)
Paper: "Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation
        for Pre-trained Models" (https://arxiv.org/abs/2512.03989)
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class VocabularyPruner:
    """
    Wrapper for vocabulary pruning operations.

    Supports multiple pruning strategies:
    - leaf_frequency: Structure-aware frequency pruning (RECOMMENDED)
    - frequency: Simple frequency-based pruning
    - last_n: Remove last N tokens by ID
    """

    def __init__(self, method: str = "leaf_frequency"):
        """
        Initialize pruner with specified method.

        Args:
            method: Pruning method ("leaf_frequency", "frequency", "last_n")
        """
        try:
            from tokenizer_extension.pruning import (
                LeafFrequencyPruner,
                FrequencyPruner,
                LastNPruner,
                PretrainedPruner,
            )
            self._LeafFrequencyPruner = LeafFrequencyPruner
            self._FrequencyPruner = FrequencyPruner
            self._LastNPruner = LastNPruner
            self._PretrainedPruner = PretrainedPruner
        except ImportError as e:
            raise ImportError(
                "tokenizer_extension failed to import. It is vendored with aligntune "
                "(third_party/tokenizer-extension), so this usually means a broken "
                "install - try reinstalling aligntune."
            ) from e

        self.method = method
        self._pruner = None
        self._create_pruner()

    def _create_pruner(self):
        """Create pruner instance based on method."""
        if self.method == "leaf_frequency":
            self._pruner = self._LeafFrequencyPruner()
            logger.info("Created leaf-based frequency pruner (structure-aware)")
        elif self.method == "frequency":
            self._pruner = self._FrequencyPruner()
            logger.info("Created simple frequency pruner")
        elif self.method == "last_n":
            self._pruner = self._LastNPruner()
            logger.info("Created last-N pruner (no training needed)")
        else:
            raise ValueError(
                f"Unknown pruning method: {self.method}. "
                "Use 'leaf_frequency', 'frequency', or 'last_n'"
            )

    def train(
        self,
        tokenizer: Any,
        corpus: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Train pruner on corpus to learn token frequencies.

        Args:
            tokenizer: Tokenizer to analyze
            corpus: Training corpus (not needed for last_n method)
            show_progress: Show progress bar

        Note:
            For last_n method, corpus is not used.
        """
        if self.method == "last_n":
            logger.info("Training last-N pruner (no corpus needed)...")
            self._pruner.train(tokenizer)
        else:
            if corpus is None:
                raise ValueError(f"{self.method} pruner requires a training corpus")
            logger.info(f"Training {self.method} pruner on {len(corpus)} samples...")
            self._pruner.train(tokenizer, corpus)

        logger.info("✓ Pruner training complete")

    def prune(
        self,
        tokenizer: Any,
        n_tokens: int,
    ) -> Dict[str, Any]:
        """
        Prune n_tokens from the tokenizer vocabulary.

        Args:
            tokenizer: Tokenizer to prune (modified in-place)
            n_tokens: Number of tokens to remove

        Returns:
            Dictionary with pruning results:
                - n_tokens_pruned: Number of tokens removed
                - new_vocab_size: New vocabulary size
                - unreachable_tokens: Count of unreachable tokens after pruning

        Note:
            Modifies tokenizer in-place.

        Examples:
            >>> pruner = VocabularyPruner(method="leaf_frequency")
            >>> pruner.train(tokenizer, corpus)
            >>> result = pruner.prune(tokenizer, n_tokens=5000)
            >>> print(f"Pruned to {result['new_vocab_size']} tokens")
        """
        if self._pruner is None:
            raise RuntimeError("Pruner not initialized. Call train() first.")

        logger.info(f"Pruning {n_tokens} tokens using {self.method} method...")

        vocab_before = len(tokenizer)

        # Perform pruning
        self._pruner.prune(tokenizer, n_tokens)

        vocab_after = len(tokenizer)
        actual_pruned = vocab_before - vocab_after

        logger.info(
            f"✓ Pruning complete: {vocab_before} → {vocab_after} "
            f"(removed {actual_pruned} tokens)"
        )

        # Check for unreachable tokens
        try:
            from tokenizer_extension.benchmarking import find_unreachable_tokens_tokenization
            unreachable = find_unreachable_tokens_tokenization(tokenizer)
            unreachable_count = len(unreachable)

            if unreachable_count == 0:
                logger.info("✓ No unreachable tokens after pruning")
            else:
                logger.warning(
                    f"⚠ {unreachable_count} unreachable tokens after pruning. "
                    "Consider using leaf_frequency method."
                )
        except Exception as e:
            logger.warning(f"Could not check unreachable tokens: {e}")
            unreachable_count = -1

        return {
            "n_tokens_pruned": actual_pruned,
            "new_vocab_size": vocab_after,
            "unreachable_tokens": unreachable_count,
            "method": self.method,
        }

    def save(self, path: str) -> None:
        """
        Save trained pruner to file.

        Args:
            path: Output file path (JSON)

        Examples:
            >>> pruner.save("leaf_freq_pruner.json")
        """
        if self._pruner is None:
            raise RuntimeError("Pruner not initialized. Call train() first.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._pruner.save(str(path))
        logger.info(f"✓ Saved pruner to {path}")

    @classmethod
    def load(cls, path: str) -> "VocabularyPruner":
        """
        Load trained pruner from file.

        Args:
            path: Path to saved pruner file

        Returns:
            VocabularyPruner instance

        Examples:
            >>> pruner = VocabularyPruner.load("leaf_freq_pruner.json")
            >>> pruner.prune(tokenizer, n_tokens=5000)
        """
        try:
            from tokenizer_extension.pruning import PretrainedPruner
        except ImportError as e:
            raise ImportError(
                "tokenizer_extension failed to import. It is vendored with aligntune "
                "(third_party/tokenizer-extension), so this usually means a broken "
                "install - try reinstalling aligntune."
            ) from e

        logger.info(f"Loading pruner from {path}...")

        # Create wrapper instance
        wrapper = cls.__new__(cls)
        wrapper._pruner = PretrainedPruner.load(path)
        wrapper.method = "pretrained"

        logger.info("✓ Pruner loaded from file")
        return wrapper


def prune_vocabulary(
    tokenizer: Any,
    corpus: Optional[List[str]],
    n_tokens: int,
    method: str = "leaf_frequency",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to prune vocabulary in one call.

    Args:
        tokenizer: Tokenizer to prune (modified in-place)
        corpus: Training corpus (not needed for last_n method)
        n_tokens: Number of tokens to remove
        method: Pruning method ("leaf_frequency", "frequency", "last_n")
        save_path: Optional path to save trained pruner

    Returns:
        Pruning results dictionary

    Examples:
        >>> result = prune_vocabulary(
        ...     tokenizer=extended_tokenizer,
        ...     corpus=hindi_corpus,
        ...     n_tokens=5000,
        ...     method="leaf_frequency",
        ... )
    """
    pruner = VocabularyPruner(method=method)
    pruner.train(tokenizer, corpus)

    if save_path:
        pruner.save(save_path)

    return pruner.prune(tokenizer, n_tokens)
