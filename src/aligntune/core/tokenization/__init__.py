"""
Tokenization module for multilingual vocabulary adaptation.

This module provides utilities for:
- Vocabulary extension (continued BPE, naive extension)
- Vocabulary pruning (leaf-based, frequency-based)
- Embedding initialization
- Tokenization evaluation metrics
- Complete tokenization training workflow

Based on research from "Teaching Old Tokenizers New Words" (Purason et al., 2025)
"""

from .config import (
    UnifiedTokenizationConfig,
    TokenizationConfig,
    VocabExtensionMethod,
)
from .trainer import TokenizationTrainer
from .vocab.continued_bpe import extend_tokenizer_continued_bpe
from .vocab.naive_extension import extend_tokenizer_naive
from .vocab.pruning_wrapper import VocabularyPruner

# Legacy Indic support (kept for backward compatibility)
from .extension import TokenizerExtender

__all__ = [
    # Main interfaces
    "UnifiedTokenizationConfig",
    "TokenizationConfig",
    "TokenizationTrainer",

    # Operations
    "extend_tokenizer_continued_bpe",
    "extend_tokenizer_naive",
    "VocabularyPruner",

    # Enums
    "VocabExtensionMethod",

    # Legacy Indic support
    "TokenizerExtender",
]
