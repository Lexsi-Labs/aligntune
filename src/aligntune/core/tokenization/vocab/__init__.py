"""
Vocabulary extension and tokenizer training modules.

Supports:
- Training tokenizers from scratch (BPE, WordPiece, Unigram)
- Naive vocabulary extension
- Continued BPE training
"""

from .trainer import train_tokenizer_from_scratch
from .detector import detect_tokenizer_type, TokenizerType
from .naive_extension import (
    extend_tokenizer_naive,
    load_tokens_from_file,
    save_tokens_to_file,
)

__all__ = [
    "train_tokenizer_from_scratch",
    "detect_tokenizer_type",
    "TokenizerType",
    "extend_tokenizer_naive",
    "load_tokens_from_file",
    "save_tokens_to_file",
]
