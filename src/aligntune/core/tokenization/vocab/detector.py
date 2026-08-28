"""
Tokenizer type detection utilities.

Detects the type of tokenizer (BPE, SentencePiece, WordPiece, Unigram)
to determine which vocabulary extension methods are supported.
"""

import logging
from enum import Enum
from typing import Union, Any

logger = logging.getLogger(__name__)


class TokenizerType(Enum):
    """Supported tokenizer types."""
    SENTENCEPIECE_BPE = "sentencepiece_bpe"  # LLaMA, Mistral, Qwen
    BPE = "bpe"                              # GPT-2, RoBERTa
    WORDPIECE = "wordpiece"                  # BERT
    UNIGRAM = "unigram"                      # T5, mBART
    UNKNOWN = "unknown"


def detect_tokenizer_type(tokenizer: Any) -> TokenizerType:
    """
    Detect the type of a HuggingFace tokenizer.

    Args:
        tokenizer: A HuggingFace tokenizer instance

    Returns:
        TokenizerType enum indicating the tokenizer type

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer_type = detect_tokenizer_type(tokenizer)
        >>> print(tokenizer_type)
        TokenizerType.SENTENCEPIECE_BPE
    """
    # Check class name
    class_name = tokenizer.__class__.__name__

    # SentencePiece-based tokenizers (most modern LLMs)
    if any(name in class_name for name in ["Llama", "Mistral", "Qwen", "Gemma"]):
        logger.info(f"Detected SentencePiece BPE tokenizer: {class_name}")
        return TokenizerType.SENTENCEPIECE_BPE

    # Check if it's a PreTrainedTokenizerFast with backend info
    if hasattr(tokenizer, "backend_tokenizer") and tokenizer.backend_tokenizer is not None:
        backend = tokenizer.backend_tokenizer
        backend_class = str(type(backend))

        # Check model type in backend
        if hasattr(backend, "model"):
            model_type = str(type(backend.model))

            if "BPE" in model_type:
                logger.info(f"Detected BPE tokenizer from backend: {model_type}")
                return TokenizerType.BPE
            elif "WordPiece" in model_type:
                logger.info(f"Detected WordPiece tokenizer from backend: {model_type}")
                return TokenizerType.WORDPIECE
            elif "Unigram" in model_type:
                logger.info(f"Detected Unigram tokenizer from backend: {model_type}")
                return TokenizerType.UNIGRAM

    # Check by known model patterns
    if any(name in class_name for name in ["GPT2", "Roberta", "BART"]):
        logger.info(f"Detected BPE tokenizer: {class_name}")
        return TokenizerType.BPE

    if any(name in class_name for name in ["Bert", "DistilBert", "Electra"]):
        logger.info(f"Detected WordPiece tokenizer: {class_name}")
        return TokenizerType.WORDPIECE

    if any(name in class_name for name in ["T5", "mBART", "Albert"]):
        logger.info(f"Detected Unigram tokenizer: {class_name}")
        return TokenizerType.UNIGRAM

    # Check vocab file names if available
    if hasattr(tokenizer, "vocab_files_names"):
        vocab_files = tokenizer.vocab_files_names
        if "tokenizer_file" in vocab_files or "vocab_file" in vocab_files:
            # Try to infer from vocab structure
            if hasattr(tokenizer, "get_vocab"):
                vocab = tokenizer.get_vocab()
                # SentencePiece typically has these special tokens
                if "▁" in str(list(vocab.keys())[:100]):  # Check first 100 tokens
                    logger.info("Detected SentencePiece BPE from vocab structure")
                    return TokenizerType.SENTENCEPIECE_BPE

    logger.warning(f"Could not determine tokenizer type for {class_name}, returning UNKNOWN")
    return TokenizerType.UNKNOWN


def supports_continued_bpe(tokenizer_type: TokenizerType) -> bool:
    """
    Check if a tokenizer type supports continued BPE training.

    Args:
        tokenizer_type: The tokenizer type

    Returns:
        True if continued BPE is supported, False otherwise
    """
    supported = tokenizer_type in [
        TokenizerType.SENTENCEPIECE_BPE,
        TokenizerType.BPE,
    ]
    return supported


def get_recommended_extension_method(tokenizer_type: TokenizerType) -> str:
    """
    Get the recommended vocabulary extension method for a tokenizer type.

    Args:
        tokenizer_type: The tokenizer type

    Returns:
        Recommended method: "continued_bpe" or "naive_extension"
    """
    if supports_continued_bpe(tokenizer_type):
        return "continued_bpe"
    else:
        return "naive_extension"
