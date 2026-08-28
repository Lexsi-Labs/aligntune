"""
Naive vocabulary extension for tokenizers.

Supports two methods:
1. Direct token addition from file (JSON/TXT)
2. Train new tokenizer + extract diff tokens
"""

import logging
import json
from typing import List, Optional, Set, Union, Dict, Any
from pathlib import Path

from .trainer import train_tokenizer_from_scratch, load_corpus_from_dataset

logger = logging.getLogger(__name__)


def extend_tokenizer_naive(
    base_tokenizer: Any,
    tokens: Optional[List[str]] = None,
    tokens_file: Optional[str] = None,
    corpus: Optional[Union[List[str], str]] = None,
    new_tokenizer_vocab_size: int = 32000,
    model_type: str = "bpe",
    **kwargs
) -> Dict[str, Any]:
    """
    Extend tokenizer vocabulary using naive extension method.

    Two modes:
    1. Direct addition: Provide tokens list or tokens_file
    2. Train + diff: Provide corpus to train new tokenizer and extract diff

    Args:
        base_tokenizer: Base tokenizer to extend (HF tokenizer)
        tokens: List of tokens to add directly (Method 1)
        tokens_file: Path to JSON/TXT file with tokens (Method 1)
        corpus: Corpus for training new tokenizer (Method 2)
        new_tokenizer_vocab_size: Vocab size for new tokenizer (Method 2)
        model_type: Type of tokenizer to train (Method 2)
        **kwargs: Additional arguments

    Returns:
        Dictionary with extension results:
            - num_added_tokens: Number of tokens added
            - added_tokens: List of added tokens
            - new_vocab_size: New vocabulary size
            - method: "direct" or "train_diff"

    Examples:
        >>> # Method 1: Direct addition from file
        >>> result = extend_tokenizer_naive(
        ...     base_tokenizer=llama_tokenizer,
        ...     tokens_file="hindi_tokens.json"
        ... )

        >>> # Method 2: Train + diff
        >>> result = extend_tokenizer_naive(
        ...     base_tokenizer=llama_tokenizer,
        ...     corpus=hindi_corpus,
        ...     new_tokenizer_vocab_size=32000
        ... )
    """
    # Determine which method to use
    if tokens is not None or tokens_file is not None:
        # Method 1: Direct addition
        return _extend_from_tokens(base_tokenizer, tokens, tokens_file)
    elif corpus is not None:
        # Method 2: Train + diff
        return _extend_from_corpus(
            base_tokenizer,
            corpus,
            new_tokenizer_vocab_size,
            model_type,
            **kwargs
        )
    else:
        raise ValueError(
            "Must provide either 'tokens'/'tokens_file' or 'corpus' for extension"
        )


def _extend_from_tokens(
    base_tokenizer: Any,
    tokens: Optional[List[str]] = None,
    tokens_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Method 1: Direct token addition from list or file.

    Args:
        base_tokenizer: Base tokenizer to extend
        tokens: List of tokens to add
        tokens_file: Path to JSON/TXT file with tokens

    Returns:
        Extension results dictionary
    """
    logger.info("Using naive extension: direct token addition")

    # Load tokens from file if provided
    if tokens_file:
        tokens = load_tokens_from_file(tokens_file)
        logger.info(f"Loaded {len(tokens)} tokens from {tokens_file}")

    if not tokens:
        raise ValueError("No tokens provided for extension")

    # Get base vocab to avoid duplicates
    base_vocab = set(base_tokenizer.get_vocab().keys())

    # Filter out tokens already in base vocab
    new_tokens = [t for t in tokens if t not in base_vocab]

    if not new_tokens:
        logger.warning("All provided tokens already exist in base vocabulary")
        return {
            "num_added_tokens": 0,
            "added_tokens": [],
            "new_vocab_size": len(base_tokenizer),
            "method": "direct",
        }

    logger.info(f"Adding {len(new_tokens)} new tokens (filtered from {len(tokens)} provided)")

    # Add tokens to base tokenizer
    num_added = base_tokenizer.add_tokens(new_tokens)

    logger.info(f"Successfully added {num_added} tokens to vocabulary")

    # Check for unreachable tokens
    unreachable_count = _count_unreachable_tokens(base_tokenizer)

    return {
        "num_added_tokens": num_added,
        "added_tokens": new_tokens[:100],  # Return first 100 for logging
        "new_vocab_size": len(base_tokenizer),
        "method": "naive_extension",
        "unreachable_tokens": unreachable_count,
    }


def _extend_from_corpus(
    base_tokenizer: Any,
    corpus: Union[List[str], str],
    new_tokenizer_vocab_size: int,
    model_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Method 2: Train new tokenizer on corpus, extract diff, add to base.

    Args:
        base_tokenizer: Base tokenizer to extend
        corpus: Training corpus (list of strings or dataset name)
        new_tokenizer_vocab_size: Vocab size for new tokenizer
        model_type: Type of tokenizer to train
        **kwargs: Additional training arguments

    Returns:
        Extension results dictionary
    """
    logger.info("Using naive extension: train + diff method")

    # Load corpus if it's a dataset name
    if isinstance(corpus, str):
        logger.info(f"Loading corpus from dataset: {corpus}")
        corpus_iter = load_corpus_from_dataset(
            dataset_name=corpus,
            text_column=kwargs.get('text_column', 'text'),
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples', None),
            streaming=kwargs.get('streaming', False),
        )
        # Convert to list for training
        corpus = list(corpus_iter)
        logger.info(f"Loaded {len(corpus)} samples from dataset")

    # Train new tokenizer from scratch
    logger.info(f"Training new {model_type} tokenizer with vocab_size={new_tokenizer_vocab_size}")
    new_tokenizer = train_tokenizer_from_scratch(
        corpus=corpus,
        vocab_size=new_tokenizer_vocab_size,
        model_type=model_type,
        special_tokens=kwargs.get('special_tokens', ["<unk>", "<s>", "</s>", "<pad>"]),
        min_frequency=kwargs.get('min_frequency', 2),
        byte_level=kwargs.get('byte_level', True),
        show_progress=kwargs.get('show_progress', True),
    )

    # Get vocabularies
    base_vocab = set(base_tokenizer.get_vocab().keys())
    new_vocab = set(new_tokenizer.get_vocab().keys())

    # Find tokens in new vocab but not in base vocab
    diff_tokens = new_vocab - base_vocab
    diff_tokens_list = list(diff_tokens)

    logger.info(f"Found {len(diff_tokens_list)} new tokens not in base vocabulary")

    if not diff_tokens_list:
        logger.warning("No new tokens found in trained tokenizer")
        return {
            "num_added_tokens": 0,
            "added_tokens": [],
            "new_vocab_size": len(base_tokenizer),
            "method": "naive_extension",
        }

    # Add diff tokens to base tokenizer
    num_added = base_tokenizer.add_tokens(diff_tokens_list)

    logger.info(f"Successfully added {num_added} tokens to vocabulary")

    # Check for unreachable tokens
    unreachable_count = _count_unreachable_tokens(base_tokenizer)

    return {
        "num_added_tokens": num_added,
        "added_tokens": diff_tokens_list[:100],  # Return first 100 for logging
        "new_vocab_size": len(base_tokenizer),
        "method": "naive_extension",
        "trained_tokenizer_vocab_size": new_tokenizer.get_vocab_size(),
        "unreachable_tokens": unreachable_count,
    }


def load_tokens_from_file(file_path: str) -> List[str]:
    """
    Load tokens from JSON or TXT file.

    Supported formats:
    - JSON: ["token1", "token2", ...] or {"tokens": ["token1", "token2", ...]}
    - TXT: One token per line

    Args:
        file_path: Path to tokens file

    Returns:
        List of tokens

    Examples:
        >>> # JSON format
        >>> tokens = load_tokens_from_file("hindi_tokens.json")

        >>> # TXT format
        >>> tokens = load_tokens_from_file("hindi_tokens.txt")
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Tokens file not found: {file_path}")

    # Determine file type by extension
    if path.suffix.lower() == '.json':
        return _load_tokens_from_json(path)
    elif path.suffix.lower() in ['.txt', '.text']:
        return _load_tokens_from_txt(path)
    else:
        # Try to guess format
        logger.warning(f"Unknown file extension {path.suffix}, attempting auto-detection")
        try:
            return _load_tokens_from_json(path)
        except json.JSONDecodeError:
            return _load_tokens_from_txt(path)


def _load_tokens_from_json(path: Path) -> List[str]:
    """Load tokens from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        # Format: ["token1", "token2", ...]
        tokens = data
    elif isinstance(data, dict):
        # Format: {"tokens": ["token1", "token2", ...]}
        if "tokens" in data:
            tokens = data["tokens"]
        else:
            raise ValueError(
                f"JSON file must contain 'tokens' key or be a list of tokens. "
                f"Found keys: {list(data.keys())}"
            )
    else:
        raise ValueError(f"Invalid JSON format. Expected list or dict, got {type(data)}")

    # Validate all items are strings
    if not all(isinstance(t, str) for t in tokens):
        raise ValueError("All tokens must be strings")

    logger.info(f"Loaded {len(tokens)} tokens from JSON file")
    return tokens


def _load_tokens_from_txt(path: Path) -> List[str]:
    """Load tokens from TXT file (one token per line)."""
    with open(path, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(tokens)} tokens from TXT file")
    return tokens


def save_tokens_to_file(tokens: List[str], file_path: str, format: str = "json") -> None:
    """
    Save tokens to file (JSON or TXT format).

    Args:
        tokens: List of tokens to save
        file_path: Output file path
        format: File format ("json" or "txt")

    Examples:
        >>> tokens = ["नमस्ते", "धन्यवाद", "मुंबई"]
        >>> save_tokens_to_file(tokens, "hindi_tokens.json", format="json")
        >>> save_tokens_to_file(tokens, "hindi_tokens.txt", format="txt")
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
    elif format.lower() == "txt":
        with open(path, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(token + '\n')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'")

    logger.info(f"Saved {len(tokens)} tokens to {path}")


def _count_unreachable_tokens(tokenizer: Any) -> int:
    """
    Count unreachable tokens in tokenizer vocabulary.

    Uses tokenizer_extension library to find tokens that exist in vocabulary
    but cannot be produced by merge rules.

    Args:
        tokenizer: HuggingFace tokenizer to check

    Returns:
        Number of unreachable tokens
    """
    try:
        from tokenizer_extension.benchmarking import find_unreachable_tokens_tokenization

        unreachable_tokens = find_unreachable_tokens_tokenization(tokenizer)
        count = len(unreachable_tokens)

        if count > 0:
            logger.warning(
                f"Found {count} unreachable tokens "
                f"({count / len(tokenizer) * 100:.2f}% of vocabulary)"
            )
        else:
            logger.info("No unreachable tokens found")

        return count

    except ImportError:
        logger.warning(
            "tokenizer-extension library not available for unreachable token check. "
            "Install with: pip install tokenizer-extension"
        )
        return -1  # Return -1 to indicate check was not performed
    except Exception as e:
        logger.warning(f"Failed to check unreachable tokens: {e}")
        return -1
