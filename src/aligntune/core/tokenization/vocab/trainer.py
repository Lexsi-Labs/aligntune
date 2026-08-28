"""
Train tokenizers from scratch using HuggingFace tokenizers library.

Supports:
- BPE training (byte-level and character-level)
- SentencePiece BPE training
- WordPiece training
- Unigram training
"""

import logging
from typing import List, Optional, Iterator, Dict, Any, Union
from pathlib import Path

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def train_tokenizer_from_scratch(
    corpus: Union[List[str], Iterator[str]],
    vocab_size: int = 32000,
    model_type: str = "bpe",
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2,
    output_dir: Optional[str] = None,
    byte_level: bool = True,
    show_progress: bool = True,
    **kwargs
) -> Tokenizer:
    """
    Train a tokenizer from scratch on a given corpus.

    Args:
        corpus: Text corpus as list of strings or iterator
        vocab_size: Target vocabulary size
        model_type: Type of tokenizer ("bpe", "wordpiece", "unigram")
        special_tokens: List of special tokens (e.g., ["<unk>", "<s>", "</s>"])
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save trained tokenizer (optional)
        byte_level: Whether to use byte-level encoding (for BPE only)
        show_progress: Show training progress bar
        **kwargs: Additional arguments for trainer

    Returns:
        Trained tokenizer

    Examples:
        >>> # Train BPE tokenizer on Hindi corpus
        >>> hindi_corpus = ["नमस्ते", "धन्यवाद", "मैं ठीक हूँ", ...]
        >>> tokenizer = train_tokenizer_from_scratch(
        ...     corpus=hindi_corpus,
        ...     vocab_size=32000,
        ...     model_type="bpe",
        ...     special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
        ... )
        >>> tokenizer.save("hindi_tokenizer.json")
    """
    logger.info(f"Training {model_type} tokenizer with vocab_size={vocab_size}")

    # Default special tokens if not provided
    if special_tokens is None:
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]

    # Create tokenizer based on model type
    if model_type.lower() == "bpe":
        tokenizer = _train_bpe_tokenizer(
            corpus=corpus,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            byte_level=byte_level,
            show_progress=show_progress,
            **kwargs
        )
    elif model_type.lower() == "wordpiece":
        tokenizer = _train_wordpiece_tokenizer(
            corpus=corpus,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            show_progress=show_progress,
            **kwargs
        )
    elif model_type.lower() == "unigram":
        tokenizer = _train_unigram_tokenizer(
            corpus=corpus,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'bpe', 'wordpiece', or 'unigram'")

    # Save tokenizer if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "tokenizer.json"
        tokenizer.save(str(save_path))
        logger.info(f"Saved trained tokenizer to {save_path}")

    return tokenizer


def _train_bpe_tokenizer(
    corpus: Union[List[str], Iterator[str]],
    vocab_size: int,
    special_tokens: List[str],
    min_frequency: int = 2,
    byte_level: bool = True,
    show_progress: bool = True,
    **kwargs
) -> Tokenizer:
    """
    Train a BPE tokenizer.

    Args:
        corpus: Training corpus
        vocab_size: Target vocabulary size
        special_tokens: Special tokens
        min_frequency: Minimum token frequency
        byte_level: Use byte-level encoding
        show_progress: Show progress bar

    Returns:
        Trained BPE tokenizer
    """
    # Initialize BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Set up normalizer (optional - can disable for some languages)
    if kwargs.get("normalize", True):
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),  # Unicode normalization
            normalizers.Lowercase() if kwargs.get("lowercase", False) else normalizers.NFD(),
        ])

    # Set up pre-tokenizer
    if byte_level:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
    else:
        # Character-level pre-tokenization (better for non-Latin scripts)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Metaspace(),
        ])
        tokenizer.decoder = decoders.Metaspace()

    # Create trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=show_progress,
        initial_alphabet=[] if byte_level else None,
    )

    # Train on corpus
    if isinstance(corpus, list):
        tokenizer.train_from_iterator(corpus, trainer=trainer)
    else:
        # Iterator
        tokenizer.train_from_iterator(corpus, trainer=trainer)

    logger.info(f"BPE tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def _train_wordpiece_tokenizer(
    corpus: Union[List[str], Iterator[str]],
    vocab_size: int,
    special_tokens: List[str],
    min_frequency: int = 2,
    show_progress: bool = True,
    **kwargs
) -> Tokenizer:
    """
    Train a WordPiece tokenizer (BERT-style).

    Args:
        corpus: Training corpus
        vocab_size: Target vocabulary size
        special_tokens: Special tokens
        min_frequency: Minimum token frequency
        show_progress: Show progress bar

    Returns:
        Trained WordPiece tokenizer
    """
    # Initialize WordPiece model
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))

    # Normalizer
    if kwargs.get("normalize", True):
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase() if kwargs.get("lowercase", False) else normalizers.NFD(),
            normalizers.StripAccents() if kwargs.get("strip_accents", False) else normalizers.NFD(),
        ])

    # Pre-tokenizer (BERT-style)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Decoder
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    # Trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=show_progress,
    )

    # Train
    if isinstance(corpus, list):
        tokenizer.train_from_iterator(corpus, trainer=trainer)
    else:
        tokenizer.train_from_iterator(corpus, trainer=trainer)

    logger.info(f"WordPiece tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def _train_unigram_tokenizer(
    corpus: Union[List[str], Iterator[str]],
    vocab_size: int,
    special_tokens: List[str],
    show_progress: bool = True,
    **kwargs
) -> Tokenizer:
    """
    Train a Unigram tokenizer (SentencePiece Unigram).

    Args:
        corpus: Training corpus
        vocab_size: Target vocabulary size
        special_tokens: Special tokens
        show_progress: Show progress bar

    Returns:
        Trained Unigram tokenizer
    """
    # Initialize Unigram model
    tokenizer = Tokenizer(Unigram())

    # Normalizer
    if kwargs.get("normalize", True):
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase() if kwargs.get("lowercase", False) else normalizers.NFD(),
        ])

    # Pre-tokenizer (Metaspace for SentencePiece compatibility)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    # Decoder
    tokenizer.decoder = decoders.Metaspace()

    # Trainer
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=show_progress,
        unk_token="<unk>",
    )

    # Train
    if isinstance(corpus, list):
        tokenizer.train_from_iterator(corpus, trainer=trainer)
    else:
        tokenizer.train_from_iterator(corpus, trainer=trainer)

    logger.info(f"Unigram tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_corpus_from_dataset(
    dataset_name: str,
    text_column: str = "text",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> Iterator[str]:
    """
    Load corpus from HuggingFace dataset for tokenizer training.

    Args:
        dataset_name: HuggingFace dataset name
        text_column: Column containing text
        split: Dataset split
        max_samples: Maximum number of samples to load
        streaming: Use streaming mode

    Yields:
        Text samples from dataset

    Examples:
        >>> corpus = load_corpus_from_dataset(
        ...     dataset_name="wikimedia/wikipedia",
        ...     text_column="text",
        ...     split="train",
        ...     max_samples=100000,
        ... )
        >>> tokenizer = train_tokenizer_from_scratch(corpus, vocab_size=32000)
    """
    from datasets import load_dataset

    logger.info(f"Loading corpus from {dataset_name} (split={split}, max_samples={max_samples})")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    # Yield text samples
    count = 0
    for sample in dataset:
        if text_column not in sample:
            logger.warning(f"Text column '{text_column}' not found in sample, skipping")
            continue

        text = sample[text_column]
        if text and isinstance(text, str):
            yield text
            count += 1

            if max_samples and count >= max_samples:
                break

    logger.info(f"Loaded {count} samples from {dataset_name}")


def convert_to_pretrained_tokenizer(
    tokenizer: Tokenizer,
    model_max_length: int = 512,
    padding_side: str = "right",
    truncation_side: str = "right",
) -> PreTrainedTokenizerFast:
    """
    Convert a trained tokenizer to HuggingFace PreTrainedTokenizerFast.

    Args:
        tokenizer: Trained tokenizer
        model_max_length: Maximum sequence length
        padding_side: Padding side ("left" or "right")
        truncation_side: Truncation side ("left" or "right")

    Returns:
        PreTrainedTokenizerFast instance

    Examples:
        >>> tokenizer = train_tokenizer_from_scratch(corpus, vocab_size=32000)
        >>> hf_tokenizer = convert_to_pretrained_tokenizer(tokenizer)
        >>> hf_tokenizer.save_pretrained("./my_tokenizer")
    """
    # Create PreTrainedTokenizerFast wrapper
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=model_max_length,
        padding_side=padding_side,
        truncation_side=truncation_side,
    )

    return hf_tokenizer
