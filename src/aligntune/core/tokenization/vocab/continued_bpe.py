"""
Continued BPE training wrapper using tokenizer-extension library.

This module provides a clean interface to the tokenizer-extension library
for continued BPE training on new domains/languages.

Credit: Based on tokenizer-extension by Purason et al. (2025)
Paper: "Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation
        for Pre-trained Models" (https://arxiv.org/abs/2512.03989)
GitHub: https://github.com/taidopurason/tokenizer-extension

Citation:
@misc{purason2025teachingoldtokenizersnew,
    title={Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation
           for Pre-trained Models},
    author={Taido Purason and Pavel Chizhov and Ivan P. Yamshchikov and Mark Fishel},
    year={2025},
    eprint={2512.03989},
    archivePrefix={arXiv},
}
"""

import logging
from typing import List, Dict, Any, Union, Iterator, Optional
from itertools import tee

logger = logging.getLogger(__name__)


def extend_tokenizer_continued_bpe(
    base_tokenizer: Any,
    corpus: Union[List[str], Iterator[str]],
    num_new_tokens: int = 20000,
    is_sentencepiece: bool = False,
    max_token_length: Optional[int] = None,
    show_progress: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Extend tokenizer vocabulary using continued BPE training.

    This method continues the BPE training process on a new corpus,
    learning new tokens and merges that are fully compatible with
    the existing vocabulary. This ensures no unreachable tokens.

    Args:
        base_tokenizer: Base HuggingFace tokenizer to extend
        corpus: Training corpus (list of strings or iterator)
        num_new_tokens: Number of new tokens to add
        is_sentencepiece: Whether base tokenizer is SentencePiece-based
        max_token_length: Maximum length of created tokens
        show_progress: Show training progress bar
        **kwargs: Additional arguments passed to train_vocab_extension

    Returns:
        Dictionary containing:
            - num_added_tokens: Number of tokens added
            - new_vocab_size: New vocabulary size
            - method: "continued_bpe"
            - vocab: New vocabulary dict
            - merges: New merge operations
            - unreachable_tokens: Count of unreachable tokens (should be 0)

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> hindi_corpus = ["नमस्ते", "धन्यवाद", ...] * 1000
        >>> result = extend_tokenizer_continued_bpe(
        ...     base_tokenizer=tokenizer,
        ...     corpus=hindi_corpus,
        ...     num_new_tokens=20000,
        ... )
        >>> print(f"Added {result['num_added_tokens']} tokens")

    Note:
        Modifies the base_tokenizer in-place.
    """
    try:
        from tokenizer_extension.train_vocab_extension import train_vocab_extension
        from tokenizer_extension.extension import extend_tokenizer
        from tokenizer_extension.benchmarking import find_unreachable_tokens_tokenization
    except ImportError as e:
        raise ImportError(
            "tokenizer_extension failed to import. It is vendored with aligntune "
            "(third_party/tokenizer-extension), so this usually means a broken "
            "install - try reinstalling aligntune."
        ) from e

    if is_sentencepiece:
        # tokenizer_extension.sentencepiece_utils does `try: import icu
        # except ImportError: icu = None` and then unconditionally calls
        # icu.Script.getScript(...) while validating merges - without PyICU
        # installed that fails with AttributeError: 'NoneType' object has
        # no attribute 'Script', deep inside training, only after the whole
        # corpus has already been processed. Fail fast instead, before that
        # work is wasted.
        try:
            import icu  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PyICU is required for continued BPE training on "
                "SentencePiece tokenizers (used to detect Unicode script "
                "boundaries when validating merges). "
                "Install with: pip install pyicu "
                "(requires the system ICU headers: "
                "apt-get install -y libicu-dev)."
            ) from e

    logger.info(
        f"Starting continued BPE training: adding {num_new_tokens} tokens "
        f"(is_sentencepiece={is_sentencepiece})"
    )

    # Estimate corpus capacity before training (prevents wasting time)
    corpus_is_iterator = hasattr(corpus, '__iter__') and not isinstance(corpus, list)

    if corpus_is_iterator:
        # Split iterator into two: one for estimation, one for training
        corpus_for_estimation, corpus = tee(corpus, 2)
    else:
        corpus_for_estimation = corpus

    logger.info("Estimating corpus capacity for requested token count...")
    fertility_before = None
    sample_texts_for_comparison = None

    try:
        from aligntune.eval.tokenization import evaluate_fertility
        from itertools import islice

        # Convert sample to list so we can evaluate before AND after on same texts
        sample_texts_for_comparison = list(islice(corpus_for_estimation, 500))

        logger.info(f"Sampling {len(sample_texts_for_comparison)} texts for fertility evaluation...")

        fertility_result_before = evaluate_fertility(
            tokenizer=base_tokenizer,
            corpus=sample_texts_for_comparison,  # Pass as list for reuse
            max_sample=len(sample_texts_for_comparison),  # Use all samples we saved
        )

        if "error" not in fertility_result_before:
            fertility_before = fertility_result_before["fertility"]
            avg_tokens_per_text = fertility_result_before["avg_tokens_per_text"]
            sample_count = fertility_result_before["sample_count"]

            # Estimate capacity using 1/1000 rule (empirical from tokenizer.md)
            # Capacity = total_corpus_tokens / 1000
            # We don't know exact corpus size, but we can estimate from sample
            logger.info(
                f"BEFORE Extension - Fertility: {fertility_before:.2f} (sample: {sample_count} texts, "
                f"avg {avg_tokens_per_text:.1f} tokens/text)"
            )

            # Heuristic warning based on fertility alone
            if fertility_before > 1.5:
                logger.warning(
                    f"⚠ HIGH FERTILITY WARNING: {fertility_before:.2f}\n"
                    f"   The tokenizer is fragmenting text heavily, suggesting:\n"
                    f"   - Corpus may be too small for {num_new_tokens} tokens\n"
                    f"   - Domain is very different from base tokenizer\n"
                    f"   Rule of thumb: Need ~1M tokens in corpus per 1k new tokens\n"
                    f"   Estimated need: ~{num_new_tokens * 1000:,} corpus tokens\n"
                    f"   Your corpus sample has: ~{avg_tokens_per_text:.1f} tokens/text\n"
                    f"   Consider: Increase corpus size or reduce num_new_tokens"
                )
            elif fertility_before > 1.3:
                logger.warning(
                    f"○ Moderate fertility: {fertility_before:.2f} - training may succeed but watch for errors"
                )
        else:
            logger.warning(f"Could not estimate fertility: {fertility_result_before.get('error')}")

    except ImportError:
        logger.warning("Fertility estimation skipped (eval module not available)")
    except Exception as e:
        logger.warning(f"Fertility estimation failed: {e}")

    # Convert iterator to list if needed
    if corpus_is_iterator:
        logger.info("Converting corpus iterator to list...")
        corpus = list(corpus)
        logger.info(f"Loaded {len(corpus)} samples")

    # Train vocabulary extension
    logger.info("Training vocabulary extension...")
    extension_result = train_vocab_extension(
        tokenizer=base_tokenizer,
        corpus=corpus,
        extension_size=num_new_tokens,
        is_sentencepiece=is_sentencepiece,
        max_token_length=max_token_length,
        sp_kwargs=kwargs.get('sp_kwargs', None),
    )

    logger.info(
        f"Training complete: {len(extension_result['vocab'])} new tokens, "
        f"{len(extension_result['merges'])} new merges"
    )

    # Apply extension to tokenizer
    logger.info("Applying extension to tokenizer...")
    extend_tokenizer(
        base_tokenizer,
        new_vocab=extension_result["vocab"],
        new_merges=extension_result["merges"],
        n_tokens=num_new_tokens,
        keep_added_token_positions=kwargs.get('keep_added_token_positions', True),
    )

    new_vocab_size = len(base_tokenizer)
    logger.info(f"Extension applied. New vocab size: {new_vocab_size}")

    # Check for unreachable tokens
    logger.info("Checking for unreachable tokens...")
    unreachable_tokens = find_unreachable_tokens_tokenization(base_tokenizer)

    if len(unreachable_tokens) == 0:
        logger.info("✓ No unreachable tokens found (good!)")
    else:
        logger.warning(
            f"⚠ Found {len(unreachable_tokens)} unreachable tokens. "
            "This is unexpected with continued BPE training."
        )

    return {
        "num_added_tokens": num_new_tokens,
        "new_vocab_size": new_vocab_size,
        "method": "continued_bpe",
        "vocab": extension_result["vocab"],
        "merges": extension_result["merges"],
        "unreachable_tokens": len(unreachable_tokens),
    }

