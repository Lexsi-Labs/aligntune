"""
Evaluation metrics for tokenization quality.

Provides metrics to assess:
- Unreachable tokens (structure validation)
- Compression ratio (efficiency)
- Fertility (tokens per word)
- Coverage (character/script coverage)
"""

import logging
from typing import List, Dict, Any, Optional, Iterable
from itertools import islice

logger = logging.getLogger(__name__)


def evaluate_unreachable_tokens(tokenizer) -> Dict[str, Any]:
    """
    Check for unreachable tokens in the tokenizer vocabulary.

    Unreachable tokens are tokens that exist in the vocabulary but cannot
    be produced by the tokenizer's merge rules (BPE structure issue).

    Args:
        tokenizer: HuggingFace tokenizer to evaluate

    Returns:
        Dictionary with:
            - unreachable_count: Number of unreachable tokens
            - unreachable_tokens: List of unreachable token strings
            - vocab_size: Total vocabulary size
            - reachable_ratio: Percentage of reachable tokens

    Examples:
        >>> result = evaluate_unreachable_tokens(tokenizer)
        >>> print(f"Unreachable: {result['unreachable_count']}/{result['vocab_size']}")
    """
    try:
        from tokenizer_extension.benchmarking import find_unreachable_tokens_tokenization

        logger.info("Checking for unreachable tokens...")
        unreachable_tokens = find_unreachable_tokens_tokenization(tokenizer)

        vocab_size = len(tokenizer)
        unreachable_count = len(unreachable_tokens)
        reachable_count = vocab_size - unreachable_count
        reachable_ratio = (reachable_count / vocab_size * 100) if vocab_size > 0 else 0.0

        result = {
            "unreachable_count": unreachable_count,
            "unreachable_tokens": list(unreachable_tokens)[:100],  # Limit output (convert set to list)
            "vocab_size": vocab_size,
            "reachable_count": reachable_count,
            "reachable_ratio": reachable_ratio,
        }

        if unreachable_count == 0:
            logger.info("✓ No unreachable tokens found (perfect structure!)")
        else:
            logger.warning(
                f"⚠ Found {unreachable_count} unreachable tokens "
                f"({100 - reachable_ratio:.2f}% of vocabulary)"
            )

        return result

    except ImportError as e:
        logger.error("tokenizer-extension library not available for evaluation")
        return {
            "unreachable_count": -1,
            "error": str(e),
            "vocab_size": len(tokenizer),
        }


def evaluate_fertility(
    tokenizer,
    corpus: Iterable[str],
    max_sample: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate tokenizer fertility on corpus sample.

    Fertility measures how many tokens are needed to represent words.
    Lower fertility = more efficient tokenization for the domain.

    Thresholds:
    - < 1.3: Good - tokenizer knows domain well
    - 1.3-1.5: Acceptable
    - > 1.5: Poor - corpus may be too small or need more tokens

    Args:
        tokenizer: HuggingFace tokenizer to evaluate
        corpus: Corpus to evaluate (accepts iterator for memory efficiency)
        max_sample: Maximum number of texts to sample (default 500)

    Returns:
        Dictionary with:
            - fertility: Tokens per word ratio
            - avg_tokens_per_text: Average tokens per text
            - avg_words_per_text: Average words per text
            - sample_count: Number of texts sampled
            - total_tokens: Total tokens in sample
            - total_words: Total words in sample
            - estimated_capacity: Estimated new tokens corpus can support (if corpus size known)

    Examples:
        >>> corpus = ["Hello world", "Testing tokenizer", ...]
        >>> result = evaluate_fertility(tokenizer, corpus)
        >>> print(f"Fertility: {result['fertility']:.2f}")
    """
    logger.info(f"Evaluating fertility on sample of {max_sample} texts...")

    # Sample corpus using islice (memory efficient - no list conversion)
    sample = islice(corpus, max_sample)

    total_tokens = 0
    total_words = 0
    sample_count = 0

    # Process sample one by one
    for text in sample:
        if not text or not isinstance(text, str):
            continue

        # Tokenize text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()

        total_tokens += len(tokens)
        total_words += len(words)
        sample_count += 1

    # Calculate metrics
    if sample_count == 0:
        logger.warning("No valid texts in corpus sample")
        return {
            "fertility": -1,
            "error": "Empty corpus sample",
            "sample_count": 0,
        }

    if total_words == 0:
        logger.warning("No words found in corpus sample")
        return {
            "fertility": -1,
            "error": "No words in corpus",
            "sample_count": sample_count,
            "total_tokens": total_tokens,
        }

    fertility = total_tokens / total_words
    avg_tokens_per_text = total_tokens / sample_count
    avg_words_per_text = total_words / sample_count

    result = {
        "fertility": fertility,
        "avg_tokens_per_text": avg_tokens_per_text,
        "avg_words_per_text": avg_words_per_text,
        "sample_count": sample_count,
        "total_tokens": total_tokens,
        "total_words": total_words,
    }

    # Log fertility assessment
    if fertility < 1.3:
        logger.info(f"✓ Good fertility: {fertility:.2f} (tokenizer knows domain well)")
    elif fertility < 1.5:
        logger.info(f"○ Acceptable fertility: {fertility:.2f}")
    else:
        logger.warning(f"⚠ High fertility: {fertility:.2f} (may need more tokens or larger corpus)")

    logger.info(f"Sample stats: {sample_count} texts, {total_tokens} tokens, {total_words} words")

    return result


def evaluate_tokenizer(
    model_name: str,
    dataset_name: str,
    *,
    metrics: List[str],
    config_name: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    eval_samples: int = 500,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """
    Run complete tokenizer evaluation.

    Args:
        model_name: Model/tokenizer identifier to load
        dataset_name: Dataset identifier/path to load
        metrics: Metrics to run. Allowed values are ``"fertility"`` and
            ``"unreachable"``.
        config_name: Optional dataset configuration
        split: Dataset split to load
        text_column: Dataset column containing text
        max_samples: Maximum corpus rows to load
        eval_samples: Maximum corpus rows used for fertility evaluation

    Returns:
        Dictionary with all evaluation metrics

    Examples:
        >>> results = evaluate_tokenizer(
        ...     model_name="gpt2",
        ...     dataset_name="my-corpus.jsonl",
        ...     text_column="text",
        ... )
        >>> print(f"Unreachable tokens: {results['unreachable']['unreachable_count']}")
    """
    requested_metrics = set(metrics)
    valid_metrics = {"fertility", "unreachable"}
    invalid_metrics = requested_metrics - valid_metrics
    if not requested_metrics:
        raise ValueError("metrics must contain 'fertility', 'unreachable', or both")
    if invalid_metrics:
        raise ValueError(
            f"Unknown tokenizer metrics: {sorted(invalid_metrics)}. "
            f"Allowed values: {sorted(valid_metrics)}"
        )

    from transformers import AutoTokenizer
    from aligntune.data import load_corpus

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    test_corpus = load_corpus(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        text_column=text_column,
        max_samples=max_samples,
    )

    results = {}

    # Metric 1: Unreachable tokens
    logger.info("=" * 60)
    logger.info("Running Tokenizer Evaluation")
    logger.info("=" * 60)

    if "unreachable" in requested_metrics:
        results["unreachable"] = evaluate_unreachable_tokens(tokenizer)

    if "fertility" in requested_metrics:
        results["fertility"] = evaluate_fertility(
            tokenizer,
            test_corpus,
            max_sample=eval_samples,
        )

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)

    return results

def compare_tokenizer_fertility(
    old_tokenizer,
    new_tokenizer,
    corpus: Iterable[str],
    eval_samples: int = 500,
    eval_percentage: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare fertility between old (base) and new (extended) tokenizer.

    Evaluates how much the tokenizer improved after vocabulary extension.
    Lower fertility = better tokenization for the domain.

    Args:
        old_tokenizer: Base tokenizer before extension
        new_tokenizer: Extended tokenizer after vocabulary addition
        corpus: Corpus to evaluate on (accepts iterator for memory efficiency)
        eval_samples: Number of samples to evaluate (default 500)
        eval_percentage: Alternatively, specify as percentage of corpus (overrides eval_samples if provided)

    Returns:
        Dictionary with:
            - old_fertility: Fertility before extension
            - new_fertility: Fertility after extension
            - fertility_reduction: Percentage reduction in fertility
            - fertility_improvement: How much fertility decreased (lower is better)
            - sample_count: Number of texts evaluated
            - metrics_before: Full metrics from old tokenizer
            - metrics_after: Full metrics from new tokenizer
            - recommendation: User-friendly interpretation

    Examples:
        >>> corpus = ["नमस्ते दोस्त", "कैसे हो", ...]
        >>> comparison = compare_tokenizer_fertility(
        ...     old_tokenizer=original_gpt2,
        ...     new_tokenizer=extended_gpt2,
        ...     corpus=corpus,
        ...     eval_samples=1000
        ... )
        >>> print(f"Fertility reduced from {comparison['old_fertility']:.2f} to {comparison['new_fertility']:.2f}")
        >>> print(f"Improvement: {comparison['fertility_reduction']:.1f}%")
    """
    logger.info("=" * 70)
    logger.info("TOKENIZER FERTILITY COMPARISON: Before vs After Extension")
    logger.info("=" * 70)

    # Determine sample size
    if eval_percentage is not None:
        # User specified percentage - we can't know total size with iterator
        logger.warning(
            f"Note: Using {eval_percentage}% - converting to absolute samples. "
            f"Assuming ~100 texts/page, using {eval_percentage * 10} samples."
        )
        actual_samples = max(100, eval_percentage * 10)
    else:
        actual_samples = eval_samples

    logger.info(f"Evaluating on {actual_samples} corpus samples...")

    # Split corpus using tee so we can evaluate both tokenizers on same texts
    from itertools import islice, tee

    corpus_iter1, corpus_iter2 = tee(corpus, 2)

    # Get sample texts
    sample_texts = list(islice(corpus_iter1, actual_samples))
    actual_sample_count = len(sample_texts)

    logger.info(f"Using {actual_sample_count} texts for comparison")

    # Evaluate BEFORE (old tokenizer)
    logger.info("Evaluating BEFORE extension (original tokenizer)...")
    metrics_before = evaluate_fertility(
        tokenizer=old_tokenizer,
        corpus=sample_texts,
        max_sample=actual_sample_count,
    )

    # Evaluate AFTER (new tokenizer)
    logger.info("Evaluating AFTER extension (extended tokenizer)...")
    metrics_after = evaluate_fertility(
        tokenizer=new_tokenizer,
        corpus=sample_texts,
        max_sample=actual_sample_count,
    )

    # Calculate comparison
    result = {
        "sample_count": actual_sample_count,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }

    # Check for errors
    if "error" in metrics_before or "error" in metrics_after:
        logger.error("Could not complete comparison due to evaluation errors")
        result["error"] = "Evaluation failed for one or both tokenizers"
        return result

    old_fertility = metrics_before["fertility"]
    new_fertility = metrics_after["fertility"]

    result["old_fertility"] = old_fertility
    result["new_fertility"] = new_fertility

    # Calculate improvement
    if old_fertility > 0:
        fertility_reduction = ((old_fertility - new_fertility) / old_fertility) * 100
        fertility_improvement = old_fertility - new_fertility
    else:
        fertility_reduction = 0
        fertility_improvement = 0

    result["fertility_reduction_percentage"] = fertility_reduction
    result["fertility_improvement_absolute"] = fertility_improvement

    # Generate recommendation
    if fertility_reduction < 0:
        recommendation = (
            f"WARNING: Fertility INCREASED from {old_fertility:.2f} to {new_fertility:.2f}\n"
            f"The extension may have hurt tokenization efficiency.\n"
            f"Consider retraining with different parameters."
        )
    elif fertility_reduction < 5:
        recommendation = (
            f"Minimal improvement: {fertility_reduction:.1f}% reduction\n"
            f"The extension provided small gains ({fertility_improvement:.2f} tokens/word).\n"
            f"Consider more training data or different token count."
        )
    elif fertility_reduction < 15:
        recommendation = (
            f"Moderate improvement: {fertility_reduction:.1f}% reduction\n"
            f"Good extension results ({fertility_improvement:.2f} tokens/word improvement).\n"
            f"Tokenizer now better understands the domain."
        )
    elif fertility_reduction < 30:
        recommendation = (
            f"Strong improvement: {fertility_reduction:.1f}% reduction\n"
            f"Significant gains ({fertility_improvement:.2f} tokens/word improvement).\n"
            f"Extension was highly successful for this domain."
        )
    else:
        recommendation = (
            f"Excellent improvement: {fertility_reduction:.1f}% reduction\n"
            f"Outstanding extension results ({fertility_improvement:.2f} tokens/word improvement).\n"
            f"Tokenizer now highly specialized for this domain."
        )

    result["recommendation"] = recommendation

    # Log detailed comparison
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Original Tokenizer (BEFORE):")
    logger.info(f"  Fertility: {old_fertility:.2f} tokens/word")
    logger.info(f"  Avg tokens/text: {metrics_before['avg_tokens_per_text']:.1f}")
    logger.info(f"  Sample: {actual_sample_count} texts")
    logger.info("")
    logger.info(f"Extended Tokenizer (AFTER):")
    logger.info(f"  Fertility: {new_fertility:.2f} tokens/word")
    logger.info(f"  Avg tokens/text: {metrics_after['avg_tokens_per_text']:.1f}")
    logger.info(f"  Sample: {actual_sample_count} texts")
    logger.info("")
    logger.info(f"IMPROVEMENT:")
    logger.info(f"  Fertility reduced by: {fertility_improvement:.2f} tokens/word")
    logger.info(f"  Percentage reduction: {fertility_reduction:.1f}%")
    logger.info("")
    logger.info(recommendation)
    logger.info("=" * 70)

    return result
