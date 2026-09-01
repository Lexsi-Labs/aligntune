from typing import Optional, Iterable, Dict, Set

from tokenizers import pre_tokenizers

from tokenizer_extension.utils import get_vocab_and_merges, get_added_tokens_vocab, disable_ignore_merges
from tokenizer_extension.sentencepiece_utils import BYTE_VOCAB

try:
    from tokenization_scorer import score as tokenization_scorer
except ImportError:
    tokenization_scorer = None

HF_BYTE_VOCAB = set(pre_tokenizers.ByteLevel.alphabet())
SP_BYTE_VOCAB = set(BYTE_VOCAB)


def find_unreachable_tokens_tokenization(tokenizer, ignore_added=True):
    with disable_ignore_merges(tokenizer):
        vocab, merges = get_vocab_and_merges(tokenizer)

        reachability = {}
        for tok in vocab:
            values = [t.value for t in tokenizer._tokenizer.model.tokenize(tok)]
            reachability[tok] = len(values) == 1 and values[0] == tok

        unreachable_tokens = set(tok for tok, value in reachability.items() if value == False)
        if ignore_added:
            unreachable_tokens = unreachable_tokens - set(get_added_tokens_vocab(tokenizer))
        return unreachable_tokens


def evaluate_tokenizer_self(
        tokenizer,
        extension_vocab: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    unreachable_tokens_tok = find_unreachable_tokens_tokenization(tokenizer)

    if extension_vocab is not None:
        unreachable_tokens_tok_ext = unreachable_tokens_tok.intersection(extension_vocab)
    else:
        unreachable_tokens_tok_ext = set()
    vocab = tokenizer.get_vocab()
    return {
        "vocab_size": len(vocab),
        "unreachable_tokens_tok": len(unreachable_tokens_tok),
        "mean_token_length": (sum(len(x) for x in vocab) / len(vocab)),
        "unreachable_tokens_tok_extension": len(unreachable_tokens_tok_ext),
        "mean_token_length_extension": (
            sum(len(x) for x in extension_vocab) / len(extension_vocab) if extension_vocab is not None else 0
        ),
    }


def evaluate_tokenizer(
        tokenizer,
        data: Iterable[str],
        ignore_empty: bool = True,
        extension_vocab: Optional[Dict[str, int]] = None,
        byte_vocab: Set[str] = None,
        is_sentencepiece: bool = False,
        return_frequencies: bool = False,
) -> Dict[str, float]:
    total_tokens = 0
    total_chars = 0
    total_words = 0
    total_examples = 0
    unknown_tokens = 0
    total_bytes = 0
    vocab = tokenizer.get_vocab()
    vocab_usage = {k: 0 for k in vocab}
    total_bytes_per_tokens = 0

    if byte_vocab is None:
        byte_vocab = SP_BYTE_VOCAB if is_sentencepiece else HF_BYTE_VOCAB

    total_byte_fallbacks = 0

    for text in data:
        if text == "":
            if ignore_empty:
                print("Empty text in dataset.")
                continue
            raise ValueError("Empty text in dataset.")

        tokens = tokenizer.tokenize(text)
        if byte_vocab is not None:
            total_byte_fallbacks += len([t for t in tokens if t in byte_vocab])

        for t in tokens:
            vocab_usage[t] += 1

        text_bytes = len(text.encode('utf-8'))
        text_tokens = len(tokens)
        total_tokens += text_tokens
        total_bytes += text_bytes
        total_bytes_per_tokens += text_bytes / text_tokens
        total_chars += len(text)
        total_words += len(text.split())
        total_examples += 1
        unknown_tokens += len([t for t in tokens if t not in vocab])

    vocab_utilisation = len(set(k for k, v in vocab_usage.items() if v != 0)) / len(vocab)
    if extension_vocab is not None and len(extension_vocab) > 0:
        extension_utilisation = len(set(k for k in extension_vocab if vocab_usage[k] != 0)) / len(extension_vocab)
        extensions_tokens = sum(vocab_usage[k] for k in extension_vocab)
    else:
        extension_utilisation = 0
        extensions_tokens = 0

    results = {
        "vocab_size": len(vocab),
        "dataset_size": total_examples,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "unknown_tokens": unknown_tokens,
        "chars_per_token": total_chars / total_tokens,
        "tokens_per_char": total_tokens / total_chars,
        "tokens_per_byte": total_tokens / total_bytes,
        "bytes_per_token": total_bytes / total_tokens,
        "average_bytes_per_token": total_bytes_per_tokens / total_examples,
        "words_per_token": total_words / total_tokens,
        "tokens_per_word": total_tokens / total_words,
        "tokens_per_sequence": total_tokens / total_examples,
        "unknown_tokens_per_tokens": unknown_tokens / total_tokens,
        "extension_usage_rate": extension_utilisation,
        "vocab_usage_rate": vocab_utilisation,
        "total_extension_tokens": extensions_tokens,
        "extension_tokens_per_token": extensions_tokens / total_tokens,
        "byte_fallback_rate": total_byte_fallbacks / total_tokens,
    }
    if return_frequencies:
        results["frequencies"] = vocab_usage

    return results


def evaluate_renyi_efficiency(
        tokenizer,
        data: Iterable[str],
) -> float:
    if tokenization_scorer is None:
        raise ImportError("tokenization_scorer is not installed, cannot compute Renyi entropy.")
    tokenized_text = [" ".join(tokenizer.tokenize(text)) for text in data]
    return tokenization_scorer(tokenized_text, metric="renyi", power=2.5)
