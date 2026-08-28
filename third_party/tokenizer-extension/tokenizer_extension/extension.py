import logging
import warnings
from itertools import islice
from typing import Optional, List, Tuple, Dict, Union
from tokenizers import pre_tokenizers
from transformers.convert_slow_tokenizer import generate_merges

from tokenizer_extension.sentencepiece_utils import read_model_proto, ILLEGAL_CHARS, save_model_proto, save_vocab_proto
from tokenizer_extension.utils import get_ordered_vocab, get_vocab_and_merges, replace_tokenizer_vocab_merges, \
    get_added_tokens_vocab, update_postprocessor_special_tokens


def get_vocab_scores(
        vocab: Union[List[str], Dict[str, int]],
        alphabet: Optional[List[str]] = None,
        added_tokens: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    if alphabet is None:
        alphabet = []

    if added_tokens is None:
        added_tokens = []

    if isinstance(vocab, dict):
        vocab = get_ordered_vocab(vocab)

    ignore_vocab = set(alphabet + added_tokens)

    score = -1
    scores = []
    for t in vocab:
        if t in ignore_vocab:
            scores.append(0)
        else:
            scores.append(score)
            score -= 1

    return [(t, s) for t, s in zip(vocab, scores)]


def extend_vocab_merges(
        new_vocab: Dict[str, int],
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        max_token_id: int,
        new_merges: Optional[List[Tuple[str, str]]] = None,
        n_tokens: Optional[int] = None,
        generate_new_merges: bool = False,
        alphabet: Optional[Tuple[List[str], str]] = "byte",
        added_tokens: Optional[List[str]] = None,
        prepend_merges: bool = False,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    if alphabet is None:
        alphabet = []
    elif isinstance(alphabet, str):
        if alphabet == "byte":
            alphabet = list(sorted(pre_tokenizers.ByteLevel.alphabet()))
        elif alphabet == "char":
            alphabet = [x for x in get_ordered_vocab(vocab) if len(x) == 1]
        else:
            raise ValueError(f"Unknown alphabet type: {alphabet}")

    new_vocab_filtered = {k: v for k, v in new_vocab.items() if k not in vocab and k not in added_tokens}
    if n_tokens is not None and len(new_vocab_filtered) < n_tokens:
        raise ValueError(
            f"Not enough new tokens to add. Found {len(new_vocab_filtered)}, but expected {n_tokens}."
        )

    new_vocab_filtered = {
        k: max_token_id + v + 1
        for v, k in enumerate(islice(get_ordered_vocab(new_vocab_filtered), n_tokens))
    }
    combined_vocab = {**vocab, **new_vocab_filtered}

    if generate_new_merges:
        if prepend_merges:
            raise ValueError("Cannot prepend merges when regenerating all merges.")
        if new_merges is not None:
            warnings.warn("new_merges is ignored when generate_new_merges is set to True.")
        combined_vocab_scores = get_vocab_scores(combined_vocab, alphabet=alphabet, added_tokens=added_tokens)
        combined_merges = generate_merges(combined_vocab, combined_vocab_scores)
        return combined_vocab, combined_merges

    if new_merges is None:
        new_vocab_scores = get_vocab_scores(new_vocab_filtered, alphabet=alphabet, added_tokens=added_tokens)
        new_merges = generate_merges(combined_vocab, new_vocab_scores)
    else:
        existing_merges = set(merges)
        new_merges = [
            (a, b) for a, b in new_merges
            if a in combined_vocab and b in combined_vocab and a + b in combined_vocab and (a, b) not in existing_merges
        ]

    combined_merges = new_merges + merges if prepend_merges else merges + new_merges

    return combined_vocab, combined_merges


def extend_tokenizer(
        tokenizer,
        new_vocab: Union[Dict[str, int], List[str]],
        new_merges: Optional[List[Tuple[str, str]]] = None,
        n_tokens: int = None,
        generate_new_merges: bool = False,
        prepend_merges: bool = False,
        alphabet: Optional[Union[List[str], str]] = "byte",
        keep_added_token_positions: bool = False
):
    """
    Extends the tokenizer with new tokens and merges inplace (changing the original tokenizer object).
    :param tokenizer: Tokenizer to extend
    :param new_vocab: New vocabulary to add
    :param new_merges: The merges to add. If None, merges will be generated based on the new_vocab, leaving existing merges intact.
    :param n_tokens: Number of tokens to add from new_vocab. If None, all tokens from new_vocab will be added.
    :param generate_new_merges: Regenerates all merges if set to True.
    :param prepend_merges: instead of appending the new merges to the end of the existing merges, prepend them to the beginning.
    :param alphabet: The alphabet to use when generating merges. Can be "byte", "char" or a list of characters.
    :param keep_added_token_positions: Keep the indices of special tokens of the original tokenizer
    :return: the tokenizer object that was extended inplace
    """
    if not isinstance(new_vocab, dict):
        new_vocab = {tok: idx for idx, tok in enumerate(new_vocab)}
    vocab, merges = get_vocab_and_merges(tokenizer)
    max_token_id = max(tokenizer._tokenizer.get_vocab(keep_added_token_positions).values())
    added_tokens_vocab = get_added_tokens_vocab(tokenizer)
    ext_vocab, ext_merges = extend_vocab_merges(
        new_vocab,
        vocab,
        merges,
        max_token_id,
        new_merges=new_merges,
        n_tokens=n_tokens,
        generate_new_merges=generate_new_merges,
        added_tokens=list(added_tokens_vocab.keys()),
        prepend_merges=prepend_merges,
        alphabet=alphabet
    )
    ext_vocab_reverse = {idx: token for token, idx in ext_vocab.items()}
    if keep_added_token_positions:
        logging.info("Adding added tokens to the vocabulary if not present to preserve their indices")
        for token, idx in added_tokens_vocab.items():
            if token in ext_vocab:
                continue
            if idx in ext_vocab_reverse:
                raise ValueError("Added token and extended vocabulary have tokens with the same indices")
            ext_vocab[token] = idx
    tokenizer = replace_tokenizer_vocab_merges(tokenizer, ext_vocab, ext_merges)
    if not keep_added_token_positions:
        tokenizer = update_postprocessor_special_tokens(tokenizer)
    if len(set(tokenizer.get_vocab().keys())) != len(set(tokenizer.get_vocab().values())):
        raise ValueError("Tokens with the same ID found in vocabulary.")
    return tokenizer


def extend_sp_model(tokenizer_prefix, extension_vocab, out_prefix, n_tokens=None):
    from sentencepiece.sentencepiece_model_pb2 import ModelProto
    model = read_model_proto(f"{tokenizer_prefix}.model")

    score = min(p.score for p in model.pieces) - 1
    vocab = {p.piece for p in model.pieces}

    tokens_to_add = get_ordered_vocab(extension_vocab)
    logging.info(f"Read {len(tokens_to_add)} tokens to add.")
    tokens_to_add = [piece for piece in tokens_to_add if piece not in ILLEGAL_CHARS and piece not in vocab]
    logging.info(f"Removed existing and illegal tokens with remaining {len(tokens_to_add)} tokens to add.")
    if n_tokens is not None:
        tokens_to_add = tokens_to_add[:n_tokens]
    logging.info(f"Adding {len(tokens_to_add)} tokens.")

    for piece in tokens_to_add:
        model.pieces.append(ModelProto.SentencePiece(piece=piece, score=score))
        vocab.add(piece)
        score -= 1

    with open(f"{out_prefix}.model", "wb") as f:
        f.write(model.SerializeToString())

    with open(f"{out_prefix}.vocab", "w", encoding="utf-8") as f:
        for p in model.pieces:
            f.write(f"{p.piece}\t{int(p.score)}\n")
