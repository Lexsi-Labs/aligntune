from collections import defaultdict
from typing import Iterable, Optional

from tqdm import tqdm
from heapq import heappush, heappop


def group_tokens(text, tokenizer):
    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    if tokenizer._tokenizer.pre_tokenizer is None:
        raise ValueError("Tokenizer must have a pre-tokenizer")

    if tokenizer._tokenizer.normalizer is not None:
        text = tokenizer._tokenizer.normalizer.normalize_str(text)

    pre_tokenized = [x[0] for x in pre_tokenizer.pre_tokenize_str(text)]

    grouped_new_words = []
    for word in pre_tokenized:
        group = [token.value for token in tokenizer._tokenizer.model.tokenize(word)]
        if len(group) > 0:
            grouped_new_words.append(group)

    return list(map(tuple, grouped_new_words))


def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    where_to_update = defaultdict(set)

    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue

        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
            where_to_update[pair].add(word)
    return pair_freqs, where_to_update


def merge_pair(a, b, splits, word_freqs, pair_freqs, queue, where_to_update):
    updated_pairs = defaultdict(int)

    for word in list(where_to_update[(a, b)]):
        freq = word_freqs[word]
        split = splits[word]
        if len(split) == 1:
            continue

        split = list(split)
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                new_token = a + b
                split = split[:i] + [new_token] + split[i + 2:]
            else:
                i += 1

        prev_pairs = list(zip(splits[word][:-1], splits[word][1:]))
        splits[word] = split
        new_pairs = list(zip(splits[word][:-1], splits[word][1:]))
        for pair in set(new_pairs) - set(prev_pairs):
            where_to_update[pair].add(word)
        for pair in set(prev_pairs) - set(new_pairs):
            where_to_update[pair].discard(word)
        for pair in prev_pairs:
            updated_pairs[pair] -= freq
        for pair in new_pairs:
            updated_pairs[pair] += freq

    for pair, change in updated_pairs.items():
        new_freq = pair_freqs.get(pair, 0) + change
        pair_freqs[pair] = new_freq
        if new_freq > 0:
            heappush(queue, (-new_freq, pair))

    del pair_freqs[(a, b)]
    del where_to_update[(a, b)]
    return splits


def train_vocab_extension(
        tokenizer,
        corpus: Iterable[str],
        extension_size: int,
        is_sentencepiece: bool = False,
        max_token_length: Optional[int] = None,
        sp_kwargs: Optional[dict] = None,
) -> dict:
    """
    :param tokenizer: The tokenizer to continually train
    :param corpus: Training corpus
    :param extension_size: The Number of tokens to add to the vocabulary
    :param is_sentencepiece: Is the tokenizer that is extended a (HuggingFace) SentencePiece tokenizer
    :param max_token_length: maximum length of a token created
    :param sp_kwargs: SentencePiece specific kwargs
    :return: Dictionary with keys: 'vocab', 'merges', 'pair_freqs', 'word_freqs'
    """
    split_freqs = defaultdict(int)

    if is_sentencepiece:
        from .sentencepiece_utils import group_tokens as group_tokens_sentencepiece
        from .sentencepiece_utils import TrainerSpec, is_valid_merge
        if sp_kwargs is None:
            sp_kwargs = {}
        cfg = TrainerSpec(**sp_kwargs)
        check_token = lambda a, b: is_valid_merge(a, b, cfg)
        group_tokens_func = lambda txt, tok: group_tokens_sentencepiece(txt, tok, **sp_kwargs)
    else:
        check_token = lambda a, b: True
        group_tokens_func = group_tokens


    for text in tqdm(corpus, desc="computing frequencies", mininterval=1):
        grouped_tokens = group_tokens_func(text, tokenizer)
        for word in grouped_tokens:
            split_freqs[word] += 1

    splits = {"".join(split): split for split in split_freqs}
    word_freqs = {"".join(split): freq for split, freq in split_freqs.items()}
    pair_freqs, where_to_update = compute_pair_freqs(splits, word_freqs)

    pair_queue = []
    for pair, freq in pair_freqs.items():
        heappush(pair_queue, (-freq, pair))

    vocab_size = extension_size
    vocab = {}
    merges = []

    with tqdm(total=vocab_size, desc="training") as pbar:
        while len(vocab) < vocab_size:
            max_freq, best_pair = heappop(pair_queue)
            max_freq = -max_freq  # Convert back to positive

            # Skip stale entries
            if best_pair is None or pair_freqs.get(best_pair, None) != max_freq:
                continue

            new_token = "".join(best_pair)
            if (max_token_length is not None and len(new_token) > max_token_length) or not check_token(*best_pair):
                del pair_freqs[best_pair]
                del where_to_update[best_pair]
                continue

            splits = merge_pair(*best_pair, splits, word_freqs, pair_freqs, pair_queue, where_to_update)
            merges.append(best_pair)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            if len(vocab) % 100 == 0 or len(vocab) == vocab_size:
                pbar.n = len(vocab)
                pbar.refresh()

    return {"vocab": vocab, "merges": merges, "pair_freqs": pair_freqs, "word_freqs": word_freqs}
