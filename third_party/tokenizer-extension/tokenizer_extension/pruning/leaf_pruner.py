from transformers import AutoTokenizer

from tokenizer_extension.benchmarking import find_unreachable_tokens_tokenization
from tokenizer_extension.pruning import TrainablePrunerBase, StaticPrunerBase, register_pruner
from tokenizer_extension.utils import get_vocab_and_merges
from tokenizer_extension.iterative_tokenizer import IterativeTokenizer
from collections import defaultdict, Counter
from heapdict import heapdict
from typing import List, Dict, Optional


def compute_token_frequencies(
    tokenizer: AutoTokenizer,
    corpus: List[str],
) -> Dict[int, int]:
    counts = Counter()
    for text in corpus:
        counts.update(tokenizer.encode(text))
    return dict(counts)


def get_vocabulary_structure(tokenizer, vocab, merges):
    it = IterativeTokenizer(vocab, merges)
    unreachable_tokens = find_unreachable_tokens_tokenization(tokenizer)
    unreachable_int = set([vocab[i] for i in unreachable_tokens])
    atomics = set(vocab.values()) - unreachable_int
    token_splits = dict()
    leaves = set([vocab[token] for token in unreachable_tokens])
    downstream_merges = defaultdict(int)
    for token in vocab.keys():
        for _, merge in it.tokenize_iteratively(token):
            if merge is None:
                continue
            (left_index, right_index), token_index = merge
            assert token_index not in token_splits or token_splits[token_index] == [left_index, right_index]
            if token_index in token_splits:
                continue
            atomics.discard(token_index)
            leaves.discard(left_index)
            leaves.discard(right_index)
            leaves.add(token_index)
            token_splits[token_index] = [left_index, right_index]
            downstream_merges[left_index] += 1
            downstream_merges[right_index] += 1
    return atomics, leaves, downstream_merges, token_splits


def leaf_pruning_order_id(
    tokenizer: AutoTokenizer,
) -> List[int]:
    vocab, merges = get_vocab_and_merges(tokenizer)
    atomics, leaves, downstream_merges, token_splits = get_vocabulary_structure(tokenizer, vocab, merges)
    pruning_order = []
    queue = heapdict()
    for token in leaves:
        queue[token] = -token
    while queue:
        token, _ = queue.popitem()
        if token in atomics or token not in leaves: continue
        if token in token_splits:
            left, right = token_splits[token]
            downstream_merges[left] -= 1
            if not downstream_merges[left]:
                leaves.add(left)
                queue[left] = -left
            downstream_merges[right] -= 1
            if not downstream_merges[right]:
                leaves.add(right)
                queue[right] = -right
        pruning_order.append(token)
    return pruning_order


def leaf_pruning_order_frequency(
    tokenizer: AutoTokenizer,
    corpus: List[str],
) -> List[int]:
    original_value = tokenizer._tokenizer.model.ignore_merges
    tokenizer._tokenizer.model.ignore_merges = False
    frequencies = compute_token_frequencies(tokenizer, corpus)
    frequencies = defaultdict(int, frequencies)
    vocab, merges = get_vocab_and_merges(tokenizer)
    atomics, leaves, downstream_merges, token_splits = get_vocabulary_structure(tokenizer, vocab, merges)
    queue = heapdict()
    for token in leaves:
        queue[token] = (frequencies[token], -token)
    pruning_order = []
    while queue:
        token, (freq, _) = queue.popitem()
        if token in atomics or token not in leaves: continue
        if token in token_splits:
            left, right = token_splits[token]
            frequencies[left] += frequencies[token]
            frequencies[right] += frequencies[token]
            frequencies[token] = 0
            downstream_merges[left] -= 1
            if not downstream_merges[left]:
                leaves.add(left)
                queue[left] = (frequencies[left], -left)
            downstream_merges[right] -= 1
            if not downstream_merges[right]:
                leaves.add(right)
                queue[right] = (frequencies[right], -right)
        pruning_order.append(token)
    tokenizer._tokenizer.model.ignore_merges = original_value
    return pruning_order


def convert_ids_to_tokens(tokenizer, ids: List[int]) -> List[str]:
    reverse_vocab = {idx: tok for tok, idx in tokenizer.get_vocab().items()}
    return [reverse_vocab[x] for x in ids]


@register_pruner("leaf_frequency")
class LeafFrequencyPruner(TrainablePrunerBase):
    def calculate_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        return convert_ids_to_tokens(tokenizer, leaf_pruning_order_frequency(tokenizer, training_data))


@register_pruner("leaf_last_n")
class LeafLastNPruner(StaticPrunerBase):
    def calculate_pruning_order(self, tokenizer, training_data: Optional[List[str]] = None) -> List[str]:
        return convert_ids_to_tokens(tokenizer, leaf_pruning_order_id(tokenizer))
