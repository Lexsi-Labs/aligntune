from collections import defaultdict
from typing import List, Optional

from tqdm import tqdm

from tokenizer_extension.iterative_tokenizer import IterativeTokenizer
from tokenizer_extension.utils import get_vocab_and_merges
from .base import TrainablePrunerBase, register_pruner


def calculate_merge_statistics(tokenizer, texts, ignore_merges=None):
    vocab, merges = get_vocab_and_merges(tokenizer)
    if ignore_merges is None:
        ignore_merges = tokenizer._tokenizer.model.ignore_merges

    step_tokenizer = IterativeTokenizer(
        vocab, merges, ignore_merges=ignore_merges, byte_fallback=tokenizer._tokenizer.model.byte_fallback,
        unk_token=tokenizer._tokenizer.model.unk_token
    )

    merge_counts = {m: 0 for m in merges}
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    token_counts = {t: 0 for t in full_vocab}

    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    normalizer = tokenizer._tokenizer.normalizer
    normalize_func = normalizer.normalize_str if normalizer is not None else lambda x: x
    for text in tqdm(texts, miniters=len(texts) // 100):
        pre_tokenized_str = pre_tokenizer.pre_tokenize_str(normalize_func(text))
        for word, _ in pre_tokenized_str:
            step_tokens = None
            for step, merge in step_tokenizer.tokenize_iteratively(word):
                step_tokens = step_tokenizer.word_to_tokens(step)
                merge_result = step_tokenizer.merge_id_to_token(merge)
                if merge_result is not None:
                    merge_counts[merge_result[0]] += 1

            if step_tokens is None:
                raise ValueError(f"No tokens returned by tokenizer for word {word}")
            for t in step_tokens:
                token_counts[t] += 1

    return token_counts, merge_counts


def merge_based_pruning_order(vocab, token_counts, merge_counts):
    token_counts = defaultdict(int, token_counts)
    for m, freq in merge_counts.items():
        for t in m:
            token_counts[t] += freq

    return [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -len(x[0]), -vocab[x[0]]))]


@register_pruner("merge_based")
class MergeBasedPruner(TrainablePrunerBase):
    def __init__(self, ignore_merges: Optional[bool] = False):
        super().__init__()
        self.ignore_merges = ignore_merges
        self._token_counts = None
        self._merge_counts = None

    def calculate_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        full_vocab = tokenizer._tokenizer.get_vocab(True)
        token_counts, merge_counts = calculate_merge_statistics(
            tokenizer=tokenizer, texts=training_data, ignore_merges=self.ignore_merges
        )
        self._token_counts = token_counts
        self._merge_counts = merge_counts
        return merge_based_pruning_order(
            vocab=full_vocab, token_counts=token_counts, merge_counts=merge_counts
        )
