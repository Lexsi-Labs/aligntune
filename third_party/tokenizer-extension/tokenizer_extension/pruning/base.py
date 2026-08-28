import json
import warnings
from itertools import islice
from typing import List, Optional, Iterable, Dict, Type
import logging

from abc import ABC, ABCMeta, abstractmethod

from tokenizers import Tokenizer
from tqdm import tqdm

from tokenizer_extension.utils import update_postprocessor_special_tokens, get_vocab_and_merges, get_ordered_vocab, \
    get_added_tokens_vocab, read_json, write_json

logger = logging.getLogger("tokenizer_extension.pruning")


def filter_special_tokens(
        tokenizer,
        tokens: List[str],
        ignore_added: bool = True,
        ignore_special: bool = True,
        ignore_tokens: List[str] = None,
) -> List[str]:
    full_vocab = tokenizer._tokenizer.get_vocab(True)

    ignore_vocab = {}
    if ignore_tokens is not None:
        ignore_vocab.update({t: full_vocab[t] for t in ignore_tokens})

    if ignore_special or ignore_added:
        ignore_vocab.update({
            x: full_vocab[x]
            for x in [tokenizer.unk_token, tokenizer.eos_token, tokenizer.bos_token, tokenizer.pad_token]
            if x is not None
        })

        if not ignore_special:
            raise ValueError(
                "Added tokens can only be ignored along with special tokens, please set ignore_special=True"
            )
        ignore_vocab.update(get_added_tokens_vocab(tokenizer, special_only=not ignore_added))

    return [x for x in tokens if x not in ignore_vocab]


def tokenizer_remove_tokens_inplace(tokenizer, tokens_to_prune: Iterable[str]):
    tokens_to_prune = set(tokens_to_prune)
    cfg = json.loads(tokenizer._tokenizer.to_str())
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    vocab, merges = get_vocab_and_merges(tokenizer)

    cfg["model"]["merges"] = [
        " ".join(m) for m in merges
        if all(t not in tokens_to_prune for t in m) and "".join(m) not in tokens_to_prune
    ]
    new_vocab_tokens = [k for k in get_ordered_vocab(full_vocab) if k not in tokens_to_prune]
    new_full_vocab = {k: i for i, k in enumerate(new_vocab_tokens)}
    cfg["model"]["vocab"] = {k: new_full_vocab[k] for k in vocab if k in new_full_vocab}

    cfg["added_tokens"] = [
        dict(token, id=new_full_vocab[token["content"]])
        for token in cfg["added_tokens"] if token["content"] in new_full_vocab
    ]

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(cfg))
    tokenizer = update_postprocessor_special_tokens(tokenizer)

    new_full_vocab = tokenizer._tokenizer.get_vocab(True)
    new_reverse_vocab = {v: k for k, v in new_full_vocab.items()}
    for k, v in sorted(full_vocab.items(), key=lambda x: x[1]):
        new_id = new_full_vocab.get(k, None)
        if new_id is None:
            logger.debug(f"Removed {k} (id={v})")
        elif v != new_id:
            logger.debug(f"Moved {k} ({v} -> {new_id})")

    logger.debug(f"Empty indices: {[i for i in range(len(new_reverse_vocab)) if i not in new_reverse_vocab]}")

    return tokenizer


class PrunerBase(ABC):
    def __init__(self):
        self._pruning_order = None

    @property
    def is_trained(self):
        return self._pruning_order is not None

    @abstractmethod
    def calculate_pruning_order(self, tokenizer, training_data: Optional[List[str]]) -> List[str]:
        raise NotImplementedError()

    def train(self, tokenizer, training_data: Optional[List[str]]):
        """
        Trains the pruner.
        :param tokenizer: The tokenizer to be pruned.
        :param training_data: The training data for the pruner. Can be None for static pruners.
        :return: The pruner object (self).
        """
        if self.is_trained:
            raise ValueError("Pruner is already trained.")
        self._pruning_order = self.calculate_pruning_order(tokenizer, training_data)
        return self

    @property
    def raw_pruning_order(self) -> List[str]:
        """
        :return: Pruning order without any filtering of special tokens.
        """
        if not self.is_trained:
            raise ValueError("Pruner is not trained yet.")
        return self._pruning_order

    def save(self, path: str):
        write_json({
            "name": self.__class__.__name__,
            "prune_order": self.raw_pruning_order
        }, path)

    def get_tokens_to_prune(
            self,
            tokenizer,
            n: Optional[int] = None,
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ) -> List[str]:
        """
        Gets the tokens to prune based on the pruning order and filtering options.
        :param tokenizer: Tokenizer to prune.
        :param n: Number of tokens to prune. If None, all tokens in the pruning order are returned.
        :param ignore_added: Ignore added tokens.
        :param ignore_special: Ignore special tokens.
        :param ignore_tokens: Custom list of tokens to ignore.
        :return: Pruning order with filtered tokens.
        """
        tokens_to_prune = list(islice(filter_special_tokens(
            tokenizer,
            self.raw_pruning_order,
            ignore_added=ignore_added,
            ignore_special=ignore_special,
            ignore_tokens=ignore_tokens,
        ), n))
        if n is not None and len(tokens_to_prune) < n:
            raise ValueError(f"Not enough tokens to prune, {len(tokens_to_prune)} < {n}")
        return tokens_to_prune

    def prune(
            self,
            tokenizer,
            n: Optional[int] = None,
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ):
        """
        Prunes the tokenizer inplace (changing the original tokenizer object).
        :param tokenizer: Tokenizer to prune.
        :param n: Number of tokens to prune. If None, all tokens in the pruning order are pruned.
        :param ignore_added: Ignore added tokens.
        :param ignore_special: Ignore special tokens.
        :param ignore_tokens: Custom list of tokens to ignore.
        :return: The pruned tokenizer.
        """
        tokens_to_prune = self.get_tokens_to_prune(
            tokenizer, n=n, ignore_added=ignore_added, ignore_special=ignore_special, ignore_tokens=ignore_tokens
        )
        tokenizer_remove_tokens_inplace(
            tokenizer=tokenizer,
            tokens_to_prune=tokens_to_prune,
        )
        return tokenizer

    def train_prune(
            self,
            tokenizer,
            training_data: Optional[List[str]] = None,
            n: Optional[int] = None,
            ignore_added: bool = True,
            ignore_special: bool = True,
            ignore_tokens: Optional[List[str]] = None
    ):
        """
        Trains and prunes the tokenizer inplace (changing the original tokenizer object).
        :param tokenizer: Tokenizer to prune.
        :param training_data: The training data for the pruner. Can be None for static pruners.
        :param n: Number of tokens to prune. If None, all tokens in the pruning order are pruned.
        :param ignore_added: Ignore added tokens.
        :param ignore_special: Ignore special tokens.
        :param ignore_tokens: Custom list of tokens to ignore.
        :return: The pruned tokenizer.
        """
        self.train(tokenizer, training_data)
        return self.prune(
            tokenizer=tokenizer,
            n=n,
            ignore_added=ignore_added,
            ignore_special=ignore_special,
            ignore_tokens=ignore_tokens
        )


PRUNER_REGISTRY: Dict[str, Type[PrunerBase]] = {}


def register_pruner(name: Optional[str] = None):
    def decorator(pruner_class: Type[PrunerBase]):
        reg_name = name or pruner_class.__name__
        PRUNER_REGISTRY[reg_name] = pruner_class
        return pruner_class

    return decorator


class TrainablePrunerBase(PrunerBase, metaclass=ABCMeta):
    def train(self, tokenizer, training_data: List[str]):
        if training_data is None:
            raise ValueError("Training data must be provided for TrainablePruner.")
        return super().train(tokenizer, training_data)


class StaticPrunerBase(PrunerBase, metaclass=ABCMeta):
    def train(self, tokenizer, training_data: Optional[List[str]] = None):
        if training_data is not None:
            warnings.warn("Training data is ignored for this pruner class.")
        return super().train(tokenizer, training_data)


class PretrainedPruner(PrunerBase):
    def __init__(self, pruning_order: List[str], name: Optional[str] = None):
        super().__init__()
        self._pruning_order = pruning_order
        self.name = name

    def calculate_pruning_order(self, tokenizer, training_data: Optional[List[str]]) -> List[str]:
        return self._pruning_order

    @classmethod
    def load(cls, path: str):
        pruner_json = read_json(path)
        return cls(pruning_order=pruner_json["prune_order"], name=pruner_json.get("name", None))


def prune_tokenizer(
        tokenizer,
        prune_ordered_tokens: List[str],
        n: Optional[int] = None,
        ignore_added: bool = True,
        ignore_special: bool = True,
        ignore_tokens: List[str] = None,
):
    return PretrainedPruner(pruning_order=prune_ordered_tokens).prune(
        tokenizer=tokenizer, n=n, ignore_added=ignore_added, ignore_special=ignore_special, ignore_tokens=ignore_tokens
    )


@register_pruner("last_n")
class LastNPruner(StaticPrunerBase):
    def calculate_pruning_order(self, tokenizer, training_data=None) -> List[str]:
        full_vocab = tokenizer._tokenizer.get_vocab(True)
        return list(reversed(get_ordered_vocab(full_vocab)))


def calculate_token_frequency(tokenizer, texts) -> Dict[str, int]:
    token_freqs = {t: 0 for t in tokenizer._tokenizer.get_vocab(True)}
    for text in tqdm(texts, miniters=len(texts) // 100):
        for token in tokenizer.tokenize(text):
            token_freqs[token] += 1
    return token_freqs


def calculate_frequency_order(tokenizer, texts) -> List[str]:
    token_counts = calculate_token_frequency(tokenizer=tokenizer, texts=texts)
    full_vocab = tokenizer._tokenizer.get_vocab(True)
    return [tok for tok, _ in sorted(token_counts.items(), key=lambda x: (x[1], -full_vocab[x[0]]))]


@register_pruner("frequency")
class FrequencyPruner(TrainablePrunerBase):
    def calculate_pruning_order(self, tokenizer, training_data: List[str]) -> List[str]:
        return calculate_frequency_order(tokenizer=tokenizer, texts=training_data)
