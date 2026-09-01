import logging
from typing import List, Optional

from .base import PretrainedPruner, LastNPruner, FrequencyPruner
from .merge_pruner import MergeBasedPruner

logger = logging.getLogger("tokenizer_extension.pruning")


# Legacy implementation
def calculate_orders(
        tokenizer,
        texts,
        calculate_token_frequency=False,
        calculate_merge_based_pruning=False,
        ignore_merges=None,
        return_counts=False
):
    orders = {
        "last_n": LastNPruner().train(tokenizer=tokenizer).raw_pruning_order
    }

    # another tokenization step on the whole dataset
    if calculate_token_frequency:
        logging.info("Calculating HF token frequency")
        orders["token_frequency"] = FrequencyPruner().train(tokenizer, texts).raw_pruning_order

    # tokenize the whole dataset
    if calculate_merge_based_pruning:
        logging.info("Calculating merge order")
        merge_pruner = MergeBasedPruner(ignore_merges=ignore_merges).train(tokenizer, texts)
        orders["least_used_token"] = merge_pruner.raw_pruning_order
        if return_counts:
            orders["_token_counts"] = merge_pruner._token_counts
            orders["_merge_counts"] = merge_pruner._merge_counts

    return orders
