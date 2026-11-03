"""Duplicate completions in a batch whose unique completions have no spans must still get an importance array."""
import numpy as np
import torch

from aligntune.backends.trl.rl.counterfact_grpo.custom_trainer import (
    compute_batched_counterfactual_importance,
)


class _Tok:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_duplicates_of_a_spanless_completion_are_filled():
    prompt_len = 2
    row = torch.tensor([5, 6, 7, 8, 9, 0])  # no calculation span, no answer marker
    batch = torch.stack([row, row.clone(), row.clone()])
    importance, stats = compute_batched_counterfactual_importance(
        model=torch.nn.Linear(1, 1), tokenizer=_Tok(), input_ids_batch=batch, prompt_len=prompt_len
    )
    assert all(isinstance(x, np.ndarray) for x in importance)
    assert len(importance) == 3
    assert stats["cache_hits"] == 2
