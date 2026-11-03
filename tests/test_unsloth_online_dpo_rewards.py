"""Unsloth Online DPO: every reward function needs a processing class.

TRL's OnlineDPOTrainer raises "The number of reward processing classes must match
the number of reward functions" when the lists differ. The trainer added a
processing class only for a reward *model*, so a run with function rewards from
``config.rewards`` (and no ``reward_model_name``) failed before the first step.
No model is loaded.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("unsloth")

from aligntune.backends.unsloth.rl.online_dpo.online_dpo import UnslothOnlineDPOTrainer  # noqa: E402


def _trainer(rewards):
    trainer = object.__new__(UnslothOnlineDPOTrainer)
    trainer.reward_model = None
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(reward_model_name=None),
        train=SimpleNamespace(reward_weights=None),
        rewards=rewards,
    )
    return trainer


@pytest.mark.parametrize("count", [1, 2])
def test_function_rewards_get_one_processing_class_each(count):
    trainer = _trainer([{"type": "length", "weight": 1.0, "params": {}}] * count)

    trainer.setup_rewards()

    assert len(trainer.reward_funcs) == count
    assert trainer.reward_processing_classes == [None] * count


def test_no_rewards_still_raises():
    with pytest.raises(ValueError, match="requires a reward_model_name"):
        _trainer([]).setup_rewards()
