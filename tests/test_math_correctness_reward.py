"""math_correctness without a reference scores each equation once, not its left side as well."""
import pytest

from aligntune.rewards.core import MathCorrectnessReward, RewardConfig, RewardType


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2 + 2 = 5", 0.0),
        ("7 - 3 = 1. 6 / 2 = 9", 0.0),
        ("2 + 2 = 4", 1.0),
        ("3 * 3 is nine and 2 + 2 = 5", 0.5),
    ],
)
def test_math_correctness_counts_each_equation_once(text, expected):
    reward = MathCorrectnessReward(RewardConfig(reward_type=RewardType.MATH_CORRECTNESS))
    assert reward.compute(text) == pytest.approx(expected)
