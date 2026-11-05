"""MathReasoningReward matches step words and units as whole words: "the" holds no unit "h"."""
import pytest

from aligntune.rewards.core import MathReasoningReward, RewardConfig, RewardType


def _reward(**params):
    params = {"check_correctness": False, "check_reasoning": False, "check_format": False, **params}
    return MathReasoningReward(RewardConfig(reward_type=RewardType.MATH_REASONING, params=params))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is four", 0.1),
        ("It's fine", 0.1),
        ("It weighs 5 kg and costs 3 dollars", 0.3),
        ("It is 12 inches", 0.2),
        ("It took 5 minutes", 0.2),
        ("It is 3 km", 0.2),
    ],
)
def test_format_counts_real_units_only(text, expected):
    assert _reward(check_format=True).compute(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Some people also reason well", 0.0),
        ("Enthusiastic people reason well", 0.0),
        ("First add, so then stop", 0.4),
        ("Thus we stop", 0.2),
    ],
)
def test_reasoning_counts_step_words_only(text, expected):
    assert _reward(check_reasoning=True).compute(text) == pytest.approx(expected)
