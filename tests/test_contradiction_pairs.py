"""Contradiction checks match whole words: "impossible" alone is not "possible" plus "impossible"."""
import pytest

from aligntune.rewards.core import (
    HallucinationReward,
    LogicalConsistencyReward,
    RewardConfig,
    RewardType,
)

CONFIDENT = "This is definitely verified. "


@pytest.mark.parametrize(
    "text, expected",
    [
        ("It is impossible to divide by zero. That is why the result is undefined.", 1.0),
        ("It is always true. It is never false.", 0.6),
    ],
)
def test_logical_consistency_contradictions(text, expected):
    reward = LogicalConsistencyReward(RewardConfig(reward_type=RewardType.LOGICAL_CONSISTENCY))
    assert reward.compute(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("It is impossible to divide by zero.", 0.5),
        ("The incorrect value was replaced.", 0.5),
        ("It is always true and never false.", 0.2),
    ],
)
def test_hallucination_contradictions(text, expected):
    reward = HallucinationReward(RewardConfig(reward_type=RewardType.HALLUCINATION))
    assert reward.compute(CONFIDENT + text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Note that I know the answer.", 0.5),
        ("That is not the answer.", 0.3),
    ],
)
def test_hallucination_context_consistency(text, expected):
    reward = HallucinationReward(RewardConfig(reward_type=RewardType.HALLUCINATION))
    assert reward.compute(CONFIDENT + text, context=["yes, that is right"]) == pytest.approx(expected)
