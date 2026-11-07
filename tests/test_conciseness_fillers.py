"""ConcisenessReward counts a filler next to punctuation, and the two-word "you know"."""
import pytest

from aligntune.rewards.core import ConcisenessReward, RewardConfig, RewardType


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Basically, it works, you know.", 0.42),
        ("Really, it works.", 0.5 - 0.2 / 3),
        ("It is very good.", 0.45),
        ("It works well.", 0.5),
    ],
)
def test_conciseness_counts_fillers(text, expected):
    reward = ConcisenessReward(RewardConfig(reward_type=RewardType.CONCISENESS))
    assert reward.compute(text) == pytest.approx(expected)
