"""TemporalConsistencyReward matches temporal pairs as whole words: "old" is not in "told"."""
import pytest

from aligntune.rewards.core import RewardConfig, RewardType, TemporalConsistencyReward


@pytest.mark.parametrize(
    "previous, text, expected",
    [
        ("He told me about the gold.", "I read the news.", 1.0),
        ("We met before lunch.", "See you this afternoon.", 1.0),
        ("That was old.", "This is new.", 0.7),
        ("That was old, I know.", "This is new, you know.", 0.7),
        ("Yesterday was fine.", "Tomorrow is better.", 0.7),
    ],
)
def test_temporal_pairs_match_whole_words(previous, text, expected):
    reward = TemporalConsistencyReward(RewardConfig(reward_type=RewardType.TEMPORAL_CONSISTENCY))
    assert reward.compute(text, previous_texts=[previous]) == pytest.approx(expected)
