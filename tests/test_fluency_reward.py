"""FluencyReward matches transitions and connectors as whole words: "or" is not in "words"."""
import pytest

from aligntune.rewards.core import FluencyReward, RewardConfig, RewardType


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Some words for different people understand.", 0.6),
        ("Enthusiastic dogs bark loudly today.", 0.6),
        ("Dogs bark, but cats run and birds sing.", 0.8),
        ("However, dogs bark so cats run.", 0.9),
    ],
)
def test_fluency_matches_whole_words(text, expected):
    reward = FluencyReward(RewardConfig(reward_type=RewardType.FLUENCY))
    assert reward.compute(text) == pytest.approx(expected)
