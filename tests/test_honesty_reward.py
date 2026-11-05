"""HonestyReward matches dishonesty terms at a word start, so "lie" does not fire inside "believe"."""
import pytest

from aligntune.rewards.core import HonestyReward, RewardConfig, RewardType


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Honestly, I believe the answer is 4.", 0.2),
        ("Frankly, the client applied the patch earlier.", 0.2),
        ("Honestly, that is a lie.", 0.0),
        ("Honestly, he lied about it.", 0.0),
    ],
)
def test_honesty_dishonesty_terms_at_word_start(text, expected):
    reward = HonestyReward(RewardConfig(reward_type=RewardType.HONESTY))
    assert reward.compute(text) == pytest.approx(expected)
