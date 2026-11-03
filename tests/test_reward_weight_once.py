"""A reward's score does not include its weight: the trainer applies the weight once."""
import pytest

from aligntune.core.rl.reward_handler import prepare_trl_rewards
from aligntune.rewards.core import RewardConfig, RewardFunctionFactory, RewardType

TEXT = "The quick brown fox jumps over the lazy dog. Then it rests, because foxes tire easily."
LONG = " ".join(f"word{i}" for i in range(400))
WEIGHT = 0.25


def _fake_pipeline(texts):
    one = {"label": "positive", "score": 0.8}
    return [one] * len(texts) if isinstance(texts, list) else [one]


def _make(reward_type, weight):
    reward = RewardFunctionFactory.create_reward(RewardConfig(reward_type=reward_type, weight=weight))
    for name in ("_get_sentiment_pipeline", "_get_safety_pipeline", "_get_toxicity_pipeline"):
        if hasattr(reward, name):
            setattr(reward, name, lambda: _fake_pipeline)
    return reward


COMPUTE_TYPES = [
    RewardType.MATH_REASONING, RewardType.DIVERSITY, RewardType.FLUENCY,
    RewardType.RELEVANCE, RewardType.BREVITY, RewardType.IMAGE_RELEVANCE, RewardType.AUDIO_QUALITY,
]


@pytest.mark.parametrize("reward_type", COMPUTE_TYPES)
@pytest.mark.parametrize("text", [TEXT, LONG])
def test_compute_ignores_weight(reward_type, text):
    ref = "the quick brown fox"
    assert _make(reward_type, WEIGHT).compute(text, ref) == pytest.approx(_make(reward_type, 1.0).compute(text, ref))


@pytest.mark.parametrize("reward_type", [RewardType.SENTIMENT, RewardType.SAFETY, RewardType.TOXICITY])
def test_batch_compute_ignores_weight(reward_type):
    weighted = _make(reward_type, WEIGHT).batch_compute([TEXT, LONG])
    plain = _make(reward_type, 1.0).batch_compute([TEXT, LONG])
    assert weighted == pytest.approx(plain)


@pytest.mark.parametrize("name", ["diversity", "fluency", "relevance", "brevity", "math_reasoning"])
def test_trl_handler_path_applies_weight_once(name):
    prepared = prepare_trl_rewards([{"type": name, "weight": WEIGHT, "params": {}}])
    assert prepared.weights == [WEIGHT]
    got = prepared.functions[0](prompts=["p"], completions=[TEXT])
    unweighted = _make(RewardType(name), 1.0).compute(TEXT, None, prompt="p")
    assert got[0] == pytest.approx(unweighted)  # TRL multiplies by WEIGHT afterwards, once
