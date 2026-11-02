"""code_verifiable reads the dataset's test columns like the other code rewards."""
from aligntune.rewards.core import RewardConfig, RewardType
from aligntune.rewards.verifiable import CodeExecutionVerifiableReward


def _reward():
    return CodeExecutionVerifiableReward(RewardConfig(reward_type=RewardType.CODE_VERIFIABLE, weight=1.0))


GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_mbpp_test_list():
    r = _reward()
    tests = ["assert add(1, 2) == 3"]
    assert r.compute(GOOD, None, test_list=tests) == 1.0
    assert r.compute(BAD, None, test_list=tests) == 0.0


def test_test_cases_column_beats_reference():
    r = _reward()
    assert r.compute(GOOD, "def add(a, b): return 0", test_cases=["assert add(2, 2) == 4"]) == 1.0
    assert r.compute(BAD, "def add(a, b): return 0", test_cases=["assert add(2, 2) == 4"]) == 0.0


def test_humaneval_test_and_entry_point():
    r = _reward()
    test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    assert r.compute(GOOD, None, test=test, entry_point="add") == 1.0
    assert r.compute(BAD, None, test=test, entry_point="add") == 0.0


def test_reference_still_works():
    r = _reward()
    assert r.compute(GOOD, ["assert add(1, 2) == 3"]) == 1.0
    assert r.compute(BAD, ["assert add(1, 2) == 3"]) == 0.0
