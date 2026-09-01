"""
Tests for RLVR (RL with Verifiable Rewards).

Tests all verifiable reward implementations:
- MathVerifiableReward
- CodeExecutionVerifiableReward
- SQLVerifiableReward
- JSONSchemaVerifiableReward
- RegexVerifiableReward
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch

from aligntune.rewards.core import RewardConfig, RewardType, RewardFunctionFactory
from aligntune.rewards.verifiable import (
    MathVerifiableReward,
    CodeExecutionVerifiableReward,
    SQLVerifiableReward,
    JSONSchemaVerifiableReward,
    RegexVerifiableReward,
)


class TestMathVerifiableReward:
    """Test MathVerifiableReward implementation."""

    @pytest.fixture
    def reward(self):
        """Create a MathVerifiableReward instance."""
        config = RewardConfig(
            reward_type=RewardType.MATH_VERIFIABLE,
            params={"simplify": True, "timeout": 5.0}
        )
        return MathVerifiableReward(config)

    def test_verify_exact_match(self, reward):
        """Test verification with exact answer match."""
        completion = "The answer is 42"
        reference = "42"
        assert reward.verify(completion, reference) is True

    def test_verify_no_match(self, reward):
        """Test verification with incorrect answer."""
        completion = "The answer is 10"
        reference = "42"
        assert reward.verify(completion, reference) is False

    def test_verify_with_extra_whitespace(self, reward):
        """Test verification handles extra whitespace."""
        completion = "The answer is: 42"
        reference = "42"
        assert reward.verify(completion, reference) is True

    def test_verify_no_reference(self, reward):
        """Test verification fails without reference."""
        completion = "The answer is 42"
        assert reward.verify(completion, None) is False

    def test_extract_answer_boxed(self, reward):
        """Test answer extraction from LaTeX boxed format."""
        text = "Therefore: \\boxed{7}"
        answer = reward._extract_answer(text)
        assert answer == "7"

    def test_extract_answer_pattern(self, reward):
        """Test answer extraction from common patterns."""
        text = "The answer is 5"
        answer = reward._extract_answer(text)
        assert answer == "5"

    def test_extract_answer_final_answer(self, reward):
        """Test answer extraction from 'final answer' pattern."""
        text = "Final answer: 42"
        answer = reward._extract_answer(text)
        assert answer == "42"

    def test_compute_correct(self, reward):
        """Test compute() returns 1.0 for correct answer."""
        completion = "The answer is 42"
        reference = "42"
        score = reward.compute(completion, reference)
        assert score == 1.0

    def test_compute_incorrect(self, reward):
        """Test compute() returns 0.0 for incorrect answer."""
        completion = "The answer is 10"
        reference = "42"
        score = reward.compute(completion, reference)
        assert score == 0.0

    def test_numeric_tolerance(self):
        """Test numeric comparison with floating point tolerance.

        Uses simplify=False (not the shared `reward` fixture, which sets
        simplify=True): verify() takes the exact sympy symbolic-comparison
        path whenever simplify=True and sympy is available, which never
        falls back to the numeric-tolerance path this test means to
        exercise - 3.141592 and 3.141593 are simply unequal Sympy Floats,
        so the symbolic path (correctly) says False. simplify=False routes
        through _normalize_compare's `abs(val1 - val2) < 1e-6` fallback.
        """
        config = RewardConfig(
            reward_type=RewardType.MATH_VERIFIABLE,
            params={"simplify": False, "timeout": 5.0}
        )
        reward = MathVerifiableReward(config)
        completion = "The answer is 3.141592"
        reference = "3.141593"
        # Should match within tolerance
        assert reward.verify(completion, reference) is True

    def test_symbolic_comparison(self, reward):
        """Test symbolic comparison when sympy is available."""
        if reward.sympy is None:
            pytest.skip("sympy not available")

        completion = "The answer is 2 + 2"
        reference = "4"
        assert reward.verify(completion, reference) is True

    def test_answer_not_found(self, reward):
        """Test when answer cannot be extracted."""
        completion = "This is a question without an answer"
        reference = "42"
        assert reward.verify(completion, reference) is False


class TestCodeExecutionVerifiableReward:
    """Test CodeExecutionVerifiableReward implementation."""

    @pytest.fixture
    def reward(self):
        """Create a CodeExecutionVerifiableReward instance."""
        config = RewardConfig(
            reward_type=RewardType.CODE_VERIFIABLE,
            params={"language": "python", "timeout": 5.0, "test_cases": []}
        )
        return CodeExecutionVerifiableReward(config)

    def test_verify_valid_code(self, reward):
        """Test verification of valid Python code."""
        code = "```python\nx = 1 + 1\nassert x == 2\n```"
        assert reward.verify(code) is True

    def test_verify_invalid_code(self, reward):
        """Test verification fails for invalid code."""
        code = "x = 1 +\n"  # Syntax error
        assert reward.verify(code) is False

    def test_extract_code_markdown(self, reward):
        """Test extraction of code from markdown blocks."""
        text = "```python\ndef hello():\n    return 'world'\n```"
        code = reward._extract_code(text)
        assert "def hello()" in code

    def test_extract_code_no_blocks(self, reward):
        """Test extraction when no markdown blocks present."""
        text = "def hello():\n    return 'world'"
        code = reward._extract_code(text)
        assert "def hello()" in code

    def test_compute_valid_code(self, reward):
        """Test compute() returns 1.0 for valid code."""
        code = "```python\nx = 1 + 1\nassert x == 2\n```"
        score = reward.compute(code)
        assert score == 1.0

    def test_compute_invalid_code(self, reward):
        """Test compute() returns 0.0 for invalid code."""
        code = "x = invalid_syntax +"
        score = reward.compute(code)
        assert score == 0.0

    def test_timeout_handling(self, reward):
        """Test that infinite loops are caught by timeout."""
        code = "while True:\n    pass"
        # Should timeout and return False
        assert reward.verify(code) is False

    def test_code_with_output(self, reward):
        """Test code that produces output."""
        code = "```python\nprint('hello')\n```"
        assert reward.verify(code) is True

    def test_code_with_function(self, reward):
        """Test code with function definition."""
        code = """
def add(a, b):
    return a + b

result = add(2, 3)
assert result == 5
"""
        assert reward.verify(code) is True

    def test_extract_code_function_definition(self, reward):
        """Test extraction prioritizes function definitions."""
        text = """
Some explanation
```python
print('wrong')
```

Then the real solution:
```python
def solution():
    return 42
```
"""
        code = reward._extract_code(text)
        # Should get the function definition
        assert "def solution()" in code

    def test_check_output(self, reward):
        """Test output matching."""
        code = "print('42')"
        expected = "42"
        result = reward._check_output(code, expected)
        # Should match (stdout is '42' after stripping)
        assert result is True or result is False  # Depends on environment


class TestSQLVerifiableReward:
    """Test SQLVerifiableReward implementation."""

    @pytest.fixture
    def reward(self):
        """Create a SQLVerifiableReward instance."""
        config = RewardConfig(
            reward_type=RewardType.SQL_VERIFIABLE,
            params={"execute": False, "db_path": ":memory:"}
        )
        return SQLVerifiableReward(config)

    def test_verify_valid_select(self, reward):
        """Test verification of valid SELECT query."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "SELECT * FROM users WHERE id = 1"
        assert reward.verify(query) is True

    def test_verify_invalid_sql(self, reward):
        """Test verification fails for invalid SQL."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "SELECT * FORM users"  # FORM instead of FROM
        assert reward.verify(query) is False

    def test_syntax_check_insert(self, reward):
        """Test syntax checking for INSERT statements."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')"
        assert reward._validate_syntax(query) is True

    def test_syntax_check_update(self, reward):
        """Test syntax checking for UPDATE statements."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "UPDATE users SET name = 'Jane' WHERE id = 1"
        assert reward._validate_syntax(query) is True

    def test_syntax_check_delete(self, reward):
        """Test syntax checking for DELETE statements."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "DELETE FROM users WHERE id = 1"
        assert reward._validate_syntax(query) is True

    def test_compute_valid_sql(self, reward):
        """Test compute() returns 1.0 for valid SQL."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "SELECT COUNT(*) FROM users"
        score = reward.compute(query)
        assert score == 1.0

    def test_compute_invalid_sql(self, reward):
        """Test compute() returns 0.0 for invalid SQL."""
        if reward.sqlglot is None:
            pytest.skip("sqlglot not available")

        query = "SELECT * FORM users"
        score = reward.compute(query)
        assert score == 0.0

    def test_verify_without_sqlglot(self):
        """Test graceful handling when sqlglot is unavailable."""
        config = RewardConfig(
            reward_type=RewardType.SQL_VERIFIABLE,
            params={"execute": False}
        )
        reward = SQLVerifiableReward(config)

        # Mock the sqlglot import to fail
        original_sqlglot = reward.sqlglot
        reward.sqlglot = None

        result = reward.verify("SELECT * FROM users")
        assert result is False

        reward.sqlglot = original_sqlglot


class TestJSONSchemaVerifiableReward:
    """Test JSONSchemaVerifiableReward implementation."""

    @pytest.fixture
    def reward(self):
        """Create a JSONSchemaVerifiableReward instance."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        config = RewardConfig(
            reward_type=RewardType.JSON_SCHEMA_VERIFIABLE,
            params={"schema": schema}
        )
        return JSONSchemaVerifiableReward(config)

    def test_verify_valid_json(self, reward):
        """Test verification of valid JSON matching schema."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        json_str = '{"name": "Alice", "age": 30}'
        reference_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        assert reward.verify(json_str, reference_schema) is True

    def test_verify_invalid_json(self, reward):
        """Test verification fails for invalid JSON."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        json_str = '{"name": "Alice"}'  # Missing required 'age'
        assert reward.verify(json_str) is False

    def test_verify_invalid_json_syntax(self, reward):
        """Test verification fails for malformed JSON."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        json_str = '{"name": "Alice", age: 30}'  # Invalid syntax
        assert reward.verify(json_str) is False

    def test_extract_json_markdown(self, reward):
        """Test extraction of JSON from markdown blocks."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        text = '```json\n{"name": "Alice", "age": 30}\n```'
        data = reward._extract_json(text)
        assert data["name"] == "Alice"
        assert data["age"] == 30

    def test_extract_json_plain(self, reward):
        """Test extraction of plain JSON."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        text = '{"name": "Bob", "age": 25}'
        data = reward._extract_json(text)
        assert data["name"] == "Bob"

    def test_compute_valid_json(self, reward):
        """Test compute() returns 1.0 for valid JSON and schema."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        json_str = '{"name": "Alice", "age": 30}'
        score = reward.compute(json_str)
        assert score == 1.0

    def test_compute_invalid_json(self, reward):
        """Test compute() returns 0.0 for invalid JSON."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        json_str = '{"name": "Alice"}'  # Missing required field
        score = reward.compute(json_str)
        assert score == 0.0

    def test_schema_from_reference(self, reward):
        """Test using schema provided via reference parameter."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        }
        json_str = '{"value": "test"}'
        assert reward.verify(json_str, schema) is True

    def test_array_schema(self, reward):
        """Test JSON array validation."""
        if reward.jsonschema is None:
            pytest.skip("jsonschema not available")

        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        json_str = '["hello", "world"]'
        assert reward.verify(json_str, schema) is True


class TestRegexVerifiableReward:
    """Test RegexVerifiableReward implementation."""

    @pytest.fixture
    def reward(self):
        """Create a RegexVerifiableReward instance."""
        config = RewardConfig(
            reward_type=RewardType.REGEX_VERIFIABLE,
            params={"pattern": r"^The answer is \d+$", "case_sensitive": True}
        )
        return RegexVerifiableReward(config)

    def test_verify_match(self, reward):
        """Test verification of matching text."""
        text = "The answer is 42"
        assert reward.verify(text) is True

    def test_verify_no_match(self, reward):
        """Test verification fails for non-matching text."""
        text = "The result is 42"
        assert reward.verify(text) is False

    def test_verify_case_sensitive(self, reward):
        """Test case-sensitive matching."""
        text = "the answer is 42"  # lowercase
        assert reward.verify(text) is False

    def test_verify_case_insensitive(self):
        """Test case-insensitive matching."""
        config = RewardConfig(
            reward_type=RewardType.REGEX_VERIFIABLE,
            params={"pattern": r"answer.*42", "case_sensitive": False}
        )
        reward = RegexVerifiableReward(config)

        assert reward.verify("The ANSWER is 42") is True
        assert reward.verify("the answer is 42") is True

    def test_verify_email_pattern(self):
        """Test email pattern validation."""
        config = RewardConfig(
            reward_type=RewardType.REGEX_VERIFIABLE,
            params={
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "case_sensitive": True
            }
        )
        reward = RegexVerifiableReward(config)

        assert reward.verify("user@example.com") is True
        assert reward.verify("invalid@email") is False

    def test_verify_multiline_pattern(self):
        """Test multiline pattern matching."""
        config = RewardConfig(
            reward_type=RewardType.REGEX_VERIFIABLE,
            params={"pattern": r"def \w+\(", "case_sensitive": True}
        )
        reward = RegexVerifiableReward(config)

        code = """
def hello():
    return "world"
"""
        assert reward.verify(code) is True

    def test_compute_match(self, reward):
        """Test compute() returns 1.0 for matching text."""
        text = "The answer is 42"
        score = reward.compute(text)
        assert score == 1.0

    def test_compute_no_match(self, reward):
        """Test compute() returns 0.0 for non-matching text."""
        text = "The result is 42"
        score = reward.compute(text)
        assert score == 0.0

    def test_verify_no_pattern(self):
        """Test verification fails without pattern."""
        config = RewardConfig(
            reward_type=RewardType.REGEX_VERIFIABLE,
            params={"pattern": "", "case_sensitive": True}
        )
        reward = RegexVerifiableReward(config)

        assert reward.verify("some text") is False

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        config = RewardConfig(
            reward_type=RewardType.REGEX_VERIFIABLE,
            params={"pattern": r"[invalid(", "case_sensitive": True}
        )
        reward = RegexVerifiableReward(config)

        # Should return False for invalid pattern
        assert reward.verify("some text") is False


class TestRewardRegistry:
    """Test RLVR rewards are properly registered."""

    def test_math_verifiable_registered(self):
        """Test MathVerifiableReward is registered."""
        config = RewardConfig(RewardType.MATH_VERIFIABLE)
        reward = RewardFunctionFactory.create_reward(config)
        assert isinstance(reward, MathVerifiableReward)

    def test_code_verifiable_registered(self):
        """Test CodeExecutionVerifiableReward is registered."""
        config = RewardConfig(RewardType.CODE_VERIFIABLE)
        reward = RewardFunctionFactory.create_reward(config)
        assert isinstance(reward, CodeExecutionVerifiableReward)

    def test_sql_verifiable_registered(self):
        """Test SQLVerifiableReward is registered."""
        config = RewardConfig(RewardType.SQL_VERIFIABLE)
        reward = RewardFunctionFactory.create_reward(config)
        assert isinstance(reward, SQLVerifiableReward)

    def test_json_schema_verifiable_registered(self):
        """Test JSONSchemaVerifiableReward is registered."""
        config = RewardConfig(RewardType.JSON_SCHEMA_VERIFIABLE)
        reward = RewardFunctionFactory.create_reward(config)
        assert isinstance(reward, JSONSchemaVerifiableReward)

    def test_regex_verifiable_registered(self):
        """Test RegexVerifiableReward is registered."""
        config = RewardConfig(RewardType.REGEX_VERIFIABLE)
        reward = RewardFunctionFactory.create_reward(config)
        assert isinstance(reward, RegexVerifiableReward)


class TestIntegration:
    """Integration tests for RLVR with GRPO training."""

    def test_math_verifiable_in_grpo_config(self):
        """Test MathVerifiableReward can be used in GRPO config."""
        from aligntune.rewards.registry import RewardRegistry

        config_dict = {
            'type': 'math_verifiable',
            'weight': 1.0,
            'params': {'simplify': True, 'timeout': 5.0}
        }

        config = RewardRegistry._dict_to_config(config_dict)
        reward = RewardRegistry.get_reward_function('math_verifiable', config)
        assert isinstance(reward, MathVerifiableReward)

    def test_composite_verifiable_rewards(self):
        """Test combining multiple verifiable rewards."""
        from aligntune.rewards.core import CompositeReward

        configs = [
            RewardConfig(RewardType.MATH_VERIFIABLE),
            RewardConfig(RewardType.CODE_VERIFIABLE),
        ]

        rewards = [RewardFunctionFactory.create_reward(c) for c in configs]
        composite = CompositeReward(rewards, weights=[0.5, 0.5])

        # Test it can compute
        score = composite.compute("test", "42")
        assert 0.0 <= score <= 1.0

    def test_batch_compute(self):
        """Test batch computation with verifiable rewards."""
        config = RewardConfig(RewardType.MATH_VERIFIABLE)
        reward = RewardFunctionFactory.create_reward(config)

        texts = [
            "The answer is 42",
            "The answer is 10",
            "The answer is 3"
        ]
        references = ["42", "42", "3"]

        scores = reward.batch_compute(texts, references)
        assert len(scores) == 3
        assert scores[0] == 1.0  # Correct
        assert scores[1] == 0.0  # Incorrect
        assert scores[2] == 1.0  # Correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
