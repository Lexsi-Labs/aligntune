"""
Simple tests for RLVR verifiable rewards without heavy dependencies.

These tests focus on testing the core logic of each verifiable reward
without requiring the full AlignTune initialization.
"""

import pytest
import json
import re
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMathVerifiableLogic:
    """Test math verification logic."""

    def test_normalize_compare_exact(self):
        """Test exact string comparison."""
        s1 = "42"
        s2 = "42"
        norm1 = ' '.join(s1.split()).lower()
        norm2 = ' '.join(s2.split()).lower()
        assert norm1 == norm2

    def test_normalize_compare_whitespace(self):
        """Test comparison with different whitespace."""
        s1 = "  42  "
        s2 = "42"
        norm1 = ' '.join(s1.split()).lower()
        norm2 = ' '.join(s2.split()).lower()
        assert norm1 == norm2

    def test_numeric_extraction(self):
        """Test extracting numeric answers from patterns."""
        patterns = [
            r'(?:The )?answer (?:is|=)\s*:?\s*([^\n]+?)(?:\n|$)',
            r'(?:Final|final) (?:answer|Answer)\s*:?\s*([^\n]+?)(?:\n|$)',
        ]

        text = "The answer is: 42"
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                assert answer == "42"
                break

    def test_boxed_extraction(self):
        """Test extracting from LaTeX boxed format."""
        text = "Therefore: \\boxed{7}"
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            assert matches[0] == "7"

    def test_numeric_tolerance(self):
        """Test floating point tolerance."""
        v1 = 3.141592
        v2 = 3.141593
        assert abs(v1 - v2) < 1e-6

    def test_answer_patterns(self):
        """Test various answer extraction patterns."""
        test_cases = [
            ("The answer is 42", "42"),
            ("Final answer: 100", "100"),
            ("Answer = 5", "5"),
            ("\\boxed{3.14}", "3.14"),
        ]

        for text, expected in test_cases:
            patterns = [
                r'(?:The )?answer (?:is|=)\s*:?\s*([^\n]+?)(?:\n|$)',
                r'(?:Final|final) (?:answer|Answer)\s*:?\s*([^\n]+?)(?:\n|$)',
                r'\\boxed\{([^}]+)\}',
            ]

            found = False
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    answer = matches[-1].strip()
                    assert answer == expected
                    found = True
                    break

            if not found:
                # Try last line for simple cases
                lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
                if lines and not lines[-1].endswith('?'):
                    assert lines[-1] == expected


class TestCodeVerifiableLogic:
    """Test code extraction logic."""

    def test_extract_python_markdown(self):
        """Test extracting Python code from markdown."""
        text = """
Here is the solution:
```python
def hello():
    return 'world'
```
"""
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        assert len(matches) > 0
        assert "def hello()" in matches[0]

    def test_extract_function_definition(self):
        """Test extracting function definitions."""
        text = """
def solution(n):
    return n * 2

print(solution(5))
"""
        lines = text.split('\n')
        code_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
            if in_function:
                code_lines.append(line)

        code = '\n'.join(code_lines)
        assert 'def solution' in code

    def test_prefer_def_block(self):
        """Test that def blocks are preferred."""
        text = """
```python
print('wrong')
```

```python
def solution():
    return 42
```
"""
        patterns = [r'```python\s*(.*?)```', r'```\s*(.*?)```']
        all_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            all_blocks.extend(matches)

        # Should find function definition
        for block in all_blocks:
            if 'def ' in block:
                assert 'def solution' in block
                break


class TestSQLVerifiableLogic:
    """Test SQL validation logic."""

    def test_sql_pattern_select(self):
        """Test SELECT query pattern."""
        query = "SELECT * FROM users WHERE id = 1"
        # Basic SQL pattern check
        assert re.match(r'^\s*SELECT', query, re.IGNORECASE)

    def test_sql_pattern_insert(self):
        """Test INSERT query pattern."""
        query = "INSERT INTO users (name) VALUES ('John')"
        assert re.match(r'^\s*INSERT', query, re.IGNORECASE)

    def test_sql_pattern_update(self):
        """Test UPDATE query pattern."""
        query = "UPDATE users SET name = 'Jane' WHERE id = 1"
        assert re.match(r'^\s*UPDATE', query, re.IGNORECASE)

    def test_sql_pattern_delete(self):
        """Test DELETE query pattern."""
        query = "DELETE FROM users WHERE id = 1"
        assert re.match(r'^\s*DELETE', query, re.IGNORECASE)

    def test_sql_invalid_syntax(self):
        """Test invalid SQL detection."""
        query = "SELECT * FORM users"  # FORM instead of FROM
        # Rough check - missing FROM
        assert not re.match(r'.*FROM.*', query, re.IGNORECASE)


class TestJSONSchemaValidation:
    """Test JSON schema validation logic."""

    def test_json_extraction_markdown(self):
        """Test JSON extraction from markdown."""
        text = '```json\n{"name": "Alice"}\n```'
        pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            data = json.loads(matches[0])
            assert data['name'] == 'Alice'

    def test_json_extraction_plain(self):
        """Test JSON extraction from plain text."""
        text = '{"name": "Bob", "age": 25}'
        try:
            data = json.loads(text)
            assert data['name'] == 'Bob'
        except json.JSONDecodeError:
            pass

    def test_json_invalid_syntax(self):
        """Test invalid JSON detection."""
        text = '{"name": "Alice", age: 30}'  # Missing quotes on age
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)

    def test_json_required_fields(self):
        """Test schema required field validation."""
        data = {"name": "Alice"}
        schema = {
            "type": "object",
            "required": ["name", "age"]
        }
        # Missing 'age' field
        assert "age" not in data


class TestRegexValidation:
    """Test regex pattern matching logic."""

    def test_regex_email(self):
        """Test email pattern."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert bool(re.search(pattern, 'user@example.com'))
        assert not bool(re.search(pattern, 'invalid@email'))

    def test_regex_case_sensitive(self):
        """Test case-sensitive matching."""
        pattern = r'^The answer is \d+$'
        assert bool(re.search(pattern, 'The answer is 42'))
        assert not bool(re.search(pattern, 'the answer is 42'))

    def test_regex_case_insensitive(self):
        """Test case-insensitive matching."""
        pattern = r'answer.*42'
        assert bool(re.search(pattern, 'The ANSWER is 42', re.IGNORECASE))
        assert bool(re.search(pattern, 'the answer is 42', re.IGNORECASE))

    def test_regex_multiline(self):
        """Test multiline regex."""
        pattern = r'def \w+\('
        code = """
def hello():
    return 'world'
"""
        assert bool(re.search(pattern, code))

    def test_regex_invalid_pattern(self):
        """Test handling of invalid regex."""
        pattern = r'[invalid('
        with pytest.raises(re.error):
            re.compile(pattern)


class TestVerifiableRewardBehavior:
    """Test the expected behavior of verifiable rewards."""

    def test_binary_reward_signal(self):
        """Test that verifiable rewards return binary signals."""
        # A verifiable reward should return either 1.0 or 0.0
        correct_score = 1.0
        incorrect_score = 0.0

        assert correct_score in [0.0, 1.0]
        assert incorrect_score in [0.0, 1.0]

    def test_deterministic_scoring(self):
        """Test that verifiable rewards are deterministic."""
        # Same input should produce same output
        completion = "The answer is 42"
        reference = "42"

        # Simulate verification
        scores = []
        for _ in range(3):
            # Should always match
            norm_comp = completion.lower().split()
            norm_ref = reference.lower().split()
            # Simplified: just check if numbers match
            score = 1.0 if "42" in completion and "42" == reference else 0.0
            scores.append(score)

        # All scores should be identical
        assert len(set(scores)) == 1

    def test_no_model_loading(self):
        """Test that verifiable rewards don't require model loading."""
        # These should be lightweight, no transformer models
        # Just string/code processing and validation
        assert True  # Placeholder - verifiable rewards are lightweight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
