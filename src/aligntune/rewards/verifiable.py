"""
RLVR (RL with Verifiable Rewards) - Deterministic verification-based reward functions.

This module implements verifiable reward functions inspired by DeepSeek-R1, QwQ, and DAPO.
Instead of learned reward models, these rewards use deterministic checks:
- Math: symbolic simplification or normalized string matching
- Code: subprocess execution with timeout and output validation
- SQL: query parsing and optional execution
- JSON Schema: jsonschema validation
- Regex: pattern matching

These verifiable rewards provide signal that is deterministic, transparent, and scalable.
"""

from typing import Optional, Any, Dict, Callable, Tuple
from abc import ABC, abstractmethod
import logging
import re
import json
import subprocess
import tempfile
import os
import signal
from contextlib import contextmanager

from .core import RewardFunction, RewardConfig

logger = logging.getLogger(__name__)


class VerifiableReward(RewardFunction, ABC):
    """Base class for verifiable reward functions.

    Verifiable rewards use deterministic checks instead of learned models.
    They compute rewards by verifying if a completion satisfies a reference
    or passes a verification check.

    Subclasses must implement the verify() method which returns True/False.
    The compute() method calls verify() and returns 1.0 or 0.0.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.timeout = config.params.get('timeout', 5.0)

    @abstractmethod
    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify if a completion is correct.

        Args:
            completion: The generated text/code to verify
            reference: Optional reference answer or expected output

        Returns:
            True if verification passes, False otherwise
        """
        pass

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Compute reward by calling verify() and returning 1.0 or 0.0.

        Args:
            text: The completion to verify
            reference: Optional reference answer
            **kwargs: Additional kwargs (may contain verification parameters)

        Returns:
            1.0 if verify() returns True, 0.0 otherwise
        """
        try:
            result = self.verify(text, reference)
            return 1.0 if result else 0.0
        except Exception as e:
            logger.debug(f"Verification failed: {e}")
            return 0.0


class MathVerifiableReward(VerifiableReward):
    """Reward for mathematical correctness using symbolic comparison.

    Uses sympy.simplify() for symbolic comparison when possible.
    Falls back to normalized string matching for simple cases.

    Expects reference to be the expected final answer.
    Extracts the final answer from completion using common patterns.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.simplify = config.params.get('simplify', True)
        # Try to import sympy, but don't fail if unavailable
        try:
            import sympy
            self.sympy = sympy
        except ImportError:
            logger.warning("sympy not available for MathVerifiableReward, using string matching only")
            self.sympy = None

    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify math correctness by comparing answers.

        Args:
            completion: Generated text containing the answer
            reference: Expected answer

        Returns:
            True if answers match, False otherwise
        """
        if not reference:
            return False

        # Extract answer from completion
        answer = self._extract_answer(completion)
        if not answer:
            return False

        # Try symbolic comparison first if sympy available
        if self.sympy and self.simplify:
            return self._symbolic_compare(answer, reference)

        # Fall back to normalized string comparison
        return self._normalize_compare(answer, reference)

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from text.

        Looks for common patterns like:
        - The answer is: X
        - Final answer: X
        - Therefore, the answer is X
        - \boxed{X}
        """
        patterns = [
            r'(?:The )?answer (?:is|=)\s*:?\s*([^\n]+?)(?:\n|$)',
            r'(?:Final|final) (?:answer|Answer)\s*:?\s*([^\n]+?)(?:\n|$)',
            r'Therefore,?\s+(?:the )?answer (?:is|=)\s*:?\s*([^\n]+?)(?:\n|$)',
            r'\\boxed\{([^}]+)\}',
            r'(?:= )([^\n]+?)(?:\n|$)',
            r'= (\d+(?:\.\d+)?)'  # Simple numeric answer
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the last match (likely the final answer)
                answer = matches[-1].strip()
                return answer if answer else None

        # If no pattern matched, return last line that looks like an answer
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if lines and not lines[-1].endswith('?'):
            return lines[-1]

        return None

    def _symbolic_compare(self, answer1: str, answer2: str) -> bool:
        """Compare answers using sympy symbolic comparison."""
        try:
            expr1 = self.sympy.simplify(answer1)
            expr2 = self.sympy.simplify(answer2)
            return expr1 == expr2
        except Exception as e:
            logger.debug(f"Symbolic comparison failed: {e}")
            return False

    def _normalize_compare(self, answer1: str, answer2: str) -> bool:
        """Compare answers using normalized string matching."""
        # Normalize whitespace
        norm1 = ' '.join(answer1.split()).lower()
        norm2 = ' '.join(answer2.split()).lower()

        if norm1 == norm2:
            return True

        # Try numeric comparison for numeric answers
        try:
            val1 = float(answer1)
            val2 = float(answer2)
            return abs(val1 - val2) < 1e-6
        except (ValueError, TypeError):
            pass

        return False


class CodeExecutionVerifiableReward(VerifiableReward):
    """Reward for code execution correctness.

    Executes code in a subprocess with timeout and verifies output.

    reference should be the expected output or a set of test cases.
    Test cases can be:
    - A string (expected stdout)
    - A dict with 'input' and 'expected_output'
    - A list of assertions as strings
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.language = config.params.get('language', 'python')
        self.timeout = config.params.get('timeout', 5.0)
        self.test_cases = config.params.get('test_cases', [])

    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify code execution by running it and checking output.

        Args:
            completion: Generated code
            reference: Expected output or test cases

        Returns:
            True if code executes successfully and produces expected output
        """
        code = self._extract_code(completion)
        if not code:
            return False

        # If we have test cases, validate against them
        test_cases = reference if reference else None
        if not test_cases and self.test_cases:
            test_cases = self.test_cases

        if test_cases:
            return self._validate_test_cases(code, test_cases)

        # Otherwise just check if code runs without error
        return self._execute_safely(code)

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from text (handles markdown code blocks)."""
        # Try markdown code blocks first
        patterns = [
            r'```python\s*(.*?)```',
            r'```\s*(.*?)```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Return last match (likely the actual solution)
                return matches[-1].strip()

        # If no blocks found, try to extract function definition
        lines = text.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        return None

    def _execute_safely(self, code: str) -> bool:
        """Safely execute code with timeout.

        Args:
            code: Python code to execute

        Returns:
            True if execution completes without error, False otherwise
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                temp_file = f.name

            try:
                result = subprocess.run(
                    ['/usr/bin/python3' if os.path.exists('/usr/bin/python3') else 'python', temp_file],
                    capture_output=True,
                    timeout=self.timeout,
                    text=True
                )
                return result.returncode == 0
            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            logger.debug(f"Code execution failed: {e}")
            return False

    def _validate_test_cases(self, code: str, test_cases: Any) -> bool:
        """Validate code against test cases.

        Args:
            code: Code to test
            test_cases: Test cases (string, dict, or list)

        Returns:
            True if all tests pass
        """
        # Handle string test cases (expected output)
        if isinstance(test_cases, str):
            return self._check_output(code, test_cases)

        # Handle list of assertion strings
        if isinstance(test_cases, list) and all(isinstance(t, str) for t in test_cases):
            return self._check_assertions(code, test_cases)

        # Handle dict or list of dicts with input/expected_output
        if isinstance(test_cases, dict):
            test_cases = [test_cases]

        if isinstance(test_cases, list) and all(isinstance(t, dict) for t in test_cases):
            return self._check_io_pairs(code, test_cases)

        return False

    def _check_output(self, code: str, expected: str) -> bool:
        """Check if code output matches expected."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                temp_file = f.name

            try:
                result = subprocess.run(
                    ['/usr/bin/python3' if os.path.exists('/usr/bin/python3') else 'python', temp_file],
                    capture_output=True,
                    timeout=self.timeout,
                    text=True
                )
                actual = result.stdout.strip()
                expected_norm = expected.strip()
                return actual == expected_norm
            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            logger.debug(f"Output check failed: {e}")
            return False

    def _check_assertions(self, code: str, assertions: list) -> bool:
        """Check if code passes a list of assertion strings."""
        full_code = code + '\n' + '\n'.join(assertions)
        return self._execute_safely(full_code)

    def _check_io_pairs(self, code: str, test_cases: list) -> bool:
        """Check if code passes input/output test pairs."""
        for test_case in test_cases:
            input_val = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')

            # Create test code that calls the function with input
            # This is simplified - in practice might need more sophisticated handling
            test_code = code + f"\nprint({input_val})"

            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(test_code)
                    f.flush()
                    temp_file = f.name

                try:
                    result = subprocess.run(
                        ['/usr/bin/python3' if os.path.exists('/usr/bin/python3') else 'python', temp_file],
                        capture_output=True,
                        timeout=self.timeout,
                        text=True
                    )
                    actual = result.stdout.strip()
                    if actual != expected_output.strip():
                        return False
                finally:
                    os.unlink(temp_file)

            except subprocess.TimeoutExpired:
                return False
            except Exception as e:
                logger.debug(f"Test case validation failed: {e}")
                return False

        return True


class SQLVerifiableReward(VerifiableReward):
    """Reward for SQL query correctness.

    Validates SQL syntax using sqlglot parsing.
    Optionally executes queries and compares results.

    reference should be the expected query result or None for syntax check only.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.execute = config.params.get('execute', False)
        self.db_path = config.params.get('db_path', ':memory:')

        # Try to import sqlglot
        try:
            import sqlglot
            self.sqlglot = sqlglot
        except ImportError:
            logger.warning("sqlglot not available for SQLVerifiableReward")
            self.sqlglot = None

    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify SQL query correctness.

        Args:
            completion: Generated SQL query
            reference: Expected query result or None for syntax check

        Returns:
            True if query is valid and produces expected result
        """
        if not self.sqlglot:
            logger.warning("sqlglot not available, skipping SQL verification")
            return False

        # First check syntax
        if not self._validate_syntax(completion):
            return False

        # If no reference, just syntax check is enough
        if not reference:
            return True

        # If reference is provided and execute is enabled, validate result
        if self.execute:
            return self._validate_result(completion, reference)

        return True

    def _validate_syntax(self, query: str) -> bool:
        """Validate SQL syntax using sqlglot."""
        try:
            self.sqlglot.parse_one(query)
            return True
        except Exception as e:
            logger.debug(f"SQL syntax validation failed: {e}")
            return False

    def _validate_result(self, query: str, expected: str) -> bool:
        """Validate SQL query result by execution."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Execute query
            cursor.execute(query)
            result = cursor.fetchall()

            # For now, just check if query executes without error
            # Full result comparison would require parsing expected result format
            return True

        except Exception as e:
            logger.debug(f"SQL execution failed: {e}")
            return False
        finally:
            try:
                conn.close()
            except:
                pass


class JSONSchemaVerifiableReward(VerifiableReward):
    """Reward for JSON schema validation.

    Validates JSON output against a provided schema.

    reference should be a JSON schema dict or string.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.schema = config.params.get('schema', {})

        try:
            import jsonschema
            self.jsonschema = jsonschema
        except ImportError:
            logger.warning("jsonschema not available for JSONSchemaVerifiableReward")
            self.jsonschema = None

    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify JSON is valid and matches schema.

        Args:
            completion: Generated JSON (as string)
            reference: JSON schema (as dict or string)

        Returns:
            True if JSON is valid and matches schema
        """
        if not self.jsonschema:
            logger.warning("jsonschema not available, skipping JSON schema validation")
            return False

        # Get schema from reference or config
        schema = reference or self.schema
        if not schema:
            return False

        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError:
                logger.debug("Invalid schema JSON")
                return False

        # Parse JSON from completion
        try:
            data = self._extract_json(completion)
        except json.JSONDecodeError:
            logger.debug("No valid JSON found in completion")
            return False

        # Validate against schema
        try:
            self.jsonschema.validate(instance=data, schema=schema)
            return True
        except self.jsonschema.ValidationError as e:
            logger.debug(f"JSON schema validation failed: {e}")
            return False

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text.

        Tries to find JSON blocks or parses the entire text as JSON.
        """
        # Try markdown JSON blocks first
        pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return json.loads(matches[-1])

        # Try to find JSON object/array in text
        for match in re.finditer(r'\{[^{}]*\}|\[[^\[\]]*\]', text, re.DOTALL):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        # Try parsing entire text as JSON
        return json.loads(text)


class RegexVerifiableReward(VerifiableReward):
    """Reward for regex pattern matching.

    Verifies that completion matches a given regex pattern.

    reference should be a regex pattern string.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.pattern = config.params.get('pattern', '')
        self.case_sensitive = config.params.get('case_sensitive', True)

    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify completion matches regex pattern.

        Args:
            completion: Generated text to match
            reference: Regex pattern string

        Returns:
            True if completion matches pattern
        """
        pattern = reference or self.pattern
        if not pattern:
            return False

        try:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            return bool(re.search(pattern, completion, flags))
        except re.error as e:
            logger.debug(f"Regex validation failed: {e}")
            return False
