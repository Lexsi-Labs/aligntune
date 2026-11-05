"""
RLVR (RL with Verifiable Rewards) - Deterministic verification-based reward functions.

This module implements verifiable reward functions inspired by DeepSeek-R1, QwQ, and DAPO.
Instead of learned reward models, these rewards use deterministic checks:
- Math: symbolic simplification or normalized string matching
- Code: execution in a child process, graded by value outside it
- SQL: query parsing and optional execution
- JSON Schema: jsonschema validation
- Regex: pattern matching

These verifiable rewards provide signal that is deterministic, transparent, and scalable.
"""

from typing import Optional, Any, Dict, Callable, Tuple
from abc import ABC, abstractmethod
import logging
import os
import re
import json
import signal
from contextlib import contextmanager

from .core import RewardFunction, RewardConfig
from . import _code_grader
from ._answers import (
    answers_match,
    clean_answer,
    extract_completion_answer,
    extract_reference_answer,
)

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
            logger.warning(f"Verification failed for {type(self).__name__}: {e}")
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

        reference = extract_reference_answer(reference)
        answer = self._extract_answer(completion)
        if not answer:
            return False
        answer = clean_answer(answer)

        # Numeric / normalized string comparison handles "48." vs "48", "1,234", fractions
        if answers_match(answer, reference):
            return True

        # Symbolic comparison for non-numeric expressions
        if self.sympy and self.simplify:
            return self._symbolic_compare(answer, reference)

        return False

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from text.

        Looks for common patterns like:
        - The answer is: X
        - Final answer: X
        - Therefore, the answer is X
        - \boxed{X}
        """
        shared = extract_completion_answer(text)
        if shared:
            return shared

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
            expr1 = self.sympy.sympify(answer1)
            expr2 = self.sympy.sympify(answer2)
            return bool(self.sympy.simplify(expr1 - expr2) == 0)
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


_NO_TESTS_WARNED = set()


def _warn_no_tests_once(name: str, kwargs: Dict[str, Any]) -> None:
    if name not in _NO_TESTS_WARNED:
        _NO_TESTS_WARNED.add(name)
        logger.warning(f"{name}: no tests reached this reward (row keys: {sorted(kwargs)[:20]}); it scores 0.0 "
                       f"without tests. Pass `reference`, `test_cases`, MBPP's `test_list`, or "
                       f"HumanEval's `test` + `entry_point`.")


class CodeExecutionVerifiableReward(VerifiableReward):
    """Reward for code execution correctness.

    Runs code in a child process with a timeout and verifies the values it returns.

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

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """Like the base class, but the row's test columns reach ``verify``."""
        try:
            return 1.0 if self.verify(text, reference, **kwargs) else 0.0
        except Exception as e:
            logger.warning(f"Verification failed for {type(self).__name__}: {e}")
            return 0.0

    def verify(self, completion: str, reference: Optional[str] = None, **row) -> bool:
        """Verify code execution by running it and checking output.

        Args:
            completion: Generated code
            reference: Expected output or test cases
            **row: Dataset columns. Tests come from ``test_cases``, MBPP's
                ``test_list``, or HumanEval's ``test`` plus ``entry_point``;
                they are used before ``reference``.

        Returns:
            True if code executes successfully and produces expected output
        """
        code = self._extract_code(completion)
        if not code:
            return False
        tests, setup = _code_grader.tests_from_columns(row)
        if tests:
            code = _code_grader.with_prompt(code, completion, row)
            if setup:
                code = code + "\n\n" + setup
            return self._validate_test_cases(code, tests)

        # If we have test cases, validate against them
        test_cases = reference if reference else None
        if not test_cases and self.test_cases:
            test_cases = self.test_cases

        if test_cases:
            return self._validate_test_cases(code, test_cases)

        # No tests: code that merely runs (a stub included) is not paid.
        _warn_no_tests_once(type(self).__name__, row)
        return False

    def _extract_code(self, text: str) -> Optional[str]:
        """Code from a completion, as the other code rewards read it (fences, imports above the first def)."""
        return _code_grader.extract_code(text) or None

    def _execute_safely(self, code: str) -> bool:
        """Whether the code runs to its end in a child process (see ``_code_grader``)."""
        status, _ = _code_grader.grade(code, [], self.timeout)
        return status == "ok"

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

        # A HumanEval check() that can't be split into asserts (loops, locals, multi-line asserts)
        if isinstance(test_cases, list) and len(test_cases) == 1 and isinstance(test_cases[0], _code_grader.HumanEvalCheck):
            return self._check_assertions(code, test_cases)

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
        """Check that the code runs to its end and prints the expected text."""
        run = _code_grader.run_candidate(code, self.timeout, capture_stdout=True)
        return run.status == "ok" and run.stdout.strip() == expected.strip()

    def _check_assertions(self, code: str, assertions: list) -> bool:
        """Check that every assertion holds, graded by value in this process."""
        status, passed = _code_grader.grade(code, assertions, self.timeout)
        return status == "ok" and all(passed)

    def _check_io_pairs(self, code: str, test_cases: list) -> bool:
        """Check each pair against the value the candidate returns.

        ``input`` is an expression string (``"add(1, 2)"``) or, as in code_correctness, the
        argument(s) for the candidate's first function (a list or tuple of several values is spread).
        Text in ``expected_output`` is compared with the printed value; any other
        ``expected_output`` is compared as a value of the same type (numbers interchange).
        """
        items = []
        for test_case in test_cases:
            input_val = test_case.get('input', '')
            if isinstance(input_val, str):
                items.append({"expr": input_val})
            else:
                items.append({"call": input_val, "no_args": True})
        run = _code_grader.run_candidate(code, self.timeout, items)
        if run.status != "ok":
            return False
        for value, test_case in zip(run.values, test_cases):
            if value is _code_grader.FAILED:
                return False
            expected = test_case.get('expected_output', '')
            if isinstance(expected, str):
                if str(value).strip() != expected.strip():
                    return False
            elif isinstance(value, bool) or isinstance(expected, bool):
                if value is not expected:
                    return False
            elif isinstance(value, (int, float)) and isinstance(expected, (int, float)):
                if value != expected:
                    return False
            elif type(value) is not type(expected) or value != expected:
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

        # If no reference, this is a syntax check only (any query that parses passes).
        if not reference:
            return True

        # A reference can only be checked by running both queries. Without execute=True we cannot
        # verify the result, so do not pay (this used to return True for any parseable query).
        if not self.execute:
            if not getattr(self, "_warned_no_execute", False):
                self._warned_no_execute = True
                logger.warning("SQLVerifiableReward: a reference was given but execute=False; "
                               "results cannot be checked, so the reward is 0. Set params.execute=True.")
            return False
        return self._validate_result(completion, reference)

    def _validate_syntax(self, query: str) -> bool:
        """Validate SQL syntax using sqlglot."""
        try:
            self.sqlglot.parse_one(query)
            return True
        except Exception as e:
            logger.debug(f"SQL syntax validation failed: {e}")
            return False

    def _connect_readonly(self):
        """Open db_path read-only, so a generated query cannot modify the user's database.
        ':memory:' gives a fresh empty database each time."""
        import sqlite3
        if self.db_path in (None, "", ":memory:"):
            return sqlite3.connect(":memory:")
        from urllib.parse import quote
        return sqlite3.connect(f"file:{quote(os.path.abspath(self.db_path))}?mode=ro", uri=True)

    def _run_query(self, query: str, timeout: float = 5.0):
        """Run one read-only statement with a time limit; returns the rows."""
        import time as _time
        conn = self._connect_readonly()
        try:
            deadline = _time.monotonic() + timeout
            conn.set_progress_handler(lambda: 1 if _time.monotonic() > deadline else 0, 10_000)
            cursor = conn.cursor()
            cursor.execute(query)  # execute() runs a single statement only
            return cursor.fetchall()
        finally:
            conn.close()

    @staticmethod
    def _normalise_rows(rows):
        return sorted((tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in rows), key=repr)

    def _validate_result(self, query: str, expected: str) -> bool:
        """Run the query and compare its rows with the reference.

        The reference may be a SQL query (run on the same database) or the expected rows as
        JSON / a Python literal, e.g. [[1, "a"], [2, "b"]]. Row order is ignored.
        """
        try:
            actual = self._run_query(query)
        except Exception as e:
            logger.debug(f"SQL execution failed: {e}")
            return False
        try:
            if isinstance(expected, str) and self._validate_syntax(expected) and \
                    not expected.lstrip().startswith(("[", "(")):
                want = self._run_query(expected)
            elif isinstance(expected, str):
                try:
                    want = json.loads(expected)
                except ValueError:
                    import ast as _ast
                    want = _ast.literal_eval(expected)
            else:
                want = expected
            return self._normalise_rows(actual) == self._normalise_rows(want)
        except Exception as e:
            logger.debug(f"SQL reference could not be evaluated: {e}")
            return False


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
