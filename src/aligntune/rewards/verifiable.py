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
import sys
from contextlib import contextmanager

from .core import RewardFunction, RewardConfig
from ._code_sandbox import (
    HumanEvalTests,
    code_runs,
    defines,
    grade_tests_detail,
    run_calls,
    run_program,
    setup_from_kwargs,
    tests_from_kwargs,
)
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
        logger.warning(f"{name}: no tests reached this reward (row keys: {sorted(kwargs)[:20]}); it can only "
                       f"check that code runs. Pass `reference`, `test_cases`, MBPP's `test_list`, or "
                       f"HumanEval's `test` + `entry_point`.")


def _first_public_function(code: str) -> Optional[str]:
    import ast
    try:
        for node in ast.parse(code).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                return node.name
    except SyntaxError:
        pass
    return None


def _is_call_expression(s: str) -> bool:
    import ast
    try:
        node = ast.parse(s.strip(), mode="eval").body
    except SyntaxError:
        return False
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name)


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

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        """1.0 if the code passes its tests, else 0.0.

        Tests come from ``reference``, else from the dataset row (``test_cases``, MBPP's
        ``test_list`` + ``test_imports``, HumanEval's ``test`` + ``entry_point``), else from
        ``params.test_cases``. Before, row columns were ignored, so on MBPP/HumanEval rows this
        reward took the no-test path and paid 1.0 for any code that ran (a stub included).
        """
        if not isinstance(text, str):
            return 0.0
        try:
            tests = reference if reference else tests_from_kwargs(kwargs, None)
            if not tests and self.test_cases:
                tests = self.test_cases
            if not tests:
                _warn_no_tests_once(type(self).__name__, kwargs)
            ok = self._verify(text, tests, setup=setup_from_kwargs(kwargs),
                              entry=kwargs.get("entry_point"), prompt=kwargs.get("prompt"))
            return 1.0 if ok else 0.0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Verification failed for {type(self).__name__}: {e}")
            return 0.0

    def verify(self, completion: str, reference: Optional[str] = None) -> bool:
        """Verify code execution by running it and checking output.

        Args:
            completion: Generated code
            reference: Expected output or test cases

        Returns:
            True if code executes successfully and produces expected output
        """
        tests = reference if reference else (self.test_cases or None)
        return self._verify(completion, tests)

    def _verify(self, completion: str, test_cases: Any, setup: str = "", entry: Optional[str] = None,
                prompt: Any = None) -> bool:
        code = self._extract_code(completion)
        if isinstance(test_cases, HumanEvalTests):
            entry = test_cases.entry_point
            prompt = test_cases.prompt
        if entry and isinstance(prompt, str) and not defines(code or "", entry) and defines(prompt, entry):
            # HumanEval-style body-only continuation (the body alone has no def to extract)
            body = code if code and "```" in completion else completion
            code = prompt + body
        if not code:
            return False

        if test_cases:
            return self._validate_test_cases(code, test_cases, setup=setup)

        # No tests anywhere: only check that the code runs (out of process). This cannot tell a
        # stub from a solution; a warning has been logged.
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

        # If no blocks found, start at the first top-level def/class/import line (keeps imports)
        lines = text.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if re.match(r'(def |async def |class |import |from \S+ import |@)', line):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        return None

    def _execute_safely(self, code: str) -> bool:
        """True if code runs to the end without raising, in a separate process
        (sys.exit()/os._exit() count as failure; the exit code is not used)."""
        try:
            return code_runs(code, timeout=self.timeout)
        except Exception as e:
            logger.debug(f"Code execution failed: {e}")
            return False

    def _validate_test_cases(self, code: str, test_cases: Any, setup: str = "") -> bool:
        """Validate code against test cases.

        Args:
            code: Code to test
            test_cases: Test cases (string, dict, or list)

        Returns:
            True if all tests pass
        """
        if isinstance(test_cases, HumanEvalTests):
            load, results = grade_tests_detail(code, test_cases, setup=setup, timeout=self.timeout)
            return load is None and bool(results) and all(results)

        # Handle string test cases (expected output)
        if isinstance(test_cases, str):
            return self._check_output(code, test_cases)

        # Handle list of assertion strings
        if isinstance(test_cases, list) and all(isinstance(t, str) for t in test_cases):
            return self._check_assertions(code, test_cases, setup=setup)

        # Handle dict or list of dicts with input/expected_output
        if isinstance(test_cases, dict):
            test_cases = [test_cases]

        if isinstance(test_cases, list) and all(isinstance(t, dict) for t in test_cases):
            return self._check_io_pairs(code, test_cases)

        return False

    def _check_output(self, code: str, expected: str) -> bool:
        """Check if the program's stdout matches expected (a stdout task by definition).

        Runs in the same sandbox as the other code rewards (limits, process-group kill, seccomp/
        Landlock where available); exiting early or raising counts as failure.
        """
        try:
            load, stdout = run_program(code, timeout=self.timeout)
            return load is None and stdout.strip() == expected.strip()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Output check failed: {e}")
            return False

    def _check_assertions(self, code: str, assertions: list, setup: str = "") -> bool:
        """All assertion strings pass, graded values out (not "the process exited 0")."""
        load, results = grade_tests_detail(code, assertions, setup=setup, timeout=self.timeout)
        return load is None and bool(results) and all(results)

    def _check_io_pairs(self, code: str, test_cases: list) -> bool:
        """Values out: the child returns the value, compared here as plain data.

        ``input`` is either a call expression string (``"add(1, 2)"``) or, as in code_correctness,
        the argument(s) for the first public function (a list/tuple of several values is spread).
        """
        func = _first_public_function(code)
        calls = []
        for tc in test_cases:
            inp = tc.get('input')
            if isinstance(inp, str) and _is_call_expression(inp):
                calls.append(inp)
            elif func is None:
                return False
            elif isinstance(inp, (list, tuple)) and len(inp) > 1:
                calls.append(f"{func}(*{inp!r})")
            elif inp is None:
                calls.append(f"{func}()")
            else:
                calls.append(f"{func}({inp!r})")
        load, results = run_calls(code, calls, timeout=self.timeout)
        if load is not None:
            return False
        for (kind, value), tc in zip(results, test_cases):
            if kind != "ok":
                return False
            expected = tc.get('expected_output', '')
            if isinstance(expected, str) and not isinstance(value, str):
                if str(value) != expected.strip():  # legacy: expected given as printed text
                    return False
            elif type(value) is not type(expected) and not (
                    isinstance(value, (int, float)) and isinstance(expected, (int, float))
                    and not isinstance(value, bool) and not isinstance(expected, bool)):
                return False
            elif value != expected:
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
