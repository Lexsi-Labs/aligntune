"""
Safe code execution environment for evaluating generated code.

Provides sandboxed execution with:
- Timeout protection
- Resource limits
- Isolated namespace
- Test case validation
- Multiple language support
"""

import ast
import sys
import signal
import io
import contextlib
import traceback
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    test_passed: bool = False
    test_results: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []


class TimeoutException(Exception):
    """Exception raised when execution times out."""
    pass


class SafeCodeExecutor:
    """
    Safe executor for generated code with multiple protection layers.
    
    Features:
    - Runs code in the AlignTune code sandbox (separate process, limits, timeout,
      seccomp/Landlock confinement on Linux)
    - Tests graded "values out" in the calling process
    - Captured stdout for test-free runs

    ``safe_builtins`` is kept for backward compatibility and is no longer used: the sandbox
    process, not a restricted namespace, is the security boundary.
    """
    
    def __init__(
        self,
        timeout: int = 5,
        max_memory_mb: int = 512,
        language: str = "python"
    ):
        """
        Initialize safe executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            language: Programming language ("python", "javascript", etc.)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.language = language.lower()
        
        # Safe builtins for Python execution
        self.safe_builtins = {
            # Basic types
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            
            # Basic functions
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            
            # Type checks
            'isinstance': isinstance,
            'type': type,
            
            # String operations
            'print': print,
            
            # Math
            'pow': pow,
            'divmod': divmod,
            
            # Special
            'None': None,
            'True': True,
            'False': False,
        }
    
    def extract_code(self, text: str) -> str:
        """
        Extract code from markdown blocks or plain text.
        
        Args:
            text: Text potentially containing code
            
        Returns:
            Extracted code string
        """
        import re
        
        # Try to find code in markdown blocks
        # Pattern 1: ```python ... ```
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Pattern 2: ```language ... ```
        pattern = r'```\w+\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Pattern 3: ``` ... ``` (no language)
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # No markdown block found, return as-is
        return text.strip()
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax without executing.
        
        Args:
            code: Python code to validate
            
        Returns:
            (is_valid, error_message)
        """
        if self.language != "python":
            # Can't validate syntax for non-Python
            return True, ""
        
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def execute(
        self,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute code safely with timeout and test cases.

        The code runs in the AlignTune code sandbox (a separate ``sys.executable -I`` process with
        resource limits, a wall-clock timeout and, on Linux, seccomp/Landlock confinement). Tests
        are graded "values out": the child only returns the values the candidate's function
        produced, as plain data, and every assert / comparison is evaluated here. Printing,
        exit codes and objects with a custom ``__eq__`` cannot pass a test.

        Args:
            code: Code to execute
            test_cases: Optional tests: MBPP assert strings, a HumanEval ``check(candidate)``
                program, or ``[{"input": ..., "expected_output": ...}, ...]``

        Returns:
            ExecutionResult with execution details
        """
        from aligntune.rewards._code_sandbox import (
            HumanEvalTests, defines, entry_name, grade_tests_detail, is_plain, run_calls, run_program,
        )

        if test_cases is None:
            test_cases = []

        # Extract code from markdown if needed
        code = self.extract_code(code)

        # Validate syntax first
        is_valid, error_msg = self.validate_syntax(code)
        if not is_valid:
            return ExecutionResult(success=False, error=error_msg)

        start = time.time()
        mem = int(self.max_memory_mb) * 1024 * 1024

        def failed(load, **kw):
            err = f"Execution timeout ({self.timeout}s exceeded)" if load == "timed out" else str(load)
            return ExecutionResult(success=False, error=err, execution_time=time.time() - start, **kw)

        if not test_cases:
            load, output = run_program(code, timeout=self.timeout, mem_bytes=mem)
            if load is not None:
                return failed(load, output=output)
            return ExecutionResult(success=True, output=output, execution_time=time.time() - start,
                                   test_passed=True)

        main = self._first_function(code)
        if isinstance(test_cases, str) and 'def check(candidate)' in test_cases:
            # HumanEval format: check(candidate) runs here, candidate calls go to the sandbox
            if main is None:
                return failed("No callable functions found in code")
            # The entry point is not passed in, and helpers are often defined before the solution:
            # try each top-level function (last defined first); every try is graded values out.
            load, results = None, [False]
            for name in self._public_functions(code)[::-1][:5]:
                load, results = grade_tests_detail(code, HumanEvalTests(test_cases, name), timeout=self.timeout)
                if load is None and results and all(results):
                    break
            test_results = [{'test_id': i, 'format': 'humaneval_check', 'passed': ok}
                            for i, ok in enumerate(results)]
        elif isinstance(test_cases, list) and isinstance(test_cases[0], str) \
                and test_cases[0].strip().startswith('assert'):
            # MBPP format: assert strings; a differently named solution is aliased to the tested name
            entry = entry_name(test_cases)
            if entry and main and not defines(code, entry):
                code = code + f"\n\n{entry} = {main}\n"
            load, results = grade_tests_detail(code, test_cases, timeout=self.timeout, entry=entry)
            test_results = [{'test_id': i, 'assertion': a, 'passed': ok}
                            for i, (a, ok) in enumerate(zip(test_cases, results))]
        else:
            # Structured dict format
            if main is None:
                return failed("No callable functions found in code")
            calls = []
            for test in test_cases:
                test_input = test.get('input', test.get('inputs'))
                if isinstance(test_input, (list, tuple)):
                    calls.append(f"{main}(*{list(test_input)!r})")
                else:
                    calls.append(f"{main}({test_input!r})")
            load, values = run_calls(code, calls, timeout=self.timeout, mem_bytes=mem)
            results, test_results = [], []
            for i, (test, (kind, value)) in enumerate(zip(test_cases, values)):
                expected = test.get('expected_output', test.get('output'))
                ok = load is None and kind == "ok" and is_plain(value) and self._compare_outputs(value, expected)
                results.append(ok)
                entry = {'test_id': i, 'input': test.get('input', test.get('inputs')), 'expected': expected,
                         'passed': ok}
                if kind == "ok":
                    entry['actual'] = value
                else:
                    entry['error'] = value
                test_results.append(entry)

        if load is not None:
            return failed(load, test_results=test_results)
        return ExecutionResult(success=True, execution_time=time.time() - start,
                               test_passed=bool(results) and all(results), test_results=test_results)

    @staticmethod
    def _public_functions(code: str) -> List[str]:
        """Names of the public top-level functions, in definition order."""
        try:
            return [n.name for n in ast.parse(code).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith('_')]
        except SyntaxError:
            return []

    @staticmethod
    def _first_function(code: str) -> Optional[str]:
        """Name of the first public top-level function (the function under test)."""
        try:
            for node in ast.parse(code).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                    return node.name
        except SyntaxError:
            pass
        return None

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """
        Compare actual and expected outputs.
        
        Args:
            actual: Actual output from function
            expected: Expected output
            
        Returns:
            True if outputs match
        """
        # Handle numeric comparison with tolerance
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            import math
            return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-9)
        
        # Handle list/tuple comparison
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        # Direct comparison
        return actual == expected
    
    def batch_execute(
        self,
        codes: List[str],
        test_cases_list: Optional[List[List[Dict[str, Any]]]] = None
    ) -> List[ExecutionResult]:
        """
        Execute multiple code samples.
        
        Args:
            codes: List of code strings
            test_cases_list: List of test case lists (one per code)
            
        Returns:
            List of ExecutionResult objects
        """
        if test_cases_list is None:
            test_cases_list = [None] * len(codes)
        
        results = []
        for code, test_cases in zip(codes, test_cases_list):
            result = self.execute(code, test_cases)
            results.append(result)
        
        return results


# ============================================================================
# SIMPLE API FOR COMMON USE CASES
# ============================================================================

def execute_code_safely(
    code: str,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 5
) -> ExecutionResult:
    """
    Simple API to execute code safely.
    
    Args:
        code: Code to execute
        test_cases: Optional test cases
        timeout: Timeout in seconds
        
    Returns:
        ExecutionResult
        
    Example:
        >>> result = execute_code_safely(
        ...     code="def add(a, b): return a + b",
        ...     test_cases=[
        ...         {"input": [1, 2], "expected_output": 3},
        ...         {"input": [10, 20], "expected_output": 30}
        ...     ]
        ... )
        >>> print(result.test_passed)
        True
    """
    executor = SafeCodeExecutor(timeout=timeout)
    return executor.execute(code, test_cases)


def validate_code_syntax(code: str) -> Tuple[bool, str]:
    """
    Validate code syntax without executing.
    
    Args:
        code: Code to validate
        
    Returns:
        (is_valid, error_message)
        
    Example:
        >>> valid, error = validate_code_syntax("def foo(): return 42")
        >>> print(valid)
        True
    """
    executor = SafeCodeExecutor()
    return executor.validate_syntax(code)