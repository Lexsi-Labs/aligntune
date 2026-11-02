"""
Safe code execution environment for evaluating generated code.

Runs code in a separate interpreter process (see ``aligntune.rewards._code_grader``)
with:
- Timeout protection
- Resource limits
- A minimal environment and an empty working directory
- Test case validation, compared here on plain values the child returns

This limits what a completion can do; it is not a sandbox. By default the code
also sees only a short list of builtins (``safe_builtins``), so ``import`` and
``open`` fail; pass ``restrict_builtins=False`` to allow everything.
"""

import ast
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from aligntune.rewards import _code_grader


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
    Runs generated code in a child process with limited builtins.

    What it limits: the code never runs in the calling process; it gets a
    timeout, CPU, file-size and (on Linux) memory limits, a minimal environment,
    an empty working directory, and by default only the builtins in
    ``safe_builtins`` (so no ``import`` and no ``open``).

    What it does not limit: this is NOT a sandbox, and the name is kept only for
    compatibility. The restricted builtins are a speed bump that determined code
    can get around. Even then the child can read the user's files, use the
    network and signal other processes. Run untrusted code in a container or VM.
    """
    
    def __init__(
        self,
        timeout: int = 5,
        max_memory_mb: int = 512,
        language: str = "python",
        restrict_builtins: bool = True,
    ):
        """
        Initialize safe executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            language: Programming language ("python", "javascript", etc.)
            restrict_builtins: Give the code only ``safe_builtins``; False
                gives it every builtin, imports included
        """
        self.restrict_builtins = restrict_builtins
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.language = language.lower()

        # Builtins the code can use by default (the list the in-process executor used)
        self.safe_builtins = {
            'int': int, 'float': float, 'str': str, 'bool': bool, 'list': list,
            'dict': dict, 'tuple': tuple, 'set': set,
            'len': len, 'sum': sum, 'max': max, 'min': min, 'abs': abs, 'round': round,
            'range': range, 'enumerate': enumerate, 'zip': zip, 'map': map,
            'filter': filter, 'sorted': sorted, 'reversed': reversed, 'any': any, 'all': all,
            'isinstance': isinstance, 'type': type, 'print': print,
            'pow': pow, 'divmod': divmod,
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
    
    def _allow(self) -> Optional[List[str]]:
        return list(self.safe_builtins) if self.restrict_builtins else None

    def _first_function(self, code: str) -> Optional[str]:
        """Name of the first public function or class the code defines at its top level."""
        try:
            body = ast.parse(code).body
        except (SyntaxError, ValueError):
            return None
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith('_'):
                return node.name
        return None

    def execute(
        self,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute code with a timeout and test cases.

        The code runs in a separate interpreter process with limited time, memory
        and environment (see ``aligntune.rewards._code_grader``); test results
        are compared here, on plain values the child returns.

        Args:
            code: Code to execute
            test_cases: Optional list of test cases

        Returns:
            ExecutionResult with execution details
        """
        if test_cases is None:
            test_cases = []

        # Extract code from markdown if needed
        code = self.extract_code(code)

        # Validate syntax first
        is_valid, error_msg = self.validate_syntax(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                error=error_msg
            )

        start = time.time()
        main_func_name = self._first_function(code)
        if test_cases and not main_func_name:
            return ExecutionResult(success=False, error="No callable functions found in code")

        # Detect test format
        # 1. MBPP: list of assertion strings ["assert func(...) == result", ...]
        # 2. HumanEval: single string with check(candidate) function
        # 3. Structured dict: [{"input": ..., "expected_output": ...}, ...]
        if isinstance(test_cases, str) and 'def check(candidate)' in test_cases:
            asserts, setup = _code_grader.humaneval_tests(test_cases, main_func_name)
            code = code + "\n\n" + setup if setup else code
            return self._run_assertions(code, list(asserts), None, start, humaneval=True)
        if test_cases and isinstance(test_cases[0], str) and test_cases[0].strip().startswith('assert'):
            renamed = []
            for assertion_str in test_cases:
                # Replace original function name with generated function name
                match = re.search(r'assert\s+(\w+)\s*\(', assertion_str)
                if match:
                    assertion_str = assertion_str.replace(match.group(1) + '(', main_func_name + '(', 1)
                renamed.append(assertion_str)
            return self._run_assertions(code, renamed, test_cases, start)
        if not test_cases:
            run = _code_grader.run_candidate(code, self.timeout, capture_stdout=True, memory_limit_mb=self.max_memory_mb, builtins_allow=self._allow())
            if run.status == "timeout":
                return ExecutionResult(success=False, error=f"Execution timeout ({self.timeout}s exceeded)")
            if run.status != "ok":
                return ExecutionResult(success=False, error="Execution failed")
            return ExecutionResult(success=True, output=run.stdout,
                                   execution_time=time.time() - start, test_passed=True)
        return self._run_structured(code, main_func_name, test_cases, start)

    def _run_assertions(self, code, asserts, originals, start, humaneval=False) -> ExecutionResult:
        status, passed = _code_grader.grade(code, asserts, self.timeout, memory_limit_mb=self.max_memory_mb, builtins_allow=self._allow())
        if status == "timeout":
            return ExecutionResult(success=False, error=f"Execution timeout ({self.timeout}s exceeded)")
        if status != "ok":
            return ExecutionResult(success=False, error="Execution failed")
        if humaneval:
            ok = all(passed)
            results = [{'test_id': 0, 'format': 'humaneval_check', 'passed': ok}]
            if not ok:
                results[0]['error'] = 'Assertion failed'
        else:
            results = []
            for i, (assertion_str, ok) in enumerate(zip(originals, passed)):
                result = {'test_id': i, 'assertion': assertion_str, 'passed': ok}
                if not ok:
                    result['error'] = 'Assertion failed'
                results.append(result)
        return ExecutionResult(success=True, execution_time=time.time() - start,
                               test_passed=all(passed), test_results=results)

    def _run_structured(self, code, main_func_name, test_cases, start) -> ExecutionResult:
        items = []
        for test in test_cases:
            test_input = test.get('input', test.get('inputs'))
            spread = "*" if isinstance(test_input, (list, tuple)) else ""
            items.append({"expr": f"{main_func_name}({spread}{test_input!r})"})
        run = _code_grader.run_candidate(code, self.timeout, items, memory_limit_mb=self.max_memory_mb, builtins_allow=self._allow())
        if run.status == "timeout":
            return ExecutionResult(success=False, error=f"Execution timeout ({self.timeout}s exceeded)")
        if run.status != "ok":
            return ExecutionResult(success=False, error="Execution failed")
        results = []
        for i, (test, actual) in enumerate(zip(test_cases, run.values)):
            test_input = test.get('input', test.get('inputs'))
            expected = test.get('expected_output', test.get('output'))
            if actual is _code_grader.FAILED:
                results.append({'test_id': i, 'input': test_input, 'expected': expected,
                                'error': 'call failed', 'passed': False})
                continue
            try:
                ok = bool(self._compare_outputs(actual, expected))
            except Exception:
                ok = False
            results.append({'test_id': i, 'input': test_input, 'expected': expected,
                            'actual': actual, 'passed': ok})
        return ExecutionResult(success=True, execution_time=time.time() - start,
                               test_passed=all(r['passed'] for r in results), test_results=results)

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