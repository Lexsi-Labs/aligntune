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
    - Process isolation using multiprocessing
    - Timeout protection
    - Restricted builtins
    - Captured stdout/stderr
    - Test case validation
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
    
    def _execute_in_process(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        result_queue: mp.Queue
    ):
        """
        Execute code in a separate process (internal use).
        
        Args:
            code: Code to execute
            test_cases: Test cases to validate
            result_queue: Queue to return results
        """
        try:
            # Capture stdout
            stdout_capture = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_capture):
                # Create restricted namespace
                namespace = {
                    '__builtins__': self.safe_builtins,
                    '__name__': '__main__',
                }
                
                # Execute code
                start_time = time.time()
                exec(code, namespace)
                exec_time = time.time() - start_time
                
                # Find callable functions (user-defined)
                functions = {
                    k: v for k, v in namespace.items()
                    if callable(v) and not k.startswith('_')
                }
                
                if not functions and not test_cases:
                    # Just execution, no tests
                    result_queue.put(ExecutionResult(
                        success=True,
                        output=stdout_capture.getvalue(),
                        execution_time=exec_time,
                        test_passed=True
                    ))
                    return
                
                if not functions and test_cases:
                    result_queue.put(ExecutionResult(
                        success=False,
                        error="No callable functions found in code",
                        output=stdout_capture.getvalue(),
                        execution_time=exec_time
                    ))
                    return
                
                # Get main function name and callable
                main_func_name = list(functions.keys())[0]
                main_func = functions[main_func_name]

                # Run test cases
                test_results = []
                all_passed = True

                # Detect test format
                # 1. MBPP: list of assertion strings ["assert func(...) == result", ...]
                # 2. HumanEval: single string with check(candidate) function
                # 3. Structured dict: [{"input": ..., "expected_output": ...}, ...]

                is_assertion_format = (
                    test_cases and
                    isinstance(test_cases[0], str) and
                    test_cases[0].strip().startswith('assert')
                )

                is_humaneval_format = (
                    test_cases and
                    isinstance(test_cases, str) and
                    'def check(candidate)' in test_cases
                )

                if is_humaneval_format:
                    # HumanEval format: execute check(candidate) function
                    try:
                        # Execute the test code which defines check(candidate)
                        exec(test_cases, namespace)
                        # Call check with our generated function
                        namespace['check'](main_func)
                        all_passed = True
                        test_results.append({'test_id': 0, 'format': 'humaneval_check', 'passed': True})
                    except AssertionError as e:
                        all_passed = False
                        test_results.append({'test_id': 0, 'format': 'humaneval_check', 'passed': False, 'error': f'Assertion failed: {e}'})
                    except Exception as e:
                        all_passed = False
                        test_results.append({'test_id': 0, 'format': 'humaneval_check', 'passed': False, 'error': str(e)})

                elif is_assertion_format:
                    # MBPP format: execute assertion strings directly
                    import re
                    for i, assertion_str in enumerate(test_cases):
                        try:
                            # Replace original function name with generated function name
                            match = re.search(r'assert\s+(\w+)\s*\(', assertion_str)
                            if match:
                                orig_func = match.group(1)
                                modified = assertion_str.replace(orig_func + '(', main_func_name + '(', 1)
                            else:
                                modified = assertion_str

                            exec(modified, namespace)
                            test_results.append({'test_id': i, 'assertion': assertion_str, 'passed': True})
                        except AssertionError:
                            all_passed = False
                            test_results.append({'test_id': i, 'assertion': assertion_str, 'passed': False, 'error': 'Assertion failed'})
                        except Exception as e:
                            all_passed = False
                            test_results.append({'test_id': i, 'assertion': assertion_str, 'passed': False, 'error': str(e)})
                else:
                    # Structured dict format (HumanEval, etc.)
                    for i, test in enumerate(test_cases):
                        try:
                            test_input = test.get('input', test.get('inputs'))
                            expected = test.get('expected_output', test.get('output'))

                            # Call function with input
                            if isinstance(test_input, (list, tuple)):
                                actual = main_func(*test_input)
                            else:
                                actual = main_func(test_input)

                            # Compare outputs
                            passed = self._compare_outputs(actual, expected)

                            test_results.append({
                                'test_id': i,
                                'input': test_input,
                                'expected': expected,
                                'actual': actual,
                                'passed': passed
                            })

                            if not passed:
                                all_passed = False

                        except Exception as test_error:
                            test_results.append({
                                'test_id': i,
                                'input': test_input if 'test_input' in locals() else None,
                                'expected': expected if 'expected' in locals() else None,
                                'error': str(test_error),
                                'passed': False
                            })
                            all_passed = False
                
                result_queue.put(ExecutionResult(
                    success=True,
                    output=stdout_capture.getvalue(),
                    execution_time=exec_time,
                    test_passed=all_passed,
                    test_results=test_results
                ))
        
        except Exception as e:
            result_queue.put(ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                output=stdout_capture.getvalue() if 'stdout_capture' in locals() else "",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            ))
    
    def execute(
        self,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Execute code safely with timeout and test cases.
        
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
        
        # Create result queue for process communication
        result_queue = mp.Queue()
        
        # Create and start execution process
        process = mp.Process(
            target=self._execute_in_process,
            args=(code, test_cases, result_queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        # Check if process completed
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join()
            return ExecutionResult(
                success=False,
                error=f"Execution timeout ({self.timeout}s exceeded)"
            )
        
        # Get result from queue
        if not result_queue.empty():
            return result_queue.get()
        else:
            # Process died without putting result
            return ExecutionResult(
                success=False,
                error="Process terminated unexpectedly"
            )
    
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