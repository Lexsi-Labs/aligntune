"""SafeCodeExecutor's default mode may import a short list of harmless modules.

Restricted builtins used to include no ``__import__``, so a solution starting with
``from typing import List`` (most HumanEval prompts) or ``import math`` failed with
"Execution failed" even when correct.
"""

from aligntune.eval.safe_executor import SAFE_IMPORTS, SafeCodeExecutor

TYPED = "from typing import List\nimport math\n\ndef total(xs: List[float]) -> float:\n    return math.fsum(xs)\n"
TESTS = ["assert total([1.0, 2.0]) == 3.0"]
HUMANEVAL = "def check(candidate):\n    assert candidate([1.0, 2.0]) == 3.0\n"


def test_default_mode_runs_a_solution_that_imports_typing_and_math():
    ex = SafeCodeExecutor()
    assert ex.execute(TYPED, TESTS).test_passed
    assert ex.execute(TYPED, HUMANEVAL).test_passed


def test_a_wrong_solution_still_fails():
    wrong = TYPED.replace("math.fsum(xs)", "0.0")
    assert not SafeCodeExecutor().execute(wrong, TESTS).test_passed


def test_modules_outside_the_list_are_refused():
    ex = SafeCodeExecutor()
    for code in ("import os\ndef f():\n    return 1\n",
                 "import subprocess\ndef f():\n    return 1\n",
                 "def f():\n    import sys\n    return 1\n",
                 "def f():\n    return __import__('os')\n"):
        assert not ex.execute(code, [{"input": None, "expected_output": 1}]).test_passed, code
    assert "os" not in SAFE_IMPORTS and "sys" not in SAFE_IMPORTS and "subprocess" not in SAFE_IMPORTS


def test_unrestricted_mode_still_allows_any_import():
    ex = SafeCodeExecutor(restrict_builtins=False)
    assert ex.execute("import os\ndef f(x):\n    return os.sep\n", [{"input": 1, "expected_output": "/"}]).test_passed
