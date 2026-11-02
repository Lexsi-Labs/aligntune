import os

import pytest

from aligntune.eval.safe_executor import SafeCodeExecutor, execute_code_safely

ADD = "def add(a, b):\n    return a + b\n"


def test_structured_tests():
    r = execute_code_safely(ADD, [{"input": [1, 2], "expected_output": 3}, {"input": [1, 2], "expected_output": 4}])
    assert r.success and not r.test_passed
    assert [t["passed"] for t in r.test_results] == [True, False]


def test_assertion_tests_rename_function():
    r = execute_code_safely(ADD, ["assert plus(1, 2) == 3"])
    assert r.success and r.test_passed


def test_humaneval_check():
    test = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    assert execute_code_safely(ADD, test).test_passed
    assert not execute_code_safely("def add(a, b):\n    return 0\n", test).test_passed


def test_no_tests_captures_output():
    r = execute_code_safely("print('hi')")
    assert r.success and r.output == "hi\n"


def test_timeout():
    r = SafeCodeExecutor(timeout=1).execute("while True:\n    pass\n")
    assert not r.success and "timeout" in r.error


def test_no_function_with_tests():
    assert not execute_code_safely("x = 1", [{"input": 1, "expected_output": 1}]).success


def test_child_does_not_inherit_the_trainers_environment(monkeypatch):
    monkeypatch.setenv("ALIGNTUNE_TEST_MARKER", "secret")
    r = SafeCodeExecutor(restrict_builtins=False).execute("import os\nprint(os.environ.get('ALIGNTUNE_TEST_MARKER', 'absent'))")
    assert r.success and r.output == "absent\n"


def test_code_runs_in_another_process():
    r = SafeCodeExecutor(restrict_builtins=False).execute("import os\nprint(os.getpid())")
    assert r.success and int(r.output) != os.getpid()


def test_import_and_open_are_refused_by_default():
    assert not execute_code_safely("import os").success
    assert not execute_code_safely("open('/etc/hosts')").success


def test_restrictions_can_be_relaxed():
    relaxed = SafeCodeExecutor(restrict_builtins=False)
    assert relaxed.execute("import os").success
    assert relaxed.execute("open('/etc/hosts').close()").success


def test_restricted_tests_still_work():
    assert execute_code_safely(ADD, ["assert add(1, 2) == 3"]).test_passed


EARLY_EXIT = "import sys\nsys.exit(0)\ndef add(a, b):\n    return a + b\n"


@pytest.mark.parametrize("relaxed", [False, True])
def test_early_exit_does_not_pass_tests(relaxed):
    r = SafeCodeExecutor(restrict_builtins=not relaxed).execute(EARLY_EXIT, ["assert add(1, 2) == 3"])
    assert not r.test_passed
    assert not any(t["passed"] for t in r.test_results)


def test_baseline_validator_needs_the_asserts_to_hold():
    import json

    from aligntune.scripts.precompute_baseline import validate_mbpp_with_execution

    gold = json.dumps(["assert add(1, 2) == 3", "assert add(2, 2) == 4"])
    assert validate_mbpp_with_execution(ADD, gold) == 1.0
    assert validate_mbpp_with_execution("def add(a, b):\n    return 0\n", gold) == 0.0
    assert validate_mbpp_with_execution(EARLY_EXIT, gold) == 0.0
    assert validate_mbpp_with_execution("print('ok')", gold) == 0.0
