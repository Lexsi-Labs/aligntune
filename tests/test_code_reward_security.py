"""Security regression tests for the code rewards (model code runs out of process, graded values out).

Adapted from the regression suite in the private report of 2 Nov 2026 (Ebin Babu Thomas, MIT).
Every "solves nothing" answer must score 0 when tests are given; every reference solution must pass.
"""
import json
import pathlib
import time

import pytest

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "code_rewards"
TASKS = json.loads((FIXTURES / "tasks.json").read_text())
EX = json.loads((FIXTURES / "exploits.json").read_text())
NOTHING = [e for e in EX["exploits"] if e["solves_nothing"]]

from aligntune.rewards.registry import RewardRegistry  # noqa: E402


def build(name):
    return RewardRegistry.get_reward_function(name, {"type": name, "weight": 1.0, "params": {}})


def render(e, t, fenced, key="template"):
    body = (e[key] if isinstance(e, dict) else e).replace("{fn}", t["entry_point"])
    return "```python\n" + body + "\n```" if fenced and (not isinstance(e, dict) or e.get("fenceable", True)) else body


CALLS = {
    "code_execution": lambda r, t, x: r.compute(x, None, test_cases=t["assert_tests"]),
    "code_correctness": lambda r, t, x: r.compute(x, None, test_cases=t["dict_tests"]),
    "code_correctness[asserts]": lambda r, t, x: r.compute(x, None, test_cases=t["assert_tests"]),
    "mbpp_reward": lambda r, t, x: r.compute(x, None, test_cases=t["assert_tests"]),
    "code_verifiable": lambda r, t, x: r.compute(x, t["assert_tests"]),
}


@pytest.mark.parametrize("fenced", [True, False])
@pytest.mark.parametrize("ex", NOTHING, ids=[e["id"] for e in NOTHING])
@pytest.mark.parametrize("reward", list(CALLS))
def test_gate1_exploits_fail_with_tests(reward, ex, fenced):
    r = build(reward.split("[")[0])
    for t in TASKS:
        s = CALLS[reward](r, t, render(ex, t, fenced))
        assert s <= 0.0, (reward, ex["id"], t["task_id"], s)


@pytest.mark.parametrize("fenced", [True, False])
@pytest.mark.parametrize("reward", list(CALLS))
def test_reference_solutions_pass(reward, fenced):
    r = build(reward.split("[")[0])
    for t in TASKS:
        text = t["reference_solution"]
        s = CALLS[reward](r, t, "```python\n" + text + "\n```" if fenced else text)
        assert s >= 0.95, (reward, t["task_id"], s)


def test_code_correctness_without_tests_pays_nothing():
    r = build("code_correctness")
    assert r.compute("") == 0.0
    assert r.compute("def f(*a, **k):\n    pass") == 0.0


def test_code_execution_without_tests_pays_nothing():
    """With no tests running is not evidence of a correct answer (a stub runs), so nothing is paid."""
    r = build("code_execution")
    assert r.compute("```python\ndef f(*a, **k):\n    pass\n```") == 0.0


@pytest.mark.parametrize("tail", ["import sys\nsys.exit(0)", "raise SystemExit(0)", "import os\nos._exit(0)"])
def test_exit_does_not_escape_or_pay(tail):
    r = build("code_execution")
    code = "```python\ndef f(*a, **k):\n    pass\n" + tail + "\n```"
    assert r.compute(code) == 0.0  # no tests: exiting is a failed run, and the caller survives
    t = TASKS[0]
    assert r.compute(render("def {fn}(*a, **k):\n    pass\n" + tail, t, True), None, test_cases=t["assert_tests"]) == 0.0


def test_cross_completion_patch_does_not_carry_over():
    t = next(x for x in TASKS if x["task_id"] == "HumanEval/2")
    r = build("code_correctness")
    a = "import math\nmath.isclose = lambda *a, **k: True\ndef truncate_number(x):\n    return 0.0\n"
    b = "def truncate_number(x):\n    return 0.0\n"
    assert r.compute(b, None, test_cases=t["dict_tests"]) == 0.0
    r.compute(a, None, test_cases=t["dict_tests"])
    assert r.compute(b, None, test_cases=t["dict_tests"]) == 0.0
    import math
    assert math.isclose(1.0, 2.0) is False  # the trainer process was never touched


def test_infinite_loop_returns_zero_within_timeout():
    t = TASKS[0]
    for name in ("code_execution", "code_correctness", "mbpp_reward"):
        r = build(name)
        t0 = time.time()
        s = CALLS[name](r, t, "```python\ndef is_not_prime(n):\n    while True:\n        pass\n```")
        assert s <= 0.0 and time.time() - t0 < 30, (name, s)


def test_tests_reach_rewards_through_trl_bridge_for_humaneval_and_mbpp():
    from aligntune.core.rl.reward_handler import prepare_trl_rewards
    he = next(x for x in TASKS if x["task_id"] == "HumanEval/0")["raw_columns"]
    prepared = prepare_trl_rewards([{"type": "code_correctness", "weight": 0.5, "params": {}},
                                    {"type": "code_execution", "weight": 0.2, "params": {}}])
    comps = [he["canonical_solution"], "```python\ndef has_close_elements(*a, **k):\n    pass\n```"]
    cols = {"test": [he["test"]] * 2, "entry_point": [he["entry_point"]] * 2, "task_id": [he["task_id"]] * 2}
    for f in prepared.functions:
        scores = f(prompts=[he["prompt"]] * 2, completions=comps, completion_ids=None, **cols)
        assert scores[0] == 1.0 and scores[1] == 0.0, (f.__name__, scores)
    mb = next(x for x in TASKS if x["task_id"] == "MBPP/3")
    p = prepare_trl_rewards([{"type": "mbpp_reward", "weight": 1.0, "params": {}}]).functions[0]
    s = p(prompts=["p", "p"], completions=["```python\n" + mb["reference_solution"] + "\n```", "I cannot."],
          completion_ids=None, test_list=[mb["assert_tests"]] * 2)
    assert s[0] >= 0.95 and s[1] < 0


# --- Counterfactual GRPO harnesses (same values-out sandbox) -------------------------------

def _cf():
    pytest.importorskip("torch")
    from aligntune.backends.trl.rl.counterfact_grpo import counterfact_grpo as cf
    return cf


@pytest.mark.parametrize("ex", NOTHING, ids=[e["id"] for e in NOTHING])
def test_counterfact_assert_harness_rejects_exploits(ex):
    cf = _cf()
    for t in TASKS:
        ok, passed, _ = cf._run_code_in_sandbox(render(ex, t, False), "", t["assert_tests"])
        assert passed == 0, (ex["id"], t["task_id"], passed)


def test_counterfact_assert_harness_passes_reference():
    cf = _cf()
    for t in TASKS:
        ok, passed, _ = cf._run_code_in_sandbox(t["reference_solution"], "", t["assert_tests"])
        assert ok and passed == len(t["assert_tests"]), (t["task_id"], passed)


def test_counterfact_evalplus_compares_values_not_stdout():
    cf = _cf()
    canonical = "def add(a, b):\n    return a + b\n"
    inputs = [[1, 2], [3, 4], [0, 0]]
    run = cf.TRLCounterFactGRPOTrainer._run_evalplus_tests  # does not use self
    assert run(None, canonical, canonical, "add", inputs) == (True, 3, [])
    fake = "def add(a, b):\n    pass\nfor i in range(9):\n    print(f'PASS:{i}')\n"
    assert run(None, fake, canonical, "add", inputs)[1] == 0
    eq = "class _E:\n    def __eq__(self, o): return True\ndef add(a, b):\n    return _E()\n"
    assert run(None, eq, canonical, "add", inputs)[1] == 0
    assert run(None, canonical + "import os\nos._exit(0)\n", canonical, "add", inputs)[1] == 0


# --- sql_verifiable ---------------------------------------------------------------------------

def _sql(tmp_path, execute=True):
    pytest.importorskip("sqlglot")
    import sqlite3
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("create table t (x int)")
    con.executemany("insert into t values (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    r = RewardRegistry.get_reward_function(
        "sql_verifiable", {"type": "sql_verifiable", "weight": 1.0,
                           "params": {"execute": execute, "db_path": str(db)}})
    return r, db


def test_sql_compares_results(tmp_path):
    r, _ = _sql(tmp_path)
    ref = "select x from t where x > 1"
    assert r.compute("select x from t where x >= 2", ref) == 1.0
    assert r.compute("select x from t", ref) == 0.0          # parses and runs, wrong rows
    assert r.compute("select x from t where x > 1", "[[2], [3]]") == 1.0


def test_sql_cannot_modify_database(tmp_path):
    import sqlite3
    r, db = _sql(tmp_path)
    assert r.compute("drop table t", "select 1") == 0.0
    assert r.compute("delete from t", "select 1") == 0.0
    assert sqlite3.connect(db).execute("select count(*) from t").fetchone()[0] == 3


def test_sql_reference_without_execute_pays_nothing(tmp_path):
    r, _ = _sql(tmp_path, execute=False)
    assert r.compute("select 42", "select x from t") == 0.0


# --- code_verifiable reads the row's tests (it paid 1.0 to a stub through TRL) ----------------

def test_code_verifiable_reads_row_tests_through_trl_bridge():
    from aligntune.core.rl.reward_handler import prepare_trl_rewards
    f = prepare_trl_rewards([{"type": "code_verifiable", "weight": 1.0, "params": {}}]).functions[0]
    mb = next(x for x in TASKS if x["task_id"] == "MBPP/3")
    stub = "```python\ndef is_not_prime(*a, **k):\n    pass\n```"
    s = f(prompts=["p"] * 3, completions=["```python\n" + mb["reference_solution"] + "\n```", stub, "x = 1"],
          completion_ids=None, test_list=[mb["assert_tests"]] * 3)
    assert s == [1.0, 0.0, 0.0], s
    he = next(x for x in TASKS if x["task_id"] == "HumanEval/0")["raw_columns"]
    s = f(prompts=[he["prompt"]] * 3, completions=[he["canonical_solution"], "def has_close_elements(*a):\n    pass", "x = 1"],
          completion_ids=None, test=[he["test"]] * 3, entry_point=[he["entry_point"]] * 3)
    assert s == [1.0, 0.0, 0.0], s


# --- HumanEval check(candidate) with loops, locals, multi-line asserts and prompt helpers ------

HE_PROMPT = '''import math


def poly(xs: list, x: float):
    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))


def find_root(lo: float, hi: float):
    """Root of x**3 - 2 in [lo, hi]."""
'''
HE_TEST = '''
METADATA = {}


def check(candidate):
    import random
    rng = random.Random(0)
    for _ in range(20):
        lo = rng.uniform(-5, 0)
        hi = rng.uniform(2, 6)
        r = candidate(lo, hi)
        assert math.fabs(poly([-2, 0, 0, 1], r)) < 1e-4
    assert candidate(0, 2) == pytest_approx(
        2 ** (1 / 3)
    ) or abs(candidate(0, 2) - 2 ** (1 / 3)) < 1e-6
    assert True
'''.replace("pytest_approx", "float")
HE_SOLUTION = '''    for _ in range(200):
        mid = (lo + hi) / 2
        if poly([-2, 0, 0, 1], mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
'''


@pytest.mark.parametrize("reward", ["code_correctness", "code_execution", "code_verifiable"])
def test_humaneval_loops_and_helpers_graded_values_out(reward):
    from aligntune.core.rl.reward_handler import prepare_trl_rewards
    f = prepare_trl_rewards([{"type": reward, "weight": 1.0, "params": {}}]).functions[0]
    comps = [
        HE_SOLUTION,                                                     # body-only continuation
        "```python\n" + HE_PROMPT + HE_SOLUTION + "```",                 # full, fenced
        "def find_root(lo, hi):\n    return 0.0",                        # wrong
        # helper override: redefining poly must not help (the trusted poly comes from the prompt)
        "def poly(xs, x):\n    return 0.0\ndef find_root(lo, hi):\n    return 0.0",
        "class _A:\n    def __eq__(s, o): return True\n    def __lt__(s, o): return True\n"
        "    def __sub__(s, o): return s\n    def __abs__(s): return s\ndef find_root(lo, hi):\n    return _A()",
    ]
    tpl = "<|im_start|>user\nComplete this:\n" + HE_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
    for prompt in (HE_PROMPT, tpl):
        s = f(prompts=[prompt] * len(comps), completions=comps, completion_ids=None,
              test=[HE_TEST] * len(comps), entry_point=["find_root"] * len(comps))
        # A body-only answer can only be completed from the raw prompt; under a chat template the
        # full answer is still graded (helpers are found inside the template) and nothing else passes.
        assert (s[0] == 1.0 or prompt is tpl) and s[1] == 1.0 and max(s[2:]) == 0.0, (reward, s)


# --- Kernel confinement of the child (skipped where the kernel offers none) -------------------

def _isolated():
    from aligntune.rewards import _code_sandbox as sb
    load, _ = sb.run_calls("import os, signal\nos.kill(os.getpid(), 0)", [], timeout=10)
    return load is not None  # signalling is refused only under the seccomp filter


@pytest.mark.parametrize("probe", [
    "import os, signal\nos.kill(os.getppid(), signal.SIGKILL)",
    "import os\nos.killpg(os.getpgid(os.getppid()), 9)",
    "import socket\nsocket.socket()",
    "import subprocess\nsubprocess.run(['true'])",
])
def test_child_cannot_signal_trainer_or_open_sockets(probe):
    if not _isolated():
        pytest.skip("seccomp not available here")
    from aligntune.rewards import _code_sandbox as sb
    load, _ = sb.run_calls(probe, [], timeout=10)
    assert load is not None  # refused, and this test process is still alive


def test_child_cannot_write_outside_its_temp_dir(tmp_path):
    from aligntune.rewards import _code_sandbox as sb
    target = tmp_path / "written_by_candidate"
    load, _ = sb.run_calls(f"open({str(target)!r}, 'w').write('x')", [], timeout=10)
    if load is None:
        target.unlink()
        pytest.skip("Landlock not available here")
    assert not target.exists()
    assert sb.run_calls("open('ok.txt', 'w').write('x')\nassert open('ok.txt').read() == 'x'", [])[0] is None


# --- Evaluation executor and code_verifiable's other paths use the same sandbox --------------

def test_safe_code_executor_grades_values_out():
    from aligntune.eval.safe_executor import SafeCodeExecutor
    ex = SafeCodeExecutor()
    mb = next(x for x in TASKS if x["task_id"] == "MBPP/3")
    assert ex.execute(mb["reference_solution"], mb["assert_tests"]).test_passed
    for e in NOTHING:
        r = ex.execute(render(e, mb, False), mb["assert_tests"])
        assert not r.test_passed, e["id"]
    he = next(x for x in TASKS if x["task_id"] == "HumanEval/0")["raw_columns"]
    assert ex.execute(he["prompt"] + he["canonical_solution"], he["test"]).test_passed
    assert not ex.execute("def has_close_elements(*a):\n    pass", he["test"]).test_passed
    assert ex.execute("def add(a, b):\n    return a + b", [{"input": [1, 2], "expected_output": 3}]).test_passed
    assert "timeout" in ex.execute("while True:\n    pass").error


def test_code_verifiable_stdout_and_io_pairs():
    r = build("code_verifiable")
    assert r.compute("```python\nprint(6 * 7)\n```", "42") == 1.0
    assert r.compute("```python\nprint(42)\nimport sys\nsys.exit(1)\n```", "42") == 0.0
    assert r.compute("```python\nwhile True: pass\n```", "42") == 0.0
    add = "```python\ndef add(a, b):\n    return a + b\n```"
    assert r.compute(add, [{"input": "add(1, 2)", "expected_output": "3"}]) == 1.0      # call expression
    assert r.compute(add, [{"input": [1, 2], "expected_output": 3}]) == 1.0           # argument values
    liar = "```python\ndef add(a, b):\n    return '3'\n```"
    assert r.compute(liar, [{"input": [1, 2], "expected_output": 3}]) == 0.0


# --- found by scoring real model completions -------------------------------------------------

def test_numpy_return_values_are_graded_as_plain_data():
    pytest.importorskip("numpy")
    from aligntune.rewards import _code_sandbox as sb
    assert sb.run_calls("import numpy", [])[0] is None  # numpy must import inside the sandbox
    r = build("code_correctness")
    tests = ["assert diff([1, 2], [1, 5]) == [0, 3]", "assert total([1, 2]) == 3"]
    ok = ("import numpy as np\ndef diff(a, b):\n    return list(np.abs(np.array(a) - np.array(b)))\n"
          "def total(a):\n    return np.sum(a)\n")
    assert r.compute(ok, None, test_cases=tests) == 1.0
    wrong = "import numpy as np\ndef diff(a, b):\n    return list(np.array(a))\ndef total(a):\n    return np.int64(0)\n"
    assert r.compute(wrong, None, test_cases=tests) == 0.0


def test_safe_code_executor_finds_the_solution_after_a_helper():
    from aligntune.eval.safe_executor import SafeCodeExecutor
    he = next(x for x in TASKS if x["task_id"] == "HumanEval/0")["raw_columns"]
    helper_first = ("def _unused():\n    pass\n\ndef dist(a, b):\n    return abs(a - b)\n\n"
                    + he["prompt"] + he["canonical_solution"])
    assert SafeCodeExecutor().execute(helper_first, he["test"]).test_passed
    stubs = "def helper(*a):\n    return True\n\ndef has_close_elements(*a):\n    return True\n"
    assert not SafeCodeExecutor().execute(stubs, he["test"]).test_passed
