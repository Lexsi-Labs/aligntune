"""Code rewards pay for correct answers, not for faking a pass.

Every reward call runs in a forked child of the test, which is killed at a hard
limit, so a completion that ends or hangs the process it is graded in can't
take pytest with it.
"""

import math
import multiprocessing
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

from aligntune.rewards.registry import RewardRegistry

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs fork")

REPO = Path(__file__).resolve().parents[1]


def _child(fn, conn):
    try:
        conn.send(("ok", fn()))
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit
        conn.send(("raised", type(e).__name__))


def in_child(fn, limit=60.0) -> Any:
    """Run ``fn`` in a forked child; return its value, or a string saying how it went wrong."""
    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(fn, send))
    proc.start()
    send.close()
    try:
        if not recv.poll(limit):
            return "hung"
        kind, value = recv.recv()
    except EOFError:
        return f"process ended with status {proc.join(5) or proc.exitcode}"
    finally:
        proc.kill()
        proc.join()
    return value if kind == "ok" else f"raised {value}"


def reward(name, weight=1.0, **params):
    return RewardRegistry.get_reward_function(name, {"type": name, "weight": weight, "params": params})


def score(name, text, *args, params=None, limit=60.0, **kwargs):
    return in_child(lambda: reward(name, **(params or {})).compute(text, *args, **kwargs), limit)


def fence(code):
    return "```python\n" + code + "\n```"


ASSERTS = [
    "assert is_not_prime(2) == False",
    "assert is_not_prime(10) == True",
    "assert is_not_prime(35) == True",
]
DICTS = [{"input": 2, "expected_output": False}, {"input": 10, "expected_output": True},
         {"input": 35, "expected_output": True}]
GOOD = fence(
    "def is_not_prime(n):\n    if n < 2:\n        return True\n"
    "    return any(n % i == 0 for i in range(2, int(n**0.5) + 1))"
)
WRONG = fence("def is_not_prime(n):\n    return False")
FLOATS = [{"input": 3.5, "expected_output": 0.5}, {"input": 1.33, "expected_output": 0.33},
          {"input": 123.456, "expected_output": 0.456}]

PRINTED_PASS = fence('import sys\ndef f(*a, **k): pass\nprint("PASS"); sys.exit(0)')
EXIT_ZERO = fence("import sys\nsys.exit(0)")
HARD_EXIT = fence("import os\nos._exit(0)")
PASS_LINES = 'for i in range(10): print(f"PASS:{i}")'
ALWAYS_EQUAL = "class _Always:\n    def __eq__(self, other): return True\ndef is_not_prime(*a, **k): return _Always()"
INT_SUBCLASS = "class B(int):\n    def __eq__(self, other): return True\ndef is_not_prime(*a, **k): return B()"
LOOP = "while True: pass"

# --------------------------------------------------------------------------- #
# Every case from the report: these failed before the change
# --------------------------------------------------------------------------- #


def test_a_completion_cannot_change_later_grades():
    poison = "import math\nmath.isclose = lambda *a, **k: True\ndef f(*a, **k): return 0.0"
    wrong = "def f(*a, **k): return 0.0"

    def run():
        r = reward("code_correctness")
        scores = [r.compute(wrong, test_cases=FLOATS), r.compute(poison, test_cases=FLOATS),
                  r.compute(wrong, test_cases=FLOATS)]
        return scores, math.isclose(1, 2)

    assert in_child(run) == ([0.0, 0.0, 0.0], False)


@pytest.mark.parametrize("text", [EXIT_ZERO, HARD_EXIT], ids=["sys_exit", "os_exit"])
@pytest.mark.parametrize("name", ["code_execution", "code_correctness", "code_verifiable"])
def test_exiting_does_not_end_the_trainer_or_score(name, text):
    if name == "code_correctness":
        bare = text.strip("`\n").removeprefix("python\n")
        assert score(name, bare, test_cases=DICTS) == 0.0
    else:
        assert score(name, text) == 0.0  # no tests: it must run to its end


@pytest.mark.parametrize("name,text,kwargs", [
    ("code_correctness", LOOP, {"test_cases": FLOATS}),
    ("code_correctness", fence(LOOP), {"test_cases": FLOATS}),
    ("code_execution", fence(LOOP), {}),
    ("code_execution", fence(LOOP), {"test_cases": ASSERTS}),
    ("mbpp_reward", fence(LOOP), {"test_list": ASSERTS}),
])
def test_an_infinite_loop_is_stopped(name, text, kwargs):
    result = score(name, text, params={"timeout": 1.0}, limit=20, **kwargs)
    assert isinstance(result, float) and result <= 0.0


@pytest.fixture
def python_on_path(tmp_path, monkeypatch):
    """A bare ``python`` on PATH, as on a machine with a virtual environment active."""
    (tmp_path / "python").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


def test_printed_pass_and_exit_zero_scores_zero(python_on_path):
    assert score("code_execution", PRINTED_PASS, test_cases=ASSERTS) == 0.0
    assert score("code_execution", PRINTED_PASS, test_cases=DICTS) == 0.0


def test_printed_pass_lines_score_no_better_than_a_wrong_answer():
    wrong = score("mbpp_reward", WRONG, test_cases=ASSERTS)
    assert score("mbpp_reward", PASS_LINES, test_cases=ASSERTS) < wrong
    assert score("mbpp_reward", "print('PASS:0')\n" + WRONG, test_cases=ASSERTS) <= wrong


def test_exit_zero_alone_does_not_pass_code_verifiable():
    assert score("code_verifiable", EXIT_ZERO, ASSERTS) == 0.0
    assert score("code_verifiable", HARD_EXIT, ASSERTS) == 0.0


@pytest.mark.parametrize("text", [ALWAYS_EQUAL, INT_SUBCLASS], ids=["always_equal", "int_subclass"])
def test_custom_equality_does_not_pass(text, python_on_path):
    assert score("code_correctness", text, test_cases=DICTS) == 0.0
    assert score("code_execution", text, test_cases=DICTS) == 0.0
    assert score("code_execution", text, test_cases=ASSERTS) == 0.0
    assert score("code_verifiable", text, ASSERTS) == 0.0
    assert score("mbpp_reward", text, test_list=ASSERTS) == 0.0


def test_operator_overloads_do_not_pass_computed_asserts(python_on_path):
    code = fence("class A(float):\n    def __sub__(self, other): return 0.0\ndef f(x): return A(9)")
    tests = ["assert abs(f(1) - 0.5) < 1e-6"]
    assert score("code_execution", code, test_cases=tests) == 0.0


@pytest.mark.parametrize("text", ["", "x = 1", GOOD], ids=["empty", "compiles", "correct"])
def test_correctness_with_no_tests_scores_zero(text):
    assert score("code_correctness", text) == 0.0


def test_no_python_on_path_still_grades(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert score("code_execution", GOOD, test_cases=ASSERTS) == 1.0
    assert score("code_verifiable", GOOD, ASSERTS) == 1.0


def test_code_correctness_strips_the_fence():
    assert score("code_correctness", GOOD, test_cases=DICTS) == 1.0


def test_imports_above_the_first_def_are_kept():
    bare = "import math\ndef f(x):\n    return math.floor(x)"
    assert score("code_execution", bare, test_cases=["assert f(1.5) == 1"]) == 1.0
    prose = "Here is my answer.\nimport math\nfrom this we get:\ndef f(x):\n    return math.floor(x)"
    assert score("code_execution", prose, test_cases=["assert f(1.5) == 1"]) == 1.0


def test_no_test_path_works_on_a_worker_thread():
    def run():
        box = []
        t = threading.Thread(target=lambda: box.append(
            reward("code_execution").compute(fence("def one():\n    return 1"), test_cases=["assert one() == 1"])))
        t.start()
        t.join()
        return box

    assert in_child(run) == [1.0]


@pytest.mark.parametrize("name", ["code_correctness", "code_quality"])
def test_weight_is_not_applied_inside_compute(name):
    kwargs: Dict[str, Any] = {"test_cases": DICTS} if name == "code_correctness" else {}

    def run():
        return (reward(name, weight=1.0).compute(GOOD, **kwargs), reward(name, weight=0.3).compute(GOOD, **kwargs))

    full, weighted = in_child(run)
    assert full > 0 and weighted == full


def test_mbpp_reward_leaves_no_temporary_file(tmp_path):
    def run():
        tempfile.tempdir = str(tmp_path)
        reward("mbpp_reward").compute(GOOD, test_cases=ASSERTS)
        return sorted(os.listdir(tmp_path))

    assert in_child(run) == []


# --------------------------------------------------------------------------- #
# What the child can and cannot see
# --------------------------------------------------------------------------- #


def test_expected_value_never_reaches_the_child():
    # Would pass if the expected value were anywhere in the child; the number is
    # split so the candidate's own source doesn't contain it.
    code = (
        "import gc\n"
        "seen = any(str(739150000 + 4628) in str(o) for o in gc.get_objects()\n"
        "           if isinstance(o, (dict, list)) and o is not globals())\n"
        "def f(): return 739150000 + 4628 if seen else 0"
    )
    assert score("code_execution", code, test_cases=["assert f() == 739154628"]) == 0.0
    assert score("code_execution", code, test_cases=[{"input": None, "expected_output": 739154628}]) == 0.0


def test_trainer_environment_and_directory_do_not_reach_the_child(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "marker-value")
    code = "import os, sys\ndef f(): return [os.environ.get('HF_TOKEN'), os.listdir('.'), sys.argv]"
    assert score("code_execution", code, test_cases=["assert f() == [None, [], ['candidate']]"]) == 1.0


def test_grandchild_dies_with_the_timeout(tmp_path):
    mark = tmp_path / "alive"
    code = (
        "import os, time\nif os.fork() == 0:\n    time.sleep(3)\n"
        f"    open({str(mark)!r}, 'w').write('x')\n    os._exit(0)\nwhile True: pass"
    )
    assert score("code_execution", code, test_cases=ASSERTS, params={"timeout": 1.0}) == 0.0
    assert score("mbpp_reward", code, test_cases=ASSERTS, params={"timeout": 1.0}) == 0.0
    in_child(lambda: __import__("time").sleep(3.5))
    assert not mark.exists()


def _writes_result(payload):
    return (
        "import gc, io, os\n"
        "for o in gc.get_objects():\n"
        "    if isinstance(o, io.TextIOWrapper) and o.name.endswith('out.json'):\n"
        f"        o.write({payload!r}); o.close()\n"
        "os._exit(0)"
    )


@pytest.mark.parametrize("payload", [
    "[" * 200000 + "]" * 200000,
    "[" + '{"list":[' * 900 + "]}" * 900 + "]",
    "[]", "[1, 2]", '{"f": "1.0"}', '[{"nope": 1}]', '[{"dict": [[[], 1]]}]', '[1.5]', '[{"f": []}]',
    "not json",
])
def test_a_tampered_result_never_raises_into_the_trainer(payload):
    code = _writes_result(payload)
    assert score("code_execution", code, test_cases=["assert f(1) == 1"]) == 0.0
    assert score("code_correctness", code, test_cases=DICTS) == 0.0


def test_result_replaced_by_a_fifo_or_link_does_not_block(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("[1]")
    for swap in ("os.mkfifo(p)", f"os.symlink({str(secret)!r}, p)"):
        code = (
            "import gc, io, os\n"
            "p = [o.name for o in gc.get_objects() if isinstance(o, io.TextIOWrapper) and o.name.endswith('out.json')][0]\n"
            f"os.unlink(p); {swap}\nos._exit(0)"
        )
        assert score("code_execution", code, test_cases=["assert f() == 1"], limit=20) == 0.0


def test_oversized_result_fails(monkeypatch):
    monkeypatch.setattr("aligntune.rewards._code_grader.MAX_RESULT_BYTES", 1024)
    assert score("code_execution", "def f(): return 'x' * 100000", test_cases=["assert f() == 'x'"]) == 0.0


# --------------------------------------------------------------------------- #
# Correct answers still score 1.0
# --------------------------------------------------------------------------- #


def test_correct_answer_scores_one_everywhere():
    assert score("code_execution", GOOD, test_cases=ASSERTS) == 1.0
    assert score("code_execution", GOOD, test_cases=DICTS) == 1.0
    assert score("code_execution", GOOD) == 0.0  # no tests: nothing is paid
    assert score("code_correctness", GOOD, test_cases=DICTS) == 1.0
    assert score("code_correctness", GOOD, test_cases=ASSERTS) == 1.0
    assert score("code_verifiable", GOOD, ASSERTS) == 1.0
    assert score("code_verifiable", GOOD) == 0.0  # no tests: nothing is paid
    assert score("mbpp_reward", GOOD, test_list=ASSERTS) > 0.9


def test_wrong_answer_scores_what_it_earns():
    assert score("code_execution", WRONG, test_cases=ASSERTS) == pytest.approx(1 / 3)
    assert score("code_correctness", WRONG, test_cases=DICTS) == pytest.approx(1 / 3)
    assert score("code_verifiable", WRONG, ASSERTS) == 0.0


def test_code_verifiable_output_and_io_pairs():
    code = "def f(x): return x * 2\nprint(f(21))"
    assert score("code_verifiable", code, "42") == 1.0
    assert score("code_verifiable", code, "41") == 0.0
    assert score("code_verifiable", code + "\nimport os\nos._exit(0)", "42") == 0.0
    assert score("code_verifiable", code, [{"input": "f(2)", "expected_output": "4"}]) == 1.0
    assert score("code_verifiable", code, [{"input": "f(2)", "expected_output": "5"}]) == 0.0


@pytest.mark.parametrize("test", [
    "assert math.isclose(area(2), 4.0, rel_tol=0.001)",
    "assert set(pair(1)) == {1, 2}",
    "assert pair(1) == (1, 2)",
    "assert data() == {'a': [1, 2.5]}",
    "assert 1 in pair(1)",
    "assert not empty()",
    "assert area(2)",
    "assert empty() is None",
    "assert area(1) < area(2) <= 4.0",
    "assert pair(1) != (2, 1)",
])
def test_assert_shapes_grade_correct_answers(test):
    code = (
        "import math\n"
        "def area(r): return float(r * r)\n"
        "def pair(n): return (n, n + 1)\n"
        "def data(): return {'a': [1, 2.5]}\n"
        "def empty(): return None"
    )
    assert score("code_execution", code, test_cases=[test]) == 1.0


def test_assert_shape_it_cant_split_fails_with_a_warning(caplog):
    def run():
        import logging
        records = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[assignment]
        logging.getLogger("aligntune.rewards._code_grader").addHandler(handler)
        value = reward("code_execution").compute(GOOD, test_cases=["print('not an assert')"])
        return value, [r.getMessage() for r in records]

    value, messages = in_child(run)
    assert value == 0.0 and any("Can't grade" in m for m in messages)


# --------------------------------------------------------------------------- #
# Dataset columns: HumanEval and MBPP
# --------------------------------------------------------------------------- #

HE_PROMPT = (
    "from typing import List\n\n\n"
    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
    '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
    "    given threshold.\n"
    "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
    "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n"
    '    """\n'
)
HE_BODY = (
    "    for idx, elem in enumerate(numbers):\n"
    "        for idx2, elem2 in enumerate(numbers):\n"
    "            if idx != idx2:\n"
    "                distance = abs(elem - elem2)\n"
    "                if distance < threshold:\n"
    "                    return True\n\n"
    "    return False\n"
)
HE_TEST = (
    "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\n"
    "def check(candidate):\n"
    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
    "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
    "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
    "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
    "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n"
    "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
)
HE_ROW: Dict[str, Any] = {"test": HE_TEST, "entry_point": "has_close_elements", "prompt": HE_PROMPT}
HE_FORMS = {
    "fenced": "Here is the function.\n" + fence(HE_PROMPT + HE_BODY),
    "bare": HE_PROMPT + HE_BODY,
    "continuation": HE_BODY,
}
STUB = "def f(*a, **k): pass"


def _recipe_rewards(path):
    import yaml

    return yaml.safe_load((REPO / path).read_text())["config"]["rewards"]


@pytest.mark.parametrize("form", sorted(HE_FORMS))
def test_humaneval_columns_reach_the_rewards(form):
    for name in ("code_execution", "code_correctness"):
        assert score(name, HE_FORMS[form], **HE_ROW) == 1.0, name
    assert score("mbpp_reward", HE_FORMS[form], **HE_ROW) > 0.9


def test_humaneval_loop_asserts_fail_with_a_warning():
    from aligntune.rewards._code_grader import humaneval_tests

    test = "import math\ndef check(candidate):\n    assert True\n    for x in range(3):\n        assert candidate(x) == x\n"
    asserts, setup = humaneval_tests(test, "f")
    assert setup == "import math" and len(asserts) == 1 and asserts[0].startswith("for x")


def test_mbpp_test_list_reaches_the_rewards():
    assert score("code_correctness", GOOD, test_list=ASSERTS) == 1.0
    assert score("code_execution", GOOD, test_list=ASSERTS) == 1.0
    assert score("mbpp_reward", GOOD, test_list=ASSERTS) > 0.9


@pytest.mark.parametrize("path", [
    "src/aligntune/recipes/configs/grpo/deepseek_humaneval.yaml",
    "src/aligntune/recipes/configs/advanced/gbmpo_l2_code.yaml",
])
def test_humaneval_recipes_rank_correct_answers_above_the_stub(path):
    import yaml

    recipe = yaml.safe_load((REPO / path).read_text())
    assert recipe["recipe"]["dataset"] == "openai/openai_humaneval"
    assert recipe["config"]["datasets"][0]["name"] == "openai/openai_humaneval"

    rewards = recipe["config"]["rewards"]

    def total(text):
        # TRL applies each weight to the reward's unweighted score
        return sum(r["weight"] * reward(r["type"], **r.get("params", {})).compute(text, **HE_ROW) for r in rewards)

    scores = in_child(lambda: {form: total(text) for form, text in {**HE_FORMS, "stub": STUB}.items()})
    for form in HE_FORMS:
        assert scores[form] > scores["stub"], scores


def test_mbpp_example_wires_mbpp_reward():
    import re

    source = (REPO / "examples/grpo_code_gen_trl_mbpp.py").read_text()
    assert re.findall(r'"type":\s*"(\w+)"', source) == ["mbpp_reward"]


@pytest.mark.parametrize("text", ["(" * 200000, "a\x00b", "[" * 5000 + "]" * 5000])
def test_source_that_breaks_the_compiler_scores_zero(text):
    assert score("code_execution", text, test_cases=ASSERTS) == 0.0
    assert score("code_execution", text) == 0.0
    assert score("code_correctness", text, test_cases=DICTS) == 0.0
    assert score("mbpp_reward", text, test_list=ASSERTS) == 0.0


_COLLIDE = "M = 2**61 - 1\ncolliding = lambda n: [k * M for k in range(1, n + 1)]\n"


def _writes_built_result(builder):
    # Builds the JSON in the child and writes it straight into the result file,
    # so the child never has to build the set itself.
    return (
        "import gc, io, json, os\n" + _COLLIDE + f"payload = json.dumps({builder})\n"
        "for o in gc.get_objects():\n"
        "    if isinstance(o, io.TextIOWrapper) and o.name.endswith('out.json'):\n"
        "        o.write(payload); o.close()\n"
        "os._exit(0)"
    )


@pytest.mark.parametrize("builder", [
    '[{"set": colliding(200000)}]',
    '[{"list": [{"frozenset": colliding(45000)} for _ in range(2)]}]',
    '[{"tuple": [{"dict": [[k, 0] for k in colliding(45000)]}]}]',
    '[{"list": [{"set": colliding(60000)} for _ in range(3)]}]',
], ids=["flat_set", "nested_frozensets", "dict_keys", "over_the_value_cap"])
def test_a_result_that_would_hang_the_decoder_scores_zero_quickly(builder):
    import time

    code = _writes_built_result(builder)
    start = time.monotonic()
    assert score("code_execution", code, test_cases=["assert f(1) == 1"], limit=30) == 0.0
    assert time.monotonic() - start < 5


def test_result_value_cap_counts_nested_values():
    from aligntune.rewards import _code_grader as g

    assert g._decode({"list": list(range(1000))}) == list(range(1000))
    nested = {"list": [{"tuple": list(range(60000))}, {"list": list(range(60000))}]}
    with pytest.raises(ValueError):
        g._decode(nested)
    code = "def f(): return [list(range(60000)), list(range(60000))]"
    assert score("code_execution", code, test_cases=["assert f()"]) == 0.0


# --- counterfactual GRPO code tasks ---------------------------------------------------------------

CF_MODULE = "aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo"
CF_TESTS = ["assert f(2) == 4", "assert f(3) == 6"]
CF_CANONICAL = "def f(x):\n    return 2 * x"
CF_FORGERIES = {
    "printed_pass": "def f(x):\n    return 0\n" + "".join(f"print('PASS:{i}')\n" for i in range(5)),
    "exit_zero": "import sys\nprint('PASS:0')\nprint('PASS:1')\nsys.exit(0)\ndef f(x):\n    return 0",
    "always_equal": "class _A:\n    def __eq__(self, o): return True\n    __hash__ = None\ndef f(x): return _A()",
    "int_subclass": "class B(int):\n    def __eq__(self, o): return True\ndef f(x): return B(0)",
}


def cf_asserts(code):
    from importlib import import_module
    return import_module(CF_MODULE)._run_code_in_sandbox(code, "", CF_TESTS, 5)


def cf_evalplus(code):
    from importlib import import_module
    trainer = import_module(CF_MODULE).TRLCounterFactGRPOTrainer
    return trainer._run_evalplus_tests(None, code, CF_CANONICAL, "f", [2, 3, [4]], 5)


@pytest.mark.parametrize("run, total", [(cf_asserts, 2), (cf_evalplus, 3)])
@pytest.mark.parametrize("name", sorted(CF_FORGERIES))
def test_counterfact_forgeries_score_zero(run, total, name):
    result = in_child(lambda: run(CF_FORGERIES[name]))
    assert not isinstance(result, str), result
    assert result[1] == 0, result


@pytest.mark.parametrize("run, total", [(cf_asserts, 2), (cf_evalplus, 3)])
def test_counterfact_correct_solution_still_passes(run, total):
    ok, passed, errors = in_child(lambda: run("def f(x):\n    return x + x"))
    assert (ok, passed, errors) == (True, total, [])


def test_counterfact_wrong_and_hanging_solutions():
    wrong = in_child(lambda: cf_evalplus("def f(x):\n    return x + 1"))
    assert wrong[:2] == (True, 0)
    hang = in_child(lambda: cf_asserts("while True:\n    pass"))
    assert hang[:2] == (False, 0) and "timed out" in hang[2][0]
    hang = in_child(lambda: cf_evalplus("while True:\n    pass"))
    assert hang[:2] == (False, 0) and "timed out" in hang[2][0]


# --- correct answers whose results are not plain builtin data, or whose function shares a builtin's name ---------

# MBPP reference solutions and tests (sanitized split), and one HumanEval-style assert that wraps the call in abs().
CORRECT = {
    "mbpp_88_counter": ("import collections\ndef freq_count(list1):\n  freq_count= collections.Counter(list1)\n  return freq_count",
                        ["assert freq_count([10,10,10,10,20,20,20,20,40,40,50,50,30])==({10: 4, 20: 4, 40: 2, 50: 2, 30: 1})",
                         "assert freq_count([1,2,3,4,3,2,4,1,3,1,4])==({1:3, 2:2,3:3,4:3})"]),
    "mbpp_126_named_sum": ("def sum(a,b): \n    sum = 0\n    for i in range (1,min(a,b)): \n        if (a % i == 0 and b % i == 0): \n"
                           "            sum += i \n    return sum",
                           ["assert sum(10,15) == 6", "assert sum(100,150) == 93", "assert sum(4,6) == 3"]),
    "mbpp_590_complex": ("import cmath\ndef polar_rect(x,y):\n cn = complex(x,y)\n cn=cmath.polar(cn)\n cn1 = cmath.rect(2, cmath.pi)\n"
                         " return (cn,cn1)",
                         ["assert polar_rect(3,4)==((5.0, 0.9272952180016122), (-2+2.4492935982947064e-16j))",
                          "assert polar_rect(4,7)==((8.06225774829855, 1.0516502125483738), (-2+2.4492935982947064e-16j))"]),
    "mbpp_737_match": ("import re \nregex = '^[aeiouAEIOU][A-Za-z0-9_]*'\ndef check_str(string): \n\treturn re.search(regex, string)",
                       ['assert check_str("annie")', 'assert not check_str("dawood")', 'assert check_str("Else")']),
    "mbpp_2_set_wrapper": ("def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ",
                           ["assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))"]),
    "mbpp_77_space_before_call": ("def is_Diff(n): \n    return (n % 11 == 0) ",
                                  ["assert is_Diff (12345) == False", "assert is_Diff(1212112) == True"]),
    "abs_wrapper": ("def truncate_number(number: float) -> float:\n    return number % 1.0",
                    ["assert abs(truncate_number(3.5) - 0.5) < 1e-6"]),
}


@pytest.mark.parametrize("case", sorted(CORRECT))
def test_correct_answers_score_full_marks(case):
    code, tests = CORRECT[case]
    assert score("code_execution", fence(code), test_cases=tests) == 1.0
    assert score("code_correctness", fence(code), test_cases=tests) == 1.0
    assert score("code_verifiable", fence(code), tests) == 1.0
    assert score("mbpp_reward", code, test_list=tests) == 1.0  # bare, so no format penalty applies
    assert score("mbpp_reward", fence(code), test_list=tests) >= 0.98  # the fence costs 0.02 (a short answer earns some back)


def test_non_data_results_cross_as_their_truth_value_only():
    from aligntune.rewards import _code_grader as g

    code = ("import re, collections\n"
            "class C(collections.Counter):\n    def __eq__(self, o): return True\n"
            "def m(s): return re.search('a', s)\n"
            "def c(): return C(a=1)\n"
            "def od(): return collections.OrderedDict([(1, 2)])\n"
            "def dd(): return collections.defaultdict(int, {1: 2})\n")
    status, passed = g.grade(code, ["assert m('a')", "assert not m('b')", "assert m('a') == True", "assert c() == {'z': 9}",
                                    "assert c()", "assert od() == {1: 2}", "assert dd() == {1: 2}"])
    # an re.Match is truthy and equals nothing; a Counter with its own __eq__ is not a Counter, so it can't pass ==
    assert (status, passed) == ("ok", [True, True, False, False, True, True, True])
    run = g.run_candidate(code + "def k(): return collections.Counter('aab')\n", 10,
                          [{"expr": "k()"}, {"expr": "od()"}, {"expr": "1+2j"}, {"expr": "m('b')"}])
    import collections

    assert type(run.values[0]) is collections.Counter and run.values[0] == {"a": 2, "b": 1}
    assert type(run.values[1]) is collections.OrderedDict and run.values[2] == 1 + 2j and run.values[3] is None
