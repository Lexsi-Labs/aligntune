"""Run model-written code in a child process and grade it by the values it returns.

The candidate runs in a fresh interpreter (``sys.executable -I``) with an empty
temporary working directory, a minimal environment (``PATH``, the locale, and
``HOME``/``TMPDIR`` pointing at that directory), no standard input, and its own
process group, which is killed when the call ends or times out. Where the
platform has ``resource``, the child also gets a CPU-time limit, a file-size
limit, no core dumps and, on Linux, an address-space (memory) limit. macOS
refuses a memory limit, so there the wall clock is the only bound on memory.

The child sends back plain data only: exact ``None``, ``bool``, ``int``,
``float``, ``str``, ``list``, ``tuple``, ``set``, ``frozenset`` and ``dict``.
No subclass and no object with its own ``__eq__`` crosses. The trainer then does
the comparison, so printed text, the exit code and a custom ``__eq__`` can't earn
a pass, and a literal expected value in a test never reaches the child.

This limits what a completion can score and keeps it out of the trainer's
process. It isn't a sandbox: the child runs as the same user, so it can read
any file that user can read, use the network, and signal other processes,
the trainer included. Everything in the child belongs to the candidate, so a
determined one can write values of its own choosing into the result. That only
scores where the value it needs is known without the expected output, such as
a test that wants ``True``. Use ``input``/``expected_output`` tests or asserts
with a literal on one side where that matters, and a container or separate user
for real isolation.
"""

import ast
import functools
import json
import logging
import math
import operator
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MAX_RESULT_BYTES = 16 * 1024 * 1024
DEFAULT_MEMORY_LIMIT_MB = 4096
_ENV_NAMES = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")

# Runs in the child. Everything it needs after the candidate has run is bound to
# a local name first. A test expression is evaluated with a copy of the builtins
# taken before the candidate ran, and every name it takes from the candidate is
# checked: a callable's return value and a variable must be plain data, or the
# test fails. The candidate can still change anything in this process; all that
# gets it is a value of its own choosing.
_HARNESS = r"""
def _main():
    import builtins, json, os, sys, types
    try:
        import resource
        for name, value in json.loads(sys.argv[3]):
            limit = getattr(resource, name, None)
            if limit is None:
                continue
            hard = resource.getrlimit(limit)[1]
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass  # macOS refuses RLIMIT_AS
    except ImportError:
        pass
    with open(sys.argv[1]) as f:
        job = json.load(f)
    os.unlink(sys.argv[1])
    out = open(sys.argv[2], "w")
    sys.argv = ["candidate"]

    _type, _callable, _isinstance, _eval, _exception = type, callable, isinstance, eval, Exception
    _list, _tuple, _set, _frozenset, _dict, _bool, _int, _str, _float = list, tuple, set, frozenset, dict, bool, int, str, float
    _repr, _module, _function, _getattr, _len = repr, types.ModuleType, types.FunctionType, getattr, len
    _dump, _close, _exit, _flush = json.dump, out.close, os._exit, sys.stdout.flush
    _clean_builtins = _dict(builtins.__dict__)

    def enc(x):
        t = _type(x)
        if x is None or t is _bool or t is _int or t is _str:
            return x
        if t is _float:
            return {"f": _repr(x)}
        if t is _list or t is _tuple or t is _set or t is _frozenset:
            return {t.__name__: [enc(i) for i in x]}
        if t is _dict:
            return {"dict": [[enc(k), enc(v)] for k, v in x.items()]}
        raise TypeError("not plain data")

    def plain(x):
        enc(x)
        return x

    def guard(v):
        if _isinstance(v, _module):
            return v
        if _callable(v):
            return lambda *a, **k: plain(v(*a, **k))
        return plain(v)

    ns = {"__name__": "__main__", "__builtins__": builtins}
    exec(compile(job["code"], "<candidate>", "exec"), ns)

    names = {"__builtins__": _clean_builtins}
    for k, v in _list(ns.items()):
        if not k.startswith("__"):
            try:
                names[k] = guard(v)
            except _exception:
                pass
    defined = [v for k, v in ns.items() if not k.startswith("_") and _type(v) is _function
               and _getattr(v, "__module__", None) == "__main__"]
    public = [v for k, v in ns.items() if not k.startswith("_") and _callable(v) and not _isinstance(v, _module)]
    target = (defined or public or [None])[0]

    values = []
    for item in job["items"]:
        try:
            if "expr" in item:
                value = _eval(item["expr"], names)
            else:
                arg = item["call"]
                f = guard(target)
                if (_type(arg) is _list or _type(arg) is _tuple) and _len(arg) > 1:
                    value = f(*arg)
                elif arg is None and item.get("no_args"):
                    value = f()
                else:
                    value = f(arg)
            values.append(enc(value))
        except _exception:
            values.append({"e": 0})
    _dump(values, out)
    _close()
    try:
        _flush()
    except _exception:
        pass
    _exit(0)

_main()
"""


class _Failed:
    """A test expression that raised or returned something that isn't plain data."""

    def __repr__(self) -> str:
        return "<failed>"


FAILED = _Failed()
_CONTAINERS = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}


MAX_RESULT_ELEMENTS = 100_000
_MAX_SAME_HASH = 64


def _hashed(items: List[Any]) -> List[Any]:
    """Refuse hash collisions before building a set or dict from the child's data.

    Values that all hash alike (multiples of ``2**61 - 1`` all hash to 0) make
    building a set or dict quadratic, which would hang the trainer. Counting the
    hash values is linear, since distinct hash values hash apart.
    """
    if items and max(Counter(map(hash, items)).values()) > _MAX_SAME_HASH:
        raise ValueError("too many colliding hashes")
    return items


def _decode(value: Any, budget: Optional[List[int]] = None) -> Any:
    """Rebuild a value the child encoded. Raises on anything unexpected, on more
    than :data:`MAX_RESULT_ELEMENTS` values in all, nested ones included, and on
    colliding hashes in a set or dict."""
    if budget is None:
        budget = [MAX_RESULT_ELEMENTS]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("too many values in the result")
    if isinstance(value, dict):
        (kind, payload), = value.items()
        if kind == "e":
            return FAILED
        if kind == "f":
            return float(payload)
        if kind == "dict":
            pairs = [(_decode(k, budget), _decode(v, budget)) for k, v in payload]
            _hashed([k for k, _ in pairs])
            return dict(pairs)
        items = [_decode(i, budget) for i in payload]
        if kind in ("set", "frozenset"):
            _hashed(items)
        return _CONTAINERS[kind](items)
    if value is None or type(value) in (bool, int, str):
        return value
    raise TypeError("not plain data")


class Run(NamedTuple):
    status: str  # "ok", "timeout" or "error"
    values: Optional[List[Any]]  # one per item; None unless status is "ok"
    stdout: str = ""


def _read_result(path: str) -> bytes:
    # The candidate can replace this file, so don't follow a link or block on a FIFO.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("result is not a regular file")
        data = f.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("result too large")
    return data


def run_candidate(
    code: str,
    timeout: float = 5.0,
    items: Sequence[Dict[str, Any]] = (),
    capture_stdout: bool = False,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> Run:
    """Run ``code`` in a child, then evaluate each item there and return the values.

    An item is ``{"expr": source}`` or ``{"call": argument}`` (the first function
    the candidate defines is called with it; a list or tuple of more than one is
    spread, and ``"no_args": True`` calls it with nothing when the argument is
    None). An item that raises or returns something other than plain data gives
    :data:`FAILED`. The status is ``"ok"`` only if the child got through the
    candidate and every item, so a crash, ``sys.exit``, ``os._exit`` and a
    tampered result are ``"error"``. Nothing the child does raises here.
    """
    proc = None
    limits = [["RLIMIT_CPU", math.ceil(timeout) + 1], ["RLIMIT_FSIZE", MAX_RESULT_BYTES], ["RLIMIT_CORE", 0]]
    if sys.platform.startswith("linux"):
        limits.append(["RLIMIT_AS", int(memory_limit_mb) << 20])
    # Two directories, so the working directory doesn't list the harness or the result.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work, \
            tempfile.TemporaryFile() as stdout:
        harness, job, result = (os.path.join(tmp, n) for n in ("h.py", "job.json", "out.json"))
        try:
            with open(harness, "w") as f:
                f.write(_HARNESS)
            with open(job, "w") as f:
                json.dump({"code": code, "items": list(items)}, f)
            env = {k: os.environ[k] for k in _ENV_NAMES if k in os.environ}
            env.update(HOME=work, TMPDIR=work)
            proc = subprocess.Popen(
                [sys.executable, "-I", harness, job, result, json.dumps(limits)],
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout if capture_stdout else subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return Run("timeout", None)
            raw = json.loads(_read_result(result))
            if not isinstance(raw, list) or len(raw) != len(items):
                return Run("error", None)
            budget = [MAX_RESULT_ELEMENTS]
            values = [_decode(v, budget) for v in raw]
            text = ""
            if capture_stdout:
                stdout.seek(0)
                text = stdout.read(MAX_RESULT_BYTES).decode("utf-8", errors="replace")
            return Run("ok", values, text)
        # The result file is the child's to write, so anything can be in it.
        except Exception:
            return Run("error", None)
        finally:
            if proc is not None:
                # A child that calls setsid leaves the group and isn't killed; a container covers that.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, AttributeError):
                    proc.kill()
                proc.wait()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

Operand = Tuple[Optional[str], Any]  # (source run in the child, None) or (None, literal kept here)


def _operand(node: ast.expr) -> Operand:
    try:
        return None, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ast.unparse(node), None


@functools.lru_cache(maxsize=4096)
def plan_assert(assertion: str) -> Optional[Tuple[Tuple[Operand, ...], Callable[..., bool]]]:
    """Split ``assert ...`` into its operands and the check the trainer applies.

    Handles comparisons (``==``, ``!=``, ``<`` and the like, ``in``, ``is``, also
    chained), ``not x``, ``math.isclose(a, b, ...)`` and bare ``assert x``. Returns
    None, and warns once per test, for anything else.
    """
    try:
        body = ast.parse(assertion.strip()).body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        body = []
    if len(body) != 1 or not isinstance(body[0], ast.Assert):
        logger.warning("Can't grade this test, so it counts as failed: %.120s", assertion)
        return None
    test = body[0].test
    if isinstance(test, ast.Compare) and all(type(op) in _COMPARE for op in test.ops):
        checks = [_COMPARE[type(op)] for op in test.ops]
        operands = tuple(_operand(n) for n in [test.left, *test.comparators])
        return operands, lambda *v: all(c(a, b) for c, a, b in zip(checks, v, v[1:]))
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return (_operand(test.operand),), operator.not_
    if isinstance(test, ast.Call) and ast.unparse(test.func) == "math.isclose" and len(test.args) == 2:
        try:
            kwargs = {k.arg: ast.literal_eval(k.value) for k in test.keywords}
        except (ValueError, TypeError, SyntaxError):
            logger.warning("Can't grade this test, so it counts as failed: %.120s", assertion)
            return None
        return tuple(_operand(a) for a in test.args), lambda a, b: math.isclose(a, b, **kwargs)
    return (_operand(test),), bool


def grade(
    code: str,
    tests: Sequence[Any],
    timeout: float = 5.0,
    compare: Callable[[Any, Any], bool] = operator.eq,
    no_args: bool = False,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> Tuple[str, List[bool]]:
    """Run ``code`` once and grade every test; return the run status and one bool per test.

    A test is an ``assert`` string (see :func:`plan_assert`) or a dict with
    ``input`` and ``expected_output``: the candidate's first function is called
    with the input in the child, and ``compare(actual, expected)`` runs here.
    Anything else fails. With no tests, status ``"ok"`` means the code ran to its end.
    """
    items: List[Dict[str, Any]] = []
    checks: List[Optional[Tuple[List[Tuple[Optional[int], Any]], Callable[..., bool]]]] = []
    for test in tests:
        if isinstance(test, str):
            plan = plan_assert(test)
            if plan is None:
                checks.append(None)
                continue
            operands, check = plan
            refs = []
            for src, literal in operands:
                if src is None:
                    refs.append((None, literal))
                else:
                    refs.append((len(items), None))
                    items.append({"expr": src})
            checks.append((refs, check))
        elif isinstance(test, dict):
            expected = test.get("expected_output")
            checks.append(([(len(items), None)], lambda actual, e=expected: compare(actual, e)))
            items.append({"call": test.get("input"), "no_args": no_args})
        else:
            logger.debug("Unknown test case format: %s", type(test).__name__)
            checks.append(None)
    run = run_candidate(code, timeout, items, memory_limit_mb=memory_limit_mb)
    if run.status != "ok":
        return run.status, [False] * len(tests)
    passed = []
    for plan in checks:
        if plan is None:
            passed.append(False)
            continue
        refs, check = plan
        args = [run.values[i] if i is not None else literal for i, literal in refs]
        try:
            passed.append(not any(a is FAILED for a in args) and bool(check(*args)))
        except Exception:  # comparing the child's values may still fail (types, recursion)
            passed.append(False)
    return run.status, passed


# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def humaneval_tests(test: str, entry_point: str) -> Tuple[Tuple[str, ...], str]:
    """Turn HumanEval's ``test`` source and ``entry_point`` into assert strings.

    Returns ``(asserts, setup)``: the asserts at the top of ``check(candidate)``
    with ``candidate`` renamed to the entry point, and the other statements (the
    module's and ``check``'s, such as imports), to run after the candidate. A
    statement that holds an assert somewhere inside, a loop for example, can't be
    split; it becomes one test that fails with a warning. ``assert True`` lines
    test nothing and are dropped.
    """
    try:
        module = ast.parse(test)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return ("<unparsable HumanEval test>",), ""

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id=entry_point, ctx=node.ctx), node) if node.id == "candidate" else node

    asserts: List[str] = []
    setup: List[str] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for stmt in node.body:
                stmt = Rename().visit(stmt)
                if isinstance(stmt, ast.Assert):
                    if not (isinstance(stmt.test, ast.Constant) and stmt.test.value):
                        asserts.append("assert " + ast.unparse(stmt.test))
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                elif any(isinstance(n, ast.Assert) for n in ast.walk(stmt)):
                    asserts.append(ast.unparse(stmt))  # not an assert string: fails with a warning
                else:
                    setup.append(ast.unparse(stmt))
        else:
            setup.append(ast.unparse(node))
    return tuple(asserts), "\n".join(setup)


def tests_from_columns(kwargs: Dict[str, Any]) -> Tuple[Optional[List[Any]], str]:
    """Find a row's tests: ``test_cases``, MBPP's ``test_list``, or HumanEval's ``test`` plus ``entry_point``.

    Returns ``(tests or None, setup code to run after the candidate)``.
    """
    for key in ("test_cases", "test_list"):
        value = kwargs.get(key)
        if value:
            return (list(value) if isinstance(value, (list, tuple)) else value), ""
    test, entry_point = kwargs.get("test"), kwargs.get("entry_point")
    if isinstance(test, str) and isinstance(entry_point, str) and entry_point:
        asserts, setup = humaneval_tests(test, entry_point)
        return list(asserts), setup
    return None, ""


def with_prompt(code: str, text: str, kwargs: Dict[str, Any]) -> str:
    """Prepend the prompt when the completion continues a HumanEval signature.

    A completion that only writes the body of the function the prompt starts
    doesn't define the entry point itself; the prompt plus the completion does.
    """
    entry_point, prompt = kwargs.get("entry_point"), kwargs.get("prompt")
    if not (isinstance(entry_point, str) and entry_point and isinstance(prompt, str)):
        return code
    defines = f"def {entry_point}("
    if defines in code or defines not in prompt:
        return code
    joined = prompt + text
    try:
        ast.parse(joined)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return code
    return joined


def extract_code(text: str) -> str:
    """Python code from a completion: the first fenced block with a ``def``, else
    the first fenced block; without a fence, the whole text if it parses, else the
    import lines plus everything from the first ``def`` on."""
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL) + re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if blocks:
        for block in blocks:
            if "def " in block:
                return block.strip()
        return blocks[0].strip()
    try:
        ast.parse(text)
        return text.strip()
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        pass
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("def ")), None)
    if start is None:
        return text.strip()
    imports = [line for line in lines[:start] if line.startswith(("import ", "from ")) and _is_import(line)]
    return "\n".join(imports + lines[start:])


def _is_import(line: str) -> bool:
    try:
        body = ast.parse(line).body
    except (SyntaxError, ValueError):
        return False
    return len(body) == 1 and isinstance(body[0], (ast.Import, ast.ImportFrom))
