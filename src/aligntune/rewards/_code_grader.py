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
process. On Linux the child also confines itself before the candidate runs
(see ``_code_confine``): seccomp refuses signals to other processes, ``ptrace``,
sockets and new processes, and Landlock limits writes to the child's own
directory and reads to the Python installation and system directories. That
is best effort. Where the kernel offers neither, the child runs as the same
user and can read any file that user can read, use the network, and signal
other processes, the trainer included; a warning says so, and
``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1`` refuses to run then. Everything in
the child belongs to the candidate, so a determined one can write values of its
own choosing into the result. That only scores where the value it needs is known
without the expected output, such as a test that wants ``True``. Use
``input``/``expected_output`` tests or asserts with a literal on one side where
that matters, and a container or separate user for stronger isolation.
"""

import ast
import builtins
import functools
import itertools
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
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

from aligntune.rewards._code_confine import (
    ISOLATE_SOURCE,
    SINGLE_THREAD_ENV,
    note_isolation_once,
    require_isolation,
)

logger = logging.getLogger(__name__)

MAX_RESULT_BYTES = 16 * 1024 * 1024
DEFAULT_MEMORY_LIMIT_MB = 4096
_ENV_NAMES = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")

# Runs in the child. Everything it needs after the candidate has run is bound to
# a local name first. Only the candidate's own calls are evaluated here (as
# separate expressions), and each result must be plain data or it is a failure.
# The trainer evaluates the rest of a test. The candidate can still change
# anything in this process; all that gets it is a value of its own choosing.
_HARNESS = ISOLATE_SOURCE + r"""
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
    # Last step before the candidate: the job is read and the result file is open, so confining
    # now costs the harness nothing. See _code_confine for what it stops.
    state = isolate(os.getcwd(), job.get("require", False))
    if job.get("require") and not (state["seccomp"] or state["landlock"]):
        os._exit(3)

    _type, _callable, _isinstance, _eval, _exception = type, callable, isinstance, eval, Exception
    _list, _tuple, _set, _frozenset, _dict, _bool, _int, _str, _float = list, tuple, set, frozenset, dict, bool, int, str, float
    _repr, _module, _function, _getattr, _len, _hasattr = repr, types.ModuleType, types.FunctionType, getattr, len, hasattr
    _dump, _close, _exit, _flush = json.dump, out.close, os._exit, sys.stdout.flush

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
        if _str(_getattr(t, "__module__", "")).split(".")[0] == "numpy" and _callable(_getattr(x, "tolist", None)):
            return enc(x.tolist())
        raise TypeError("not plain data")

    allow = job.get("builtins")
    ns = {"__name__": "__main__",
          "__builtins__": builtins if allow is None else {k: _getattr(builtins, k) for k in allow if _hasattr(builtins, k)}}
    exec(compile(job["code"], "<candidate>", "exec"), ns)

    defined = [v for k, v in ns.items() if not k.startswith("_") and _type(v) is _function
               and _getattr(v, "__module__", None) == "__main__"]
    public = [v for k, v in ns.items() if not k.startswith("_") and _callable(v) and not _isinstance(v, _module)]
    target = (defined or public or [None])[0]

    values = []
    for item in job["items"]:
        try:
            if "expr" in item:
                value = _eval(item["expr"], ns)
            else:
                arg = item["call"]
                f = target
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


def _collides(members: Any) -> bool:
    """Whether more than :data:`_MAX_SAME_HASH` distinct members share one hash.

    Distinct values that all hash alike (multiples of ``2**61 - 1`` all hash to 0)
    make building a set or dict quadratic, which would hang the trainer. Equal
    values are cheap to insert, so they don't count. Everything here is linear:
    members are grouped by hash, and a crowded group is counted by ``repr``,
    which is conservative (``1``, ``1.0`` and ``True`` count as three).
    """
    def h(x: Any) -> int:
        try:
            return hash(x)
        except TypeError:  # unhashable, so it can't become a member or a key
            return id(x)

    buckets: Dict[int, List[Any]] = {}
    for m in members:
        buckets.setdefault(h(m), []).append(m)
    return any(len(b) > _MAX_SAME_HASH and len({(type(x).__name__, repr(x)) for x in b}) > _MAX_SAME_HASH
               for b in buckets.values())


def _hashed(items: List[Any]) -> List[Any]:
    """Refuse hash collisions before building a set or dict from the child's data."""
    if _collides(items):
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
    builtins_allow: Optional[Sequence[str]] = None,
) -> Run:
    """Run ``code`` in a child, then evaluate each item there and return the values.

    An item is ``{"expr": source}`` or ``{"call": argument}`` (the first function
    the candidate defines is called with it; a list or tuple of more than one is
    spread, and ``"no_args": True`` calls it with nothing when the argument is
    None). An item that raises or returns something other than plain data gives
    :data:`FAILED`. The status is ``"ok"`` only if the child got through the
    candidate and every item, so a crash, ``sys.exit``, ``os._exit`` and a
    tampered result are ``"error"``. Nothing the child does raises here.

    With ``builtins_allow``, the candidate sees only those builtins (no
    ``__import__``, so no imports). That narrows what a completion can reach
    by name; it does not stop a determined one.
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
                json.dump({"code": code, "items": list(items), "builtins": None if builtins_allow is None else list(builtins_allow),
                           "require": require_isolation()}, f)
            env = {k: os.environ[k] for k in _ENV_NAMES if k in os.environ}
            env.update(HOME=work, TMPDIR=work, **SINGLE_THREAD_ENV)
            note_isolation_once()
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

class _Fail(Exception):
    """A part of the test could not be evaluated, which fails it."""


class _Undecomposable(Exception):
    pass


_FUNCS = {
    n: getattr(builtins, n)
    for n in ("len sum min max abs sorted set frozenset list tuple dict int float str bool isinstance "
              "any all round reversed zip enumerate divmod").split()
}
_MATH = {n: getattr(math, n) for n in "isclose sqrt floor ceil fabs isnan isinf isfinite pi e inf nan".split()}
_METHODS = frozenset(
    "count index keys values items get startswith endswith lower upper strip lstrip rstrip split join "
    "isdigit isalpha isalnum isupper islower find issubset issuperset union intersection "
    "difference symmetric_difference copy".split()
)
_PLAIN = (bool, int, float, str, list, tuple, set, frozenset, dict, type(None))
_BUILTIN_NAMES = frozenset(dir(builtins))
# Every container, string and repeat the trainer builds or reads while evaluating a test is held to
# this many elements, and an assert may visit this many nodes, so a forged value or a big literal
# can't make the trainer allocate or loop for long.
_MAX_REPEAT = 10**5
_MAX_NODES = 10_000


class _Vals(list):
    """The child's values for one evaluation, plus the count of nodes visited."""

    visited = 0
    touched = 0


_VIEWS = (type({}.keys()), type({}.values()), type({}.items()))


def _tame(x: Any) -> Any:
    """A dict view or an iterator (zip, enumerate, reversed...) as a list, read up to the element cap."""
    if type(x) in _VIEWS or hasattr(x, "__next__"):
        out = list(itertools.islice(x, _MAX_REPEAT + 1))
        if len(out) > _MAX_REPEAT:
            raise _Fail
        return out
    return x


def _small(v: Any, *things: Any, hashed: bool = False) -> None:
    """Refuse a big container or string, a total past the cap for this assert, and hash collisions.

    Sets, dicts and dict views are always checked; a list or tuple only with ``hashed``, when its
    elements are about to become set members or dict keys.
    """
    for t in things:
        if type(t) in (str, list, tuple, set, frozenset, dict) or type(t) in _VIEWS:
            if len(t) > _MAX_REPEAT:
                raise _Fail
            v.touched += len(t)
            if type(t) is not str and (hashed or type(t) not in (list, tuple)) and _collides(t):
                raise _Fail
    if v.touched > 10 * _MAX_REPEAT:
        raise _Fail


def _dict_keys_collide(arg: Any) -> bool:
    """For ``dict(pairs)``: whether the keys that would be inserted hash alike.

    Every element must be a list or tuple of two items, so no key hides in another kind of pair.
    """
    if type(arg) not in (list, tuple, set, frozenset):
        return False
    keys = []
    for p in arg:
        if type(p) not in (list, tuple) or len(p) != 2:
            raise _Fail
        keys.append(p[0])
    return _collides(keys)


Ev = Callable[[List[Any]], Any]


def _is_chain(node: ast.expr) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _candidate(node: ast.expr) -> bool:
    """Whether ``node`` is the candidate's: a name it defines, or a call to one (``f(1)``, ``mod.f(1)``)."""
    if isinstance(node, ast.Call):
        return _is_chain(node.func) and _candidate(node.func)
    if isinstance(node, ast.Attribute):
        return _is_chain(node) and _candidate(node.value)
    if isinstance(node, ast.Name):
        if node.id in _FUNCS or node.id == "math":
            return False
        if node.id in _BUILTIN_NAMES:
            raise _Undecomposable(node.id)
        return True
    return False


_SET_LIKE = (set, frozenset) + _VIEWS
_SET_METHODS = frozenset("union intersection difference symmetric_difference issubset issuperset".split())


def _binop(op: ast.operator, a: Any, b: Any, v: Any) -> Any:
    _small(v, a, b, hashed=type(a) in _SET_LIKE or type(b) in _SET_LIKE)
    if isinstance(op, ast.Mult):
        for seq, n in ((a, b), (b, a)):
            if isinstance(seq, (str, list, tuple)) and type(n) in (int, bool):
                if len(seq) * abs(n) > _MAX_REPEAT:
                    raise _Fail
    elif isinstance(op, ast.Add):
        if isinstance(a, (str, list, tuple)) and isinstance(b, (str, list, tuple)) and len(a) + len(b) > _MAX_REPEAT:
            raise _Fail
    elif isinstance(op, ast.Pow):
        if type(a) is int and type(b) is int and (abs(b) > 1000 or a.bit_length() * abs(b) > 10_000):
            raise _Fail
    elif isinstance(op, ast.Mod):
        if type(a) not in (int, float) or type(b) not in (int, float):  # no printf-style formatting
            raise _Fail
    result = _BINOPS[type(op)](a, b)
    _small(v, result)
    return result


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert, ast.Not: operator.not_}


def _compile(node: ast.expr, calls: List[str]) -> Ev:
    run = _compile_node(node, calls)

    def counted(v: Any) -> Any:
        v.visited += 1
        if v.visited > _MAX_NODES:
            raise _Fail
        return run(v)

    return counted


def _compile_node(node: ast.expr, calls: List[str]) -> Ev:
    """Turn a test expression into a function of the child's values.

    Each maximal candidate call becomes one expression for the child; everything
    else is evaluated here, on plain data, by this whitelist.
    """
    try:
        literal = ast.literal_eval(node)
        return lambda v: literal
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if _candidate(node):
        calls.append(ast.unparse(node))
        i = len(calls) - 1

        def get(v: List[Any]) -> Any:
            if v[i] is FAILED:
                raise _Fail
            return v[i]

        return get
    if isinstance(node, ast.BoolOp):
        subs = [_compile(x, calls) for x in node.values]
        if isinstance(node.op, ast.And):
            def conj(v: List[Any]) -> Any:
                r: Any = True
                for s in subs:
                    r = s(v)
                    if not r:
                        break
                return r

            return conj

        def disj(v: List[Any]) -> Any:  # an `or` holds if any side holds, even if another raised
            failed, r = False, False
            for s in subs:
                try:
                    r = s(v)
                except (_Fail, TypeError, ValueError, RecursionError, OverflowError, ZeroDivisionError):
                    failed = True
                    continue
                if r:
                    return r
            if failed:
                raise _Fail
            return r

        return disj
    if isinstance(node, ast.UnaryOp):
        x, op = _compile(node.operand, calls), _UNARY[type(node.op)]
        return lambda v: op(x(v))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _compile(node.left, calls), _compile(node.right, calls)
        return lambda v: _binop(node.op, a(v), b(v), v)
    if isinstance(node, ast.Compare):
        ops = [_COMPARE[type(op)] for op in node.ops]
        parts = [_compile(x, calls) for x in [node.left, *node.comparators]]

        def compare(v: List[Any]) -> bool:
            left = parts[0](v)
            for op, part in zip(ops, parts[1:]):
                right = part(v)
                if not op(left, right):
                    return False
                left = right
            return True

        return compare
    if isinstance(node, ast.IfExp):
        t, a, b = (_compile(x, calls) for x in (node.test, node.body, node.orelse))
        return lambda v: a(v) if t(v) else b(v)
    if isinstance(node, ast.Subscript):
        value, index = _compile(node.value, calls), _compile_slice(node.slice, calls)
        return lambda v: value(v)[index(v)]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and not any(isinstance(x, ast.Starred) for x in node.elts):
        items, make = [_compile(x, calls) for x in node.elts], {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)]
        return lambda v: make(i(v) for i in items)
    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        pairs = [(_compile(cast(ast.expr, k), calls), _compile(x, calls)) for k, x in zip(node.keys, node.values)]
        return lambda v: {k(v): x(v) for k, x in pairs}
    if isinstance(node, ast.Name) and node.id in _FUNCS:
        thing = _FUNCS[node.id]
        return lambda v: thing
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math" \
            and node.attr in _MATH:
        thing = _MATH[node.attr]
        return lambda v: thing
    if isinstance(node, ast.Call) and not any(isinstance(a, ast.Starred) for a in node.args) \
            and all(k.arg for k in node.keywords):
        args = [_compile(a, calls) for a in node.args]
        kwargs = [(str(k.arg), _compile(k.value, calls)) for k in node.keywords]
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FUNCS:
            def call(v: List[Any]) -> Any:
                values = [_tame(a(v)) for a in args]
                if func.id != "len":
                    _small(v, *values, hashed=func.id in ("set", "frozenset", "dict"))
                if func.id == "dict" and values and _dict_keys_collide(values[0]):
                    raise _Fail
                return _FUNCS[func.id](*values, **{k: a(v) for k, a in kwargs})

            return call
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math" \
                and func.attr in _MATH and callable(_MATH[func.attr]):
            return lambda v: _MATH[func.attr](*(a(v) for a in args), **{k: a(v) for k, a in kwargs})
        if isinstance(func, ast.Attribute) and func.attr in _METHODS:
            receiver = _compile(func.value, calls)

            def method(v: List[Any]) -> Any:
                r = receiver(v)
                if type(r) not in _PLAIN:
                    raise _Fail
                values = [_tame(a(v)) for a in args]
                _small(v, r, *values, hashed=func.attr in _SET_METHODS)
                if func.attr == "join" and (len(values) != 1 or not hasattr(values[0], "__len__")
                                            or len(r) * len(values[0]) > _MAX_REPEAT):
                    raise _Fail
                result = getattr(r, func.attr)(*values, **{k: a(v) for k, a in kwargs})
                _small(v, result)
                return result

            return method
    raise _Undecomposable(ast.unparse(node))


def _compile_slice(node: ast.expr, calls: List[str]) -> Ev:
    if isinstance(node, ast.Slice):
        parts = [_compile(x, calls) if x is not None else (lambda v: None) for x in (node.lower, node.upper, node.step)]
        return lambda v: slice(*(p(v) for p in parts))
    return _compile(node, calls)


@functools.lru_cache(maxsize=4096)
def plan_assert(assertion: str) -> Optional[Tuple[Tuple[str, ...], Callable[[List[Any]], bool]]]:
    """Split ``assert ...`` into the candidate calls the child runs and the check the trainer applies.

    The child only runs the candidate's own calls and returns plain data. The
    trainer evaluates the rest (``and``, ``or``, ``not``, chained comparisons,
    arithmetic, whitelisted builtins, ``math`` functions and methods of plain
    data). Returns None, and warns once per test, for anything else.
    """
    try:
        body = ast.parse(assertion.strip()).body
        if len(body) == 1 and isinstance(body[0], ast.Assert):
            calls: List[str] = []
            ev = _compile(body[0].test, calls)
            return tuple(calls), lambda values: bool(ev(_Vals(values)))
    except (SyntaxError, ValueError, MemoryError, RecursionError, _Undecomposable):
        pass
    logger.warning("Can't grade this test, so it counts as failed: %.120s", assertion)
    return None


def _bound_names(code: str) -> set:
    """Names the candidate binds at module level by ``def``, ``class`` or assignment."""
    try:
        body = ast.parse(code).body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return set()
    names = set()
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def _bound_builtins(code: str, test: str) -> bool:
    used = {n.id for n in ast.walk(ast.parse(test.strip())) if isinstance(n, ast.Name)}
    return bool(used & (set(_FUNCS) | {"math"}) & _bound_names(code))


def grade(
    code: str,
    tests: Sequence[Any],
    timeout: float = 5.0,
    compare: Callable[[Any, Any], bool] = operator.eq,
    no_args: bool = False,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    builtins_allow: Optional[Sequence[str]] = None,
) -> Tuple[str, List[bool]]:
    """Run ``code`` once and grade every test; return the run status and one bool per test.

    A test is an ``assert`` string (see :func:`plan_assert`) or a dict with
    ``input`` and ``expected_output``: the candidate's first function is called
    with the input in the child, and ``compare(actual, expected)`` runs here.
    Anything else fails. With no tests, status ``"ok"`` means the code ran to its end.
    """
    items: List[Dict[str, Any]] = []
    checks: List[Optional[Tuple[List[int], Callable[[List[Any]], bool]]]] = []
    for test in tests:
        if isinstance(test, str):
            plan = plan_assert(test)
            if plan is None:
                checks.append(None)
                continue
            calls, check = plan
            if _bound_builtins(code, test):
                logger.warning("Test uses a builtin's name that the candidate defines, so it counts as failed: %.120s", test)
                checks.append(None)
                continue
            refs = list(range(len(items), len(items) + len(calls)))
            items.extend({"expr": src} for src in calls)
            checks.append((refs, check))
        elif isinstance(test, dict):
            expected = test.get("expected_output")
            checks.append(([len(items)], lambda v, e=expected: v[0] is not FAILED and bool(compare(v[0], e))))
            items.append({"call": test.get("input"), "no_args": no_args})
        else:
            logger.debug("Unknown test case format: %s", type(test).__name__)
            checks.append(None)
    run = run_candidate(code, timeout, items, memory_limit_mb=memory_limit_mb, builtins_allow=builtins_allow)
    if run.status != "ok":
        return run.status, [False] * len(tests)
    assert run.values is not None  # status "ok" always comes with values
    passed = []
    for plan in checks:
        if plan is None:
            passed.append(False)
            continue
        refs, check = plan
        try:
            passed.append(bool(check([run.values[i] for i in refs])))
        except Exception:  # _Fail, or the child's values did not fit the test (types, recursion)
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
