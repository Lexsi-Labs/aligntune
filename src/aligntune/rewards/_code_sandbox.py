"""Out-of-process execution for the code rewards.

Model-generated code never runs in the trainer process. It runs in a fresh ``sys.executable -I``
child with a temp working directory, a minimal environment, resource limits, and a wall-clock
timeout that kills the child's whole process group.

Assertion tests are graded "values out": the child runs only the candidate code plus the calls the
asserts make to the candidate's function, and sends back the results as plain data (None, bool,
int, float, complex, str, bytes, list, tuple, dict, set). The assert itself is evaluated here, in
the trainer process, on fresh built-in objects. So printing "PASS", exiting with status 0, or
returning an object whose ``__eq__`` is always True does not pass a test, and the expected values
never reach the child.
"""
import ast
import builtins
import json
import os
import shutil
import subprocess
import sys
import tempfile

_BUILTINS = set(dir(builtins))

_CHILD = r'''
import json, os, sys, resource
def enc(v, d=0):
    if d > 60: raise TypeError("too deep")
    if v is None or isinstance(v, bool) or (isinstance(v, str) and type(v) is str): return v
    if isinstance(v, str): return str.__str__(v)
    if isinstance(v, int): return {"i": int.__repr__(v)}
    if isinstance(v, float): return {"f": float.__repr__(v)}
    if isinstance(v, complex): return {"c": [float.__repr__(v.real), float.__repr__(v.imag)]}
    if isinstance(v, (bytes, bytearray)): return {"b": bytes(v).hex()}
    if isinstance(v, list): return {"l": [enc(x, d + 1) for x in list.__iter__(v)]}
    if isinstance(v, tuple): return {"t": [enc(x, d + 1) for x in tuple.__iter__(v)]}
    if isinstance(v, frozenset): return {"F": [enc(x, d + 1) for x in frozenset.__iter__(v)]}
    if isinstance(v, set): return {"s": [enc(x, d + 1) for x in set.__iter__(v)]}
    if isinstance(v, dict): return {"d": [[enc(k, d + 1), enc(x, d + 1)] for k, x in dict.items(v)]}
    raise TypeError("not plain data: " + type(v).__name__)
req = json.loads(sys.stdin.read())
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
for name, val in req["limits"].items():
    try:
        resource.setrlimit(getattr(resource, name), (val, val))
    except Exception:
        pass
out = {"load": None, "calls": []}
ns = {"__name__": "__candidate__"}
try:
    exec(compile(req["setup"] + "\n" + req["code"], "<candidate>", "exec"), ns)
except BaseException as e:
    out["load"] = type(e).__name__ + ": " + str(e)[:200]
if out["load"] is None:
    for src in req["calls"]:
        try:
            val = eval(src, ns)
            try:
                out["calls"].append({"ok": enc(val)})
            except TypeError:  # not plain data (e.g. re.Match): only its truth value crosses
                out["calls"].append({"ok": {"o": bool(val)}})
        except BaseException as e:
            out["calls"].append({"err": type(e).__name__ + ": " + str(e)[:200]})
with open(req["result"], "w") as f:
    f.write(json.dumps(out))
'''


def _dec(v, d=0):
    if d > 60:
        raise ValueError("too deep")
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, dict) and len(v) == 1:
        (tag, x), = v.items()
        if tag == "i":
            return int(x)
        if tag == "f":
            return float(x)
        if tag == "c":
            return complex(float(x[0]), float(x[1]))
        if tag == "b":
            return bytes.fromhex(x)
        if tag in ("l", "t", "s", "F"):
            items = [_dec(y, d + 1) for y in x]
            return {"l": list, "t": tuple, "s": set, "F": frozenset}[tag](items)
        if tag == "d":
            return {_dec(k, d + 1): _dec(y, d + 1) for k, y in x}
        if tag == "o" and isinstance(x, bool):
            return _Opaque(x)
    raise ValueError("malformed value")


class _Opaque:
    """A non-data return value. Supports truth testing only (`assert f(x)` / `assert not f(x)`, which
    True/False would pass too); it equals nothing but itself."""
    __slots__ = ("_truthy",)

    def __init__(self, truthy):
        self._truthy = truthy

    def __bool__(self):
        return self._truthy

    __hash__ = object.__hash__


def run_calls(code, calls, setup="", timeout=5.0, mem_bytes=1024**3):
    """Run ``code`` in a child process, then evaluate each call expression in ``calls`` there.

    Returns ``(load_error, results)``: ``load_error`` is None when the code ran to the end without
    raising (``SystemExit`` included), and ``results[i]`` is ``("ok", value)`` or ``("err", reason)``.
    """
    tmp = tempfile.mkdtemp(prefix="aligntune_code_")
    result = os.path.join(tmp, "result.json")
    req = {"code": code, "setup": setup or "", "calls": list(calls), "result": result,
           "limits": {"RLIMIT_CPU": int(timeout) + 1, "RLIMIT_AS": mem_bytes, "RLIMIT_FSIZE": 10 * 1024**2,
                      "RLIMIT_NPROC": 0, "RLIMIT_CORE": 0}}
    env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "LANG": "C.UTF-8"}
    proc = None
    try:
        proc = subprocess.Popen([sys.executable, "-I", "-c", _CHILD], stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp, env=env,
                                start_new_session=True)
        proc.communicate(json.dumps(req).encode("utf-8", "surrogatepass"), timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        shutil.rmtree(tmp, ignore_errors=True)
        return "timed out", [("err", "timed out")] * len(calls)
    except Exception as e:  # noqa: BLE001
        _kill(proc)
        shutil.rmtree(tmp, ignore_errors=True)
        return f"could not run: {e}", [("err", "could not run")] * len(calls)
    _kill(proc)  # also reaps anything the candidate left in its process group
    try:
        with open(result) as f:
            out = json.loads(f.read(8_000_000))
        results = []
        for item in out.get("calls", []):
            try:
                results.append(("ok", _dec(item["ok"])) if "ok" in item else ("err", str(item.get("err"))))
            except Exception:  # noqa: BLE001
                results.append(("err", "malformed value"))
        load = out.get("load")
    except Exception:  # noqa: BLE001 -- no result file: the child exited early (sys.exit/os._exit) or crashed
        load, results = "exited before reporting", []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    results += [("err", load or "missing")] * (len(calls) - len(results))
    return load, results[:len(calls)]


def _kill(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, 9)
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def code_runs(code, timeout=5.0):
    """True if ``code`` runs to the end in a child process without raising (exit calls count as failure)."""
    load, _ = run_calls(code, [], timeout=timeout)
    return load is None


def entry_name(assertions):
    """The function the asserts test: the first called plain name in the first assert that is not a
    builtin, else the first called name (MBPP 126 calls its function ``sum``)."""
    for a in assertions:
        try:
            tree = ast.parse(str(a))
        except SyntaxError:
            continue
        names = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        for n in names:
            if n not in _BUILTINS:
                return n
        return names[0] if names else None
    return None


class _CallLifter(ast.NodeTransformer):
    """Replace each call to a candidate function with a placeholder name; remember its source."""

    def __init__(self, names, calls):
        self.names, self.calls = names, calls

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.names:
            self.calls.append(ast.unparse(node))
            return ast.copy_location(ast.Name(f"__aligntune_v{len(self.calls) - 1}", ast.Load()), node)
        return self.generic_visit(node)


def grade_assertions(code, assertions, setup="", timeout=5.0, entry=None):
    """Grade ``code`` against MBPP-style assert strings, values out. Returns a list of bools."""
    return grade_assertions_detail(code, assertions, setup, timeout, entry)[1]


def grade_assertions_detail(code, assertions, setup="", timeout=5.0, entry=None):
    """Like grade_assertions, but returns ``(load_error, list_of_bools)``; load_error is None when the
    candidate code ran, "timed out" on timeout, else the reason it did not run."""
    entry = entry or entry_name(assertions)
    names = {entry} if entry else set()
    # Other non-builtin names the asserts call (helpers the candidate must define). Builtins other
    # than the entry stay trusted, so redefining `set` or `sorted` cannot make both sides agree.
    for a in assertions:
        try:
            for n in ast.walk(ast.parse(str(a))):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id not in _BUILTINS:
                    names.add(n.func.id)
        except SyntaxError:
            pass
    trusted = {"__builtins__": builtins}
    if setup:
        try:
            exec(compile(setup, "<setup>", "exec"), trusted)
        except Exception:  # noqa: BLE001
            pass
    # Standard-library modules the asserts use without importing them (`math.isclose`,
    # `sys.getsizeof`): the tests used to borrow the candidate's own imports.
    for a in assertions:
        try:
            for n in ast.walk(ast.parse(str(a))):
                if (isinstance(n, ast.Name) and n.id not in trusted and n.id not in names
                        and n.id in getattr(sys, "stdlib_module_names", ())):
                    trusted[n.id] = __import__(n.id)
        except Exception:  # noqa: BLE001
            pass
    names -= {k for k in trusted if k != entry}
    calls, tests = [], []
    for a in assertions:
        try:
            node = ast.parse(str(a).strip()).body[0]
        except (SyntaxError, IndexError):
            tests.append(None)
            continue
        if not isinstance(node, ast.Assert):
            tests.append(None)
            continue
        start = len(calls)
        expr = _CallLifter(names, calls).visit(node.test)
        tests.append((ast.fix_missing_locations(ast.Expression(expr)), range(start, len(calls))))
    load, results = run_calls(code, calls, setup=setup, timeout=timeout) if calls else ("no call to grade", [])
    out = []
    for t in tests:
        if t is None or load is not None:
            out.append(False)
            continue
        expr, idx = t
        if not idx or any(results[i][0] != "ok" for i in idx):
            out.append(False)
            continue
        env = dict(trusted)
        env.update({f"__aligntune_v{i}": results[i][1] for i in idx})
        try:
            out.append(bool(eval(compile(expr, "<assert>", "eval"), env)))
        except Exception:  # noqa: BLE001
            out.append(False)
    return load, out


def humaneval_assertions(test_src, entry_point):
    """HumanEval's ``check(candidate)`` program -> its single-line asserts, calling the entry point.
    Asserts that use loop variables or locals of ``check`` cannot be lifted and will fail."""
    out = []
    for line in str(test_src).splitlines():
        s = line.strip()
        if s.startswith("assert ") and not s.startswith("assert True"):
            out.append(s.replace("candidate(", f"{entry_point}("))
    return out


def tests_from_kwargs(kwargs, default=None):
    """Find tests in whatever the dataset row provides: ``test_cases`` (AlignTune's own key),
    MBPP's ``test_list``, or HumanEval's ``test`` + ``entry_point``."""
    tc = kwargs.get("test_cases")
    if tc:
        return list(tc) if isinstance(tc, (list, tuple)) else tc
    if kwargs.get("test_list"):
        return list(kwargs["test_list"])
    if kwargs.get("test") and kwargs.get("entry_point"):
        return humaneval_assertions(kwargs["test"], kwargs["entry_point"])
    return default


def setup_from_kwargs(kwargs):
    for key in ("test_imports", "test_setup_code"):
        v = kwargs.get(key)
        if v:
            return "\n".join(v) if isinstance(v, (list, tuple)) else str(v)
    return ""


def is_plain(v, d=0):
    """True for plain data only (what can be compared safely with ==)."""
    if d > 60:
        return False
    if v is None or type(v) in (bool, int, float, complex, str, bytes):
        return True
    if type(v) in (list, tuple, set, frozenset):
        return all(is_plain(x, d + 1) for x in v)
    if type(v) is dict:
        return all(is_plain(k, d + 1) and is_plain(x, d + 1) for k, x in v.items())
    return False
