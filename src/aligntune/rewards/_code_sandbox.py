"""Out-of-process execution for the code rewards.

Model-generated code never runs in the trainer process. It runs in a fresh ``sys.executable -I``
child with a temp working directory, a minimal environment, resource limits, and a wall-clock
timeout that kills the child's whole process group.

On Linux the child also confines itself before it runs the candidate (best effort, see
``isolation_status``):

* a seccomp filter makes ``kill``/``tgkill``/``pidfd_send_signal``, ``ptrace``, ``socket``,
  ``io_uring_setup``, ``fork``/``clone`` and ``execve`` fail with EPERM, so the candidate cannot
  signal or trace the trainer, open network connections or start other programs;
* a Landlock ruleset allows writes only inside the child's temp directory and reads only from the
  Python installation and system directories (not the user's home), and on kernels with Landlock
  ABI 4+ blocks TCP bind/connect.

Set ``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1`` to refuse to run candidate code (score 0) when the
kernel offers neither.

Assertion tests are graded "values out": the child runs only the candidate code plus the calls the
asserts make to the candidate's function, and sends back the results as plain data (None, bool,
int, float, complex, str, bytes, list, tuple, dict, set). The assert itself is evaluated here, in
the trainer process, on fresh built-in objects. So printing "PASS", exiting with status 0, or
returning an object whose ``__eq__`` is always True does not pass a test, and the expected values
never reach the child.

HumanEval's ``check(candidate)`` programs (loops, locals, multi-line asserts) are graded the same
way through a :class:`CandidateSession`: ``check`` runs here, and each call to ``candidate`` is
forwarded to the child, which returns the value as plain data.
"""
import ast
import builtins
import json
import logging
import os
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time

logger = logging.getLogger(__name__)

_BUILTINS = set(dir(builtins))
_MAX_FRAME = 8_000_000

# Plain-data codec. Executed in the child (for return values and call arguments) and here (to
# encode call arguments). ``enc`` refuses anything that is not plain data and never calls a method
# the candidate could have overridden.
_CODEC = r'''
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
    if str(getattr(type(v), "__module__", "")).split(".")[0] == "numpy" and callable(getattr(v, "tolist", None)):
        return enc(v.tolist(), d + 1)  # numpy scalar/array -> Python values; the result is checked again here
    raise TypeError("not plain data: " + type(v).__name__)

def dec_trusted(v):
    if v is None or isinstance(v, (bool, str)): return v
    (tag, x), = v.items()
    if tag == "i": return int(x)
    if tag == "f": return float(x)
    if tag == "c": return complex(float(x[0]), float(x[1]))
    if tag == "b": return bytes.fromhex(x)
    if tag in ("l", "t", "s", "F"):
        return {"l": list, "t": tuple, "s": set, "F": frozenset}[tag](dec_trusted(y) for y in x)
    if tag == "d": return {dec_trusted(k): dec_trusted(y) for k, y in x}
    raise ValueError("bad value")
'''

# Self-confinement, run in the child after its own imports and before the candidate.
_ISOLATE = r'''
def isolate(tmp, require):
    st = {"seccomp": False, "landlock": 0}
    try:
        import ctypes, platform
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        libc.prctl.restype = ctypes.c_int
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError("no_new_privs")
    except Exception:
        return st
    # --- Landlock: writes only under tmp, reads only from Python + system dirs, no TCP ---
    try:
        def sc(*a):
            return libc.syscall(*[ctypes.c_long(x) if isinstance(x, int) else x for x in a])
        abi = sc(444, ctypes.c_void_p(0), 0, 1)  # LANDLOCK_CREATE_RULESET_VERSION
        if abi >= 1:
            fs = (1 << 13) - 1
            if abi >= 2: fs |= 1 << 13
            if abi >= 3: fs |= 1 << 14
            if abi >= 5: fs |= 1 << 15
            net = 3 if abi >= 4 else 0
            scoped = 3 if abi >= 6 else 0
            size = 24 if abi >= 6 else (16 if abi >= 4 else 8)
            attr = (ctypes.c_uint64 * 3)(fs, net, scoped)
            rfd = sc(444, ctypes.byref(attr), size, 0)
            if rfd < 0:
                raise OSError("create_ruleset")
            class PB(ctypes.Structure):
                _pack_ = 1
                _fields_ = [("allowed", ctypes.c_uint64), ("fd", ctypes.c_int32)]
            file_ok = 1 | 2 | 4 | (1 << 14) | (1 << 15)
            def allow(path, access):
                try:
                    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                except OSError:
                    return
                try:
                    import stat
                    a = access & fs
                    if not stat.S_ISDIR(os.fstat(fd).st_mode):
                        a &= file_ok
                    if a:
                        pb = PB(a, fd)
                        sc(445, rfd, 1, ctypes.byref(pb), 0)  # LANDLOCK_RULE_PATH_BENEATH
                finally:
                    os.close(fd)
            read = 1 | 4 | 8  # EXECUTE | READ_FILE | READ_DIR
            ro = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                  "/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/dev", "/proc", "/sys",
                  "/opt", "/nix", "/run/current-system"}
            ro |= {p for p in sys.path if p and os.path.isabs(p)}
            for p in ro:
                allow(p, read)
            allow(tmp, fs)
            allow("/dev/null", 2 | 4 | (1 << 14))
            if sc(446, rfd, 0) == 0:  # landlock_restrict_self
                st["landlock"] = abi
            os.close(rfd)
    except Exception:
        pass
    # --- seccomp: no signals to others, no ptrace, no sockets, no new processes or programs ---
    try:
        m = platform.machine()
        if m in ("x86_64", "AMD64"):
            arch, x32 = 0xC000003E, True
            deny = [62, 200, 234, 129, 297, 424, 434, 438, 101, 310, 311, 41, 425,
                    56, 57, 58, 435, 59, 322]
        elif m in ("aarch64", "arm64"):
            arch, x32 = 0xC00000B7, False
            deny = [129, 130, 131, 138, 240, 424, 434, 438, 117, 270, 271, 198, 425,
                    220, 435, 221, 281]
        else:
            deny = None
        if deny:
            LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
            ALLOW, EPERM = 0x7FFF0000, 0x00050001
            prog = [(LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, EPERM), (LD, 0, 0, 0)]
            checks = ([(JGE, 0x40000000)] if x32 else []) + [(JEQ, n) for n in deny]
            deny_at = len(prog) + len(checks) + 1
            for code, k in checks:
                prog.append((code, deny_at - len(prog) - 1, 0, k))
            prog += [(RET, 0, 0, ALLOW), (RET, 0, 0, EPERM)]
            class F(ctypes.Structure):
                _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte),
                            ("k", ctypes.c_uint32)]
            class P(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(F))]
            arr = (F * len(prog))(*[F(c, jt, jf, k) for c, jt, jf, k in prog])
            fp = P(len(prog), arr)
            if libc.prctl(22, 2, ctypes.addressof(fp), 0, 0) == 0:  # PR_SET_SECCOMP, FILTER
                st["seccomp"] = True
    except Exception:
        pass
    return st
'''

_CHILD = r'''
import json, os, sys, resource, struct, io
''' + _CODEC + _ISOLATE + r'''
pin = os.fdopen(os.dup(0), "rb", buffering=0)
pout = os.fdopen(os.dup(1), "wb", buffering=0)
_nul = os.open(os.devnull, os.O_RDWR)
os.dup2(_nul, 0)
os.dup2(_nul, 1)
def rd():
    h = b""
    while len(h) < 4:
        c = pin.read(4 - len(h))
        if not c: raise EOFError
        h += c
    n = struct.unpack(">I", h)[0]
    b = b""
    while len(b) < n:
        c = pin.read(n - len(b))
        if not c: raise EOFError
        b += c
    return json.loads(b)
def wr(o):
    b = json.dumps(o).encode()
    pout.write(struct.pack(">I", len(b)) + b)
def out(v):
    try:
        return {"ok": enc(v)}
    except TypeError:  # not plain data (e.g. re.Match): only its truth value crosses
        return {"ok": {"o": bool(v)}}
req = rd()
for name, val in req["limits"].items():
    try:
        resource.setrlimit(getattr(resource, name), (val, val))
    except Exception:
        pass
if req["mode"] == "program":
    _o = os.open(os.path.join(req["tmp"], "stdout.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(_o, 1)
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", closefd=False), encoding="utf-8", errors="replace")
    sys.stdin = io.StringIO(req.get("stdin") or "")
iso = isolate(req["tmp"], req["require"])
load = None
if req["require"] and not (iso["seccomp"] or iso["landlock"]):
    load = "isolation unavailable"
ns = {"__name__": "__candidate__"}
if load is None:
    try:
        exec(compile(req["setup"] + "\n" + req["code"], "<candidate>", "exec"), ns)
    except BaseException as e:
        load = type(e).__name__ + ": " + str(e)[:200]
    try:
        sys.stdout.flush()
    except BaseException:
        pass
mode = req["mode"]
if mode == "calls":
    calls = []
    if load is None:
        for src in req["calls"]:
            try:
                calls.append(out(eval(src, ns)))
            except BaseException as e:
                calls.append({"err": type(e).__name__ + ": " + str(e)[:200]})
    wr({"load": load, "calls": calls, "iso": iso})
elif mode == "program":
    wr({"load": load, "iso": iso})
else:  # serve: answer calls until the trainer closes the pipe
    wr({"load": load, "iso": iso})
    while load is None:
        try:
            msg = rd()
        except EOFError:
            break
        try:
            fn = ns[msg["fn"]]
            v = fn(*dec_trusted(msg["args"]), **dec_trusted(msg["kwargs"]))
            r = out(v)
        except BaseException as e:
            r = {"err": type(e).__name__ + ": " + str(e)[:200]}
        try:
            sys.stdout.flush()
        except BaseException:
            pass
        wr(r)
'''

_codec_ns = {}
exec(_CODEC, _codec_ns)  # our own constant source: gives the trainer-side encoder for call arguments
_enc = _codec_ns["enc"]

_ISO_WARNED = []


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


class SandboxTimeout(Exception):
    pass


class SandboxClosed(Exception):
    pass


def _require_isolation():
    return os.environ.get("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "").strip().lower() in ("1", "true", "yes")


def _note_isolation(iso):
    if _ISO_WARNED or not isinstance(iso, dict):
        return
    if not (iso.get("seccomp") and iso.get("landlock")):
        _ISO_WARNED.append(True)
        logger.warning(
            "Code-reward sandbox: kernel isolation is partial (seccomp=%s, landlock ABI=%s). Candidate "
            "code still runs in a separate process with limits, but may be able to signal other "
            "processes or read/write the user's files. Set ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1 to refuse.",
            iso.get("seccomp"), iso.get("landlock"))


class _Child:
    """One sandboxed child process and its framed JSON channel (stdin/stdout of the child)."""

    def __init__(self, req, timeout, mem_bytes):
        self.tmp = tempfile.mkdtemp(prefix="aligntune_code_")
        self.deadline = time.monotonic() + timeout
        self.proc = None
        self._buf = b""
        req = dict(req, tmp=self.tmp, require=_require_isolation(),
                   limits={"RLIMIT_CPU": int(timeout) + 1, "RLIMIT_AS": mem_bytes,
                           "RLIMIT_FSIZE": 10 * 1024 ** 2, "RLIMIT_NPROC": 0, "RLIMIT_CORE": 0})
        env = {"PATH": "/usr/bin:/bin", "HOME": self.tmp, "TMPDIR": self.tmp, "LANG": "C.UTF-8",
               # The child may not start threads; BLAS libraries start a pool when numpy is imported
               # and fail (numpy then fails to import) unless told to use the calling thread only.
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1"}
        try:
            self.proc = subprocess.Popen([sys.executable, "-I", "-c", _CHILD], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.tmp,
                                         env=env, start_new_session=True)
            self.send(req)
        except BaseException:
            self.close()
            raise

    def send(self, obj):
        b = json.dumps(obj).encode("utf-8", "surrogatepass")
        try:
            self.proc.stdin.write(struct.pack(">I", len(b)) + b)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise SandboxClosed(str(e))

    def _read(self, n):
        fd = self.proc.stdout.fileno()
        while len(self._buf) < n:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise SandboxTimeout("timed out")
            r, _, _ = select.select([fd], [], [], left)
            if not r:
                raise SandboxTimeout("timed out")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise SandboxClosed("exited before reporting")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def recv(self):
        n = struct.unpack(">I", self._read(4))[0]
        if n > _MAX_FRAME:
            raise SandboxClosed("reply too large")
        try:
            return json.loads(self._read(n))
        except ValueError:
            raise SandboxClosed("malformed reply")

    def close(self):
        p = self.proc
        if p is not None:
            for s in (p.stdin, p.stdout):
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                os.killpg(p.pid, 9)
            except Exception:  # noqa: BLE001
                pass
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
            try:
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def run_calls(code, calls, setup="", timeout=5.0, mem_bytes=1024**3):
    """Run ``code`` in a child process, then evaluate each call expression in ``calls`` there.

    Returns ``(load_error, results)``: ``load_error`` is None when the code ran to the end without
    raising (``SystemExit`` included), and ``results[i]`` is ``("ok", value)`` or ``("err", reason)``.
    """
    calls = list(calls)
    child = None
    try:
        child = _Child({"mode": "calls", "code": code, "setup": setup or "", "calls": calls}, timeout, mem_bytes)
        out = child.recv()
    except SandboxTimeout:
        return "timed out", [("err", "timed out")] * len(calls)
    except SandboxClosed as e:
        return str(e), [("err", str(e))] * len(calls)
    except Exception as e:  # noqa: BLE001
        return f"could not run: {e}", [("err", "could not run")] * len(calls)
    finally:
        if child is not None:
            child.close()
    if not isinstance(out, dict):
        return "malformed reply", [("err", "malformed reply")] * len(calls)
    _note_isolation(out.get("iso"))
    results = []
    for item in out.get("calls", []) if isinstance(out.get("calls"), list) else []:
        try:
            results.append(("ok", _dec(item["ok"])) if "ok" in item else ("err", str(item.get("err"))))
        except Exception:  # noqa: BLE001
            results.append(("err", "malformed value"))
    load = out.get("load")
    load = None if load is None else str(load)
    results += [("err", load or "missing")] * (len(calls) - len(results))
    return load, results[:len(calls)]


def run_program(code, stdin="", timeout=5.0, mem_bytes=1024**3, max_output=1_000_000):
    """Run ``code`` as a program in a child process. Returns ``(load_error, stdout_text)``;
    ``load_error`` is None when it ran to the end without raising (exits count as errors)."""
    child = None
    try:
        child = _Child({"mode": "program", "code": code, "setup": "", "stdin": stdin}, timeout, mem_bytes)
        out = child.recv()
        if not isinstance(out, dict):
            return "malformed reply", ""
        _note_isolation(out.get("iso"))
        try:
            with open(os.path.join(child.tmp, "stdout.txt"), "rb") as f:
                text = f.read(max_output).decode("utf-8", "replace")
        except OSError:
            text = ""
        load = out.get("load")
        return (None if load is None else str(load)), text
    except SandboxTimeout:
        return "timed out", ""
    except SandboxClosed as e:
        return str(e), ""
    except Exception as e:  # noqa: BLE001
        return f"could not run: {e}", ""
    finally:
        if child is not None:
            child.close()


class CandidateError(Exception):
    """The candidate's function raised (or could not be reached)."""


class CandidateSession:
    """A child process that keeps the candidate loaded and answers calls, values out.

    ``call(name, args, kwargs)`` sends plain-data arguments and returns the plain-data value the
    candidate's function produced (or raises CandidateError). One wall-clock budget (``timeout``)
    covers loading and every call; once it is spent, or the child dies, every later call fails.
    """

    def __init__(self, code, setup="", timeout=5.0, mem_bytes=1024**3, max_calls=20000):
        self.load = None
        self.calls = 0
        self.max_calls = max_calls
        self._dead = None
        self._child = None
        try:
            self._child = _Child({"mode": "serve", "code": code, "setup": setup or ""}, timeout, mem_bytes)
            out = self._child.recv()
            _note_isolation(out.get("iso") if isinstance(out, dict) else None)
            load = out.get("load") if isinstance(out, dict) else "malformed reply"
            self.load = None if load is None else str(load)
        except SandboxTimeout:
            self.load = "timed out"
        except SandboxClosed as e:
            self.load = str(e)
        except Exception as e:  # noqa: BLE001
            self.load = f"could not run: {e}"
        if self.load is not None:
            self._dead = self.load
            self.close()

    def call(self, name, args=(), kwargs=None):
        self.calls += 1
        if self._dead:
            raise CandidateError(self._dead)
        if self.calls > self.max_calls:
            self._dead = "too many calls"
            self.close()
            raise CandidateError(self._dead)
        try:
            payload = {"fn": name, "args": _enc(list(args)), "kwargs": _enc(dict(kwargs or {}))}
        except TypeError as e:
            raise CandidateError(f"argument is not plain data: {e}")
        try:
            self._child.send(payload)
            r = self._child.recv()
        except (SandboxTimeout, SandboxClosed) as e:
            self._dead = str(e)
            self.close()
            raise CandidateError(self._dead)
        if isinstance(r, dict) and "ok" in r:
            try:
                return _dec(r["ok"])
            except Exception:  # noqa: BLE001
                raise CandidateError("malformed value")
        raise CandidateError(str(r.get("err") if isinstance(r, dict) else "malformed reply"))

    def close(self):
        if self._child is not None:
            self._child.close()
            self._child = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def code_runs(code, timeout=5.0):
    """True if ``code`` runs to the end in a child process without raising (exit calls count as failure)."""
    load, _ = run_calls(code, [], timeout=timeout)
    return load is None


def defines(code, name):
    """True if ``code`` has a ``def name(`` line."""
    return bool(name) and re.search(r"^\s*(async\s+)?def\s+" + re.escape(name) + r"\s*\(", code, re.M) is not None


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


# --- HumanEval check(candidate) programs ------------------------------------------------------

class HumanEvalTests:
    """HumanEval's ``test`` program (defines ``check(candidate)``) plus its ``entry_point``, and the
    task prompt when available (helpers such as ``poly`` or ``encode_cyclic`` are taken from it)."""

    def __init__(self, test, entry_point, prompt=None):
        self.test = str(test)
        self.entry_point = str(entry_point)
        self.prompt = prompt

    def __bool__(self):
        return True

    def __repr__(self):
        return f"HumanEvalTests(entry_point={self.entry_point!r})"


def _prompt_text(prompt):
    if prompt is None:
        return ""
    if isinstance(prompt, (list, tuple)):
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in prompt)
    return str(prompt)


def prompt_helpers(prompt, entry_point):
    """Imports and helper definitions from the task prompt's code (everything but the entry point).
    The prompt may be wrapped in a chat template: the longest parseable code region is used."""
    lines = _prompt_text(prompt).splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"(from \S+ import |import |def |async def |class |@)", l)]
    if not starts:
        return ""
    s = starts[0]
    tree = None
    for e in range(len(lines), s, -1):
        try:
            tree = ast.parse("\n".join(lines[s:e]))
            break
        except SyntaxError:
            continue
    if tree is None:
        return ""
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name != entry_point)]
    return ast.unparse(ast.Module(body=keep, type_ignores=[])) if keep else ""


class _AssertRecorder(ast.NodeTransformer):
    """``assert T, M`` -> ``try: ok = bool(T) except Exception: ok = False`` + ``__aligntune_rec(ok)``.
    Every assert is checked (a failure does not stop ``check``), and each counts once."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return node

    def visit_Assert(self, node):
        ok = "__aligntune_ok"
        body = ast.parse(
            f"try:\n    {ok} = bool(__aligntune_T)\nexcept Exception:\n    {ok} = False\n__aligntune_rec({ok})"
        ).body
        body[0].body[0].value.args[0] = node.test
        for n in body:
            ast.copy_location(n, node)
        return body


class _CandidateProxy:
    def __init__(self, session, name):
        self._session, self._name = session, name

    def __call__(self, *args, **kwargs):
        return self._session.call(self._name, args, kwargs)


_HELPER_WARNED = []


def grade_humaneval_detail(code, tests, setup="", timeout=5.0):
    """Run HumanEval's ``check(candidate)`` here with ``candidate`` forwarded to a sandboxed child.

    Returns ``(load_error, list_of_bools)``, one bool per assert that was reached after at least one
    new call to the candidate (so ``assert True`` lines, which pass for anything, are not counted).
    An exception outside an assert (e.g. the candidate raising in ``solution = candidate(x)``) fails
    the whole task.
    """
    entry = tests.entry_point
    trusted = {"__builtins__": builtins, "__name__": "__aligntune_check__",
               "__aligntune_rec": lambda ok: None}  # module-level asserts in the test file: not counted
    helpers = prompt_helpers(tests.prompt, entry)
    if helpers:
        try:
            exec(compile(helpers, "<prompt>", "exec"), trusted)
        except Exception:  # noqa: BLE001
            pass
    try:
        tree = ast.fix_missing_locations(_AssertRecorder().visit(ast.parse(tests.test)))
        exec(compile(tree, "<check>", "exec"), trusted)
        check = trusted["check"]
    except Exception as e:  # noqa: BLE001
        return f"bad test program: {e}", [False]
    records, state = [], {"seen": 0}
    session = CandidateSession(code, setup=setup, timeout=timeout)
    if session.load is not None:
        return session.load, [False]
    proxy = _CandidateProxy(session, entry)

    def rec(ok):
        if session.calls > state["seen"]:
            state["seen"] = session.calls
            records.append(bool(ok))

    trusted["__aligntune_rec"] = rec
    trusted[entry] = proxy
    try:
        check(proxy)
    except NameError as e:
        if not _HELPER_WARNED:
            _HELPER_WARNED.append(True)
            logger.warning(f"HumanEval check() needs a helper the prompt did not provide ({e}); "
                           f"pass the raw task prompt so helpers can be loaded from it.")
        return None, [False]
    except Exception:  # noqa: BLE001
        return None, [False]
    finally:
        session.close()
    return None, records or [False]


def humaneval_assertions(test_src, entry_point):
    """HumanEval's ``check(candidate)`` program -> its single-line asserts, calling the entry point.
    Kept for callers that need assert strings; graders use :func:`grade_humaneval_detail`."""
    out = []
    for line in str(test_src).splitlines():
        s = line.strip()
        if s.startswith("assert ") and not s.startswith("assert True"):
            out.append(s.replace("candidate(", f"{entry_point}("))
    return out


def grade_tests_detail(code, tests, setup="", timeout=5.0, entry=None):
    """Grade against assert strings or HumanEvalTests. Returns ``(load_error, list_of_bools)``."""
    if isinstance(tests, HumanEvalTests):
        return grade_humaneval_detail(code, tests, setup=setup, timeout=timeout)
    return grade_assertions_detail(code, tests, setup=setup, timeout=timeout, entry=entry)


def tests_from_kwargs(kwargs, default=None):
    """Find tests in whatever the dataset row provides: ``test_cases`` (AlignTune's own key),
    MBPP's ``test_list``, or HumanEval's ``test`` + ``entry_point`` (returned as HumanEvalTests)."""
    tc = kwargs.get("test_cases")
    if tc:
        if isinstance(tc, str) and kwargs.get("entry_point") and "def check(" in tc:
            return HumanEvalTests(tc, kwargs["entry_point"], kwargs.get("prompt"))
        return list(tc) if isinstance(tc, (list, tuple)) else tc
    if kwargs.get("test_list"):
        return list(kwargs["test_list"])
    if kwargs.get("test") and kwargs.get("entry_point"):
        return HumanEvalTests(kwargs["test"], kwargs["entry_point"], kwargs.get("prompt"))
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
