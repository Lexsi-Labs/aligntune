"""Kernel confinement for the child process that runs model-written code (Linux).

The graders start the candidate in a fresh ``sys.executable -I`` process with limits on
CPU, memory and file size. This adds what limits can't: before the candidate runs, the
child confines itself with

* seccomp: no ``kill``/``tgkill``/``pidfd_send_signal`` (a completion could otherwise
  SIGKILL the trainer through ``os.getppid()``), no ``ptrace``, no sockets, no
  ``io_uring_setup``, and no new processes or programs (``fork``/``clone``/``execve``);
* Landlock: writes only under its own temporary directory, reads only from the Python
  installation and system directories (not the user's home), and no TCP on ABI 4+.

Both are best effort. Without them (non-Linux, an old kernel, some containers) the child
is still a separate process with resource limits, and a one-time warning says so. Set
``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1`` to refuse to run candidate code (score 0)
where the kernel offers neither. Threads can't be started in confined code; BLAS
libraries are told to use one thread so ``numpy`` still imports.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Dict

logger = logging.getLogger(__name__)

# BLAS libraries start threads on import, which the confined child may not do.
SINGLE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
}

# Defines ``isolate(tmp, require)`` for the child; ``tmp`` is the one writable directory.
ISOLATE_SOURCE = "import os, sys\n" + r'''
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


def require_isolation() -> bool:
    return os.environ.get("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "").strip().lower() in ("1", "true", "yes")


_probed: Dict[str, bool] = {}


def note_isolation_once() -> None:
    """Warn once, the first time a candidate is about to run, if the kernel can't fully confine it."""
    if _probed:
        return
    _probed["done"] = True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(
                [sys.executable, "-I", "-c", ISOLATE_SOURCE + "import json\nprint(json.dumps(isolate(os.getcwd(), False)))"],
                cwd=tmp, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL,
            ).stdout
        state = json.loads(out.strip().splitlines()[-1])
    except Exception:
        state = {}
    if not (state.get("seccomp") and state.get("landlock")):
        logger.warning(
            "Code-reward sandbox: kernel isolation is partial (seccomp=%s, landlock ABI=%s). Candidate "
            "code still runs in a separate process with limits, but may be able to signal other "
            "processes or read/write the user's files. Set ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1 to refuse.",
            state.get("seccomp"), state.get("landlock"))
