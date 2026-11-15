"""The child that runs model-written code confines itself (Linux: seccomp + Landlock).

AlignTune starts candidate children in two places: ``_code_grader.run_candidate`` (asserts, input/output tests)
and ``_code_sandbox`` (a HumanEval ``check(candidate)`` that runs here while the candidate answers calls from its
own process). Every probe goes through both. The escapes are tried with the real parent process (this pytest process): reading its memory through
``/proc/<ppid>/mem``, ``process_vm_readv`` and ``ptrace``, and signalling it. With confinement active none may
work, and a candidate that tries scores 0. A control run with the confinement switched off shows the probe
really reads the parent's memory (it is skipped where the kernel's Yama setting blocks the read anyway).
Honest candidates must score what they did before. On other platforms only the process limits apply, so the
Linux-only cases are skipped and the rest checks that behaviour there is unchanged.
"""

import ctypes
import json
import logging
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

from aligntune.rewards import _code_confine, _code_grader, _code_sandbox
from aligntune.rewards.registry import RewardRegistry

LINUX = sys.platform.startswith("linux")


def _isolation_state():
    out = subprocess.run(
        [sys.executable, "-I", "-c", _code_confine.ISOLATE_SOURCE + "import json\nprint(json.dumps(isolate(os.getcwd(), False)))"],
        cwd=tempfile.gettempdir(), capture_output=True, text=True,
    ).stdout
    state = json.loads(out.strip().splitlines()[-1])
    return bool(state.get("seccomp")), int(state.get("landlock") or 0)


SECCOMP, LANDLOCK = _isolation_state() if LINUX else (False, 0)
# Landlock also refuses ptrace and process_vm_readv between domains; signals only from ABI 6 on.
BLOCKS_MEMORY_READS = LANDLOCK >= 1
BLOCKS_PTRACE = SECCOMP or LANDLOCK >= 1
BLOCKS_SIGNALS = SECCOMP or LANDLOCK >= 6


# ALIGNTUNE_EXPECT_ISOLATION=1 (for CI and the Linux runs): the module fails instead of skipping its Linux
# tests when seccomp or Landlock are not active here (conftest.py also turns any other skip into a failure).
if os.environ.get("ALIGNTUNE_EXPECT_ISOLATION", "").strip().lower() in ("1", "true", "yes") and not (LINUX and SECCOMP and LANDLOCK):
    raise RuntimeError(
        "ALIGNTUNE_EXPECT_ISOLATION is set but seccomp=%s and Landlock ABI=%s are not both active here (Linux: %s)"
        % (SECCOMP, LANDLOCK, LINUX)
    )


def _reward():
    return RewardRegistry.get_reward_function("code_execution", {"type": "code_execution", "weight": 1.0,
                                                                  "params": {"timeout": 30.0}})


def _fenced(code):
    return "```python\n" + code + "\n```"


# A check() with a loop can't be split into asserts, so it runs here and forwards each call to a _code_sandbox child.
CHECK = "def check(candidate):\n    for _ in range(1):\n        assert candidate(1) == %r\n"


def _scores(code, expected):
    """The same completion through both children: an assert (``_code_grader``) and a HumanEval check (``_code_sandbox``)."""
    text = _fenced(code)
    return (
        _reward().compute(text, test_cases=[f"assert f(1) == {expected!r}"]),
        _reward().compute(text, test=CHECK % (expected,), entry_point="f"),
    )


def _rc(code, timeout, exprs=("f(1)",), call=None):
    """``_code_grader.run_candidate``: the values, or None unless the child got through."""
    items = [{"call": call}] if call is not None else [{"expr": e} for e in exprs]
    run = _code_grader.run_candidate(code, timeout, items)
    return run.values if run.status == "ok" else None


def _rc_many(code, jobs, timeout):
    """The ``_code_sandbox`` child: one list of values per job, or None for every job unless the candidate loaded."""
    flat = [e for job in jobs for e in job]
    load, results = _code_sandbox.run_calls(code, flat, timeout=timeout)
    if load is not None:
        return [None] * len(jobs)
    out, i = [], 0
    for job in jobs:
        out.append([v if kind == "ok" else _code_grader.FAILED for kind, v in results[i:i + len(job)]])
        i += len(job)
    return out


# The two children's sources, and the lines the tests below swap out: the isolate() call and the report.
CHILDREN = {
    (_code_grader, "_HARNESS"): ('state = isolate(os.getcwd(), job.get("require", False))', "state = %s",
                                 "    if report >= 0:\n        os.write(report, json.dumps(state).encode())\n        os.close(report)\n",
                                 "    os.close(report)\n"),
    (_code_sandbox, "_CHILD"): ('iso = isolate(req["tmp"], req["require"])', "iso = %s",
                                "if _report >= 0:\n    os.write(_report, json.dumps(iso).encode())\n    os.close(_report)\n",
                                "os.close(_report)\n"),
}
_ORIGINAL = {key: getattr(*key) for key in CHILDREN}


def _patch_children(monkeypatch, edit):
    """Set both children's source to ``edit(source, call, stub, report, no_report)``."""
    for (module, attr), parts in CHILDREN.items():
        source = _ORIGINAL[(module, attr)]
        assert parts[0] in source and parts[2] in source
        monkeypatch.setattr(module, attr, edit(source, *parts))


def _replace_in_children(monkeypatch, old, new):
    _patch_children(monkeypatch, lambda source, *parts: (source.replace(old, new) if old in source else pytest.fail(old)))


def _patch_both(monkeypatch, name, value):
    """A name both children's starters import from _code_confine."""
    monkeypatch.setattr(_code_grader, name, value)
    monkeypatch.setattr(_code_sandbox, name, value)


@pytest.fixture
def unconfined(monkeypatch):
    """Both children with the isolate() call replaced by a stub that reports no isolation (and the opt-out that lets a
    Linux grader run a candidate without seccomp, which is refused by default)."""
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")
    _patch_children(monkeypatch, lambda source, call, stub, *_: source.replace(call, stub % '{"seccomp": False, "landlock": 0}'))


@pytest.fixture(scope="module", autouse=True)
def parent_is_ptraceable():
    """Let any same-user process ptrace this one (PR_SET_PTRACER_ANY), as on a kernel without Yama (WSL2, many
    containers). Without it Yama may stop the probes by itself and the confined tests would prove nothing.
    Only this process is affected, only for this module."""
    if not LINUX:
        yield
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(0x59616D61, ctypes.c_ulong(-1 & (2**64 - 1)), 0, 0, 0)  # PR_SET_PTRACER, PR_SET_PTRACER_ANY
    yield
    libc.prctl(0x59616D61, 0, 0, 0, 0)


@pytest.fixture(scope="module")
def sentinel():
    """A value that lives only in this process's memory, as an expected answer does in the trainer."""
    value = "ALIGNTUNE_SENTINEL_" + os.urandom(16).hex()
    buf = ctypes.create_string_buffer(value.encode())  # a raw copy at a known address, for process_vm_readv
    yield value, ctypes.addressof(buf)  # one value for the whole module, so a stale copy in memory is the same value
    del buf


# --- the seccomp program itself, run through a tiny BPF interpreter (any platform, both architectures) ----------------

ALLOW, EPERM = 0x7FFF0000, 0x00050001
X86, ARM = 0xC000003E, 0xC00000B7
def _tables():
    ns = {}
    exec(_code_confine.ISOLATE_SOURCE, ns)  # the deny list lives in the child's source, in one place
    return ns


TABLES = _tables()
DENY = {name: nrs for _, group in TABLES["SECCOMP_DENY"] for name, nrs in group.items()}  # name: (x86_64, aarch64)
SELF_ONLY = TABLES["SECCOMP_SELF_ONLY"]  # name: (x86_64, aarch64, pid argument)
CMD_DENY = TABLES["SECCOMP_CMD_DENY"]  # name: (x86_64, aarch64, command argument, commands)
# What Python, numpy and the C library really call (read write mmap ... getrandom futex clock_gettime getrlimit).
PLAIN = {
    "x86_64": [0, 1, 2, 3, 9, 10, 11, 12, 13, 25, 28, 35, 39, 63, 97, 110, 160, 202, 204, 217, 228, 231, 257, 318],
    "aarch64": [63, 64, 56, 57, 222, 226, 215, 214, 134, 216, 233, 101, 172, 160, 163, 173, 164, 98, 123, 61, 113, 94, 278],
}


def _programs():
    return {"x86_64": (X86, TABLES["seccomp_program"]("x86_64")), "aarch64": (ARM, TABLES["seccomp_program"]("aarch64"))}


def _run_bpf(prog, arch, nr, args=()):
    data = {0: nr & 0xFFFFFFFF, 4: arch}
    for i, a in enumerate(args):
        data[16 + 8 * i], data[20 + 8 * i] = a & 0xFFFFFFFF, (a >> 32) & 0xFFFFFFFF
    acc, pc = 0, 0
    while True:
        code, jt, jf, k = prog[pc]
        if code == 0x20:
            acc = data.get(k, 0)
        elif code == 0x15:
            pc += jt if acc == k else jf
        elif code == 0x35:
            pc += jt if acc >= k else jf
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")
        pc += 1
        assert pc < len(prog), "ran off the end of the program"


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_seccomp_program_decisions(machine):
    arch, prog = _programs()[machine]
    col = 0 if machine == "x86_64" else 1
    assert len(prog) < 255
    for nr in PLAIN[machine]:
        assert _run_bpf(prog, arch, nr, (0, 0, 0)) == ALLOW, nr
    refused = {nrs[col] for nrs in DENY.values() if nrs[col] is not None}
    for name, nrs in DENY.items():
        if nrs[col] is not None:
            assert _run_bpf(prog, arch, nrs[col], (0, 0, 0)) == EPERM, name
    # nothing else is refused by number alone: the other calls are decided by their arguments below
    arg_decided = {v[col] for v in SELF_ONLY.values()} | {v[col] for v in CMD_DENY.values()}
    for nr in range(0, 460):
        if nr not in refused and nr not in arg_decided:
            assert _run_bpf(prog, arch, nr, (0, 0, 0)) == ALLOW, nr
    for name, (*nrs, i) in SELF_ONLY.items():
        nr = nrs[col]
        assert _run_bpf(prog, arch, nr, (0,)) == ALLOW, name  # the caller itself
        for other in (1, 1234, 0xFFFFFFFF, 1 << 32, -1 & (2**64 - 1)):
            assert _run_bpf(prog, arch, nr, (other,)) == EPERM, (name, other)
    for name, (*nrs, i, cmds_bad) in CMD_DENY.items():
        nr = nrs[col]
        assert _run_bpf(prog, arch, nr, (3, 3, 0)) == ALLOW  # F_GETFL, a termios ioctl would be 0x5401
        assert _run_bpf(prog, arch, nr, (3, 0x5401, 0)) == ALLOW
        for cmd in cmds_bad:
            assert _run_bpf(prog, arch, nr, (3, cmd, 0)) == EPERM, cmd
            assert _run_bpf(prog, arch, nr, (3, cmd | (1 << 40), 0)) == EPERM, cmd  # the kernel reads 32 bits
    fcntl = CMD_DENY["fcntl"][col]
    assert _run_bpf(prog, arch, fcntl, (3, 4, 0o4000)) == ALLOW  # F_SETFL
    assert _run_bpf(prog, arch, fcntl, (3, 1, 1)) == ALLOW  # F_SETFD


def test_seccomp_program_refuses_the_wrong_architecture_and_x32():
    arch, prog = _programs()["x86_64"]
    assert _run_bpf(prog, ARM, 0, (0,)) == EPERM
    assert _run_bpf(prog, arch, 0x40000000 | 0, (0,)) == EPERM  # x32 numbers reach the same kernel code


# Calls newer than some distributions' kernel headers: a header that lacks them is old, not wrong.
NEWER_THAN_OLD_HEADERS = {
    "pidfd_send_signal", "pidfd_open", "pidfd_getfd", "process_madvise", "process_mrelease", "clone3", "io_uring_setup",
    "io_uring_enter", "io_uring_register", "open_tree", "move_mount", "fsopen", "fsconfig", "fsmount", "fspick",
    "mount_setattr", "quotactl_fd", "kexec_file_load", "finit_module", "userfaultfd", "bpf", "kcmp", "sched_setattr",
    "execveat", "perf_event_open", "fanotify_init", "name_to_handle_at", "open_by_handle_at", "setns", "prlimit64",
}


@pytest.mark.skipif(not LINUX, reason="kernel headers")
def test_syscall_numbers_match_the_kernel_headers_where_installed():
    import re

    every = {n: nrs for n, nrs in DENY.items()}
    every.update({n: v[:2] for n, v in SELF_ONLY.items()})
    every.update({n: v[:2] for n, v in CMD_DENY.items()})
    found, skipped = 0, set()
    for path, col in (("/usr/include/x86_64-linux-gnu/asm/unistd_64.h", 0), ("/usr/include/asm/unistd_64.h", 0),
                      ("/usr/include/asm-generic/unistd.h", 1)):
        if not os.path.exists(path):
            continue
        text = open(path).read()
        for name, nrs in every.items():
            m = re.search(r"#define __NR3264_%s\s+(\d+)" % name, text) or re.search(r"#define __NR_%s\s+(\d+)" % name, text)
            if m is None:
                if nrs[col] is None:
                    continue  # fork, vfork, chown and lchown only exist on x86_64
                assert name in NEWER_THAN_OLD_HEADERS, (path, name)  # an old header may lack a new call, nothing else
                skipped.add(name)
                continue
            assert int(m.group(1)) == nrs[col], (path, name)
            found += 1
    if not found:
        pytest.skip("no kernel headers installed")


# --- the probes: each returns the sentinel if it got it, and raises if the kernel said no -----------------------

SCAN_PROC_MEM = textwrap.dedent('''
    import os, re

    def _read():
        pid = os.getppid()
        pat = re.compile(rb"ALIGNTUNE_SENT" + rb"INEL_[0-9a-f]{32}")
        with open("/proc/%d/maps" % pid) as maps, open("/proc/%d/mem" % pid, "rb", 0) as mem:
            for line in maps:
                parts = line.split()
                if not parts[1].startswith("rw") or (len(parts) > 5 and parts[5] != "[heap]"):
                    continue  # anonymous memory and the heap hold Python objects
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                while lo < hi:
                    n = min(hi - lo, 1 << 24)
                    try:
                        mem.seek(lo)
                        hit = pat.search(mem.read(n))
                    except (OSError, ValueError, OverflowError):
                        break
                    if hit:
                        return hit.group(0).decode()
                    lo += max(n - 64, 1)
        raise LookupError("not found")

    def f(x):
        return _read()
''')

PREAD_AT_ADDRESS = textwrap.dedent('''
    import os

    def f(x):
        fd = os.open("/proc/%%d/mem" %% os.getppid(), os.O_RDONLY)
        return os.pread(fd, %(size)d, %(addr)d).decode()
''')

PROCESS_VM_READV = textwrap.dedent('''
    import ctypes, os

    class IOV(ctypes.Structure):
        _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]

    def f(x):
        libc = ctypes.CDLL(None, use_errno=True)
        buf = ctypes.create_string_buffer(%(size)d)
        local, remote = IOV(ctypes.addressof(buf), %(size)d), IOV(%(addr)d, %(size)d)
        n = libc.process_vm_readv(os.getppid(), ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
        if n <= 0:
            raise OSError(ctypes.get_errno(), "process_vm_readv")
        return buf.raw[:n].decode()
''')

# PTRACE_SEIZE does not stop the parent, so a successful attach is harmless here and is released when the child exits.
PTRACE = textwrap.dedent('''
    import ctypes, os

    def f(x):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
        if libc.ptrace(0x4206, os.getppid(), None, None) != 0:  # PTRACE_SEIZE
            raise OSError(ctypes.get_errno(), "ptrace")
        return "attached"
''')

def _ctypes_call(call):
    return "import ctypes, os, resource\nlibc = ctypes.CDLL(None, use_errno=True)\ndef f(x):\n" + textwrap.indent(call, "    ")


KILL_PARENT = {
    "os.kill": "os.kill(os.getppid(), signal.SIGUSR1)",
    "os.killpg": "os.killpg(os.getpgid(os.getppid()), signal.SIGUSR1)",
    "pidfd_send_signal": "signal.pidfd_send_signal(os.pidfd_open(os.getppid()), signal.SIGUSR1)",
}


@pytest.fixture
def no_hold(monkeypatch):
    """The trainer's non-dumpable hold alone would stop these reads; switch it off to test the filter and Landlock."""
    _patch_both(monkeypatch, "hold_parent", lambda: False)


@pytest.fixture
def landlock_only(monkeypatch):
    """Both layers' state with seccomp left out: the child skips its filter (and reports so)."""
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")  # a Linux grader refuses a child without seccomp by default
    _replace_in_children(monkeypatch, "prog = seccomp_program(platform.machine())", "prog = None")


@pytest.mark.skipif(not (LINUX and BLOCKS_MEMORY_READS), reason="needs Landlock to refuse reads of another process's memory")
def test_the_parents_memory_cannot_be_read_through_proc(sentinel, no_hold):
    value, _ = sentinel
    assert _scores(SCAN_PROC_MEM, value) == (0.0, 0.0)


@pytest.mark.skipif(not (LINUX and BLOCKS_MEMORY_READS), reason="needs Landlock to refuse reads of another process's memory")
def test_the_parents_memory_cannot_be_read_at_a_known_address_through_proc(sentinel, no_hold):
    value, addr = sentinel
    assert _scores(PREAD_AT_ADDRESS % {"addr": addr, "size": len(value)}, value) == (0.0, 0.0)


@pytest.mark.skipif(not (LINUX and BLOCKS_PTRACE), reason="no seccomp or Landlock here")
def test_the_parents_memory_cannot_be_read_with_process_vm_readv(sentinel, no_hold):
    value, addr = sentinel
    code = PROCESS_VM_READV % {"addr": addr, "size": len(value)}
    assert _scores(code, value) == (0.0, 0.0)


@pytest.mark.skipif(not (LINUX and BLOCKS_PTRACE), reason="no seccomp or Landlock here")
def test_the_parent_cannot_be_traced(no_hold):
    assert _scores(PTRACE, "attached") == (0.0, 0.0)


@pytest.mark.skipif(not LINUX, reason="ptrace is Linux-only")
def test_control_unconfined_candidate_can_trace_the_parent(unconfined, no_hold):
    """Without confinement and without the hold the same attach works (given a kernel that allows it), so the blocked
    result above is the filter's or Landlock's doing."""
    if _scores(PTRACE, "attached") != (1.0, 1.0):
        pytest.skip("this kernel refuses ptrace attach even without protection (Yama or a container)")


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_landlock_alone_blocks_every_memory_read_and_ptrace(sentinel, landlock_only, no_hold):
    """No seccomp filter (the child reports none) and no hold: Landlock's own ptrace rule refuses all four."""
    value, addr = sentinel
    for code in (SCAN_PROC_MEM, PREAD_AT_ADDRESS % {"addr": addr, "size": len(value)},
                 PROCESS_VM_READV % {"addr": addr, "size": len(value)}):
        assert _scores(code, value) == (0.0, 0.0)
    assert _scores(PTRACE, "attached") == (0.0, 0.0)


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
def test_seccomp_alone_blocks_process_vm_readv_and_ptrace(sentinel, seccomp_only, no_hold):
    """No Landlock and no hold: the filter still refuses process_vm_readv and ptrace (reads through /proc are Landlock's or the hold's)."""
    value, addr = sentinel
    assert _scores(PROCESS_VM_READV % {"addr": addr, "size": len(value)}, value) == (0.0, 0.0)
    assert _scores(PTRACE, "attached") == (0.0, 0.0)


@pytest.mark.skipif(not (LINUX and BLOCKS_SIGNALS), reason="no seccomp here")
@pytest.mark.parametrize("how", sorted(KILL_PARENT))
def test_the_parent_cannot_be_signalled(how):
    if how == "pidfd_send_signal" and not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("signal.pidfd_send_signal is not available in this Python build")
    hits = []
    old = signal.signal(signal.SIGUSR1, lambda *a: hits.append(1))
    try:
        code = f"import os, signal\n{KILL_PARENT[how]}\ndef f(x):\n    return 1"
        assert _scores(code, 1) == (0.0, 0.0)  # the signal call raises, so the candidate never gets to answer
    finally:
        signal.signal(signal.SIGUSR1, old)
    assert hits == []


# --- syscalls that act on OTHER processes (same user, so the kernel would allow them) ----------------------------

ARCH = 0 if platform.machine() in ("x86_64", "AMD64") else 1
# Never called without the filter in place: they would start a process or change the machine.
NOT_CALLED_UNCONFINED = {"clone", "clone3", "fork", "vfork", "acct", "msgget", "reboot", "kexec_load", "kexec_file_load",
                         "swapon", "swapoff", "init_module", "finit_module", "delete_module", "pivot_root", "chroot",
                         "mount", "umount2", "setns", "unshare", "setuid", "setgid", "setreuid", "setregid",
                         "setresuid", "setresgid", "setfsuid", "setfsgid", "setgroups", "capset"}
# Calls that can answer ENOSYS on a kernel built without them.
MAY_BE_MISSING = {"kexec_load", "kexec_file_load", "init_module", "finit_module", "delete_module", "acct", "userfaultfd",
                  "fanotify_init", "quotactl_fd", "mq_open", "mq_unlink", "mq_timedsend", "mq_timedreceive", "mq_notify",
                  "mq_getsetattr", "io_uring_setup", "io_uring_enter", "io_uring_register", "bpf", "perf_event_open",
                  "swapon", "swapoff", "mount_setattr", "move_mount", "open_tree", "fsopen", "fsconfig", "fsmount",
                  "fspick", "process_mrelease", "process_madvise", "pidfd_send_signal", "pidfd_open", "pidfd_getfd",
                  "quotactl", "syslog", "kcmp", "open_by_handle_at", "name_to_handle_at"}
SYSCALL_PROBE = (
    "import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
    "def f(x):\n    out = []\n    for nr in %r:\n        rc = libc.syscall(nr, 0, 0, 0, 0, 0, 0)\n"
    "        out.append(ctypes.get_errno() if rc == -1 else 0)\n    return out\n"
)


@pytest.mark.skipif(not (LINUX and SECCOMP) or platform.machine() not in ("x86_64", "AMD64", "aarch64", "arm64"),
                    reason="needs seccomp on a supported architecture")
def test_every_listed_syscall_answers_eperm_in_the_confined_child():
    """Each call is made with all-zero arguments through libc.syscall, after the filter is in place (so none of them runs)."""
    names = [n for n, nrs in sorted(DENY.items()) if nrs[ARCH] is not None and n not in ("clone", "clone3", "fork", "vfork")]
    code = SYSCALL_PROBE % ([DENY[n][ARCH] for n in names],)
    out = _rc(code, 30, ("f(1)",))
    assert out is not None
    assert {n: e for n, e in zip(names, out[0]) if e != 1} == {}  # EPERM for all of them


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
def test_pid_and_command_calls_are_refused_for_others_and_allowed_for_the_caller():
    code = textwrap.dedent('''
        import ctypes, os
        libc = ctypes.CDLL(None, use_errno=True)
        def call(nr, *a):
            rc = libc.syscall(nr, *[ctypes.c_long(x) for x in a] + [ctypes.c_long(0)] * (6 - len(a)))
            return -ctypes.get_errno() if rc == -1 else 0
        def f(x):
            out = {}
            for name, nr in %(self_only)r.items():
                out[name + " other"] = call(nr, os.getppid())
            return out
    ''') % {"self_only": {n: v[ARCH] for n, v in SELF_ONLY.items()}}
    out = _rc(code, 20, ("f(1)",))
    assert out is not None and set(out[0].values()) == {-1}  # EPERM for every other process


@pytest.mark.skipif(not LINUX, reason="Linux syscalls")
def test_control_listed_syscalls_exist_on_this_kernel(unconfined):
    """Without the filter a wrong number would answer ENOSYS: so every number is a real call here (or a call this kernel lacks)."""
    names = [n for n, nrs in sorted(DENY.items()) if nrs[ARCH] is not None and n not in NOT_CALLED_UNCONFINED]
    code = SYSCALL_PROBE % ([DENY[n][ARCH] for n in names],)
    out = _rc(code, 30, ("f(1)",))
    assert out is not None
    enosys = {n for n, e in zip(names, out[0]) if e == 38}
    assert enosys <= MAY_BE_MISSING, enosys - MAY_BE_MISSING


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
def test_the_parents_resource_limits_cannot_be_changed():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    code = f"import os, resource\nresource.prlimit(os.getppid(), resource.RLIMIT_NOFILE, ({soft - 1}, {hard}))\ndef f(x):\n    return 1\n"
    assert _scores(code, 1) == (0.0, 0.0)
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (soft, hard)


@pytest.mark.skipif(not LINUX, reason="Linux syscalls")
def test_control_unconfined_candidate_can_change_the_parents_limits(unconfined):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    code = f"import os, resource\nresource.prlimit(os.getppid(), resource.RLIMIT_NOFILE, ({soft - 1}, {hard}))\ndef f(x):\n    return 1\n"
    try:
        assert _scores(code, 1) == (1.0, 1.0)
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == (soft - 1, hard)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
def test_the_parents_scheduling_cannot_be_changed():
    before = (os.sched_getscheduler(os.getpid()), os.sched_getaffinity(0))
    attacks = (
        "os.sched_setscheduler(os.getppid(), os.SCHED_BATCH, os.sched_param(0))",
        "os.sched_setparam(os.getppid(), os.sched_param(0))",
        f"os.sched_setaffinity(os.getppid(), {{{min(before[1])}}})",
    )
    for attack in attacks:
        assert _scores(f"import os\n{attack}\ndef f(x):\n    return 1\n", 1) == (0.0, 0.0), attack
    assert (os.sched_getscheduler(os.getpid()), os.sched_getaffinity(0)) == before


@pytest.mark.skipif(not LINUX, reason="Linux syscalls")
def test_control_unconfined_candidate_can_change_the_parents_scheduling(unconfined):
    before = os.sched_getscheduler(os.getpid())
    try:
        code = "import os\nos.sched_setscheduler(os.getppid(), os.SCHED_BATCH, os.sched_param(0))\ndef f(x):\n    return 1\n"
        assert _scores(code, 1) == (1.0, 1.0)
        assert os.sched_getscheduler(os.getpid()) == os.SCHED_BATCH
    finally:
        os.sched_setscheduler(os.getpid(), before, os.sched_param(0))


FCNTL_OWNER = {
    "F_SETOWN": "fcntl.fcntl(r, fcntl.F_SETOWN, os.getppid())",
    "F_SETOWN_EX": "fcntl.fcntl(r, 15, struct.pack('ii', 1, os.getppid()))",  # F_OWNER_PID
}


def _sigio_candidate(how):
    # the kernel signals the file's owner (here the parent) with SIGUSR1 when the pipe becomes readable
    return (
        "import fcntl, os, struct\nr, w = os.pipe()\n"
        f"{FCNTL_OWNER[how]}\n"
        "fcntl.fcntl(r, 10, 10)  # F_SETSIG, SIGUSR1\n"
        "fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK | os.O_ASYNC)\n"
        "os.write(w, b'x')\n"
        "def f(x):\n    return 1\n"
    )


# FIOSETOWN does the same on files that support it (ttys, sockets); a pipe answers ENOTTY, so compare the errors.
FIOSETOWN_ERRNO = (
    "import fcntl, os, struct\nr, w = os.pipe()\nerr = 0\ntry:\n"
    "    fcntl.ioctl(r, 0x8901, struct.pack('i', os.getppid()))\nexcept OSError as e:\n    err = e.errno\n"
    "def f(x):\n    return err\n"
)


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
def test_fiosetown_is_refused():
    assert _rc(FIOSETOWN_ERRNO, 20, ("f(1)",)) == [1]  # EPERM, from the filter


@pytest.mark.skipif(not LINUX, reason="Linux ioctl")
def test_control_fiosetown_is_not_refused_without_confinement(unconfined):
    out = _rc(FIOSETOWN_ERRNO, 20, ("f(1)",))
    assert out is not None and out != [1]


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
@pytest.mark.parametrize("how", sorted(FCNTL_OWNER))
def test_the_parent_cannot_be_signalled_through_a_file_owner(how):
    hits = []
    old = signal.signal(signal.SIGUSR1, lambda *a: hits.append(1))
    try:
        assert _scores(_sigio_candidate(how), 1) == (0.0, 0.0)
        time.sleep(0.2)
    finally:
        signal.signal(signal.SIGUSR1, old)
    assert hits == []


@pytest.mark.skipif(not LINUX, reason="Linux fcntl")
@pytest.mark.parametrize("how", sorted(FCNTL_OWNER))
def test_control_unconfined_candidate_can_signal_the_parent_through_a_file_owner(how, unconfined):
    hits = []
    old = signal.signal(signal.SIGUSR1, lambda *a: hits.append(1))
    try:
        assert _scores(_sigio_candidate(how), 1) == (1.0, 1.0)
        time.sleep(0.5)
    finally:
        signal.signal(signal.SIGUSR1, old)
    assert hits, "the signal never arrived, so the confined test above proves nothing"


def _can_ptrace_anything():
    """Root, or a process with CAP_SYS_PTRACE, reads any process of its own user whatever the dumpable flag says."""
    try:
        with open("/proc/self/status") as f:
            caps = int(next(line for line in f if line.startswith("CapEff:")).split()[1], 16)
        return bool(caps & (1 << 19))
    except Exception:
        return os.geteuid() == 0


def _dumpable():
    return ctypes.CDLL(None).prctl(3, 0, 0, 0, 0)  # PR_GET_DUMPABLE


# --- the controls: with the confinement off, the same probes do work (when the kernel allows them at all) --------

@pytest.mark.skipif(not LINUX, reason="/proc/<pid>/mem is Linux-only")
def test_control_unconfined_candidate_reads_the_parents_memory(sentinel, unconfined, monkeypatch):
    value, addr = sentinel
    _patch_both(monkeypatch, "hold_parent", lambda: False)  # the hold alone would stop the read too
    if _scores(SCAN_PROC_MEM, value) != (1.0, 1.0):
        pytest.skip("this kernel refuses the read even without confinement (Yama ptrace_scope > 0, or a container)")
    # the probe works, so the reward in the confined tests above being 0 is the confinement's doing
    assert _scores(PROCESS_VM_READV % {"addr": addr, "size": len(value)}, value) == (1.0, 1.0)


@pytest.mark.skipif(not LINUX or _can_ptrace_anything(), reason="needs a user without CAP_SYS_PTRACE (root reads any process)")
def test_the_hold_alone_stops_an_unconfined_candidate_reading_the_parents_memory(sentinel, unconfined, monkeypatch):
    """No seccomp, no Landlock (the child is stubbed): only the trainer's non-dumpable flag stands in the way."""
    value, addr = sentinel
    request = PREAD_AT_ADDRESS % {"addr": addr, "size": len(value)}
    with monkeypatch.context() as m:
        m.setattr(_code_grader, "hold_parent", lambda: False)
        m.setattr(_code_sandbox, "hold_parent", lambda: False)
        if _scores(request, value) != (1.0, 1.0):
            pytest.skip("this kernel refuses the read even without any protection (Yama ptrace_scope > 0, or a container)")
    assert _scores(request, value) == (0.0, 0.0)


@pytest.mark.skipif(not LINUX, reason="signals to the parent are blocked by seccomp (Linux)")
def test_control_unconfined_candidate_can_signal_the_parent(unconfined):
    hits = []
    old = signal.signal(signal.SIGUSR1, lambda *a: hits.append(1))
    try:
        code = "import os, signal\nos.kill(os.getppid(), signal.SIGUSR1)\ndef f(x):\n    return 1"
        assert _scores(code, 1) == (1.0, 1.0)
    finally:
        signal.signal(signal.SIGUSR1, old)
    assert hits, "the signal never arrived, so the confined test above proves nothing"


# --- without Landlock: the grading process is non-dumpable while a candidate runs ---------------------------------

@pytest.fixture
def seccomp_only(monkeypatch):
    """The state of a kernel without Landlock: the child skips its Landlock rules (and reports so)."""
    _replace_in_children(monkeypatch, "abi = sc(444, ctypes.c_void_p(0), 0, 1)", "abi = 0")


needs_unprivileged_seccomp = pytest.mark.skipif(
    not (LINUX and SECCOMP) or (LINUX and _can_ptrace_anything()),
    reason="needs seccomp and a user without CAP_SYS_PTRACE (root reads any process regardless)",
)


@needs_unprivileged_seccomp
def test_control_without_landlock_and_without_the_hold_the_parent_is_readable(sentinel, seccomp_only, monkeypatch):
    value, addr = sentinel
    _patch_both(monkeypatch, "hold_parent", lambda: False)
    if _scores(PREAD_AT_ADDRESS % {"addr": addr, "size": len(value)}, value) != (1.0, 1.0):
        pytest.skip("this kernel refuses the read even without Landlock (Yama ptrace_scope > 0 or similar)")


@needs_unprivileged_seccomp
def test_without_landlock_the_parent_is_not_readable_while_a_candidate_runs(sentinel, seccomp_only):
    value, addr = sentinel
    assert _scores(PREAD_AT_ADDRESS % {"addr": addr, "size": len(value)}, value) == (0.0, 0.0)
    assert _scores(SCAN_PROC_MEM, value) == (0.0, 0.0)


@pytest.mark.skipif(not LINUX, reason="PR_SET_DUMPABLE is Linux-only")
def test_the_dumpable_flag_is_restored_after_graders_run_concurrently(seccomp_only):
    import concurrent.futures

    before = _dumpable()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        scores = list(pool.map(lambda _: _scores(HONEST, 2), range(4)))
    assert scores == [(1.0, 1.0)] * 4
    assert _dumpable() == before
    assert _code_confine._hold == {"n": 0, "restore": False}


@pytest.mark.skipif(not LINUX, reason="PR_SET_DUMPABLE is Linux-only")
def test_the_hold_is_released_even_when_the_candidate_hangs(seccomp_only):
    before = _dumpable()
    assert _rc("while True: pass", 1, ("1",)) is None
    assert _dumpable() == before


# --- the hold on the trainer's dumpable flag: taken whatever the kernel offers, counted, restored -----------------

SLEEPER = "import time\ntime.sleep(%s)\ndef f(x):\n    return 1"


@pytest.mark.skipif(not LINUX, reason="PR_SET_DUMPABLE is Linux-only")
def test_the_hold_is_taken_and_restored_with_the_real_kernel_state():
    """Not only where Landlock is missing: on a host that has both layers the flag drops for the run too."""
    import threading

    before = _dumpable()
    seen = []
    worker = threading.Thread(target=lambda: seen.append(_rc(SLEEPER % 1.5, 30, ("f(1)",))))
    worker.start()
    during = set()
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        during.add(_dumpable())
        time.sleep(0.02)
    worker.join()
    assert seen == [[1]]
    assert 0 in during  # dropped while the child was alive
    assert _dumpable() == before and _code_confine._hold == {"n": 0, "restore": False}


@pytest.mark.skipif(not LINUX, reason="PR_SET_DUMPABLE is Linux-only")
def test_nested_and_concurrent_candidates_keep_the_count_right():
    import threading

    assert _code_confine.hold_parent() and _code_confine.hold_parent()  # nested holds
    assert _dumpable() == 0 and _code_confine._hold["n"] == 2
    _code_confine.release_parent()
    assert _dumpable() == 0  # still held by the outer one
    results, samples = [], []
    threads = [threading.Thread(target=lambda d=d: results.append(_rc(SLEEPER % d, 30, ("f(1)",))))
               for d in (1.5, 1.8, 2.1)]
    for t in threads:
        t.start()
    time.sleep(0.6)
    for _ in range(25):  # all three children are alive in this window
        samples.append(_dumpable())
        time.sleep(0.02)
    for t in threads:
        t.join()
    assert results == [[1]] * 3 and set(samples) == {0}
    assert _dumpable() == 0  # the outer hold is still there
    _code_confine.release_parent()
    assert _dumpable() == 1 and _code_confine._hold == {"n": 0, "restore": False}


@pytest.mark.skipif(not LINUX, reason="PR_SET_DUMPABLE is Linux-only")
def test_a_process_that_was_not_dumpable_to_begin_with_stays_so():
    libc = ctypes.CDLL(None)
    libc.prctl(4, 0, 0, 0, 0)
    try:
        assert _code_confine.hold_parent()
        _code_confine.release_parent()
        assert _dumpable() == 0  # never flipped back on what we did not flip off
    finally:
        libc.prctl(4, 1, 0, 0, 0)


# --- ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION -------------------------------------------------------------------------

HONEST = "def f(x):\n    return x + 1"


def _pretend_isolation(monkeypatch, seccomp, landlock, allow_no_seccomp=True):
    """What the child reports (and checks itself against): both children with isolate() replaced by a fixed state.
    A Linux grader refuses a child without seccomp by default; ``allow_no_seccomp`` sets the opt-out so the tests
    that pretend one layer is missing can look at the other behaviour."""
    if allow_no_seccomp:
        monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")
    state = '{"seccomp": %s, "landlock": %d}' % (bool(seccomp), 4 if landlock else 0)
    _patch_children(monkeypatch, lambda source, call, stub, *_: source.replace(call, stub % state))


def _no_report(monkeypatch):
    """Both children close the report pipe without writing their state."""
    _patch_children(monkeypatch, lambda source, call, stub, report, no_report: source.replace(report, no_report))


@pytest.mark.parametrize("value, expected", [
    (None, ""), ("", ""), ("0", ""), ("false", ""), ("no", ""), ("off", ""),
    ("1", "full"), ("true", "full"), (" YES ", "full"), ("full", "full"), ("any", "any"), ("ANY", "any"),
    ("banana", "full"),  # a typo must not silently turn the requirement off
])
def test_require_isolation_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", raising=False)
    else:
        monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", value)
    assert _code_confine.require_isolation() == expected


@pytest.mark.parametrize("seccomp, landlock", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("value, needs", [(None, "nothing"), ("1", "both"), ("true", "both"), ("full", "both"), ("any", "one")])
def test_require_isolation_modes(monkeypatch, value, needs, seccomp, landlock):
    """``1``/``true``/``full`` need seccomp AND Landlock, ``any`` needs one, unset runs the candidate anyway.
    The decision is made from what the child reports, not from anything the grader probed beforehand."""
    if value is None:
        monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", raising=False)
    else:
        monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", value)
    _pretend_isolation(monkeypatch, seccomp, landlock)
    runs = {"nothing": True, "both": seccomp and landlock, "one": seccomp or landlock}[needs]
    assert _scores(HONEST, 2) == ((1.0, 1.0) if runs else (0.0, 0.0))
    assert (_rc(HONEST, 10, ("f(1)",)) is not None) == runs
    assert (_rc_many(HONEST, [("f(1)",)], 10) != [None]) == runs


def test_an_unmet_requirement_says_which_layer_is_missing(monkeypatch, caplog):
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    monkeypatch.setattr(_code_confine, "_unmet_logged", set())
    _pretend_isolation(monkeypatch, True, False)
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _scores(HONEST, 2) == (0.0, 0.0)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1" in text and "Landlock" in text and "scores 0" in text
    assert "seccomp or" not in text.replace("both seccomp and", "")
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "any")
    _pretend_isolation(monkeypatch, False, False)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _scores(HONEST, 2) == (0.0, 0.0)
    assert "seccomp or Landlock" in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("value, needs_both", [("1", True), ("any", False)])
def test_require_isolation_follows_the_real_kernel(monkeypatch, value, needs_both):
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", value)
    runs = (SECCOMP and LANDLOCK) if needs_both else (SECCOMP or LANDLOCK)
    assert _scores(HONEST, 2) == ((1.0, 1.0) if runs else (0.0, 0.0))


# --- no seccomp: refused by default on Linux, with an opt-out ---------------------------------------------------

@pytest.mark.skipif(not LINUX, reason="the default refusal is Linux-only")
@pytest.mark.parametrize("which", ["run_candidate", "sandbox"])
def test_without_seccomp_candidates_are_refused_by_default_on_linux(monkeypatch, caplog, tmp_path, which):
    marker = tmp_path / "ran"
    code = f"open({str(marker)!r}, 'w').write('x')\ndef f(x):\n    return 1"
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", raising=False)
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", raising=False)
    _pretend_isolation(monkeypatch, False, True, allow_no_seccomp=False)  # Landlock yes, seccomp no
    monkeypatch.setattr(_code_confine, "_unmet_logged", set())
    monkeypatch.setattr(_code_confine, "_partial_logged", {})
    run = (lambda: _rc(code, 10, ("f(1)",))) if which == "run_candidate" else (
        lambda: _rc_many(code, [("f(1)",)], 10))
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert run() in (None, [None])
        assert run() in (None, [None])
    assert not marker.exists()  # the child exited before the candidate ran
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # one clear error, not one per call
    text = errors[0].getMessage()
    assert "seccomp is not active" in text and "ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP=1" in text and "scores 0" in text


@pytest.mark.skipif(not LINUX, reason="the default refusal is Linux-only")
@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_the_opt_out_restores_the_best_effort_behaviour(monkeypatch, tmp_path, value):
    marker = tmp_path / "ran"
    code = f"open({str(marker)!r}, 'w').write('x')\ndef f(x):\n    return 1"
    _pretend_isolation(monkeypatch, False, True, allow_no_seccomp=False)
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", value)
    assert _rc(code, 10, ("f(1)",)) == [1]
    assert marker.exists()


@pytest.mark.skipif(not LINUX, reason="the default refusal is Linux-only")
def test_a_linux_child_that_reports_nothing_is_refused_by_default(monkeypatch):
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", raising=False)
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", raising=False)
    monkeypatch.setattr(_code_confine, "_unmet_logged", set())
    _no_report(monkeypatch)
    assert _rc(HONEST, 10, ("f(1)",)) is None
    assert _rc_many(HONEST, [("f(1)",)], 10) == [None]


def test_other_platforms_keep_best_effort_with_a_warning(monkeypatch):
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", raising=False)
    for platform_name, refuses in (("linux", True), ("darwin", False), ("win32", False), ("freebsd14", False)):
        monkeypatch.setattr(sys, "platform", platform_name)
        assert _code_confine.refuses_without_seccomp() is refuses, platform_name
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")
    assert _code_confine.refuses_without_seccomp() is False
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "0")
    assert _code_confine.refuses_without_seccomp() is True


@pytest.mark.skipif(LINUX, reason="only meaningful off Linux: a host without seccomp by design keeps running candidates")
def test_a_host_without_seccomp_still_grades_here_with_limits_and_a_warning():
    assert _scores(HONEST, 2) == (1.0, 1.0)


# --- capabilities: dropped before the candidate runs --------------------------------------------------------------

class _FakeLibc:
    def __init__(self):
        self.prctl_calls, self.syscalls = [], []

    def prctl(self, *args):
        self.prctl_calls.append(args)
        return 0

    def syscall(self, nr, header, data):
        import struct

        self.syscalls.append((nr.value, ctypes.string_at(header.value, 8), ctypes.string_at(data.value, 24)))
        return 0


def _status_file(tmp_path, **caps):
    lines = ["Name:\tpython"] + ["%s:\t%016x" % (k, v) for k, v in caps.items()]
    path = tmp_path / "status"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize("machine, nr", [("x86_64", 126), ("aarch64", 91)])
def test_drop_capabilities_with_a_mocked_kernel(tmp_path, monkeypatch, machine, nr):
    import struct

    monkeypatch.setattr(platform, "machine", lambda: machine)
    last_cap = tmp_path / "cap_last_cap"
    last_cap.write_text("40\n")
    drop = _tables()["drop_capabilities"]
    libc = _FakeLibc()
    done = drop(libc, status_path=_status_file(tmp_path, CapInh=0, CapPrm=0, CapEff=0, CapBnd=0, CapAmb=0), last_cap_path=str(last_cap))
    assert done is True
    assert [c[1] for c in libc.prctl_calls if c[0] == 24] == list(range(41))  # PR_CAPBSET_DROP for every cap
    assert (47, 4, 0, 0, 0) in libc.prctl_calls  # PR_CAP_AMBIENT_CLEAR_ALL
    (called_nr, header, data), = libc.syscalls  # capset(version 3, this process, all zero)
    assert called_nr == nr and header == struct.pack("Ii", 0x20080522, 0) and data == bytes(24)


def test_drop_capabilities_says_so_when_something_is_left_or_cannot_be_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    drop = _tables()["drop_capabilities"]
    last_cap = tmp_path / "cap_last_cap"
    last_cap.write_text("40\n")
    left = _status_file(tmp_path, CapInh=0, CapPrm=0x1FFFFFFFFFF, CapEff=0x1FFFFFFFFFF, CapBnd=0, CapAmb=0)
    assert drop(_FakeLibc(), status_path=left, last_cap_path=str(last_cap)) is False
    assert drop(_FakeLibc(), status_path=_status_file(tmp_path, CapEff=0, CapPrm=0, CapInh=0, CapAmb=1), last_cap_path=str(last_cap)) is False
    assert drop(_FakeLibc(), status_path=str(tmp_path / "missing"), last_cap_path=str(last_cap)) is False
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")  # no capset number known
    assert drop(_FakeLibc(), status_path=_status_file(tmp_path, CapEff=0, CapPrm=0, CapInh=0, CapAmb=0), last_cap_path=str(last_cap)) is False


def test_the_report_carries_caps_dropped_and_full_mode_needs_it():
    assert _code_confine.parse_report(b'{"seccomp": true, "landlock": 4, "caps_dropped": true}') == {"seccomp": True, "landlock": 4, "caps_dropped": True}
    assert _code_confine.parse_report(b'{"seccomp": true, "landlock": 4, "caps_dropped": "yes"}') is None
    both = {"seccomp": True, "landlock": 4}
    assert _code_confine.isolation_met("full", both) and _code_confine.isolation_met("full", dict(both, caps_dropped=True))
    assert not _code_confine.isolation_met("full", dict(both, caps_dropped=False))
    assert _code_confine.isolation_met("any", dict(both, caps_dropped=False))


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs a Linux kernel with seccomp")
def test_the_real_child_drops_its_capabilities_and_reports_it():
    out = subprocess.run(
        [sys.executable, "-I", "-c", _code_confine.ISOLATE_SOURCE + "import json\nprint(json.dumps(isolate(os.getcwd(), False)))"],
        cwd=tempfile.gettempdir(), capture_output=True, text=True,
    ).stdout
    assert json.loads(out.strip().splitlines()[-1])["caps_dropped"] is True
    code = textwrap.dedent('''
        def f(x):
            caps = {}
            for line in open("/proc/self/status"):
                if line.startswith("Cap"):
                    k, _, v = line.partition(":")
                    caps[k] = int(v.strip(), 16)
            return [caps["CapEff"], caps["CapPrm"], caps["CapInh"], caps.get("CapAmb", 0)]
    ''')
    assert _rc(code, 20, ("f(1)",)) == [[0, 0, 0, 0]]


def _root_like_available():
    if os.geteuid() == 0:
        return "root"
    try:
        ok = subprocess.run(["unshare", "-r", "true"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    return "userns" if ok else ""


ROOT_LIKE = _root_like_available() if LINUX else ""

ROOT_SCRIPT = textwrap.dedent('''
    import json, os, subprocess, sys, time
    sys.path.insert(0, SRC)
    import logging
    logging.disable(logging.CRITICAL)
    from aligntune.rewards import _code_confine, _code_grader as g, _code_sandbox as sb
    out = {"euid": os.geteuid()}
    sentinel = "SENTINEL_" + os.urandom(16).hex()
    sibling = subprocess.Popen([sys.executable, "-c", (
        "import ctypes, sys, time\\nctypes.CDLL(None).prctl(4, 0, 0, 0, 0)\\n"  # PR_SET_DUMPABLE 0: reading it needs CAP_SYS_PTRACE
        "buf = ctypes.create_string_buffer(%r.encode())\\nprint(ctypes.addressof(buf), flush=True)\\ntime.sleep(120)" % sentinel)],
        stdout=subprocess.PIPE, text=True)
    addr = int(sibling.stdout.readline())
    read = "import os\\ndef f(x):\\n    fd = os.open('/proc/%d/mem', os.O_RDONLY)\\n    return os.pread(fd, %d, %d).decode()\\n" % (sibling.pid, len(sentinel), addr)
    status = ("def f(x):\\n    return [int(l.split(':')[1], 16) for l in open('/proc/self/status') if l.startswith(('CapEff', 'CapPrm'))]\\n")
    syscalls = ("import os, time, stat\\ndef f(x):\\n    out = []\\n"
                "    try:\\n        os.mknod('node', stat.S_IFCHR | 0o600, os.makedev(1, 3))\\n        out.append('mknod worked')\\n    except OSError as e:\\n        out.append('mknod ' + str(e.errno))\\n"
                "    try:\\n        time.clock_settime(time.CLOCK_REALTIME, time.time())\\n        out.append('clock_settime worked')\\n    except OSError as e:\\n        out.append('clock_settime ' + str(e.errno))\\n"
                "    return out\\n")
    originals = {(g, "_HARNESS"): g._HARNESS, (sb, "_CHILD"): sb._CHILD}
    no_landlock = "abi = sc(444, ctypes.c_void_p(0), 0, 1)"
    keep_caps = 'st["caps_dropped"] = drop_capabilities(libc)'
    os.environ["ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP"] = "1"

    def use(landlock=True, drop=True):
        for (m, n), h in originals.items():
            h = h if landlock else h.replace(no_landlock, "abi = 0")
            h = h if drop else h.replace(keep_caps, 'st["caps_dropped"] = False')
            setattr(m, n, h)

    def run(code):  # both children must agree; a call that raised comes back as "FAILED"
        r = g.run_candidate(code, 30, [{"expr": "f(1)"}])
        a = None if r.status != "ok" else ["FAILED" if x is g.FAILED else x for x in r.values]
        load, res = sb.run_calls(code, ["f(1)"], timeout=30)
        b = None if load is not None else [v if k == "ok" else "FAILED" for k, v in res]
        return a if a == b else {"grader": a, "sandbox": b}

    use(landlock=False, drop=False)  # control: seccomp only, capabilities kept
    out["control_caps"] = run(status)
    out["control_reads_other_process"] = run(read)
    out["control_syscalls"] = run(syscalls)
    use(landlock=False, drop=True)  # capabilities dropped, no Landlock: only the capability stands in the way
    out["dropped_caps"] = run(status)
    out["dropped_reads_other_process"] = run(read)
    out["dropped_syscalls"] = run(syscalls)
    use(landlock=True, drop=True)  # everything
    out["full_reads_other_process"] = run(read)
    out["full_ordinary_answer"] = run("def f(x):\\n    return x + 1\\n")
    out["sentinel"] = sentinel
    sibling.kill()
    print("JSON:" + json.dumps(out))
''')


@pytest.mark.skipif(not (LINUX and SECCOMP and ROOT_LIKE), reason="needs root, or a user namespace (unshare -r), to hold capabilities")
def test_as_root_a_confined_candidate_has_no_capabilities_and_cannot_read_another_process():
    """Real capabilities: root, or root in a user namespace. Without the drop the candidate keeps CAP_SYS_PTRACE and reads a
    non-dumpable sibling's memory (control); with it, it cannot. An ordinary answer still scores 1."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    script = "SRC = %r\n" % src + ROOT_SCRIPT
    cmd = [sys.executable, "-P", "-c", script]
    if ROOT_LIKE == "userns":
        cmd = ["unshare", "-r"] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("JSON:")), None)
    assert line, proc.stderr[proc.stderr.rfind("Traceback"):][-2500:] or proc.stderr[-1500:]
    out = json.loads(line[5:])
    assert out["euid"] == 0
    # control: seccomp only, capabilities kept: the capability reads the sibling's memory
    assert out["control_caps"][0][0] != 0, "the control holds no capability, so the test proves nothing"
    assert out["control_reads_other_process"] == [out["sentinel"]]
    # with the drop: no capability, and the read fails (the candidate raised, so there is no result)
    assert out["dropped_caps"] == [[0, 0]]
    assert out["dropped_reads_other_process"] in (None, ["FAILED"])
    assert out["full_reads_other_process"] in (None, ["FAILED"])
    assert out["full_ordinary_answer"] == [2]
    # mknod of a device and clock_settime are refused (as real root they work without the drop)
    assert out["dropped_syscalls"] == [["mknod 1", "clock_settime 1"]]
    if ROOT_LIKE == "root":
        assert out["control_syscalls"] == [["mknod worked", "clock_settime worked"]]


# --- what the confined child may read -------------------------------------------------------------------------

def _under(path, roots):
    path = os.path.realpath(path)
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in map(os.path.realpath, roots))


ALLOWED_ROOTS = ["/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/etc", "/dev", "/proc", "/sys", "/nix",
                 sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]


def _isolated_script(body, python=None, extra_path=None):
    """Run ``body`` in a ``-I`` child after ``isolate()`` (optionally with a directory on its sys.path)."""
    prefix = f"import sys\nsys.path.append({extra_path!r})\n" if extra_path else ""
    with tempfile.TemporaryDirectory() as work:
        return subprocess.run(
            [python or sys.executable, "-I", "-c", prefix + _code_confine.ISOLATE_SOURCE + "isolate(os.getcwd(), False)\n" + body],
            cwd=work, capture_output=True, text=True,
        )


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_a_sys_path_entry_is_not_readable(tmp_path):
    if _under(tmp_path, ALLOWED_ROOTS):
        pytest.skip("the temporary directory is under a directory the child may read anyway")
    (tmp_path / "answers.txt").write_text("EXPECTED")
    read = f"print(open({str(tmp_path / 'answers.txt')!r}).read())"
    denied = _isolated_script(read, extra_path=str(tmp_path))  # like an editable install or a PYTHONPATH entry
    assert "PermissionError" in denied.stderr and "EXPECTED" not in denied.stdout
    control = subprocess.run([sys.executable, "-I", "-c", read], capture_output=True, text=True)
    assert control.stdout.strip() == "EXPECTED"


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_a_candidate_cannot_read_a_checkout_the_trainers_cwd_or_list_the_home_directory(tmp_path, monkeypatch):
    import pwd

    home = pwd.getpwuid(os.getuid()).pw_dir
    checkout = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    (tmp_path / "answers.txt").write_text("EXPECTED")
    monkeypatch.chdir(tmp_path)  # the trainer's working directory
    targets = {
        "checkout": os.path.join(checkout, "pyproject.toml"),
        "cwd": str(tmp_path / "answers.txt"),
        "home": home,
    }
    if any(_under(path, ALLOWED_ROOTS) for path in targets.values()):
        pytest.skip("one of the paths is under a directory the child may read anyway (interpreter prefix or /usr)")
    code = (
        "import os\ndef f(x):\n    out = []\n"
        + "".join(
            f"    try:\n        {'os.listdir' if k == 'home' else 'open'}({path!r})\n        out.append(True)\n"
            "    except OSError:\n        out.append(False)\n"
            for k, path in targets.items()
        )
        + "    return out\n"
    )
    assert _rc(code, 20, ("f(1)",)) == [[False, False, False]]


ETC_FILE_NOT_ON_THE_LIST = next(
    (p for p in ("/etc/passwd", "/etc/hosts", "/etc/group", "/etc/hostname", "/etc/fstab", "/etc/shells") if os.path.isfile(p)), None
)


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_proc_dev_etc_and_sys_are_narrowed_to_what_a_python_child_needs():
    """The trainer's /proc entry, a listing of /dev, /etc/passwd and the like are refused; what libraries read still works."""
    ppid_files = ("environ", "mem", "cmdline", "maps", "status", "stat", "comm")
    code = textwrap.dedent('''
        import os

        def ok(fn):
            try:
                fn()
                return True
            except OSError:
                return False

        def read(path, mode="rb", n=16):
            def go():
                with open(path, mode) as fh:
                    fh.read(n)
            return go

        def f(x):
            ppid = os.getppid()
            out = {}
            for name in %(ppid_files)r:
                out["parent " + name] = ok(read("/proc/%%d/%%s" %% (ppid, name)))
            out["parent dir"] = ok(lambda: os.listdir("/proc/%%d" %% ppid))
            out["list /proc"] = ok(lambda: os.listdir("/proc"))
            out["list /dev"] = ok(lambda: os.listdir("/dev"))
            out["list /sys"] = ok(lambda: os.listdir("/sys"))
            out["list /etc"] = ok(lambda: os.listdir("/etc"))
            out["etc file"] = ok(read(%(etc)r)) if %(etc)r else False
            out["own status"] = ok(read("/proc/self/status", "r"))
            out["own fd dir"] = ok(lambda: os.listdir("/proc/self/fd"))
            out["cpuinfo"] = ok(read("/proc/cpuinfo", "r"))
            out["meminfo"] = ok(read("/proc/meminfo", "r"))
            out["urandom"] = ok(lambda: os.urandom(8)) and ok(read("/dev/urandom"))
            out["dev zero"] = ok(read("/dev/zero"))
            out["dev null"] = ok(lambda: open("/dev/null", "w").write("x"))
            out["ld.so.cache"] = ok(read("/etc/ld.so.cache")) if os.path.exists("/etc/ld.so.cache") else True
            out["cpu topology"] = ok(lambda: os.listdir("/sys/devices/system/cpu")) if os.path.exists("/sys/devices/system/cpu") else True
            return out
    ''') % {"ppid_files": ppid_files, "etc": ETC_FILE_NOT_ON_THE_LIST}
    result = _rc(code, 30, ("f(1)",))
    assert result is not None
    got = result[0]
    refused = [k for k, v in got.items() if k.startswith("parent ") or k in ("parent dir", "list /proc", "list /dev", "list /sys", "list /etc", "etc file")]
    allowed = [k for k in got if k not in refused]
    assert {k: got[k] for k in refused} == {k: False for k in refused}
    assert {k: got[k] for k in allowed} == {k: True for k in allowed}


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
def test_ownership_xattrs_and_truncate_by_path_are_refused_but_chmod_utime_and_copy2_work():
    """Landlock doesn't cover these; the filter refuses chown, xattr writes and truncate(path). chmod and timestamps are NOT
    prevented (shutil.copy2 and common code use them)."""
    code = textwrap.dedent('''
        import os, shutil

        def denied(fn):
            try:
                fn()
                return "allowed"
            except PermissionError:
                return "EPERM"
            except OSError as e:
                return "other %d" % e.errno

        def f(x):
            with open("data.txt", "w") as fh:
                fh.write("hello world")
            out = {}
            out["chown"] = denied(lambda: os.chown("data.txt", -1, -1))
            out["lchown"] = denied(lambda: os.lchown("data.txt", -1, -1))
            fd = os.open("data.txt", os.O_RDWR)
            out["fchown"] = denied(lambda: os.fchown(fd, -1, -1))
            out["fchownat"] = denied(lambda: os.chown("data.txt", -1, -1, follow_symlinks=False))
            out["setxattr"] = denied(lambda: os.setxattr("data.txt", "user.k", b"v"))
            out["fsetxattr"] = denied(lambda: os.setxattr(fd, "user.k", b"v"))
            out["lsetxattr"] = denied(lambda: os.setxattr("data.txt", "user.k", b"v", follow_symlinks=False))
            out["removexattr"] = denied(lambda: os.removexattr("data.txt", "user.k"))
            out["fremovexattr"] = denied(lambda: os.removexattr(fd, "user.k"))
            out["lremovexattr"] = denied(lambda: os.removexattr("data.txt", "user.k", follow_symlinks=False))
            out["truncate path"] = denied(lambda: os.truncate("data.txt", 3))
            os.close(fd)
            # what still works
            with open("data.txt", "r+") as fh:
                fh.truncate(5)  # ftruncate
            out["ftruncate"] = open("data.txt").read()
            os.chmod("data.txt", 0o600)
            os.utime("data.txt", (1, 1))
            shutil.copy2("data.txt", "copy.txt")
            out["copy2"] = open("copy.txt").read()
            out["utime"] = int(os.stat("copy.txt").st_mtime)
            return out
    ''')
    result = _rc(code, 30, ("f(1)",))
    assert result is not None
    got = result[0]
    denied = ["chown", "lchown", "fchown", "fchownat", "setxattr", "fsetxattr", "lsetxattr", "removexattr", "fremovexattr",
              "lremovexattr", "truncate path"]
    assert {k: got[k] for k in denied} == {k: "EPERM" for k in denied}
    assert got["ftruncate"] == "hello" and got["copy2"] == "hello" and got["utime"] == 1


@pytest.mark.skipif(not (LINUX and LANDLOCK) or not os.path.isdir("/opt"), reason="needs Landlock and an /opt directory")
def test_opt_is_not_readable_unless_it_is_under_the_interpreter_prefix():
    if _under("/opt", ALLOWED_ROOTS) or any(_under(p, ["/opt"]) for p in (sys.prefix, sys.base_prefix)):
        pytest.skip("the interpreter lives under /opt, so its prefix (part of /opt) is readable by design")
    code = "import os\ndef f(x):\n    try:\n        os.listdir('/opt')\n        return True\n    except OSError:\n        return False\n"
    assert _rc(code, 20, ("f(1)",)) == [False]


def test_a_prefix_that_contains_the_home_or_working_directory_is_warned_about_once(monkeypatch, tmp_path, caplog):
    sub = tmp_path / "home" / "me"
    sub.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    monkeypatch.setattr(sys, "exec_prefix", str(tmp_path))
    monkeypatch.setattr(sys, "base_exec_prefix", str(tmp_path))
    monkeypatch.setenv("HOME", str(sub))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_code_confine, "_prefix_warned", {})
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        _code_confine.warn_broad_prefix()
        _code_confine.warn_broad_prefix()
    warned = [r.getMessage() for r in caplog.records if "interpreter prefix" in r.getMessage()]
    assert len(warned) == 2  # the home directory and the working directory, each once
    assert any("home directory" in m for m in warned) and any("working directory" in m for m in warned)
    # a prefix that contains neither: no warning
    monkeypatch.setattr(sys, "prefix", "/nonexistent-prefix-for-the-test")
    monkeypatch.setattr(sys, "base_prefix", "/nonexistent-prefix-for-the-test")
    monkeypatch.setattr(sys, "exec_prefix", "/nonexistent-prefix-for-the-test")
    monkeypatch.setattr(sys, "base_exec_prefix", "/nonexistent-prefix-for-the-test")
    monkeypatch.setattr(_code_confine, "_prefix_warned", {})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        _code_confine.warn_broad_prefix()
    assert not [r for r in caplog.records if "interpreter prefix" in r.getMessage()]


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_dev_urandom_and_os_urandom_work_in_the_confined_child():
    code = "import os, random\ndef f(x):\n    return [len(os.urandom(16)), len(open('/dev/urandom', 'rb').read(8)), 0 <= random.SystemRandom().random() < 1]\n"
    assert _rc(code, 20, ("f(1)",)) == [[16, 8, True]]


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_a_virtualenv_under_the_home_directory_is_still_readable(tmp_path):
    import pwd
    import shutil
    import venv

    home = pwd.getpwuid(os.getuid()).pw_dir
    try:
        root = tempfile.mkdtemp(dir=home, prefix=".aligntune_venv_test_")
    except OSError:
        pytest.skip("the home directory is not writable here")
    try:
        venv.EnvBuilder(with_pip=False, clear=True).create(os.path.join(root, "env"))
        python = os.path.join(root, "env", "bin", "python")
        purelib = subprocess.run([python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                                 capture_output=True, text=True).stdout.strip()
        os.makedirs(os.path.join(purelib, "fakepkg"), exist_ok=True)
        with open(os.path.join(purelib, "fakepkg", "__init__.py"), "w") as f:
            f.write("VALUE = 7\n")
        with open(os.path.join(root, "sibling.txt"), "w") as f:
            f.write("EXPECTED")
        out = _isolated_script(
            "import json, re, fakepkg\n"
            "res = [fakepkg.VALUE, os.path.isfile(os.path.join(sys.prefix, 'pyvenv.cfg')) and bool(open(os.path.join(sys.prefix, 'pyvenv.cfg')).read())]\n"
            f"try:\n    open({os.path.join(root, 'sibling.txt')!r}).read()\n    res.append('LEAK')\nexcept OSError:\n    res.append('denied')\n"
            f"try:\n    os.listdir({home!r})\n    res.append('LEAK')\nexcept OSError:\n    res.append('denied')\n"
            "print(json.dumps(res))\n",
            python=python,
        )
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout.strip().splitlines()[-1]) == [7, True, "denied", "denied"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("state, warns", [
    ({"seccomp": True, "landlock": 4}, False), ({"seccomp": True, "landlock": 0}, True),
    ({"seccomp": False, "landlock": 4}, True), (None, True),
])
def test_partial_isolation_warns_once(monkeypatch, caplog, state, warns):
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")  # a Linux grader refuses states without seccomp by default
    monkeypatch.setattr(_code_confine, "_partial_logged", {})
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _code_confine.judge_report(state) is True
        assert _code_confine.judge_report(state) is True
    warned = [r for r in caplog.records if "kernel isolation is partial" in r.getMessage()]
    assert len(warned) == (1 if warns else 0)
    if state is None or not state["seccomp"]:
        assert "start processes" in warned[0].getMessage() or state is None


def test_the_decision_uses_the_childs_report_not_a_probe(monkeypatch, caplog):
    """Whatever the kernel really offers, a child that reports one layer is refused under REQUIRE=1, and a child that
    reports nothing at all (it died before reporting) is refused too; without REQUIRE both still run or fail on their own."""
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    _pretend_isolation(monkeypatch, True, False)  # the real kernel may well have Landlock; the child says it did not
    assert _rc(HONEST, 10, ("f(1)",)) is None
    # a child that never writes its status
    _no_report(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        monkeypatch.setattr(_code_confine, "_unmet_logged", set())
        assert _rc(HONEST, 10, ("f(1)",)) is None
        assert _rc_many(HONEST, [("f(1)",)], 10) == [None]
    assert "no report" in " ".join(r.getMessage() for r in caplog.records)
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION")
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "1")  # no report counts as no seccomp, refused on Linux by default
    assert _rc(HONEST, 10, ("f(1)",)) == [2]  # nothing required: it runs


@pytest.mark.parametrize("which", ["run_candidate", "sandbox"])
def test_a_child_that_failed_a_required_layer_never_runs_the_candidate(monkeypatch, tmp_path, which):
    marker = tmp_path / "ran"
    code = f"open({str(marker)!r}, 'w').write('x')\ndef f(x):\n    return 1"
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    _pretend_isolation(monkeypatch, True, False)  # the child sees that Landlock failed
    if which == "run_candidate":
        assert _rc(code, 10, ("f(1)",)) is None
    else:
        assert _rc_many(code, [("f(1)",)], 10) == [None]
    assert not marker.exists()
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION")
    _pretend_isolation(monkeypatch, True, False)
    _rc(code, 10, ("f(1)",))  # control: with nothing required it does run
    assert marker.exists()


def test_a_report_that_is_not_exactly_a_plain_state_counts_as_no_report():
    for data in (b"", b"not json", b"[]", b'{"seccomp": "yes", "landlock": 4}', b'{"seccomp": true}',
                 b'{"seccomp": true, "landlock": "4"}', b'{"seccomp": true, "landlock": true}', b"\xff\xfe"):
        assert _code_confine.parse_report(data) is None, data
    assert _code_confine.parse_report(b'{"seccomp": true, "landlock": 4, "extra": 1}') == {"seccomp": True, "landlock": 4}


def test_the_report_pipe_is_read_with_a_size_cap_and_a_timeout():
    def pipe_with(data=b"", close=True):
        r, w = os.pipe()
        os.write(w, data)
        if close:
            os.close(w)
            w = None
        return r, w

    r, w = pipe_with(b'{"seccomp": true, "landlock": 4}')
    assert _code_confine.read_report(r, 1.0) == {"seccomp": True, "landlock": 4}
    os.close(r)
    r, w = pipe_with(b"x" * 10000)  # too big
    assert _code_confine.read_report(r, 1.0) is None
    os.close(r)
    r, w = pipe_with(b"", close=False)  # a child that hangs before reporting
    start = time.monotonic()
    timed_out = _code_confine.read_report(r, 0.3)
    assert not timed_out and isinstance(timed_out, _code_confine._TimedOut) and time.monotonic() - start < 2
    os.close(r)
    os.close(w)
    r, w = pipe_with(b"")  # a child that died before reporting
    assert _code_confine.read_report(r, 1.0) is None
    os.close(r)


def test_the_report_timeout_is_a_constant_that_scales_with_the_candidates_timeout():
    assert _code_confine.REPORT_TIMEOUT == 10.0
    assert _code_confine.report_timeout(1) == 10.0 and _code_confine.report_timeout(5) == 10.0
    assert _code_confine.report_timeout(30) == 60.0
    assert _code_confine.report_timeout(10_000) == 120.0


def test_a_child_that_does_not_report_in_time_is_logged_as_such(monkeypatch, caplog):
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    monkeypatch.setattr(_code_confine, "_unmet_logged", set())
    monkeypatch.setattr(_code_confine, "_partial_logged", {})
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _code_confine.judge_report(_code_confine._TimedOut()) is False
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "did not report in time" in text and "reported nothing" not in text
    monkeypatch.setattr(_code_confine, "_unmet_logged", set())
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _code_confine.judge_report(None) is False
    assert "reported nothing" in " ".join(r.getMessage() for r in caplog.records)


def test_a_child_that_hangs_before_reporting_scores_zero_under_require_and_the_run_is_not_stuck(monkeypatch):
    """The harness never writes its report and never closes the pipe's write end... until it exits: a short timeout is honoured."""
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    monkeypatch.setattr(_code_confine, "REPORT_TIMEOUT", 0.5)
    monkeypatch.setattr(_code_confine, "report_timeout", lambda t: 0.5)
    _patch_children(monkeypatch, lambda source, *_: "import time\ntime.sleep(60)\n" + source)
    start = time.monotonic()
    assert _rc(HONEST, 10, ("f(1)",)) is None
    assert _rc_many(HONEST, [("f(1)",)], 10) == [None]
    assert time.monotonic() - start < 15


def _many_descriptors(count):
    """Raise the soft limit if it allows and open ``count`` descriptors, so the next ones are numbered above 1024."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = count + 200
    if soft < want:
        if hard != resource.RLIM_INFINITY and hard < want:
            pytest.skip("the hard descriptor limit is too low here")
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    return (soft, hard), [os.open(os.devnull, os.O_RDONLY) for _ in range(count)]


@pytest.mark.parametrize("which", ["run_candidate", "sandbox"])
def test_a_trainer_with_many_open_files_still_gets_its_candidates_graded(monkeypatch, which):
    """select.select() fails for descriptors >= 1024 (FD_SETSIZE); the report is read with selectors instead. The report
    pipe is forced above 1024 here (a trainer that holds this many files gets such descriptors for its temporary files)."""
    import resource

    if os.name == "nt":
        pytest.skip("POSIX descriptors")
    import fcntl

    limits, held = _many_descriptors(1100)
    real_pipe = os.pipe

    def high_pipe():
        r, w = real_pipe()
        r2, w2 = fcntl.fcntl(r, fcntl.F_DUPFD_CLOEXEC, 1200), fcntl.fcntl(w, fcntl.F_DUPFD_CLOEXEC, 1200)
        os.close(r)
        os.close(w)
        seen.append((r2, w2))
        return r2, w2

    seen = []
    try:
        assert max(held) >= 1024
        monkeypatch.setattr(os, "pipe", high_pipe)
        if which == "run_candidate":
            assert _rc(HONEST, 20, ("f(1)",)) == [2]
        else:
            assert _rc_many(HONEST, [("f(1)",), ("f(2)",)], 20) == [[2], [3]]
        assert _scores(HONEST, 2) == (1.0, 1.0)
        assert seen and all(r >= 1024 for r, _ in seen)  # the reports really came over descriptors above 1024
    finally:
        monkeypatch.undo()
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)


class _AsWindows:
    """The ``os`` module as seen on Windows by the grader: ``os.name == "nt"``, everything else as it is here."""

    name = "nt"

    def __getattr__(self, attr):
        return getattr(os, attr)


def test_on_windows_no_report_pipe_is_passed_and_the_run_goes_on(monkeypatch, caplog):
    """``pass_fds`` is POSIX-only: without it every candidate would score 0 there. The child reports nothing."""
    monkeypatch.setattr(_code_confine, "os", _AsWindows())
    monkeypatch.delenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", raising=False)
    seen = {}
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    assert _rc(HONEST, 20, ("f(1)",)) == [2]
    assert "pass_fds" not in seen
    # ... and the requirement is refused, with a clear reason
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    monkeypatch.setattr(_code_confine, "_unmet_logged", set())
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _rc(HONEST, 20, ("f(1)",)) is None
    assert "no report channel" in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("which", ["run_candidate", "sandbox"])
def test_the_report_cannot_be_forged_or_found_by_the_candidate(monkeypatch, tmp_path, which):
    """The child closes the report pipe before the candidate runs: no report pipe is left to find, and whatever the
    candidate writes to the descriptors it can find changes nothing about what the grader believes (seccomp is off
    here). The sandbox child's only pipes are its stdin and stdout, the channel its values go back on by design."""
    import stat

    waited = []
    monkeypatch.setattr(_code_confine, "group_gone", lambda pgid, wait=1.0: waited.append(pgid) or True)
    _pretend_isolation(monkeypatch, False, False)
    code = textwrap.dedent('''
        import os, stat
        fifos = 0
        for fd in range(3, 256):
            try:
                if stat.S_ISFIFO(os.fstat(fd).st_mode):
                    fifos += 1
                    os.write(fd, b'{"seccomp": true, "landlock": 4}')
            except OSError:
                pass
        for root, dirs, files in os.walk("."):
            for name in files:
                open(os.path.join(root, name), "w").write('{"seccomp": true, "landlock": 4}')
        def f(x):
            return fifos
    ''')
    if which == "run_candidate":
        assert _rc(code, 10, ("f(1)",)) == [0]  # no pipe found
    else:
        assert _rc_many(code, [("f(1)",)], 10) in ([None], [[2]])  # stdin and stdout: the reply is garbled, or a value
    assert len(waited) == 1  # the grader still believed "no seccomp" and cleaned up accordingly


def test_without_seccomp_the_group_is_killed_and_the_hold_waits_for_it(monkeypatch, caplog):
    """Where the child says seccomp is not active it could have forked: the grader waits for the process group to be gone."""
    waited, scanned = [], []
    monkeypatch.setattr(_code_confine, "group_gone", lambda pgid, wait=1.0: waited.append(pgid) or True)
    monkeypatch.setattr(_code_confine, "kill_marked", lambda marker, wait=2.0: scanned.append(marker) or True)
    _pretend_isolation(monkeypatch, False, True)
    monkeypatch.setattr(_code_confine, "_partial_logged", {})
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        assert _rc(HONEST, 10, ("f(1)",)) == [2]
    assert len(waited) == 1 and len(scanned) == 1 and len(scanned[0]) == 16
    assert "start processes" in " ".join(r.getMessage() for r in caplog.records)
    _pretend_isolation(monkeypatch, True, True)
    _rc(HONEST, 10, ("f(1)",))
    assert len(waited) == 1 and len(scanned) == 1  # with seccomp active there is nothing to wait for or scan for


def _alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _setsid_candidate(pidfile):
    return textwrap.dedent(f'''
        import os, time
        if os.fork() == 0:
            os.setsid()  # out of the process group
            open({str(pidfile)!r}, "w").write(str(os.getpid()))
            time.sleep(60)
            os._exit(0)
        time.sleep(0.5)
        def f(x):
            return 1
    ''')


@pytest.mark.skipif(not (LINUX and os.path.isdir("/proc")), reason="needs /proc")
@pytest.mark.parametrize("which", ["run_candidate"])  # the sandbox child also has RLIMIT_NPROC 0, so it can't fork at all
def test_without_seccomp_a_process_that_left_the_group_is_found_by_the_marker_and_killed(monkeypatch, tmp_path, which):
    """Seccomp is mocked off, so the candidate can fork and setsid() out of the process group. The grader finds it by the
    run's marker in its environment and kills it, and keeps the trainer non-dumpable until then."""
    _pretend_isolation(monkeypatch, False, False)
    pidfile = tmp_path / "pid"
    seen = []
    real = _code_confine.kill_marked

    def spy(marker, wait=2.0):
        seen.append(_dumpable())  # the hold is still on while the scan runs
        return real(marker, wait)

    monkeypatch.setattr(_code_confine, "kill_marked", spy)

    def run():
        if which == "run_candidate":
            return _rc(_setsid_candidate(pidfile), 10, ("f(1)",))
        return _rc_many(_setsid_candidate(pidfile), [("f(1)",)], 10)

    before = _dumpable()
    assert run() in ([1], [[1]])
    assert seen == [0] and _dumpable() == before  # scanned while held, released after
    pid = int(pidfile.read_text())
    assert not _alive(pid)

    # control: without the scan the escaped process lives on (it is killed here)
    monkeypatch.setattr(_code_confine, "kill_marked", lambda marker, wait=2.0: True)
    run()
    pid = int(pidfile.read_text())
    try:
        assert _alive(pid), "the control process was not alive, so the check above proves nothing"
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


@pytest.mark.skipif(not (LINUX and os.path.isdir("/proc")), reason="needs /proc")
@pytest.mark.skipif(not LINUX, reason="PR_SET_DUMPABLE is Linux-only")
def test_the_hold_is_kept_while_survivors_remain_for_a_bounded_time_then_released_with_one_warning(monkeypatch, caplog):
    """A leftover that can't be killed keeps the trainer non-dumpable until the budget is spent, then it is released
    anyway and the warning is logged once."""
    _pretend_isolation(monkeypatch, False, False)
    monkeypatch.setattr(_code_confine, "REAP_BUDGET", 0.6)
    monkeypatch.setattr(_code_confine, "_survivors_logged", {})
    states = []

    def stuck(marker, wait=2.0):
        states.append(("scan", _dumpable()))
        time.sleep(wait)
        return False

    def never_gone(pgid, wait=1.0):
        states.append(("group", _dumpable()))
        time.sleep(wait)
        return False

    monkeypatch.setattr(_code_confine, "kill_marked", stuck)
    monkeypatch.setattr(_code_confine, "group_gone", never_gone)
    before = _dumpable()
    with caplog.at_level(logging.WARNING, logger=_code_confine.logger.name):
        start = time.monotonic()
        assert _rc(HONEST, 10, ("f(1)",)) == [2]
        assert _rc(HONEST, 10, ("f(1)",)) == [2]
        assert time.monotonic() - start < 8  # bounded: about the budget per run, not forever
    assert states and all(d == 0 for _, d in states)  # held the whole time it waited
    assert _dumpable() == before  # released after
    assert len([r for r in caplog.records if "survived the cleanup" in r.getMessage()]) == 1


@pytest.mark.skipif(not (LINUX and os.path.isdir("/proc")), reason="needs /proc")
def test_kill_marked_kills_only_processes_that_carry_the_marker():
    marker = os.urandom(8).hex()
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    mine = subprocess.Popen(sleeper, env={**os.environ, "ALIGNTUNE_GRADER_RUN": marker})
    other = subprocess.Popen(sleeper, env={**os.environ, "ALIGNTUNE_GRADER_RUN": os.urandom(8).hex()})
    plain = subprocess.Popen(sleeper)
    try:
        time.sleep(0.3)
        assert _code_confine.kill_marked(marker) is True
        mine.wait(timeout=5)
        assert mine.returncode == -signal.SIGKILL and other.poll() is None and plain.poll() is None
        assert _code_confine.kill_marked("no-such-marker", wait=0.2) is True
    finally:
        for proc in (mine, other, plain):
            proc.kill()


def test_group_gone():
    assert _code_confine.group_gone(2**22 + 12345, wait=0.2) is True  # no such group
    if hasattr(os, "killpg"):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            assert _code_confine.group_gone(proc.pid, wait=0.1) is False  # still alive
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            assert _code_confine.group_gone(proc.pid, wait=1.0) is True
        finally:
            proc.kill()


# --- honest candidates still score what they earned (threads are blocked, BLAS runs on one) -----------------------

def test_plain_code_still_runs():
    assert _scores(HONEST, 2) == (1.0, 1.0)


def _importable_by_the_child(module):
    """The child runs ``python -I``, which ignores the user site and PYTHONPATH: a module installed only there isn't importable."""
    return subprocess.run([sys.executable, "-I", "-c", f"import {module}"], capture_output=True).returncode == 0


def test_numpy_imports_and_works_with_one_blas_thread():
    pytest.importorskip("numpy")
    if not _importable_by_the_child("numpy"):
        pytest.skip("numpy is installed where `python -I` can't see it (user site)")
    code = (
        "import numpy as np\n"
        "def f(x):\n"
        "    a = np.arange(9, dtype=float).reshape(3, 3) + np.eye(3)\n"
        "    return float(np.linalg.solve(a, np.ones(3)).sum()) + int(np.dot(a, a).trace())\n"
    )
    import numpy as np

    a = np.arange(9, dtype=float).reshape(3, 3) + np.eye(3)
    want = float(np.linalg.solve(a, np.ones(3)).sum()) + int(np.dot(a, a).trace())
    values = _rc(code, 60, ("f(1)",))
    assert values is not None and values[0] == pytest.approx(want)


def test_pandas_and_scipy_import_and_work_when_installed():
    pytest.importorskip("pandas")
    if not _importable_by_the_child("pandas"):
        pytest.skip("pandas is installed where `python -I` can't see it (user site)")
    code = "import pandas as pd\ndef f(x):\n    return int(pd.DataFrame({'a': [1, 2, 3]}).a.sum())\n"
    assert _rc(code, 120, ("f(1)",)) == [6]
    if pytest.importorskip("scipy"):
        code = "from scipy import linalg\nimport numpy as np\ndef f(x):\n    return round(float(linalg.det(np.eye(3) * 2)), 6)\n"
        assert _rc(code, 120, ("f(1)",)) == [8.0]


@pytest.mark.skipif(not (LINUX and SECCOMP), reason="needs seccomp")
@pytest.mark.parametrize("name, body", [
    ("thread", "import threading\nt = threading.Thread(target=lambda: 1)\nt.start()\nt.join()"),
    ("fork", "import os\nos.fork()"),
    ("subprocess", "import subprocess\nsubprocess.run(['true'])"),
    ("os.system", "import os\nassert os.system('true') == 0"),  # returns a status instead of raising
    ("socket", "import socket\nsocket.socket()"),
])
def test_candidates_cannot_start_threads_processes_or_sockets(name, body):
    assert _scores(body + "\ndef f(x):\n    return 1", 1) == (0.0, 0.0)


@pytest.mark.parametrize("module", [
    "json", "re", "math", "random", "itertools", "functools", "collections", "heapq", "bisect", "string", "textwrap",
    "datetime", "decimal", "fractions", "statistics", "hashlib", "base64", "struct", "zlib", "csv", "io", "copy",
    "operator", "enum", "dataclasses", "typing", "sqlite3", "unicodedata", "pathlib", "tempfile", "uuid", "time",
    "threading", "queue", "concurrent.futures", "multiprocessing", "subprocess", "socket", "ctypes", "ssl", "getpass",
])
def test_stdlib_modules_still_import(module):
    assert _rc(f"import {module}\ndef f(x):\n    return 1", 30, call=1) == [1]


def test_stdlib_work_gives_the_same_answers():
    code = textwrap.dedent('''
        import collections, datetime, decimal, fractions, functools, hashlib, heapq, itertools, json, math, random, re, sqlite3

        @functools.lru_cache(None)
        def fib(n):
            return n if n < 2 else fib(n - 1) + fib(n - 2)

        def f(x):
            random.seed(7)
            db = sqlite3.connect(":memory:")
            db.execute("create table t (a int)")
            db.executemany("insert into t values (?)", [(i,) for i in range(5)])
            return [fib(30), sorted(collections.Counter("banana").items()), json.loads(json.dumps({"a": [1, 2]})),
                    str(decimal.Decimal("1.10") + decimal.Decimal("2.20")), str(fractions.Fraction(3, 6)),
                    hashlib.sha256(b"x").hexdigest()[:8], heapq.nsmallest(2, [5, 3, 9, 1]),
                    list(itertools.permutations("ab")), re.sub(r"\\d+", "#", "a1b22"),
                    db.execute("select sum(a) from t").fetchone()[0], math.comb(10, 3),
                    datetime.date(2024, 2, 29).isoformat(), len(random.sample(range(100), 5))]
    ''')
    assert _rc(code, 30, ("f(1)",)) == [[
        832040, [("a", 3), ("b", 1), ("n", 2)], {"a": [1, 2]}, "3.30", "1/2", "2d711642", [1, 3],
        [("a", "b"), ("b", "a")], "a#b#", 10, 120, "2024-02-29", 5]]


def test_files_in_its_own_directory_and_the_temp_dir_work():
    code = textwrap.dedent('''
        import os, pathlib, tempfile

        def f(x):
            with open("scratch.txt", "w") as fh:
                fh.write("hi")
            with tempfile.NamedTemporaryFile("w+") as tmp:
                tmp.write("yo")
                tmp.seek(0)
                got = tmp.read()
            d = tempfile.mkdtemp()
            pathlib.Path(d, "a").write_text("z")
            return [open("scratch.txt").read(), got, pathlib.Path(d, "a").read_text(), sorted(os.listdir("."))[:1]]
    ''')
    assert _rc(code, 30, ("f(1)",)) == [["hi", "yo", "z", ["scratch.txt"]]]


@pytest.mark.skipif(not LINUX, reason="Linux scheduling calls")
def test_a_candidates_own_limits_scheduling_and_file_flags_still_work():
    code = textwrap.dedent('''
        import fcntl, os, resource, sys

        def f(x):
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            own = resource.prlimit(0, resource.RLIMIT_NOFILE)
            os.sched_setaffinity(0, os.sched_getaffinity(0))
            os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
            r, w = os.pipe()
            flags = fcntl.fcntl(r, fcntl.F_GETFL)
            fcntl.fcntl(r, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            fcntl.fcntl(w, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
            fcntl.flock(w, fcntl.LOCK_EX)
            return [soft > 0, own == (soft, hard), os.isatty(0), bool(fcntl.fcntl(r, fcntl.F_GETFL) & os.O_NONBLOCK)]
    ''')
    assert _rc(code, 30, ("f(1)",)) == [[True, True, False, True]]


def test_deep_recursion_and_a_few_seconds_of_cpu_still_work():
    code = textwrap.dedent('''
        import sys
        sys.setrecursionlimit(20000)

        def depth(n):
            return 0 if n == 0 else 1 + depth(n - 1)

        def f(x):
            total = 0
            for i in range(4_000_000):
                total += i * i % 7
            return [depth(8000), total]
    ''')
    values = _rc(code, 60, ("f(1)",))
    assert values == [[8000, sum(i * i % 7 for i in range(4_000_000))]]


def test_one_child_answers_every_assert_with_the_same_confinement():
    code = "def f(x):\n    return 2 * x"
    tests = ["assert f(2) == 4", "assert f(3) == 6", "assert f(5) == 11"]
    assert _code_grader.grade(code, tests, 10) == ("ok", [True, True, False])
    assert _rc_many(code, [("f(2)",), ("f(3)",)], 10) == [[4], [6]]


@pytest.mark.skipif(not (LINUX and LANDLOCK), reason="needs Landlock")
def test_writes_stay_inside_the_childs_own_directory(tmp_path):
    secret, written = tmp_path / "secret.txt", tmp_path / "written.txt"
    secret.write_text("S3CRET")
    if _under(tmp_path, ALLOWED_ROOTS):
        pytest.skip("the temporary directory is under a directory the child may read anyway")
    for run in (lambda c: _rc(c, 20), lambda c: _rc_many(c, [("f(1)",)], 20)[0]):
        assert run(f"v = open({str(secret)!r}).read()\ndef f(x): return v") is None
        assert run(f"open({str(written)!r}, 'w').write('x')\ndef f(x): return 1") is None
        assert not written.exists()
        assert run("open('scratch.txt', 'w').write('hi')\ndef f(x): return open('scratch.txt').read()") == ["hi"]
        home_probe = os.path.join(os.path.expanduser("~"), "_aligntune_confine_probe.txt")
        assert run(f"open({home_probe!r}, 'w').write('x')\ndef f(x): return 1") is None
        assert not os.path.exists(home_probe)
