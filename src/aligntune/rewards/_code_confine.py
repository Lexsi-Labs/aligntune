"""Kernel confinement for the child process that runs model-written code (Linux).

The graders start the candidate in a fresh ``sys.executable -I`` process with limits on
CPU, memory and file size. This adds what limits can't: before the candidate runs, the
child confines itself with

* seccomp: no ``kill``/``tgkill``/``pidfd_send_signal`` (a completion could otherwise
  SIGKILL the trainer through ``os.getppid()``), no ``ptrace``, no sockets, no
  ``io_uring``, no new processes or programs (``fork``/``clone``/``execve``), no SysV IPC or
  POSIX message queues, and nothing that changes another process: ``prlimit64`` and the
  ``sched_set*`` calls only on the caller (pid 0), no ``setpriority``/``ioprio_set``/
  ``perf_event_open``/``keyctl``, and no ``F_SETOWN``/``FIOSETOWN`` (they make the kernel signal
  another process). It also refuses the file changes Landlock doesn't cover: ``chown``,
  extended-attribute writes and ``truncate(2)`` by path;
* capabilities: all dropped (bounding, effective, permitted, inheritable and ambient sets) before
  anything else, so a trainer that runs as root does not hand them over; the child reports
  ``caps_dropped``;
* Landlock: writes (create, modify, delete, truncate on open) only under its own temporary
  directory; reads only from the interpreter's prefixes and library paths (stdlib,
  site-packages), the system library directories, its own ``/proc/<pid>`` and a short list of
  files (``/dev/urandom``, ``/proc/cpuinfo``, ``/etc/ld.so.cache`` and the like). So not the
  user's home, a repository checkout, any other ``sys.path`` entry, ``/etc/passwd``, a listing
  of ``/dev`` or ``/proc``, or another process's ``/proc/<pid>`` (the trainer's ``mem``,
  ``maps``, ``environ`` and ``cmdline``); and no TCP on ABI 4+. Landlock does not cover
  ``chmod`` or timestamp changes: those are NOT prevented on files the user can write.
  If the interpreter's prefix contains the home directory or the working directory, all of it is
  readable by candidates (a one-time warning says so).

Both are best effort. Without them (non-Linux, an old kernel, some containers) the child
is still a separate process with resource limits, and a one-time warning says so. On Linux,
though, a child that reports no seccomp is refused by default (the candidate is not run, one error
is logged); ``ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP=1`` restores the best-effort behaviour. Without
seccomp it can also start processes, so the grader kills its process group, then finds and
kills every process that still carries the run's marker in its environment (Linux, best
effort: that is why ``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1`` exists). Each child writes what
isolation it really got over a pipe that it closes before the candidate runs (the candidate
can't write to it), and the grader decides on that report, not on any earlier probe (on
Windows there is no pipe: the child reports nothing). On Linux the grading process also marks
itself non-dumpable for as long as a candidate child is alive, whatever the kernel offers.

``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION`` refuses to run candidate code (score 0, with a warning that
says which layer is missing) when the child didn't get the isolation asked for (the child itself
exits before the candidate runs): ``1``, ``true``,
``yes`` or ``full`` need BOTH seccomp and Landlock; ``any`` needs at least one. Unset (or ``0``,
``false``, ``no``, ``off``) keeps the best-effort default. Threads can't be started in confined
code; BLAS libraries are told to use one thread so ``numpy`` still imports.
"""

import json
import logging
import os
import selectors
import sys
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# BLAS libraries start threads on import, which the confined child may not do.
SINGLE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
}

# Defines ``isolate(tmp, require)`` for the child; ``tmp`` is the one writable directory.
ISOLATE_SOURCE = "import os, sys\n" + r'''
# The seccomp deny list, in one place. name -> (x86_64, aarch64); None: that architecture has no such call.
# The numbers are checked against the kernel headers (tests/test_code_confinement.py).
SECCOMP_DENY = [
    ("signals to other processes (a completion could SIGKILL the trainer through os.getppid())", {
        "kill": (62, 129), "tkill": (200, 130), "tgkill": (234, 131), "rt_sigqueueinfo": (129, 138),
        "rt_tgsigqueueinfo": (297, 240), "pidfd_send_signal": (424, 424)}),
    ("handles on other processes", {
        "pidfd_open": (434, 434), "pidfd_getfd": (438, 438), "process_madvise": (440, 440),
        "process_mrelease": (448, 448)}),
    ("reading or tracing other processes", {
        "ptrace": (101, 117), "process_vm_readv": (310, 270), "process_vm_writev": (311, 271),
        "kcmp": (312, 272), "perf_event_open": (298, 241)}),
    ("new processes, threads and programs", {
        "clone": (56, 220), "clone3": (435, 435), "fork": (57, None), "vfork": (58, None),
        "execve": (59, 221), "execveat": (322, 281)}),
    ("sockets and io_uring (which can do what the filter refuses)", {
        "socket": (41, 198), "io_uring_setup": (425, 425), "io_uring_enter": (426, 426),
        "io_uring_register": (427, 427)}),
    ("SysV IPC and POSIX message queues: segments and queues the trainer or NCCL created are reachable by any "
     "process of the same user", {
        "shmget": (29, 194), "shmat": (30, 196), "shmctl": (31, 195), "shmdt": (67, 197),
        "semget": (64, 190), "semop": (65, 193), "semctl": (66, 191), "semtimedop": (220, 192),
        "msgget": (68, 186), "msgsnd": (69, 189), "msgrcv": (70, 188), "msgctl": (71, 187),
        "mq_open": (240, 180), "mq_unlink": (241, 181), "mq_timedsend": (242, 182),
        "mq_timedreceive": (243, 183), "mq_notify": (244, 184), "mq_getsetattr": (245, 185)}),
    ("namespaces, mounts and other system administration", {
        "unshare": (272, 97), "setns": (308, 268), "mount": (165, 40), "umount2": (166, 39),
        "pivot_root": (155, 41), "chroot": (161, 51), "open_by_handle_at": (304, 265),
        "name_to_handle_at": (303, 264), "move_mount": (429, 429), "open_tree": (428, 428),
        "fsopen": (430, 430), "fsconfig": (431, 431), "fsmount": (432, 432), "fspick": (433, 433),
        "mount_setattr": (442, 442), "quotactl": (179, 60), "quotactl_fd": (443, 443),
        "swapon": (167, 224), "swapoff": (168, 225), "reboot": (169, 142), "kexec_load": (246, 104),
        "kexec_file_load": (320, 294), "init_module": (175, 105), "finit_module": (313, 273),
        "delete_module": (176, 106), "acct": (163, 89), "syslog": (103, 116)}),
    ("kernel objects shared with the trainer (the user's keyrings, bpf, userfaultfd, fanotify)", {
        "bpf": (321, 280), "userfaultfd": (323, 282), "fanotify_init": (300, 262), "add_key": (248, 217),
        "request_key": (249, 218), "keyctl": (250, 219)}),
    ("credentials (no-ops without privilege, refused anyway)", {
        "setuid": (105, 146), "setgid": (106, 144), "setreuid": (113, 145), "setregid": (114, 143),
        "setresuid": (117, 147), "setresgid": (119, 149), "setfsuid": (122, 151), "setfsgid": (123, 152),
        "setgroups": (116, 159), "capset": (126, 91)}),
    ("scheduling and I/O priority that can name a whole user (setpriority(PRIO_USER, 0))", {
        "setpriority": (141, 140), "ioprio_set": (251, 30)}),
    ("file changes Landlock does not cover: ownership, extended attributes and truncate(2) by path (ftruncate and "
     "O_TRUNC are covered; chmod and utime are not refused, shutil.copy2 and much other code use them)", {
        "chown": (92, None), "fchown": (93, 55), "lchown": (94, None), "fchownat": (260, 54),
        "setxattr": (188, 5), "lsetxattr": (189, 6), "fsetxattr": (190, 7),
        "removexattr": (197, 14), "lremovexattr": (198, 15), "fremovexattr": (199, 16),
        "truncate": (76, 45)}),
]
# Allowed only on the caller itself: name -> (x86_64, aarch64, index of the pid argument); pid 0 is the caller.
SECCOMP_SELF_ONLY = {
    "prlimit64": (302, 261, 0), "sched_setparam": (142, 118, 0), "sched_setscheduler": (144, 119, 0),
    "sched_setaffinity": (203, 122, 0), "sched_setattr": (314, 274, 0), "migrate_pages": (256, 238, 0),
    "move_pages": (279, 239, 0),
}
# Refused for some commands: name -> (x86_64, aarch64, index of the command argument, commands). The kernel reads
# the command as 32 bits. F_SETOWN, F_SETSIG, F_SETOWN_EX and FIOSETOWN/SIOCSPGRP make the kernel signal another
# process (SIGIO, or the F_SETSIG signal) when a pipe or file becomes ready.
SECCOMP_CMD_DENY = {
    "fcntl": (72, 25, 1, (8, 10, 15)),
    "ioctl": (16, 29, 1, (0x8901, 0x8902)),
}


def seccomp_program(m):
    """The BPF program, a list of (code, jt, jf, k), for machine ``m``; None if the machine is unknown."""
    if m in ("x86_64", "AMD64"):
        arch, x32, col = 0xC000003E, True, 0
    elif m in ("aarch64", "arm64"):
        arch, x32, col = 0xC00000B7, False, 1
    else:
        return None
    deny = [nrs[col] for _, group in SECCOMP_DENY for nrs in group.values() if nrs[col] is not None]
    pid0 = [(v[col], v[2]) for v in SECCOMP_SELF_ONLY.values()]
    cmds = [(v[col], v[2], v[3]) for v in SECCOMP_CMD_DENY.values()]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    ALLOW, EPERM = 0x7FFF0000, 0x00050001
    D = -1  # a jump to the final EPERM, patched below
    blocks = [(nr, [(LD, 0, 0, 16 + 8 * i), (JEQ, 0, D, 0), (LD, 0, 0, 20 + 8 * i), (JEQ, 0, D, 0),
                    (RET, 0, 0, ALLOW)]) for nr, i in pid0]
    blocks += [(nr, [(LD, 0, 0, 16 + 8 * i)] + [(JEQ, D, 0, v) for v in vals] + [(RET, 0, 0, ALLOW)])
               for nr, i, vals in cmds]
    n_checks = (1 if x32 else 0) + len(deny) + len(blocks)
    head = [(LD, 0, 0, 4), (JEQ, 1, 0, arch), (RET, 0, 0, EPERM), (LD, 0, 0, 0)]
    body_at = len(head) + n_checks + 1  # after the default ALLOW
    deny_at = body_at + sum(len(b) for _, b in blocks)
    prog = list(head)
    if x32:
        prog.append((JGE, deny_at - len(prog) - 1, 0, 0x40000000))
    for n in deny:
        prog.append((JEQ, deny_at - len(prog) - 1, 0, n))
    start = body_at
    for nr, b in blocks:
        prog.append((JEQ, start - len(prog) - 1, 0, nr))
        start += len(b)
    prog.append((RET, 0, 0, ALLOW))
    for nr, b in blocks:
        for code, jt, jf, k in b:
            here = len(prog)
            prog.append((code, deny_at - here - 1 if jt == D else jt, deny_at - here - 1 if jf == D else jf, k))
    prog.append((RET, 0, 0, EPERM))
    assert len(prog) == deny_at + 1 and deny_at < 255  # jump offsets are one byte
    return prog


def drop_capabilities(libc, status_path="/proc/self/status", last_cap_path="/proc/sys/kernel/cap_last_cap"):
    """Give up every capability before the candidate runs, so a trainer that runs as root does not hand them over.

    Clears the bounding set (when we hold CAP_SETPCAP), then the effective, permitted and inheritable sets with
    capset(), then the ambient set. Returns True if the status file then shows no effective, permitted, inheritable
    or ambient capability (also when there were none to begin with); False if any is left or it can't be checked.
    The bounding set of an unprivileged process stays full: with no_new_privs and no execve it gives nothing."""
    try:
        import ctypes, platform, struct
        try:
            with open(last_cap_path) as f:
                last = int(f.read())
        except Exception:
            last = 40
        for cap in range(last + 1):
            libc.prctl(24, cap, 0, 0, 0)  # PR_CAPBSET_DROP (EPERM without CAP_SETPCAP: nothing to drop then)
        libc.prctl(47, 4, 0, 0, 0)  # PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL
        nr = {"x86_64": 126, "AMD64": 126, "aarch64": 91, "arm64": 91}.get(platform.machine())
        if nr is None:
            return False
        header = ctypes.create_string_buffer(struct.pack("Ii", 0x20080522, 0))  # _LINUX_CAPABILITY_VERSION_3, this process
        data = ctypes.create_string_buffer(24)  # two (effective, permitted, inheritable) u32 triples, all zero
        libc.syscall(ctypes.c_long(nr), ctypes.c_void_p(ctypes.addressof(header)), ctypes.c_void_p(ctypes.addressof(data)))
        seen = {}
        with open(status_path) as f:
            for line in f:
                if line.startswith("Cap"):
                    key, _, value = line.partition(":")
                    seen[key] = int(value.strip(), 16)
        return all(seen.get(k, 0) == 0 for k in ("CapInh", "CapPrm", "CapEff", "CapAmb")) and "CapEff" in seen
    except Exception:
        return False


def isolate(tmp, require):
    st = {"seccomp": False, "landlock": 0, "caps_dropped": False}
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
    st["caps_dropped"] = drop_capabilities(libc)  # first: seccomp will refuse capset, and Landlock needs no capability
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
            # Trees: the interpreter's own prefixes and library paths (and nothing else from sys.path: an
            # editable install or a PYTHONPATH-style entry is often a checkout under the user's home, and a
            # candidate needs none of it), the system library directories, this process's own /proc entry
            # (/proc/self is a symlink to it) and the CPU topology that BLAS and OpenMP read.
            trees = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                     "/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", "/nix", "/run/current-system",
                     "/proc/%d" % os.getpid(), "/sys/devices/system/cpu", "/etc/ld.so.conf.d"}
            import sysconfig
            for key in ("stdlib", "platstdlib", "purelib", "platlib"):
                try:
                    trees.add(sysconfig.get_path(key))
                except Exception:
                    pass  # a missing sysconfig path means one fewer readable directory; Landlock stays default-deny
            trees.discard(None)
            # Single files: what the dynamic loader, the C library and Python read. Not /proc/<other pid>
            # (so not the trainer's mem, maps, environ or cmdline), not a listing of /dev or /etc, not
            # /etc/passwd.
            files = ["/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/localtime", "/etc/timezone", "/etc/os-release",
                     "/etc/nsswitch.conf", "/dev/urandom", "/dev/random", "/dev/zero",
                     "/proc/cpuinfo", "/proc/meminfo", "/proc/stat", "/proc/version", "/proc/loadavg",
                     "/proc/uptime", "/proc/sys/vm/overcommit_memory"]
            for p in trees:
                allow(p, read)
            for p in files:
                allow(p, 4)  # READ_FILE
            allow(tmp, fs)
            allow("/dev/null", 2 | 4 | (1 << 14))
            if sc(446, rfd, 0) == 0:  # landlock_restrict_self
                st["landlock"] = abi
            os.close(rfd)
    except Exception:
        pass  # best effort: Landlock unavailable or refused, so the child reports landlock=0 and is never counted as confined
    # --- seccomp: no signals to others, no ptrace, no sockets, no new processes or programs,
    #     and nothing that changes or reads another process (limits, scheduling, memory, perf) ---
    try:
        prog = seccomp_program(platform.machine())
        if prog:
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
        pass  # best effort: seccomp unavailable or refused, so the child reports seccomp=False
    return st
'''


_unrecognised_logged = set()


def require_isolation() -> str:
    """``""`` (best effort), ``"full"`` (seccomp and Landlock both) or ``"any"`` (at least one)."""
    value = os.environ.get("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return ""
    if value == "any":
        return "any"
    if value not in ("1", "true", "yes", "full") and value not in _unrecognised_logged:
        _unrecognised_logged.add(value)
        logger.warning("Unrecognised ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=%r: treating it as 'full' "
                       "(1, true, yes, full: seccomp and Landlock; any: at least one; 0: off).", value)
    return "full"


def isolation_met(mode: str, state: Dict[str, Any]) -> bool:
    if mode == "full":
        return bool(state.get("seccomp") and state.get("landlock") and state.get("caps_dropped", True))
    return bool(state.get("seccomp") or state.get("landlock"))


def allow_no_seccomp() -> bool:
    """``ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP=1`` (or true, yes): run candidates on Linux even where seccomp is not active."""
    return os.environ.get("ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP", "").strip().lower() in ("1", "true", "yes")


def refuses_without_seccomp() -> bool:
    """Linux only (not macOS or Windows, which have no seccomp by design): candidate code is not run, and scores 0,
    unless the child reports that seccomp is active, or the opt-out above is set."""
    return sys.platform.startswith("linux") and os.name != "nt" and not allow_no_seccomp()


_unmet_logged = set()
_partial_logged: Dict[str, bool] = {}


def judge_report(state: Any) -> bool:
    """Judge what a candidate child reported about its own isolation; True if its result may be used.

    ``state`` is the child's ``{"seccomp": bool, "landlock": ABI}`` (what its own ``isolate()`` really achieved,
    written before the candidate ran), None if it reported nothing, or what ``read_report`` returns on a timeout. Logs once if the isolation is partial.
    With ``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION`` set, an unmet requirement (including no report at all) is
    logged once per reason and refused: the child itself exits without running the candidate in that case, so
    this is the second check. Nothing here is taken from a probe: only from the child that ran the candidate."""
    st = state if isinstance(state, dict) else {}
    no_report = state is None or isinstance(state, _TimedOut)
    nothing = ("the child did not report in time" if isinstance(state, _TimedOut)
               else "this platform has no report channel" if os.name == "nt"
               else "the child reported nothing")
    if not (st.get("seccomp") and st.get("landlock")) and not _partial_logged:
        _partial_logged["done"] = True
        logger.warning(
            "AlignTune code-reward sandbox: kernel isolation is partial (seccomp=%s, landlock ABI=%s%s). Candidate "
            "code still runs in a separate process with limits, but may be able to signal other "
            "processes or read/write the user's files%s. Set ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1 (both layers) or "
            "=any (at least one) to refuse to run candidate code without them.",
            st.get("seccomp"), st.get("landlock"), "; " + nothing if no_report else "",
            "; without seccomp it can also start processes" if not st.get("seccomp") else "")
    if st.get("caps_dropped") is False and st.get("seccomp") and "caps" not in _partial_logged:
        _partial_logged["caps"] = True
        logger.warning("AlignTune code-reward sandbox: the candidate's process still holds Linux capabilities after the "
                       "attempt to drop them (caps_dropped=false); ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1 refuses to run it.")
    if refuses_without_seccomp() and not st.get("seccomp"):
        why = ("seccomp is not active in the grading child on this Linux host (%s), so a candidate could start processes "
               "and signal other programs. Candidate code is not run and scores 0. Set ALIGNTUNE_SANDBOX_ALLOW_NO_SECCOMP=1 "
               "to run candidates without seccomp anyway (best effort), or run where the kernel allows seccomp filters."
               % (nothing if no_report else "the kernel refused the filter or does not have it"))
        if why not in _unmet_logged:
            _unmet_logged.add(why)
            logger.error("AlignTune code-reward sandbox: " + why)
        return False
    mode = require_isolation()
    if not mode or isolation_met(mode, st):
        return True
    missing = [name for name, on in (("seccomp", st.get("seccomp")), ("Landlock", st.get("landlock"))) if not on]
    if mode == "any":
        missing = ["seccomp and Landlock (neither is available)"]
    why = ("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=%s needs %s, but the candidate's process got %s; candidate code "
           "is not run and scores 0." % (os.environ.get("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", ""),
                                         "both seccomp and Landlock" if mode == "full" else "seccomp or Landlock",
                                         "no report (%s)" % nothing if no_report else " and ".join(missing)))
    if why not in _unmet_logged:
        _unmet_logged.add(why)
        logger.warning(why)
    return False


_MAX_REPORT_BYTES = 4096
REPORT_TIMEOUT = 10.0  # seconds a child gets to report (interpreter start-up and isolate()); see report_timeout


def report_timeout(candidate_timeout: float) -> float:
    """How long to wait for a child's report: ``REPORT_TIMEOUT``, more when the candidate's own timeout is long
    (a slow or loaded machine starts a child slowly), at most two minutes."""
    return min(max(REPORT_TIMEOUT, 2.0 * float(candidate_timeout)), 120.0)


class _TimedOut(dict):
    """What ``read_report`` returns when the child did not report in time: falsy, like no report, but the log can say why."""


def parse_report(data: bytes) -> Any:
    """The state in ``data`` (what a child wrote to the report pipe); None unless it is exactly a plain state."""
    try:
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("seccomp"), bool) and type(obj.get("landlock")) is int:
            state = {"seccomp": obj["seccomp"], "landlock": obj["landlock"]}
            if "caps_dropped" in obj:
                if not isinstance(obj["caps_dropped"], bool):
                    return None
                state["caps_dropped"] = obj["caps_dropped"]
            return state
    except Exception:
        pass  # not a plain state report: None, which every caller treats as no isolation
    return None


def read_report(fd: int, timeout: float = REPORT_TIMEOUT) -> Any:
    """Read the child's report from the read end of its report pipe (at most 4 KiB, until it closes the pipe or
    ``timeout`` passes). The state if there is a valid one. None if the child died before reporting or sent anything
    but a plain state; an empty ``_TimedOut`` if it did not report in time. The child writes the report right after
    ``isolate()`` and closes the pipe before the candidate runs, so the candidate can't write to it or know its number.

    Uses ``selectors`` (epoll, kqueue or poll), not ``select.select``, which fails for descriptors >= 1024: a
    trainer with many open files would otherwise get no report from any child."""
    data, deadline = b"", time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            while len(data) <= _MAX_REPORT_BYTES:
                left = deadline - time.monotonic()
                if left <= 0 or not sel.select(left):
                    return _TimedOut()
                chunk = os.read(fd, _MAX_REPORT_BYTES + 1 - len(data))
                if not chunk:
                    break  # the child closed its end
                data += chunk
    except (OSError, ValueError):
        return None
    return parse_report(data) if len(data) <= _MAX_REPORT_BYTES else None


_prefix_warned: Dict[str, bool] = {}


def warn_broad_prefix() -> None:
    """Warn once if an interpreter prefix contains the home directory or the working directory: the child may read
    its whole prefix, so a candidate could then read everything under the home directory or the checkout."""
    if _prefix_warned:
        return
    _prefix_warned["done"] = True
    try:
        inside = {"the home directory": os.path.expanduser("~"), "the working directory": os.getcwd()}
        prefixes = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
        for what, path in inside.items():
            path = os.path.realpath(path)
            for prefix in sorted(prefixes):
                if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                    logger.warning(
                        "AlignTune code-reward sandbox: the interpreter prefix %s contains %s (%s), so a candidate "
                        "can read all of it. Run the trainer from an interpreter whose prefix is a virtualenv or "
                        "system directory of its own.", prefix, what, path)
                    break
    except Exception:
        pass  # a warning only


def kill_marked(marker: str, wait: float = 2.0) -> bool:
    """Linux: kill every process whose initial environment carries ``ALIGNTUNE_GRADER_RUN=<marker>`` (a candidate's
    descendants, which inherit it) and return True once none is left; False if some survive ``wait`` seconds.

    Best effort: a process that rewrites its own environment block or starts a program with another
    environment isn't found. Without seccomp a candidate can do both, which is what
    ``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1`` is for. Nothing to do (True) where there is no /proc."""
    import signal
    import time

    if not os.path.isdir("/proc"):
        return True
    needle = b"ALIGNTUNE_GRADER_RUN=" + marker.encode()  # the variable the grader puts in the child's environment
    me = os.getpid()
    deadline = time.monotonic() + wait
    while True:
        found = []
        for name in os.listdir("/proc"):
            if not name.isdigit() or int(name) == me:
                continue
            try:
                with open("/proc/%s/environ" % name, "rb") as f:
                    if needle in f.read():
                        found.append(int(name))
            except OSError:
                continue  # gone, or not ours to read
        if not found:
            return True
        for pid in found:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass  # already gone, or not ours to signal: nothing left to kill
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)


def group_gone(pgid: int, wait: float = 1.0) -> bool:
    """Wait briefly for every process of group ``pgid`` to be gone (after it was killed); True if it is."""
    import time
    deadline = time.monotonic() + wait
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except (OSError, AttributeError):
            return True  # not ours to signal any more (or no process groups here): nothing of ours is in it
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)


# Defence in depth for the trainer's memory. Landlock refuses a confined process any ptrace-checked access to
# the trainer (``/proc/<pid>/mem``, ``process_vm_readv``) even where Yama allows it, but nothing here relies on
# it being there: the grading process also drops its "dumpable" flag for as long as any candidate child is
# alive, which a process of the same user without CAP_SYS_PTRACE can't read or write through /proc either
# (``mem``, ``environ``, ``oom_score_adj``). The flag is restored afterwards; concurrent graders share one hold,
# counted under a lock. The only cost is that debuggers and profilers of the trainer can't attach for the
# seconds a candidate runs. Where seccomp is missing the candidate can fork, so the grader keeps the hold
# after the child's death until its process group is gone and no process carrying the run's environment
# marker is left, for at most about 5 seconds; if something survives that, it logs once and releases the hold
# anyway (a trainer must not stay non-dumpable for good because of one stuck process).
_hold_lock = threading.Lock()
_hold: Dict[str, Any] = {"n": 0, "restore": False}


def hold_parent() -> bool:
    """Call before starting a candidate child (Linux, unconditionally); pair with ``release_parent()``."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        with _hold_lock:
            if _hold["n"] == 0:
                # PR_GET_DUMPABLE == 1 (not 0, not 2) is the only state it is safe to flip and flip back
                _hold["restore"] = libc.prctl(3, 0, 0, 0, 0) == 1 and libc.prctl(4, 0, 0, 0, 0) == 0
            _hold["n"] += 1
        return True
    except Exception:
        return False


def release_parent() -> None:
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        with _hold_lock:
            _hold["n"] -= 1
            if _hold["n"] == 0 and _hold["restore"]:
                libc.prctl(4, 1, 0, 0, 0)  # PR_SET_DUMPABLE
                _hold["restore"] = False
    except Exception:
        pass  # restoring is best effort; if it fails the trainer stays non-dumpable, the safe direction


# --- starting and cleaning up a candidate child (shared by _code_grader and _code_sandbox) ---------------------------

def start_child(argv, timeout: float, env: Dict[str, str], first_input: bytes = b"", **popen: Any):
    """Start a candidate child; return it, the marker in its environment, and the isolation it reported.

    The child gets the write end of a pipe as its last argument, writes its state there right after ``isolate()``
    and closes it before the candidate runs; the report is read here before anything else happens (waiting at most
    ``report_timeout(timeout)``). ``first_input`` is written to the child's stdin first, for a child that reads its
    job from there before it confines itself. A child that sent no valid report counts as having no isolation. On
    Windows there is no pipe to pass (``pass_fds`` is POSIX-only): the child gets -1, reports nothing and the run
    goes on, unless ``ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION`` refuses it."""
    import subprocess

    warn_broad_prefix()
    marker = os.urandom(8).hex()  # a run-unique value in the child's environment, inherited by its descendants
    pipe = os.name != "nt"
    r, w = os.pipe() if pipe else (-1, -1)
    try:
        proc = subprocess.Popen(list(argv) + [str(w)], env=dict(env, ALIGNTUNE_GRADER_RUN=marker),
                                start_new_session=True, **popen, **({"pass_fds": (w,)} if pipe else {}))
    except BaseException:
        if pipe:
            os.close(r)
            os.close(w)
        raise
    if pipe:
        os.close(w)  # the child's copy is the only one left: EOF means it closed it or died
    try:
        if first_input:
            proc.stdin.write(first_input)
            proc.stdin.flush()
        state = read_report(r, report_timeout(timeout)) if pipe else None
    except BaseException:
        kill_group(proc)
        raise
    finally:
        if pipe:
            os.close(r)
    return proc, marker, state


def kill_group(proc) -> None:
    """Kill the child's process group (a child that calls setsid leaves it: see ``reap``) and wait for the child."""
    import signal

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()
    proc.wait()


REAP_BUDGET = 5.0  # seconds the hold waits for a candidate's leftovers (see hold_parent)
_survivors_logged: Dict[str, bool] = {}


def reap(proc, marker: str, state: Any) -> None:
    """Kill the child's process group. Where the child says seccomp was not active (or said nothing) it could have
    forked or called setsid: wait until the group is gone, then find and kill every process that still carries the
    run's marker in its environment (Linux). The caller releases the trainer's non-dumpable hold only after this
    returns, so the hold lasts until nothing of the candidate is left, but for at most ``REAP_BUDGET`` seconds: if
    something survives that, a warning is logged once and the hold is released anyway. Best effort."""
    kill_group(proc)
    if not (state or {}).get("seccomp"):
        deadline = time.monotonic() + REAP_BUDGET
        gone = kill_marked(marker, wait=REAP_BUDGET) if marker else True
        gone = group_gone(proc.pid, wait=max(deadline - time.monotonic(), 0.05)) and gone
        if not gone and not _survivors_logged:
            _survivors_logged["done"] = True
            logger.warning("AlignTune code-reward sandbox: a process started by the candidate survived the cleanup "
                           "(the kernel offers no seccomp here); the trainer's protection is released anyway. Set "
                           "ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION=1 to refuse to run candidates without seccomp.")
