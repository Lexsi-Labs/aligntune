"""The child that runs model-written code confines itself (Linux: seccomp + Landlock).

Every case goes through the release grader's ``run_candidate``. The escapes are
tried with real paths and the real parent process; none of them may succeed, and
the code must still be able to work inside its own directory.
"""

import os
import sys
import tempfile

import pytest

from aligntune.rewards import _code_confine, _code_grader

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="seccomp and Landlock are Linux-only")


def run(code, **kw):
    return _code_grader.run_candidate(code, timeout=20, items=[{"call": 1}], **kw)


def _confinement_available():
    import json, subprocess
    out = subprocess.run(
        [sys.executable, "-I", "-c", _code_confine.ISOLATE_SOURCE + "import json\nprint(json.dumps(isolate(os.getcwd(), False)))"],
        cwd=tempfile.gettempdir(), capture_output=True, text=True,
    ).stdout
    state = json.loads(out.strip().splitlines()[-1])
    return bool(state.get("seccomp")), bool(state.get("landlock"))


SECCOMP, LANDLOCK = _confinement_available()


def test_plain_code_still_runs():
    assert run("def f(x):\n    return x + 1").values == [2]


@pytest.mark.skipif(not SECCOMP, reason="no seccomp here")
@pytest.mark.parametrize("name, code", [
    ("signal the parent", "import os, signal\nos.kill(os.getppid(), signal.SIGKILL)\ndef f(x): return 1"),
    ("open a socket", "import socket\nsocket.socket()\ndef f(x): return 1"),
    ("start a process", "import subprocess\nsubprocess.run(['true'])\ndef f(x): return 1"),
    ("fork", "import os\nos.fork()\ndef f(x): return 1"),
])
def test_seccomp_blocks(name, code):
    assert run(code).status == "error"
    assert os.getpid()  # and the trainer is still here


@pytest.mark.skipif(not LANDLOCK, reason="no Landlock here")
def test_landlock_keeps_writes_and_reads_inside_the_child_directory(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("S3CRET")
    written = tmp_path / "written.txt"
    # tmp_path is outside the system directories and outside the child's own directory
    assert run(f"v = open({str(secret)!r}).read()\ndef f(x): return v").status == "error"
    assert run(f"open({str(written)!r}, 'w').write('x')\ndef f(x): return 1").status == "error"
    assert not written.exists()
    assert run("open('scratch.txt', 'w').write('hi')\ndef f(x): return open('scratch.txt').read()").values == ["hi"]


@pytest.mark.skipif(not LANDLOCK, reason="no Landlock here")
def test_home_directory_is_not_readable_or_writable():
    home = os.path.expanduser("~")
    probe = os.path.join(home, "_aligntune_confine_probe.txt")
    assert run(f"open({probe!r}, 'w').write('x')\ndef f(x): return 1").status == "error"
    assert not os.path.exists(probe)
    assert run(f"import os\nv = len(os.listdir({home!r}))\ndef f(x): return v").status == "error"


def test_require_isolation_runs_when_the_kernel_offers_it(monkeypatch):
    monkeypatch.setenv("ALIGNTUNE_SANDBOX_REQUIRE_ISOLATION", "1")
    result = run("def f(x):\n    return x + 1")
    if SECCOMP or LANDLOCK:
        assert result.values == [2]
    else:
        assert result.status == "error"  # refuses to run unconfined


def test_numpy_imports_and_its_results_are_plain_data():
    pytest.importorskip("numpy")
    assert run("import numpy as np\ndef f(x):\n    return np.array([1, 2, 3])").values == [[1, 2, 3]]
    assert run("import numpy as np\ndef f(x):\n    return np.int64(7)").values == [7]
