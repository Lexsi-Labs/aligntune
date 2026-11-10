"""AlignTune's calls into huggingface_hub, bound to the installed version's real signatures.

The network is off: ``HfApi`` and ``login`` are replaced by recorders that bind the
arguments AlignTune passed to the signature of the real function, so an argument the
installed huggingface_hub does not accept (``private`` on ``upload_file``,
``write_permission`` on ``login``) fails here the way it fails for a user.
"""
import inspect
from pathlib import Path

import pytest

hub = pytest.importorskip("huggingface_hub")


def _binding_recorder(real, calls, name):
    sig = inspect.signature(real)

    def record(*args, **kwargs):
        # bind() raises TypeError for an argument the real function does not take.
        sig.bind(*args, **kwargs)
        calls.append((name, kwargs))

    return record


class _FakeApi:
    calls: list = []

    def __init__(self, token=None):
        for method in ("create_repo", "upload_file", "upload_folder"):
            real = getattr(hub.HfApi, method)
            setattr(self, method, _binding_recorder(real.__get__(self, _FakeApi), self.calls, method))


@pytest.fixture
def api(monkeypatch):
    _FakeApi.calls = []
    monkeypatch.setattr("aligntune.core.export.hf_hub.HfApi", _FakeApi)
    return _FakeApi


def _checkpoint(tmp_path, adapter):
    ckpt = tmp_path / "ckpt"
    (ckpt / "model").mkdir(parents=True)
    if adapter:
        (ckpt / "model" / "adapter_config.json").write_text("{}")
        (ckpt / "model" / "adapter_model.safetensors").write_bytes(b"x")
    else:
        (ckpt / "model" / "config.json").write_text("{}")
        (ckpt / "tokenizer").mkdir()
        (ckpt / "tokenizer" / "tokenizer.json").write_text("{}")
    return ckpt


@pytest.mark.parametrize("private", [True, False])
def test_full_model_upload_sets_visibility_when_creating_the_repo(api, tmp_path, private):
    from aligntune.core.export.hf_hub import HFHubExporter

    exporter = HFHubExporter(repo_id="user/model", private=private, token="t")
    url = exporter.export_model(_checkpoint(tmp_path, adapter=False))

    assert url == "https://huggingface.co/user/model"
    names = [name for name, _ in api.calls]
    assert names[0] == "create_repo" and api.calls[0][1] == {"private": private, "exist_ok": True}
    assert names.count("upload_folder") == 1 and names.count("upload_file") == 1  # the tokenizer file
    assert all("private" not in kwargs for name, kwargs in api.calls if name.startswith("upload"))


def test_adapter_only_upload_binds_to_upload_file(api, tmp_path):
    from aligntune.core.export.hf_hub import HFHubExporter

    exporter = HFHubExporter(repo_id="user/adapter", adapter_only=True, private=True, token="t")
    exporter.export_model(_checkpoint(tmp_path, adapter=True))

    uploaded = sorted(kwargs["path_in_repo"] for name, kwargs in api.calls if name == "upload_file")
    assert uploaded == ["adapter_config.json", "adapter_model.safetensors"]


def test_setup_hf_auth_logs_in_with_the_token_the_real_login_accepts(monkeypatch):
    import aligntune.utils.auth as auth

    calls = []
    monkeypatch.setattr(auth, "login", _binding_recorder(hub.login, calls, "login"))
    users = iter([Exception("not logged in"), {"name": "someone"}])

    def whoami():
        user = next(users)
        if isinstance(user, Exception):
            raise user
        return user

    monkeypatch.setattr(auth, "whoami", whoami)

    assert auth.setup_hf_auth(token="hf_test", write_permission=True) is True
    assert calls == [("login", {"token": "hf_test"})]


def test_no_user_facing_text_names_the_removed_huggingface_cli_command():
    root = Path(__file__).resolve().parents[1] / "src" / "aligntune"
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py") if "huggingface-cli" in p.read_text()]
    assert offenders == [], "huggingface_hub 2.x no longer installs huggingface-cli; say `hf auth login`"
