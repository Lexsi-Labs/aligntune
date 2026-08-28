"""
Shared pytest configuration and fixtures.

Usage:
    pytest                           # uses default model
    pytest --model Qwen/Qwen3-0.6B  # override model for all tests

Also contains test-layer compatibility patches that bridge behavioral changes
between the trl version pinned by the source code's assumptions and the trl
version actually installed, without touching src/.
"""

import dataclasses
import functools
import inspect
import logging
import os
import re

import pytest
import trl

logger = logging.getLogger(__name__)


# huggingface_hub>=0.26 removed HfFolder (deprecated since 0.19) in favor of
# get_token()/save_token(), but src/aligntune/core/export/hf_hub.py still
# imports HfFolder directly, which breaks collection of any test that touches
# aligntune.core.export. Polyfill a minimal HfFolder shim so those tests can
# run against the installed huggingface_hub version without touching src/.
import huggingface_hub

if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolderShim:
        @staticmethod
        def get_token():
            return huggingface_hub.get_token()

        @staticmethod
        def save_token(token):
            return huggingface_hub.login(token=token, add_to_git_credential=False)

    huggingface_hub.HfFolder = _HfFolderShim

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name used by tests that load a real model (default: {DEFAULT_MODEL})",
    )


def pytest_configure(config):
    # Expose the model as an env var so module-level code (e.g. parametrize lists)
    # can read it without needing a fixture.
    try:
        model = config.getoption("--model")
    except ValueError:
        model = DEFAULT_MODEL
    os.environ["ALIGNTUNE_TEST_MODEL"] = model


_UNSLOTH_SIGNATURE = re.compile(
    r"aligntune\.backends\.unsloth"
    r"|backend\s*=\s*['\"]unsloth['\"]"
    r"|BACKEND\s*=\s*['\"]unsloth['\"]"
    r"|use_unsloth\s*=\s*True"
    r"|^\s*(?:from|import)\s+unsloth\b",
    re.MULTILINE,
)
_unsloth_file_cache = {}


def pytest_collection_modifyitems(config, items):
    """Run every Unsloth-touching test file after all non-Unsloth files.

    Unsloth monkey-patches transformers attention/norm classes at the class
    level (process-wide), not per-instance. Once a test loads a model via
    Unsloth, any TRL-only test that later loads the same architecture family
    in the same pytest process can inherit the patched class without the
    internal buffers Unsloth's own loader sets up, and crash with e.g.
    "AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'"
    or "RuntimeError: Unsloth: You must specify a `formatting_func`" - even
    though the failing test itself never touches Unsloth. Deferring all
    Unsloth-touching files to the end keeps every non-Unsloth test running
    against unpatched classes, matching how they behave in isolation.

    Detection is content-based (not just filename) - some files touch the
    Unsloth backend (e.g. via `aligntune.backends.unsloth` or a module-level
    `BACKEND = "unsloth"` constant) without "unsloth" appearing in their
    filename.
    """
    def touches_unsloth(item):
        path = str(item.fspath)
        if path not in _unsloth_file_cache:
            try:
                with open(path, encoding="utf-8") as fh:
                    content = fh.read()
            except OSError:
                content = ""
            _unsloth_file_cache[path] = bool(_UNSLOTH_SIGNATURE.search(content))
        return _unsloth_file_cache[path]

    unsloth_items = [item for item in items if touches_unsloth(item)]
    other_items = [item for item in items if not touches_unsloth(item)]
    items[:] = other_items + unsloth_items


@pytest.fixture(scope="session")
def model_name(request):
    """HuggingFace model name for tests that load real weights.

    Override on the command line:
        pytest --model Qwen/Qwen3-0.6B
    """
    return request.config.getoption("--model")


@pytest.fixture(autouse=True, scope="session")
def patch_dpo_config_for_new_trl():
    """trl>=1.0 DPOConfig dropped max_prompt_length/max_completion_length in favor of
    max_length + truncation_mode. SPIN (src/aligntune/backends/trl/rl/spin/spin.py)
    still passes max_prompt_length, which raises TypeError on newer trl.
    """
    original_init = trl.DPOConfig.__init__
    supported = inspect.signature(original_init).parameters

    if "max_prompt_length" in supported:
        yield  # trl<1.0 already supports the kwarg — nothing to patch
        return

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        kwargs.pop("max_prompt_length", None)
        kwargs.pop("max_completion_length", None)
        kwargs.setdefault("truncation_mode", "keep_start")
        return original_init(self, *args, **kwargs)

    trl.DPOConfig.__init__ = patched_init
    yield
    trl.DPOConfig.__init__ = original_init


# trl==1.7.0 (installed) dropped GRPOConfig kwargs that
# src/aligntune/backends/{trl,unsloth}/rl/grpo/grpo.py still pass (e.g.
# max_prompt_length), which raises TypeError before training can start.
# pyproject.toml pins trl==0.23.0, so this only bites when the environment
# drifts to a newer trl. We defensively drop any kwarg GRPOConfig's dataclass
# fields don't recognize, rather than hardcoding one bad key, so this keeps
# working if trl drops/renames more fields later.
_GRPOConfig = trl.GRPOConfig
_valid_fields = {f.name for f in dataclasses.fields(_GRPOConfig)}
_original_grpoconfig_init = _GRPOConfig.__init__


@functools.wraps(_original_grpoconfig_init)
def _patched_grpoconfig_init(self, *args, **kwargs):
    unsupported = {k: v for k, v in kwargs.items() if k not in _valid_fields}
    if unsupported:
        logger.warning(
            "GRPOConfig (trl==%s) does not accept %s - dropping for test compatibility",
            trl.__version__,
            sorted(unsupported),
        )
        kwargs = {k: v for k, v in kwargs.items() if k in _valid_fields}
    _original_grpoconfig_init(self, *args, **kwargs)


_GRPOConfig.__init__ = _patched_grpoconfig_init


