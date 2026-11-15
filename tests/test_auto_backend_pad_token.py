"""backend="auto" uses TRL when Unsloth would refuse the tokenizer.

Unsloth needs a pad token distinct from eos. For a tokenizer with none and no
reserved or unk token to reuse (SmolLM2, for example) it stops with "Could not
find a valid pad token". The recommended backend is stubbed to Unsloth and the
tokenizer is a small local one, so nothing is downloaded. Unsloth is never
imported: the check loads Unsloth's ``pad_token.py`` by path, because importing
``unsloth_zoo`` or probing Unsloth patches TRL (and needs a GPU).
"""

from unittest import mock

import importlib.util
import sys

import pytest

tokenizers = pytest.importorskip("tokenizers")
from transformers import PreTrainedTokenizerFast

from aligntune.core.backend_factory import (
    BackendConfig,
    BackendFactory,
    BackendType,
    create_rl_trainer,
    create_sft_trainer,
)


@pytest.fixture(autouse=True)
def unsloth_could_run():
    # Unsloth installed with a CUDA GPU; the check itself never imports Unsloth.
    if importlib.util.find_spec("unsloth_zoo") is None:
        pytest.skip("unsloth_zoo is not installed")
    with mock.patch("aligntune.core.backend_factory._unsloth_could_run", return_value=True):
        yield


def _tokenizer(**special):
    vocab = {"</s>": 0, "<unk>": 1, "<pad>": 2, "hello": 3}
    model = tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>" if "unk_token" in special else None)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer(model), eos_token="</s>", **special)


def _chosen(tokenizer, factory, explicit_backend=None, **kwargs):
    recommended = lambda training_type, algorithm=None: BackendConfig(training_type, BackendType.UNSLOTH, algorithm)
    with mock.patch.object(BackendFactory, "get_recommended_backend", side_effect=recommended), \
            mock.patch.object(BackendFactory, "create_trainer", new=lambda config, backend_config: backend_config), \
            mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
            mock.patch("aligntune.core.backend_factory._enable_unsloth_backend"):
        return factory(model_name="m", dataset_name="d", backend=explicit_backend or "auto", **kwargs).backend


@pytest.mark.parametrize("factory, kwargs", [(create_sft_trainer, {}), (create_rl_trainer, {"algorithm": "dpo"})])
def test_auto_uses_trl_when_unsloth_has_no_pad_token(factory, kwargs):
    unsloth_loaded = {name for name in sys.modules if name.split(".")[0] in ("unsloth", "unsloth_zoo")}
    assert _chosen(_tokenizer(), factory, **kwargs) == BackendType.TRL
    # Neither Unsloth nor its zoo package was imported (both patch TRL on import).
    assert {name for name in sys.modules if name.split(".")[0] in ("unsloth", "unsloth_zoo")} == unsloth_loaded


@pytest.mark.parametrize("special", [{"pad_token": "<pad>"}, {"unk_token": "<unk>"}])
def test_auto_keeps_unsloth_when_it_can_pad(special):
    # A real pad token, or an unk token Unsloth can reuse as pad.
    assert _chosen(_tokenizer(**special), create_sft_trainer) == BackendType.UNSLOTH


def test_explicit_unsloth_is_not_overridden():
    with mock.patch("transformers.AutoTokenizer.from_pretrained") as load, \
            mock.patch.object(BackendFactory, "create_trainer", new=lambda config, backend_config: backend_config), \
            mock.patch("aligntune.core.backend_factory._enable_unsloth_backend"):
        backend = create_sft_trainer(model_name="m", dataset_name="d", backend="unsloth").backend
    assert backend == BackendType.UNSLOTH
    load.assert_not_called()
