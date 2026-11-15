"""``chat_template: "auto"`` keeps the tokenizer's template on every backend.

Config files default ``chat_template`` to ``"auto"``. The model loader treated
it as a named Unsloth template and imported ``unsloth.chat_templates`` even on
the TRL backend. Importing Unsloth replaces TRL's ``SFTTrainer`` (on macOS with
MLX the replacement then fails), and ``"auto"`` is not an Unsloth template name,
so nothing was applied anyway.
"""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from aligntune.core.model_loader import setup_tokenizer
from aligntune.core.registry import TaskType
from aligntune.utils.config_utils import parse_config_to_unified


@pytest.fixture
def unsloth_templates():
    """A stand-in ``unsloth.chat_templates`` that records calls."""
    get_chat_template = mock.Mock(side_effect=lambda tokenizer, **_: tokenizer)
    fake = types.ModuleType("unsloth.chat_templates")
    fake.get_chat_template = get_chat_template
    with mock.patch.dict(sys.modules, {"unsloth": types.ModuleType("unsloth"), "unsloth.chat_templates": fake}):
        yield get_chat_template


def _tokenizer():
    return SimpleNamespace(
        pad_token="<pad>", eos_token="</s>", chat_template="{{ messages }}", apply_chat_template=None
    )


def _config(chat_template):
    return SimpleNamespace(dataset=SimpleNamespace(chat_template=chat_template))


@pytest.mark.parametrize("use_unsloth", [False, True])
def test_auto_keeps_the_tokenizer_template(unsloth_templates, use_unsloth):
    tokenizer = setup_tokenizer(_tokenizer(), _config("auto"), TaskType.SFT, use_unsloth=use_unsloth)

    unsloth_templates.assert_not_called()
    assert tokenizer.chat_template == "{{ messages }}"


def test_named_template_does_not_import_unsloth_on_trl(unsloth_templates):
    tokenizer = setup_tokenizer(_tokenizer(), _config("chatml"), TaskType.SFT)

    unsloth_templates.assert_not_called()
    assert tokenizer.chat_template == "{{ messages }}"


def test_named_template_is_applied_on_unsloth(unsloth_templates):
    setup_tokenizer(_tokenizer(), _config("chatml"), TaskType.SFT, use_unsloth=True)

    unsloth_templates.assert_called_once()
    assert unsloth_templates.call_args.kwargs["chat_template"] == "chatml"


def test_parser_keeps_a_dataset_level_chat_template():
    config = {"model": {"name_or_path": "m"}, "datasets": [{"name": "d", "chat_template": "chatml"}]}

    assert parse_config_to_unified(config, training_type="sft")["chat_template"] == "chatml"


def test_parser_top_level_chat_template_wins():
    config = {"chat_template": "llama-3", "model": {"name_or_path": "m"},
              "datasets": [{"name": "d", "chat_template": "chatml"}]}

    assert parse_config_to_unified(config, training_type="sft")["chat_template"] == "llama-3"
