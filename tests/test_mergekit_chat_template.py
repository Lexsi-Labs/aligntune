"""mergekit drops the chat template from the merged model; restore_chat_template puts it back.

Uses the Qwen2.5-0.5B-Instruct tokenizer (no weights, no mergekit run).
"""

import pytest
from transformers import AutoTokenizer

from aligntune.core.merge.mergekit_merger import restore_chat_template

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHAT = [{"role": "user", "content": "hi"}]


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)  # fresh each time: _save changes its template


@pytest.fixture
def template():
    return AutoTokenizer.from_pretrained(MODEL).chat_template


def _save(tokenizer, path, template):
    tokenizer.chat_template = template
    tokenizer.save_pretrained(path)
    return str(path)


def test_template_is_copied_into_a_merged_model_that_lacks_one(tokenizer, template, tmp_path):
    source = _save(tokenizer, tmp_path / "source", template)
    merged = _save(tokenizer, tmp_path / "merged", None)
    assert AutoTokenizer.from_pretrained(merged).chat_template is None

    assert restore_chat_template(source, merged) is True

    restored = AutoTokenizer.from_pretrained(merged)
    assert restored.chat_template == template
    assert restored.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)


def test_existing_template_is_left_alone(tokenizer, tmp_path):
    source = _save(tokenizer, tmp_path / "source", "SOURCE {{ messages }}")
    merged = _save(tokenizer, tmp_path / "merged", "MERGED {{ messages }}")

    assert restore_chat_template(source, merged) is False
    assert AutoTokenizer.from_pretrained(merged).chat_template == "MERGED {{ messages }}"


def test_nothing_to_copy_and_a_bad_path_do_not_raise(tokenizer, tmp_path):
    bare = _save(tokenizer, tmp_path / "bare", None)
    other = _save(tokenizer, tmp_path / "other", None)
    assert restore_chat_template(bare, other) is False
    assert restore_chat_template(bare, tmp_path / "does-not-exist") is False
