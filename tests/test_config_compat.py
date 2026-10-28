"""
Tests for issue #38: merged configs must stay readable by transformers 4.x
loaders (top-level rope_theta / rope_scaling / torch_dtype), without changing
how transformers 5 reads them.
"""

import json

import pytest

from aligntune.utils.config_compat import add_legacy_config_fields, restore_base_config


def _write(path, cfg):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg))


def _read(path):
    return json.loads((path / "config.json").read_text())


def test_adds_rope_theta_and_torch_dtype(tmp_path):
    _write(tmp_path, {
        "model_type": "qwen2",
        "dtype": "float16",
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    })
    added = add_legacy_config_fields(tmp_path)
    cfg = _read(tmp_path)
    assert cfg["rope_theta"] == 1000000.0
    assert cfg["torch_dtype"] == "float16"
    assert "rope_scaling" not in cfg  # default rope type has no scaling block
    assert cfg["rope_parameters"] == {"rope_theta": 1000000.0, "rope_type": "default"}
    assert set(added) == {"rope_theta", "torch_dtype"}


def test_non_default_rope_type_writes_rope_scaling(tmp_path):
    _write(tmp_path, {
        "model_type": "llama",
        "rope_parameters": {
            "rope_theta": 500000.0, "rope_type": "llama3", "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    })
    add_legacy_config_fields(tmp_path)
    cfg = _read(tmp_path)
    assert cfg["rope_theta"] == 500000.0
    assert cfg["rope_scaling"] == {
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
    }


def test_per_layer_type_rope_parameters_use_full_attention(tmp_path):
    _write(tmp_path, {
        "model_type": "gemma3_text",
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    })
    add_legacy_config_fields(tmp_path)
    assert _read(tmp_path)["rope_theta"] == 1000000.0


def test_nested_text_config(tmp_path):
    _write(tmp_path, {
        "model_type": "some_vlm",
        "text_config": {"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}},
    })
    added = add_legacy_config_fields(tmp_path)
    assert _read(tmp_path)["text_config"]["rope_theta"] == 1000000.0
    assert "text_config.rope_theta" in added


def test_existing_keys_are_not_overwritten(tmp_path):
    _write(tmp_path, {
        "rope_theta": 123.0,
        "torch_dtype": "bfloat16",
        "dtype": "float16",
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    })
    assert add_legacy_config_fields(tmp_path) == []
    cfg = _read(tmp_path)
    assert cfg["rope_theta"] == 123.0
    assert cfg["torch_dtype"] == "bfloat16"


def test_missing_config_is_a_noop(tmp_path):
    assert add_legacy_config_fields(tmp_path) == []


def test_transformers5_reads_patched_config_identically(tmp_path):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        dtype="float16",
    )
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    cfg.save_pretrained(before_dir)
    cfg.save_pretrained(after_dir)
    add_legacy_config_fields(after_dir)
    patched = _read(after_dir)
    assert patched["rope_theta"] == 1000000.0
    assert patched["torch_dtype"] == "float16"

    before = transformers.AutoConfig.from_pretrained(before_dir).to_dict()
    after = transformers.AutoConfig.from_pretrained(after_dir).to_dict()
    for d in (before, after):
        d.pop("transformers_version", None)
        d.pop("_name_or_path", None)  # the directory it was loaded from
    assert before == after


def test_restore_base_config_keeps_merged_vocab_and_dtype(tmp_path):
    base = tmp_path / "base"
    merged = tmp_path / "merged"
    _write(base, {"model_type": "qwen2", "rope_theta": 1000000.0, "vocab_size": 100, "torch_dtype": "bfloat16"})
    _write(merged, {
        "model_type": "qwen2", "vocab_size": 104, "dtype": "float16",
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    })
    assert restore_base_config(merged, str(base)) is True
    cfg = _read(merged)
    assert cfg["rope_theta"] == 1000000.0
    assert cfg["vocab_size"] == 104
    assert cfg["torch_dtype"] == "float16"
    assert "rope_parameters" not in cfg


def test_restore_base_config_missing_base_returns_false(tmp_path):
    merged = tmp_path / "merged"
    _write(merged, {"vocab_size": 10})
    assert restore_base_config(merged, str(tmp_path / "does-not-exist")) is False
    assert _read(merged) == {"vocab_size": 10}


# --- tokenizer_config.json -------------------------------------------------

from aligntune.utils.config_compat import add_legacy_tokenizer_fields  # noqa: E402


def _write_tok(path, cfg):
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer_config.json").write_text(json.dumps(cfg))


def _read_tok(path):
    return json.loads((path / "tokenizer_config.json").read_text())


def test_list_extra_special_tokens_become_additional_special_tokens(tmp_path):
    _write_tok(tmp_path, {"extra_special_tokens": ["<|im_start|>", "<|im_end|>"], "eos_token": "<|im_end|>"})
    assert add_legacy_tokenizer_fields(tmp_path) is True
    cfg = _read_tok(tmp_path)
    assert cfg["additional_special_tokens"] == ["<|im_start|>", "<|im_end|>"]
    assert "extra_special_tokens" not in cfg
    assert cfg["eos_token"] == "<|im_end|>"


def test_existing_additional_special_tokens_are_kept_and_merged(tmp_path):
    _write_tok(tmp_path, {"additional_special_tokens": ["<a>"], "extra_special_tokens": ["<a>", "<b>"]})
    add_legacy_tokenizer_fields(tmp_path)
    assert _read_tok(tmp_path)["additional_special_tokens"] == ["<a>", "<b>"]


def test_dict_extra_special_tokens_are_left_alone(tmp_path):
    # The dict form (named model-specific tokens) is what transformers 4.x expects.
    original = {"extra_special_tokens": {"image_token": "<image>"}}
    _write_tok(tmp_path, original)
    assert add_legacy_tokenizer_fields(tmp_path) is False
    assert _read_tok(tmp_path) == original


def test_tokenizers_backend_class_becomes_pretrained_tokenizer_fast(tmp_path):
    # transformers 5 saves Llama 3 style tokenizers with a class 4.x doesn't have.
    _write_tok(tmp_path, {"tokenizer_class": "TokenizersBackend", "eos_token": "<|end_of_text|>"})
    assert add_legacy_tokenizer_fields(tmp_path) is True
    assert _read_tok(tmp_path) == {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|end_of_text|>"}


def test_model_specific_tokenizer_class_is_left_alone(tmp_path):
    original = {"tokenizer_class": "Qwen2Tokenizer"}
    _write_tok(tmp_path, original)
    assert add_legacy_tokenizer_fields(tmp_path) is False
    assert _read_tok(tmp_path) == original


def test_missing_tokenizer_config_is_a_noop(tmp_path):
    assert add_legacy_tokenizer_fields(tmp_path) is False


def test_transformers5_loads_patched_tokenizer_identically(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", eos_token="<eos>",
        extra_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    fast.save_pretrained(before_dir)
    fast.save_pretrained(after_dir)
    if not isinstance(_read_tok(after_dir).get("extra_special_tokens"), list):
        pytest.skip("this transformers version does not save extra_special_tokens as a list")
    assert add_legacy_tokenizer_fields(after_dir) is True

    before = transformers.AutoTokenizer.from_pretrained(before_dir)
    after = transformers.AutoTokenizer.from_pretrained(after_dir)
    text = "hello <|im_start|> world <|im_end|> <eos>"
    assert before(text)["input_ids"] == after(text)["input_ids"]
    assert sorted(before.all_special_tokens) == sorted(after.all_special_tokens)
    assert len(before) == len(after)
