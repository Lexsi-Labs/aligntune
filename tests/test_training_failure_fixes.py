"""Small CPU-only checks for the PPO, Unsloth Online DPO, ORPO, ES and SFT device-map fixes."""
from types import SimpleNamespace

from datasets import Dataset

from aligntune.core.rl.prompt_utils import drop_rows_without_completion_tokens, truncate_prompts


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_truncate_prompts_cuts_only_long_string_prompts():
    data = Dataset.from_dict({"prompt": ["a b c d e", "a b"]})
    out = truncate_prompts(data, _Tok(), 3)
    assert out["prompt"] == ["a b c", "a b"]


def test_truncate_prompts_cuts_chat_messages_and_ignores_missing_column():
    chats = Dataset.from_dict({"prompt": [[{"role": "user", "content": "a b c d e f"}]]})
    out = truncate_prompts(chats, _Tok(), 3)["prompt"][0][0]["content"].split()
    assert 0 < len(out) <= 3
    other = Dataset.from_dict({"text": ["a b c d"]})
    assert truncate_prompts(other, _Tok(), 2) is other
    assert truncate_prompts(None, _Tok(), 2) is None


def test_drop_rows_helper_is_shared_with_the_trl_trainer():
    from aligntune.backends.trl.rl.orpo.orpo import drop_rows_without_completion_tokens as trl_version
    assert trl_version is drop_rows_without_completion_tokens


def _sft_config(device_map):
    from aligntune.core.sft.training_args_builder import build_sft_config
    cfg = SimpleNamespace(
        model=SimpleNamespace(device_map=device_map),
        train=SimpleNamespace(),
        logging=SimpleNamespace(),
    )
    return build_sft_config(cfg, train_dataset=None, output_dir="/tmp/port_test_sft")


def test_device_map_cpu_trains_on_the_cpu():
    assert _sft_config("cpu").use_cpu is True


def test_other_device_maps_leave_use_cpu_alone():
    assert _sft_config("auto").use_cpu is False


def test_save_model_compat_sets_missing_attribute_only():
    from aligntune.utils.trainer_compat import ensure_save_model_compat

    bare = SimpleNamespace()
    assert ensure_save_model_compat(bare).is_distributed_loading_by_transformers is False
    already = SimpleNamespace(is_distributed_loading_by_transformers=True)
    assert ensure_save_model_compat(already).is_distributed_loading_by_transformers is True
    assert ensure_save_model_compat(None) is None


def test_counterfact_grpo_tolerates_chat_message_prompts():
    """Chat-message (list) prompts are unhashable; the gold lookup must not raise."""
    from types import SimpleNamespace

    from aligntune.backends.trl.rl.counterfact_grpo.counterfact_grpo import (
        TRLCounterFactGRPOTrainer,
    )

    fake = SimpleNamespace(prompt_to_data={"2+2?": {"gold_answer": "4"}})
    lookup = TRLCounterFactGRPOTrainer._data_for_prompt
    assert lookup(fake, "2+2?") == {"gold_answer": "4"}
    assert lookup(fake, "unknown") == {}
    assert lookup(fake, [{"role": "user", "content": "2+2?"}]) == {}


def test_es_renders_chat_message_prompts_to_text():
    """ES must hand text, not message lists, to the tokenizer."""
    from types import SimpleNamespace

    from aligntune.backends.es.trainer import ESTrainerBackend

    msgs = [{"role": "user", "content": "hi"}]
    tok = SimpleNamespace(
        chat_template="x",
        apply_chat_template=lambda m, tokenize, add_generation_prompt: "RENDERED:" + m[0]["content"],
    )
    fake = SimpleNamespace(tokenizer=tok)
    render = ESTrainerBackend._render_prompt
    assert render(fake, "plain") == "plain"
    assert render(fake, msgs) == "RENDERED:hi"
    fake_nochat = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None))
    assert render(fake_nochat, msgs) == "user: hi"
