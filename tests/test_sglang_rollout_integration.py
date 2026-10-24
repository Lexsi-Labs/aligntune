"""
GPU integration test: weights synced by SGLangRollout really change what the
SGLang server computes, and match the trainer's (LoRA) policy.

Compares per-token logprobs of a fixed continuation (greedy text comparisons
are too sensitive to bf16 kernel differences). Needs a running server for
the same model and a CUDA GPU, e.g.:
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000
    SGLANG_URL=http://127.0.0.1:30000 pytest tests/test_sglang_rollout_integration.py
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from aligntune.backends.trl.rl.rollout.sglang_rollout import SGLangRollout

SGLANG_URL = os.environ.get("SGLANG_URL")
MODEL = os.environ.get("SGLANG_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(not SGLANG_URL or not torch.cuda.is_available(), reason="needs SGLANG_URL and a GPU"),
]


@torch.no_grad()
def hf_logprobs(model, seq, start):
    logits = model(torch.tensor([seq], device="cuda")).logits[0, :-1].float()
    logps = logits.log_softmax(-1).gather(-1, torch.tensor(seq[1:], device="cuda")[:, None])[:, 0]
    return logps[start - 1:].cpu()


def sglang_logprobs(rollout, seq, start):
    out = rollout.client._request("POST", "/generate", {
        "input_ids": seq,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": start - 1,
    })
    return torch.tensor([lp for lp, _, _ in out["meta_info"]["input_token_logprobs"] if lp is not None])


def test_synced_lora_policy_is_what_sglang_computes(tmp_path):
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name three colors."}], add_generation_prompt=True, tokenize=True,
        return_dict=True,
    )["input_ids"]
    seq = prompt + tokenizer("Red, green and blue are three common colors people see every day.")["input_ids"]
    start = len(prompt)

    rollout = SGLangRollout(SGLANG_URL, sync_dir=str(tmp_path))
    base = hf_logprobs(model, seq, start)
    noise = (sglang_logprobs(rollout, seq, start) - base).abs().mean()  # bf16 kernel mismatch
    assert noise < 0.1, f"server does not start at the base weights (error {noise:.3f})"

    torch.manual_seed(0)
    policy = peft.get_peft_model(model, peft.LoraConfig(
        r=8, lora_alpha=2, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False,
    ))
    target = hf_logprobs(policy, seq, start)
    shift = (target - base).abs().mean()
    assert shift > 0.5, f"random LoRA too weak to test sync (shift={shift:.3f})"

    trainer = SimpleNamespace(
        model=policy,
        is_fsdp_enabled=False,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m, is_main_process=True, wait_for_everyone=lambda: None),
        _dist=SimpleNamespace(gather_params=lambda params: nullcontext()),
        _log_metric=lambda name, value: None,
    )
    rollout.sync_weights(trainer)

    error = (sglang_logprobs(rollout, seq, start) - target).abs().mean()
    print(f"noise {noise:.3f}, policy shift {shift:.3f}, SGLang-vs-policy error after sync {error:.3f} nats/token")
    assert error < 0.2 * shift, f"synced server logprobs off by {error:.3f} (policy shift {shift:.3f})"
