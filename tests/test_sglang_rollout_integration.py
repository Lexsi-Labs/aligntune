"""
GPU integration test: weights synced by SGLangRollout really change what the
SGLang server computes, and match the trainer's (LoRA) policy. Covers whichever
sync mode the server supports: run it once against a plain server (full-weight
sync) and once against one launched with --enable-lora (adapter sync).

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
# LoRA layers for the test; GPT-2, for example, needs "c_attn,c_fc".
TARGETS = os.environ.get("SGLANG_TEST_TARGETS", "q_proj,v_proj,down_proj").split(",")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(
        not SGLANG_URL or not torch.cuda.is_available(), reason="needs SGLANG_URL and a GPU"
    ),
]


@torch.no_grad()
def hf_logprobs(model, seq, start):
    logits = model(torch.tensor([seq], device="cuda")).logits[0, :-1].float()
    logps = logits.log_softmax(-1).gather(-1, torch.tensor(seq[1:], device="cuda")[:, None])[:, 0]
    return logps[start - 1 :].cpu()


def sglang_logprobs(rollout, seq, start):
    payload = {
        "input_ids": seq,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": start - 1,
    }
    if rollout.lora_name:  # adapter-sync mode (server launched with --enable-lora)
        payload["lora_path"] = rollout.lora_name
    out = rollout.client._request("POST", "/generate", payload)
    return torch.tensor(
        [lp for lp, _, _ in out["meta_info"]["input_token_logprobs"] if lp is not None]
    )


def test_synced_lora_policy_is_what_sglang_computes(tmp_path):
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Same dtype as the server, which serves the checkpoint dtype (float32 for GPT-2).
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").cuda().eval()
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Name three colors."}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )["input_ids"]
    else:  # base model without a chat template
        prompt = tokenizer("Q: Name three colors.\nA:")["input_ids"]
    # Score the model's own greedy answer: fixed text can be very unlikely under the model (log
    # probs near -30 for instruction-tuned models like Gemma 3), where bf16 rounding alone
    # produces large server-vs-trainer differences.
    seq = model.generate(
        torch.tensor([prompt], device="cuda"), max_new_tokens=24, do_sample=False
    )[0].tolist()
    start = len(prompt)

    rollout = SGLangRollout(SGLANG_URL, sync_dir=str(tmp_path))
    base = hf_logprobs(model, seq, start)
    noise = (sglang_logprobs(rollout, seq, start) - base).abs().mean()  # bf16 kernel mismatch
    assert noise < 0.1, f"server does not start at the base weights (error {noise:.3f})"

    # Use the weakest random LoRA that clearly moves the policy: the server's bf16 error grows
    # faster than the LoRA does, and models differ in how strong a random LoRA has to be.
    for alpha in (1, 2, 4, 8, 16, 32, 64):
        torch.manual_seed(0)
        policy = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=8,
                lora_alpha=alpha,
                target_modules=TARGETS,
                init_lora_weights=False,
            ),
        )
        target = hf_logprobs(policy, seq, start)
        shift = (target - base).abs().mean()
        if shift > 0.5:
            break
        model = policy.unload()
    assert shift > 0.5, f"random LoRA too weak to test sync (shift={shift:.3f})"

    trainer = SimpleNamespace(
        model=policy,
        is_fsdp_enabled=False,
        accelerator=SimpleNamespace(
            unwrap_model=lambda m: m, is_main_process=True, wait_for_everyone=lambda: None
        ),
        _dist=SimpleNamespace(gather_params=lambda params: nullcontext()),
        _log_metric=lambda name, value: None,
    )
    rollout.sync_weights(trainer)

    error = (sglang_logprobs(rollout, seq, start) - target).abs().mean()
    mode = "adapter" if rollout.lora_name else "full-weight"
    print(
        f"[{mode} sync] noise {noise:.3f}, policy shift {shift:.3f}, "
        f"SGLang-vs-policy error after sync {error:.3f} nats/token"
    )
    assert (
        error < 0.2 * shift
    ), f"synced server logprobs off by {error:.3f} (policy shift {shift:.3f})"
