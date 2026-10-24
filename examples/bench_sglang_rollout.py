"""
Rollout-only A/B benchmark: transformers `generate` vs SGLang for one GRPO
generation batch (8 GSM8K prompts x 8 samples, 256 new tokens), alternating
the two backends round by round so background load hits both equally.
Also times the SGLang weight sync (LoRA fold + safetensors write + reload).

    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000 --skip-tokenizer-init --sleep-on-idle
    python examples/bench_sglang_rollout.py --sglang-url http://127.0.0.1:30000
"""

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from types import SimpleNamespace

import peft
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from aligntune.backends.trl.rl.rollout.sglang_rollout import SGLangRollout

parser = argparse.ArgumentParser()
parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--prompts", type=int, default=8)
parser.add_argument("--generations", type=int, default=8)
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
model = peft.get_peft_model(model, peft.LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)).eval()

questions = load_dataset("openai/gsm8k", "main", split="train")["question"]
system = "Solve the math problem with brief step-by-step reasoning. End with 'The answer is: <number>'."


def batch(round_idx):
    start = round_idx * args.prompts
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
        )["input_ids"]
        for q in questions[start:start + args.prompts]
    ]
    return [p for p in prompts for _ in range(args.generations)]  # TRL's repeat layout


@torch.no_grad()
def hf_rollout(prompt_ids):
    tensors = [torch.tensor(p) for p in prompt_ids]
    width = max(len(t) for t in tensors)
    input_ids = torch.stack([torch.cat([torch.full((width - len(t),), tokenizer.pad_token_id), t]) for t in tensors])
    mask = (torch.arange(width)[None, :] >= (width - torch.tensor([len(t) for t in tensors]))[:, None]).long()
    out = model.generate(
        input_ids=input_ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=args.max_new_tokens, do_sample=True,
        temperature=1.0, top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )[:, width:]
    is_eos = out == tokenizer.eos_token_id
    lengths = torch.where(is_eos.any(1), is_eos.int().argmax(1) + 1, torch.tensor(out.size(1), device=out.device))
    return int(lengths.sum())


trainer = SimpleNamespace(
    model=model, is_fsdp_enabled=False,
    accelerator=SimpleNamespace(unwrap_model=lambda m: m, is_main_process=True, wait_for_everyone=lambda: None),
    _dist=SimpleNamespace(gather_params=lambda params: nullcontext()),
    _log_metric=lambda name, value: None,
)
rollout = SGLangRollout(args.sglang_url)
params = {"max_new_tokens": args.max_new_tokens, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
          "repetition_penalty": 1.0, "stop_token_ids": [tokenizer.eos_token_id]}

results = {"hf": [], "sglang": [], "sync_s": []}
for r in range(args.rounds + 1):  # round 0 is warmup
    prompt_ids = batch(r)

    torch.cuda.synchronize()
    t = time.perf_counter()
    hf_tokens = hf_rollout(prompt_ids)
    torch.cuda.synchronize()
    hf_s = time.perf_counter() - t

    t = time.perf_counter()
    rollout.sync_weights(trainer)
    sync_s = time.perf_counter() - t

    t = time.perf_counter()
    outputs = rollout._generate_groups(prompt_ids, params)
    sg_s = time.perf_counter() - t
    sg_tokens = sum(len(o["output_ids"]) for o in outputs)

    print(f"round {r}: hf {hf_s:.2f}s {hf_tokens / hf_s:.0f} tok/s | sglang {sg_s:.2f}s {sg_tokens / sg_s:.0f} tok/s "
          f"| sync {sync_s:.2f}s", flush=True)
    if r:
        results["hf"].append({"s": hf_s, "tokens": hf_tokens})
        results["sglang"].append({"s": sg_s, "tokens": sg_tokens})
        results["sync_s"].append(sync_s)

summary = {
    backend: {
        "median_s": statistics.median(x["s"] for x in results[backend]),
        "median_tokens_per_s": statistics.median(x["tokens"] / x["s"] for x in results[backend]),
    }
    for backend in ("hf", "sglang")
}
summary["median_sync_s"] = statistics.median(results["sync_s"])
print(json.dumps(summary, indent=2))
