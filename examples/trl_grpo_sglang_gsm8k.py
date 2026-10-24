"""
GRPO on a GSM8K subset with SGLang rollouts, plus a timing harness to compare
against the default transformers generation path.

1. Start SGLang (any venv; it only talks HTTP + reads the weight-sync dir):
     python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
         --port 30000 --mem-fraction-static 0.25 --skip-tokenizer-init --sleep-on-idle
2. Train:
     python examples/trl_grpo_sglang_gsm8k.py --rollout-backend sglang
     python examples/trl_grpo_sglang_gsm8k.py --rollout-backend hf   # baseline

Each run writes <output_dir>/bench.json with per-step reward, step time and
rollout throughput. Rollout time is measured at TRL's `_generate` boundary
(tokenize + weight sync + generate), so both backends are timed the same way.
"""

import argparse
import json
import time
from pathlib import Path

from transformers import TrainerCallback
from trl import GRPOTrainer

from aligntune.core.backend_factory import create_rl_trainer

parser = argparse.ArgumentParser()
parser.add_argument("--rollout-backend", choices=["hf", "sglang"], default="sglang")
parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--max-steps", type=int, default=60)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()
output_dir = args.output_dir or f"output/grpo_gsm8k_{args.rollout_backend}"

rollouts = []
_generate = GRPOTrainer._generate


def timed_generate(self, prompts):
    start = time.perf_counter()
    out = _generate(self, prompts)
    rollouts.append(
        {
            "step": self.state.global_step,
            "seconds": time.perf_counter() - start,
            "completion_tokens": int(out[4]),  # total_completion_tokens
        }
    )
    return out


GRPOTrainer._generate = timed_generate


class StepTimer(TrainerCallback):
    def __init__(self):
        self.steps = []

    def on_step_begin(self, args, state, control, **kwargs):
        self._start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        self.steps.append(time.perf_counter() - self._start)


trainer = create_rl_trainer(
    model_name=args.model,
    dataset_name="openai/gsm8k",
    config_name="main",
    split="train",
    max_samples=2000,
    column_mapping={"question": "prompt", "answer": "response"},
    system_prompt=(
        "Solve the math problem with brief step-by-step reasoning. "
        "End with 'The answer is: <number>'."
    ),
    algorithm="grpo",
    backend="trl",
    output_dir=output_dir,
    rollout_backend=args.rollout_backend,
    sglang_server_url=args.sglang_url,
    max_steps=args.max_steps,
    batch_size=8,
    gradient_accumulation_steps=8,  # 64 completions = 8 prompts x 8 per optimizer step
    num_generations=8,
    learning_rate=3e-5,
    beta=0.0,
    max_completion_length=256,
    temperature=1.0,
    top_p=1.0,
    use_peft=True,
    lora_r=16,
    lora_alpha=32,
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    rewards=[{"type": "math_verifiable", "weight": 1.0, "params": {}}],
    bf16=True,
    use_gradient_checkpointing=True,
    max_seq_length=768,
    logging_steps=1,
    save_steps=10_000,
    seed=42,
)
timer = StepTimer()
trainer.setup_model()
trainer.setup_data()
trainer.setup_rewards()
trainer.setup_trainer()
trainer.trainer.add_callback(timer)
trainer.trainer.train()

logs = [h for h in trainer.trainer.state.log_history if "reward" in h]
# Skip step 1: it includes CUDA-graph/JIT warmup on both backends.
steady = rollouts[2:]
tokens = sum(r["completion_tokens"] for r in steady)
seconds = sum(r["seconds"] for r in steady)
summary = {
    "rollout_backend": args.rollout_backend,
    "steps": len(timer.steps),
    "mean_step_s": sum(timer.steps[1:]) / max(len(timer.steps) - 1, 1),
    "rollout_tokens_per_s": tokens / max(seconds, 1e-9),
    "mean_rollout_s": seconds / max(len(steady), 1),
    "reward_first5": sum(h["reward"] for h in logs[:5]) / max(len(logs[:5]), 1),
    "reward_last5": sum(h["reward"] for h in logs[-5:]) / max(len(logs[-5:]), 1),
    "rewards": [h["reward"] for h in logs],
    "step_s": timer.steps,
    "rollouts": rollouts,
    "log_history": trainer.trainer.state.log_history,
}
Path(output_dir).mkdir(parents=True, exist_ok=True)
(Path(output_dir) / "bench.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
