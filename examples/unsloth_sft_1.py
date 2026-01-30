# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import json
import torch
from aligntune.core.backend_factory import create_sft_trainer
from aligntune.eval.runner import EvalConfig, run_eval

def _npc_processing_fn(example: dict) -> dict:
    """
    MobileGameNPC sample:
      - player: user message
      - alien: assistant message

    Normalize to AlignTune eval schema:
      - prompt: player
      - completion: alien
    """
    return {
        "prompt": str(example.get("player", "")).strip(),
        "completion": str(example.get("alien", "")).strip(),
    }


def run_evaluation_with_runner(
    model_path: str,
    output_dir: str,
    is_pre_training: bool = False,
    use_lora: bool = False,
    base_model: str | None = None,
) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"{'PRE' if is_pre_training else 'POST'}-TRAINING EVALUATION (EvalConfig)")
    print(f"{'=' * 60}\n")

    eval_output_dir = str(
        Path(output_dir)
        / f"eval_{'pre' if is_pre_training else 'post'}_training_runner"
    )

    # Small dataset: evaluate on the train split (DataManager may create validation/test too)
    cfg = EvalConfig(
        model_path=model_path,
        output_dir=eval_output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        split="train",
        # Use text metrics; provide an explicit list so RLEvaluator doesn't default to RL metrics.
        task_type="generic",
        metrics=["rouge", "bleu", "perplexity"],
        batch_size=2,
        max_samples=50,
        # NPC replies are short; keep generations bounded.
        max_length=128,
        temperature=0.1,
        do_sample=False,
        system_prompt=SYSTEM_PROMPT,
        processing_fn=_npc_processing_fn,
        use_lora=use_lora,
        base_model=base_model,
        use_unsloth=True
    )

    try:
        torch.cuda.empty_cache()
        results = run_eval(cfg)

        # Flatten common keys for printing convenience
        summary = {
            "perplexity": results.get("perplexity", 0.0),
            "rouge_l": results.get("rouge_l", results.get("rouge", 0.0)),
            "bleu": results.get("bleu", 0.0),
            "total_samples": results.get("total", cfg.max_samples or 0),
            "output_dir": eval_output_dir,
        }
        print(
            f"✅ Perplexity: {summary['perplexity']:.4f}, "
            f"ROUGE-L: {summary['rouge_l']:.4f}, BLEU: {summary['bleu']:.4f}"
        )
        return summary
    except Exception as e:  # noqa: BLE001
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        torch.cuda.empty_cache()

# -----------------------------
# Hard-coded example settings
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "bebechien/MobileGameNPC"
DATASET_CONFIG = "martian"  # "martian" or "venusian"
OUTPUT_DIR = "output_gemma_npc_evalconfig"

SYSTEM_PROMPT = (
    "You are a game NPC. Reply in character, matching the speaking style shown in examples."
)
print(
    "SFT Training for Gemma NPC (MobileGameNPC) - Direct API "
    "(TRL Backend, EvalConfig-based evaluation)"
)
results: dict = {
    "model": MODEL_NAME,
    "dataset": f"{DATASET_NAME}:{DATASET_CONFIG}",
    "output_dir": OUTPUT_DIR,
}

# -----------------------------
# Pre-training evaluation
# -----------------------------
print("\nPRE-TRAINING EVALUATION")
print("=" * 60)
pre_eval = run_evaluation_with_runner(
    model_path=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    is_pre_training=True,
    use_lora=False,
    base_model=None,
)
if pre_eval:
    results["pre_training_eval"] = pre_eval

torch.cuda.empty_cache()

# -----------------------------
# Training (full fine-tune by default)
# -----------------------------
print("\nTRAINING")
print("=" * 60)

bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
fp16_ok = bool(torch.cuda.is_available() and not bf16_ok)

trainer = create_sft_trainer(
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    backend="unsloth",
    output_dir=OUTPUT_DIR,
    task_type="instruction_following",
    # Dataset config/subset
    subset=DATASET_CONFIG,
    num_epochs=10,
    # Training duration
    max_steps=200,
    # Batch settings
    batch_size=2,
    gradient_accumulation_steps=1,
    # Optimizer / LR schedule
    learning_rate=5e-5,
    warmup_ratio=0.1,
    # Sequence length / data
    max_seq_length=512,
    max_samples=2000,
    # Prompts / columns
    system_prompt=SYSTEM_PROMPT,
    # Map dataset columns -> expected trainer columns
    column_mapping={"player": "instruction", "alien": "response"},
    # Full fine-tune (no LoRA / no quantization by default)
    use_peft=False,
    bf16=bf16_ok,
    fp16=fp16_ok,
    gradient_checkpointing=False,
    # Saving / logging
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    seed=42,
)

training_results = trainer.train()
results["training"] = {
    "final_loss": (
        training_results.get("train_loss", 0.0)
        if isinstance(training_results, dict)
        else 0.0
    ),
    "model_path": OUTPUT_DIR,
}

# Free training model before evaluation
import gc

del trainer
gc.collect()
torch.cuda.empty_cache()

# -----------------------------
# Post-training evaluation
# -----------------------------
print("\nPOST-TRAINING EVALUATION")
print("=" * 60)
post_eval = run_evaluation_with_runner(
    model_path=OUTPUT_DIR,
    output_dir=OUTPUT_DIR,
    is_pre_training=False,
    use_lora=False,
    base_model=MODEL_NAME,
)
if post_eval:
    results["post_training_eval"] = post_eval

# -----------------------------
# Summary
# -----------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUTPUT_DIR) / "full_results.json", "w") as f:
    json.dump(results, f, indent=2)

