from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict
import torch
from aligntune.core.backend_factory import create_sft_trainer
from aligntune.core.rl.registries import DatasetRegistry
from aligntune.eval.runner import EvalConfig, run_eval


def _ensure_dataset_downloaded() -> None:
    print(f"⬇️  Downloading dataset to {DATA_PATH}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)  # noqa: S310


def _register_trialbench_loader() -> None:
    """
    TRL backend SFT loader supports DatasetRegistry loaders.
    We register a loader name and then train with dataset_name=<that name>.
    """

    def _loader(name: str, split: str = "train", **kwargs) -> Any:
        from datasets import load_dataset

        # This JSONL is a single split; ignore HF config/subset.
        ds = load_dataset("json", data_files=str(DATA_PATH), split=split)
        return ds

    DatasetRegistry.register_loader("trialbench_txgemma_adverse_event", _loader)


def _eval_processing_fn(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Eval expects prompt/completion.
    TrialBench JSONL has:
      - input_text
      - output_text
    """
    return {
        "prompt": str(example.get("input_text", "")).strip(),
        "completion": str(example.get("output_text", "")).strip(),
    }

def run_evaluation_with_runner(model_path: str, is_pre_training: bool, base_model: str | None) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"{'PRE' if is_pre_training else 'POST'}-TRAINING EVALUATION (EvalConfig)")
    print(f"{'=' * 60}\n")

    eval_output_dir = str(
        Path(OUTPUT_DIR) / f"eval_{'pre' if is_pre_training else 'post'}_training_runner"
    )

    cfg = EvalConfig(
        model_path=model_path,
        output_dir=eval_output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Use local JSONL path so DataManager uses JSONLoader
        dataset_name=str(DATA_PATH),
        dataset_config=None,
        split="train[0:200]",
        task_type="generic",
        metrics=["accuracy", "perplexity"],
        batch_size=2,
        max_samples=200,
        # Outputs are short labels ("Yes"/"No")
        max_length=8,
        do_sample=False,
        processing_fn=_eval_processing_fn,
        use_lora=(model_path != HF_MODEL_ID),
        base_model=base_model,
    )

    try:
        torch.cuda.empty_cache()
        results = run_eval(cfg)
        summary = {
            "accuracy": results.get("accuracy", 0.0),
            "perplexity": results.get("perplexity", 0.0),
            "total_samples": results.get("total", cfg.max_samples or 0),
            "output_dir": eval_output_dir,
        }
        print(f"✅ Accuracy: {summary['accuracy']:.4f}, Perplexity: {summary['perplexity']:.4f}")
        return summary
    except Exception as e:  # noqa: BLE001
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        torch.cuda.empty_cache()

HF_MODEL_ID = "google/txgemma-2b-predict"
OUTPUT_DIR = "output_txgemma_trialbench_evalconfig"

DATA_URL = (
    "https://storage.googleapis.com/healthai-us/txgemma/datasets/"
    "trialbench_adverse-event-rate-prediction_train.jsonl"
)
DATA_DIR = "/content"
DATA_PATH = "/content/trialbench_adverse-event-rate-prediction_train.jsonl"

MAX_STEPS = 50
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 512

# QLoRA (4-bit) + LoRA params
USE_PEFT = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "o_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": False,
}

results: dict = {
    "model": HF_MODEL_ID,
    "dataset": str(DATA_PATH),
    "output_dir": OUTPUT_DIR,
}

print("TxGemma TrialBench SFT (AlignTune TRL backend)")
_ensure_dataset_downloaded()
_register_trialbench_loader()

# -----------------------------
# Pre-training evaluation
# -----------------------------
pre_eval = run_evaluation_with_runner(model_path=HF_MODEL_ID, is_pre_training=True, base_model=None)
if pre_eval:
    results["pre_training_eval"] = pre_eval

# -----------------------------
# Training (QLoRA)
# -----------------------------
bf16_ok = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
fp16_ok = bool(torch.cuda.is_available() and not bf16_ok)

trainer = create_sft_trainer(
    model_name=HF_MODEL_ID,
    dataset_name="trialbench_txgemma_adverse_event",
    backend="trl",
    output_dir=OUTPUT_DIR,
    task_type="instruction_following",
    # Training duration
    max_steps=MAX_STEPS,
    # Batch settings
    batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    # Optimizer / LR schedule
    learning_rate=LEARNING_RATE,
    warmup_steps=2,
    optimizer="adamw_8bit",  # Alternative to paged_adamw_8bit (avoids optim_args parsing issues)
    # Sequence length / data
    max_seq_length=MAX_SEQ_LENGTH,
    max_samples=5000,
    # Map json fields to expected columns
    column_mapping={"input_text": "instruction", "output_text": "response"},
    # QLoRA
    use_peft=USE_PEFT,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    lora_target_modules=LORA_TARGET_MODULES,
    quantization=QUANTIZATION,
    bf16=bf16_ok,
    fp16=fp16_ok,
    gradient_checkpointing=False,
    # Saving / logging
    save_steps=25,
    save_total_limit=2,
    logging_steps=5,
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
post_eval = run_evaluation_with_runner(model_path=OUTPUT_DIR, is_pre_training=False, base_model=HF_MODEL_ID)
if post_eval:
    results["post_training_eval"] = post_eval

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUTPUT_DIR) / "full_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(results)

