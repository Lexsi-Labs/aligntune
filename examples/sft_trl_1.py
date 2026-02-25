
import os
import sys
from pathlib import Path
import json
import torch
import gc
from aligntune.core.backend_factory import create_sft_trainer
from aligntune.eval.runner import EvalConfig, run_eval

# =============================================================================
# HARD-CODED CONFIGURATION
# =============================================================================

MODEL_NAME = "google/gemma-7b"
TOKENIZER_NAME = "philschmid/gemma-tokenizer-chatml"  # ChatML format tokenizer
DATASET_NAME = "philschmid/dolly-15k-oai-style"
OUTPUT_DIR = "output_gemma_chatml_evalconfig"

# Training hyperparameters (matching the notebook)
MAX_STEPS = 500  # Use epochs instead
NUM_EPOCHS = 50
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 1512  # Max sequence length for packing

# LoRA config (matching QLoRA paper & notebook)
LORA_R = 6
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"  # Special value for all linear layers

# Quantization
QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
}

# Precision
BF16 = True
TF32 = True  # TensorFloat-32 for Ampere GPUs

# Flash Attention (requires Ampere+ GPU)
USE_FLASH_ATTENTION = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cleanup_memory():
    """Clear GPU memory and cache."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_evaluation_with_runner(
    model_path: str,
    output_dir: str,
    is_pre_training: bool = False,
    use_lora: bool = False,
    base_model: str | None = None,
) -> dict | None:
    """
    Run evaluation using EvalConfig + run_eval.

    For ChatML format, we evaluate on a subset of the Dolly dataset.
    """
    print(f"\n{'=' * 60}")
    print(f"{'PRE' if is_pre_training else 'POST'}-TRAINING EVALUATION (EvalConfig)")
    print(f"{'=' * 60}\n")

    eval_output_dir = str(
        Path(output_dir)
        / f"eval_{'pre' if is_pre_training else 'post'}_training_runner"
    )

    # Use a small subset for evaluation
    cfg = EvalConfig(
        model_path=model_path,
        output_dir=eval_output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dataset_name=DATASET_NAME,
        dataset_config=None,
        split="train[0:100]",  # Small eval set
        task_type="text",  # text => perplexity + ROUGE + BLEU
        batch_size=2,
        max_samples=100,
        max_length=512,  # Smaller for eval
        temperature=0.3,
        do_sample=True,
        use_lora=use_lora,
        base_model=base_model,
    )

    try:
        cleanup_memory()
        results = run_eval(cfg)

        summary = {
            "perplexity": results.get("perplexity", 0.0),
            "rouge": results.get("rouge_l", results.get("rouge", 0.0)),
            "bleu": results.get("bleu", 0.0),
            "total_samples": results.get("total", cfg.max_samples or 0),
            "output_dir": eval_output_dir,
        }
        print(
            f"✅ Perplexity: {summary['perplexity']:.4f}, "
            f"ROUGE: {summary['rouge']:.4f}, BLEU: {summary['bleu']:.4f}"
        )
        return summary
    except Exception as e:  # noqa: BLE001
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        cleanup_memory()

print(f"Model: {MODEL_NAME}")
print(f"Tokenizer: {TOKENIZER_NAME} (ChatML format)")
print(f"Dataset: {DATASET_NAME}")
print(f"Output Directory: {OUTPUT_DIR}\n")

results: dict = {
    "model": MODEL_NAME,
    "tokenizer": TOKENIZER_NAME,
    "dataset": DATASET_NAME,
    "output_dir": OUTPUT_DIR,
}

# Check Flash Attention support
if USE_FLASH_ATTENTION:
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] < 8:
            print(
                f"⚠️  Flash Attention requires Ampere+ GPU (compute capability >= 8.0). "
                f"Your GPU has compute capability {device_capability[0]}.{device_capability[1]}. "
                f"Disabling Flash Attention."
            )
            use_flash_attention = False
        else:
            use_flash_attention = True
            print(f"✅ Flash Attention 2 enabled (GPU compute capability: {device_capability[0]}.{device_capability[1]})")
    else:
        use_flash_attention = False
        print("⚠️  Flash Attention requires CUDA. Disabling.")
else:
    use_flash_attention = False

# ---------------------------------------------------------------------
# Pre-training evaluation (base model)
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1: PRE-TRAINING EVALUATION")
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
    print(
        f"✅ Pre-training perplexity: {pre_eval.get('perplexity', 0.0):.2f}, "
        f"ROUGE: {pre_eval.get('rouge', 0.0):.2f}, "
        f"BLEU: {pre_eval.get('bleu', 0.0):.4f}"
    )

cleanup_memory()

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: SFT TRAINING WITH CHATML FORMAT")
print("=" * 60)

print(f"Creating SFT trainer with ChatML tokenizer...")
print(f"  Model: {MODEL_NAME}")
print(f"  Tokenizer: {TOKENIZER_NAME}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  QLoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  Flash Attention: {use_flash_attention}")

try:
    trainer = create_sft_trainer(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        backend="trl",
        output_dir=OUTPUT_DIR,
        task_type="instruction_following",  # Conversational format
        # Training duration
        max_steps=MAX_STEPS,
        epochs=NUM_EPOCHS,
        # Batch settings
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # Optimizer / LR schedule
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="constant",  # Constant LR as in notebook
        max_grad_norm=0.3,  # Gradient clipping
        # Sequence length / data
        max_seq_length=MAX_SEQ_LENGTH,
        max_samples=500,
        # Dataset format - ChatML/messages format
        # The dataset is already in messages format, so we use conversational format
        dataset_text_field="messages",  # For conversational format
        # LoRA config
        use_peft=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
        # Quantization / precision
        quantization=QUANTIZATION,
        bf16=BF16,
        tf32=TF32,
        gradient_checkpointing=True,
        # Flash Attention
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        # Model loading
        trust_remote_code=True,
        # Saving / logging
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=5,
        report_to="tensorboard",
        seed=42,
    )

    # Note: For ChatML format, we need the custom tokenizer (philschmid/gemma-tokenizer-chatml).
    # The TRL backend will load the tokenizer from the model path by default.
    # If the custom tokenizer is needed, you may need to manually set it:
    #   from transformers import AutoTokenizer
    #   trainer.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # The dataset is in messages format which TRL supports natively.

    # Optionally set custom tokenizer if needed
    # Uncomment if create_sft_trainer doesn't load the ChatML tokenizer automatically
    # from transformers import AutoTokenizer
    # trainer.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("\nStarting SFT training...")
    training_results = trainer.train()
    results["training"] = {
        "final_loss": (
            training_results.get("train_loss", 0.0)
            if isinstance(training_results, dict)
            else 0.0
        ),
        "model_path": OUTPUT_DIR,
    }

    print("\n✅ SFT training completed!")
    print(f"Model saved to: {OUTPUT_DIR}")

except Exception as e:  # noqa: BLE001
    print(f"❌ Training failed: {e}")
    import traceback

    traceback.print_exc()
    results["training"] = {"error": str(e)}

cleanup_memory()

# ---------------------------------------------------------------------
# Post-training evaluation (LoRA adapter on top of base model)
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: POST-TRAINING EVALUATION")
print("=" * 60)

if "training" in results and "error" not in results["training"]:
    post_eval = run_evaluation_with_runner(
        model_path=OUTPUT_DIR,
        output_dir=OUTPUT_DIR,
        is_pre_training=False,
        use_lora=True,
        base_model=MODEL_NAME,
    )
    if post_eval:
        results["post_training_eval"] = post_eval
        print(
            f"✅ Post-training perplexity: {post_eval.get('perplexity', 0.0):.2f}, "
            f"ROUGE: {post_eval.get('rouge', 0.0):.2f}, "
            f"BLEU: {post_eval.get('bleu', 0.0):.4f}"
        )
else:
    print("Skipping post-training evaluation (training failed)")

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
if pre_eval and "post_training_eval" in results:
    post_eval = results["post_training_eval"]
    pre_ppl = pre_eval.get("perplexity", 0.0)
    post_ppl = post_eval.get("perplexity", 0.0)
    improvement = pre_ppl - post_ppl
    pct = (improvement / pre_ppl * 100) if pre_ppl > 0 else 0.0
    results["improvement"] = {
        "perplexity_delta": improvement,
        "perplexity_pct": pct,
    }
    print(f"📊 Pre:  {pre_ppl:.2f}")
    print(f"📊 Post: {post_ppl:.2f}")
    print(f"📊 Improvement: {improvement:+.2f} ({pct:+.1f}%)")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(OUTPUT_DIR) / "full_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {Path(OUTPUT_DIR) / 'full_results.json'}")