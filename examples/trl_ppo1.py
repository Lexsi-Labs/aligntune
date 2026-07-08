
# Import required libraries
import gc
import json
from pathlib import Path
import torch
from aligntune.core.backend_factory import create_rl_trainer
from aligntune.eval.runner import EvalConfig, run_eval

# Configuration
MODEL_NAME = "EleutherAI/pythia-1.4b"
DATASET = "CarperAI/openai_summarize_tldr"
OUTPUT_DIR = "./output"
BACKEND = "trl"  # 2-5x faster training

def cleanup_memory():
    """Clear GPU memory and cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("✓ Memory cleaned")
def run_evaluation(model_path, output_dir, eval_name, use_lora=False, base_model=None):
    """Evaluate model on dataset with text generation metrics"""
    print(f"\n>>> Evaluating {eval_name.upper()} model...")

    eval_config = EvalConfig(
        model_path=model_path,
        output_dir=f"{output_dir}/eval_{eval_name}",
        device="cuda",
        task_type="text",
        dataset_name=DATASET,
        split="test",
        column_mapping={
            "prompt": "prompt",
            "label": "response"
        },
        max_length=512,
        temperature=0.1,
        batch_size=8,
        max_samples=100,
        use_lora=use_lora,
        base_model=base_model,
        use_unsloth=False,
    )

    try:
        results = run_eval(eval_config)

        summary = {
            'perplexity': results.get('perplexity', 0.0),
            'rouge_l': results.get('rouge_l', results.get('rouge', 0.0)),
            'bleu': results.get('bleu', 0.0),
            'total_samples': results.get('total', 50)
        }

        print(f"✅ Perplexity: {summary['perplexity']:.4f}, "
              f"ROUGE-L: {summary['rouge_l']:.4f}, BLEU: {summary['bleu']:.4f}")

        return summary

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        cleanup_memory()



ppo_output_dir = f"{OUTPUT_DIR}/ppo_summarization_tldr_optimized"

ppo_trainer = create_rl_trainer(
    # ============================================================================
    # MODEL & DATASET
    # ============================================================================
    model_name="EleutherAI/pythia-1.4b",
    dataset_name="CarperAI/openai_summarize_tldr",
    split="train[:1000]",  # Increased from 5000

    # ============================================================================
    # ALGORITHM & BACKEND
    # ============================================================================
    algorithm="ppo",
    backend="trl",
    output_dir=ppo_output_dir,

    # ============================================================================
    # REWARD MODEL
    # ============================================================================
    reward_model_name="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
    reward_value_model="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",

    # ============================================================================
    # CORE TRAINING PARAMETERS
    # ============================================================================
    num_epochs=1,
    batch_size=8,                      # Increased from 4 (adjust based on GPU)
    mini_batch_size=4,                  # Increased from 1
    gradient_accumulation_steps=1,      # Reduced since batch_size increased
    max_steps=-1,                       # -1 means train for full epoch
    learning_rate=1.41e-5,              # Optimized for PPO (was 3e-6)

    # ============================================================================
    # PPO-SPECIFIC HYPERPARAMETERS
    # ============================================================================
    kl_coef=0.05,                       # KL divergence coefficient (reduced from 0.1)
    cliprange=0.2,                      # PPO clip range (epsilon)
    cliprange_value=0.2,                # Value function clipping
    vf_coef=0.5,                        # Value function loss coefficient (increased)
    num_ppo_epochs=4,                   # PPO optimization epochs per batch

    # Adaptive KL Control
    adap_kl_ctrl=True,                  # Enable adaptive KL controller
    init_kl_coef=0.2,                   # Initial KL coefficient
    target_kl=6.0,                      # Target KL divergence

    # ============================================================================
    # OPTIMIZER SETTINGS
    # ============================================================================
    optimizer_type="adamw",
    weight_decay=0.01,                  # L2 regularization
    max_grad_norm=0.5,                  # Gradient clipping
    warmup_steps=100,                   # Learning rate warmup
    lr_scheduler_type="cosine",         # Learning rate schedule

    # ============================================================================
    # GENERATION PARAMETERS
    # ============================================================================
    response_length=128,
    temperature=1.0,                    # Increased for more diversity during training
    top_p=0.9,                          # Nucleus sampling
    top_k=50,                           # Top-k filtering
    max_seq_length=512,

    # ============================================================================
    # REWARD PROCESSING
    # ============================================================================
    use_score_scaling=True,             # Scale rewards
    use_score_norm=True,                # Normalize rewards
    whiten_rewards=True,                # Whiten rewards (standardize)

    # ============================================================================
    # LORA CONFIGURATION
    # ============================================================================
    use_lora=True,
    lora_r=128,                         # Increased rank (was 64)
    lora_alpha=256,                     # 2x rank (was 64)
    lora_dropout=0.05,                  # Reduced dropout (was 0.1)
    lora_target_modules=[
        "query_key_value",              # Attention layers
        "dense",                        # Output projection
        "dense_h_to_4h",               # FFN up-projection
        "dense_4h_to_h"                # FFN down-projection
    ],

    # ============================================================================
    # QUANTIZATION
    # ============================================================================
    quantization={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },

    # ============================================================================
    # EVALUATION & CHECKPOINTING
    # ============================================================================
    eval_strategy="steps",
    eval_steps=50,                      # More frequent evaluation (was 20)
    save_strategy="steps",
    save_steps=100,                     # Save checkpoints
    save_total_limit=3,                 # Keep only 3 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="reward/mean",

    # ============================================================================
    # LOGGING
    # ============================================================================
    logging_steps=10,                   # Log every 10 steps
    loggers=["tensorboard"],            # Add "wandb" if using Weights & Biases
    report_to=["tensorboard"],
    logging_first_step=True,

    # ============================================================================
    # MIXED PRECISION & PERFORMANCE
    # ============================================================================
    fp16=False,                         # Use bf16 instead if available
    bf16=True,                          # Better for training stability

    # ============================================================================
    # SAMPLING & GENERATION CONTROL
    # ============================================================================
    do_sample=True,
    num_beams=1,                        # Sampling, not beam search
    early_stopping=False,

    # ============================================================================
    # ADDITIONAL STABILITY OPTIONS
    # ============================================================================
    remove_unused_columns=False,
    dataloader_num_workers=4,           # Parallel data loading
    dataloader_pin_memory=True,
    seed=42,                            # Reproducibility
)

# ============================================================================
# TRAINING EXECUTION
# ============================================================================
print("=" * 80)
print("Starting PPO Training with Optimized Configuration")
print("=" * 80)
print(f"Output Directory: {ppo_output_dir}")
print(f"Dataset: train[:1000] samples")
print(f"Batch Size: 16 (mini_batch: 4)")
print(f"Learning Rate: 1.41e-5")
print(f"LoRA Config: r={128}, alpha={256}")
print("=" * 80)

training_results = ppo_trainer.train()

print("\n" + "=" * 80)
print("✓ Training Completed Successfully")
print("=" * 80)
print(f"Final Loss:      {training_results.get('final_loss', 'N/A')}")
print(f"Training Time:   {training_results.get('training_time', 0):.2f}s")
print(f"Total Steps:     {training_results.get('total_steps', 'N/A')}")
print(f"Best Checkpoint: {ppo_output_dir}")
print("=" * 80)

cleanup_memory()

# Evaluate base model
print("[1/2] Evaluating Base Model...")
base_results = run_evaluation(
    model_path=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    eval_name="base",
    use_lora=False
)


cleanup_memory()

# Evaluate PPO model
ppo_output_dir = "/content/output/ppo_summarization_tldr_optimized"
# ppo_ouptut_dir = "zlyngkhoi/PPO_TRL_summarization"
print("[2/2] Evaluating PPO Model...")
ppo_results = run_evaluation(
    model_path=ppo_output_dir,
    output_dir=OUTPUT_DIR,
    eval_name="ppo",
    use_lora=True,
    base_model=MODEL_NAME
)



cleanup_memory()

# ============================================================================
# RESULTS COMPARISON: BASE MODEL vs PPO-TRAINED MODEL
# ============================================================================

print("\n" + "="*70)
print("TRAINING RESULTS COMPARISON")
print("="*70 + "\n")

# Extract metrics
base_perplexity = base_results.get('perplexity', 0.0)
base_rouge = base_results.get('rouge_l', 0.0)
base_bleu = base_results.get('bleu', 0.0)
base_samples = base_results.get('total_samples', 0)

ppo_perplexity = ppo_results.get('perplexity', 0.0)
ppo_rouge = ppo_results.get('rouge_l', 0.0)
ppo_bleu = ppo_results.get('bleu', 0.0)
ppo_samples = ppo_results.get('total_samples', 0)

# Calculate improvements
perplexity_change = ppo_perplexity - base_perplexity
rouge_change = ppo_rouge - base_rouge
bleu_change = ppo_bleu - base_bleu

# Percentage changes
perplexity_pct = ((ppo_perplexity - base_perplexity) / base_perplexity * 100) if base_perplexity != 0 else 0
rouge_pct = ((ppo_rouge - base_rouge) / base_rouge * 100) if base_rouge != 0 else 0
bleu_pct = ((ppo_bleu - base_bleu) / base_bleu * 100) if base_bleu != 0 else 0

print("📊 BASE MODEL RESULTS")
print("-" * 70)
print(f"   Perplexity:    {base_perplexity:.4f}")
print(f"   ROUGE-L:       {base_rouge:.4f}")
print(f"   BLEU:          {base_bleu:.6f}")
print(f"   Total Samples: {base_samples}")

print("\n📊 PPO-TRAINED MODEL RESULTS")
print("-" * 70)
print(f"   Perplexity:    {ppo_perplexity:.4f}")
print(f"   ROUGE-L:       {ppo_rouge:.4f}")
print(f"   BLEU:          {ppo_bleu:.6f}")
print(f"   Total Samples: {ppo_samples}")

print("\n📈 IMPROVEMENT ANALYSIS")
print("-" * 70)

# Perplexity (lower is better)
if perplexity_change < 0:
    print(f"   ✅ Perplexity:    {perplexity_change:.4f} ({perplexity_pct:.2f}%) - IMPROVED")
elif perplexity_change > 0:
    print(f"   ❌ Perplexity:    {perplexity_change:+.4f} ({perplexity_pct:+.2f}%) - DEGRADED")
else:
    print(f"   ➖ Perplexity:    No change")

# ROUGE-L (higher is better)
if rouge_change > 0:
    print(f"   ✅ ROUGE-L:       {rouge_change:+.4f} ({rouge_pct:+.2f}%) - IMPROVED")
elif rouge_change < 0:
    print(f"   ❌ ROUGE-L:       {rouge_change:.4f} ({rouge_pct:.2f}%) - DEGRADED")
else:
    print(f"   ➖ ROUGE-L:       No change")

# BLEU (higher is better)
if bleu_change > 0:
    print(f"   ✅ BLEU:          {bleu_change:+.6f} ({bleu_pct:+.2f}%) - IMPROVED")
elif bleu_change < 0:
    print(f"   ❌ BLEU:          {bleu_change:.6f} ({bleu_pct:.2f}%) - DEGRADED")
else:
    print(f"   ➖ BLEU:          No change")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Overall assessment
improved_metrics = 0
degraded_metrics = 0

if perplexity_change < 0:  # Lower perplexity is better
    improved_metrics += 1
elif perplexity_change > 0:
    degraded_metrics += 1

if rouge_change > 0:  # Higher ROUGE is better
    improved_metrics += 1
elif rouge_change < 0:
    degraded_metrics += 1

if bleu_change > 0:  # Higher BLEU is better
    improved_metrics += 1
elif bleu_change < 0:
    degraded_metrics += 1

print(f"\n   Improved Metrics:  {improved_metrics}/3")
print(f"   Degraded Metrics:  {degraded_metrics}/3")
print(f"   Unchanged Metrics: {3 - improved_metrics - degraded_metrics}/3")

if improved_metrics > degraded_metrics:
    print("\n   🎉 Overall: PPO training IMPROVED model performance!")
elif degraded_metrics > improved_metrics:
    print("\n   ⚠️  Overall: PPO training DEGRADED model performance")
    print("   💡 Consider: Adjusting reward weights, learning rate, or training steps")
else:
    print("\n   ➖ Overall: Mixed results - some metrics improved, some degraded")

