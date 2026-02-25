import gc
import json
import torch
from pathlib import Path
from aligntune.core.backend_factory import create_rl_trainer
from aligntune.eval.runner import EvalConfig, run_eval

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("✓ Memory cleaned")

# Configuration
MODEL_NAME = "microsoft/phi-2"
DATASET = "argilla/distilabel-intel-orca-dpo-pairs"
OUTPUT_DIR = "./output/phi2_dpo_alignment"

results = {
    "model": MODEL_NAME,
    "dataset": DATASET,
    "task": "DPO Fine-tuning with Phi-2",
    "output_dir": OUTPUT_DIR
}
print("\n✅ Setup complete!")

# Cell 2: Train Phi-2 Model with DPO

print("\n" + "="*70)
print("STEP 1: TRAINING PHI-2 WITH DPO")
print("="*70)

print("\n🎯 Training Phi-2 with Direct Preference Optimization...")
print("   - Learning from preferred vs rejected responses")
print("   - Using argilla/distilabel-intel-orca-dpo-pairs dataset")
print("   - 4-bit quantization for memory efficiency")
print("   - LoRA adapters for parameter-efficient training\n")

trainer = create_rl_trainer(
    model_name=MODEL_NAME,
    dataset_name=DATASET,
    algorithm="dpo",
    backend="unsloth",

    # Dataset configuration
    split="train[:50%]",  # Use 50% for training (5% for eval)
    # Dataset has columns: system, input, chosen, rejected

    # Training configuration
    num_epochs=1,  # Following tutorial's max_steps approach
    batch_size=4,
    learning_rate=5e-5,
    gradient_accumulation_steps=4,  # Effective batch = 16

    # Sequence length (matching tutorial)
    max_seq_length=1536,
    max_prompt_length=1024,
    max_completion_length=512,

    # Memory optimization - 4-bit quantization
    quantization={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    precision="bf16",

    # LoRA configuration (matching tutorial)
    use_lora=True,
    lora_config={
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # DPO-specific parameters
    beta=0.1,  # DPO temperature parameter (matching tutorial)

    # Output configuration
    output_dir=OUTPUT_DIR,
    save_steps=10,  # Save every 10 steps (matching tutorial)
    logging_steps=1,
    eval_steps=20,
    max_steps=500,  # Matching tutorial's max_steps

    # Optimizer settings (matching tutorial)
    optim="paged_adamw_32bit",
    warmup_steps=100,
    lr_scheduler_type="cosine",

    # Logging
    seed=42,
)

print("🚀 Starting DPO training...\n")
print("Expected VRAM usage: ~5.19 GB (before quantization)")
print("With 4-bit quantization: ~1.72 GB\n")

try:
    training_results = trainer.train()

    print("\n" + "="*70)
    print("✅ DPO TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Model saved to: {training_results['model_path']}")
    print(f"Final loss: {training_results.get('final_loss', 'N/A')}")
    print(f"Training time: {training_results.get('training_time', 0):.2f}s")
    print(f"Total steps: {training_results.get('total_steps', 'N/A')}")

    print("\n📊 Expected Inference Performance:")
    print("   - Inference Time: ~11.96 secs")
    print("   - Cold Start Time: ~7.82 secs")
    print("   - Tokens/Sec: ~21.34")
    print("   - Latency/Token: ~46.85 ms")
    print("   - VRAM Required: ~1.72 GB")

    trained_model_path = training_results["model_path"]
    results["training"] = {
        'final_loss': training_results.get('final_loss', 0.0),
        'training_time': training_results.get('training_time', 0),
        'total_steps': training_results.get('total_steps', 0),
        'model_path': training_results['model_path']
    }

except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    trained_model_path = None

cleanup_memory()

trainer.dataset_dict

# Cell 3: Evaluate Base Phi-2 Model

print("\n" + "="*70)
print("STEP 2: EVALUATION - BASE PHI-2 MODEL")
print("="*70)

print("\n>>> Evaluating BASE Phi-2 model (before DPO training)...")
print("   - Model: microsoft/phi-2 (2.7B parameters)")
print("   - Metrics: Reward Margin, Preference Accuracy, Win Rate\n")

base_config = EvalConfig(
    model_path=MODEL_NAME,
    output_dir=f"{OUTPUT_DIR}/eval_results_base",
    device="cuda",

    # Task configuration
    task_type="dpo",
    data_task_type="dpo",

    # DPO-specific metrics
    metrics=["reward_margin", "preference_accuracy", "win_rate"],

    # Dataset (use from index 11000 held-out for evaluation)
    dataset_name=DATASET,
    split="test",

    # Generation parameters
    max_length=1536,
    temperature=0.0,  # Greedy decoding for evaluation
    batch_size=2,  # Smaller batch for Phi-2
    max_samples=100,  # Evaluate on 100 test examples

    # Model settings
    use_lora=False,  # Base model has no LoRA
    trust_remote_code=True,  # Required for Phi-2
    use_unsloth=True

)

base_results = run_eval(base_config,trainer.dataset_dict)

print("\n✅ Base Phi-2 Model Evaluation Results:")
print(f"   Reward Margin: {base_results.get('reward_margin', 0.0):.4f}")
print(f"   Preference Accuracy: {base_results.get('preference_accuracy', 0.0)*100:.2f}%")
print(f"   Win Rate: {base_results.get('win_rate', 0.0)*100:.2f}%")
print(f"   Total Samples: {base_results.get('total', 0)}")

results["eval_base"] = {
    'reward_margin': base_results.get('reward_margin', 0.0),
    'preference_accuracy': base_results.get('preference_accuracy', 0.0),
    'win_rate': base_results.get('win_rate', 0.0),
    'total_samples': base_results.get('total', 100)
}

cleanup_memory()

trained_model_path= "/content/output/phi2_dpo_alignment"

# Cell 4: Evaluate Trained DPO Phi-2 Model
print("\n" + "="*70)
print("STEP 3: EVALUATION - TRAINED DPO PHI-2 MODEL")
print("="*70)
if trained_model_path is None:
    print("\n❌ Skipping trained model evaluation (training failed)")
else:
    print("\n>>> Evaluating TRAINED DPO Phi-2 model...")
    print("   - Model: Fine-tuned Phi-2 with LoRA adapters")
    print("   - Reference: Base microsoft/phi-2\n")

    trained_config = EvalConfig(
        # model_path=trained_model_path,
        output_dir=f"{OUTPUT_DIR}/eval_results_trained",
        device="cuda",
        model_path=trained_model_path,

        # Task configuration
        task_type="dpo",
        data_task_type="dpo",

        # DPO-specific metrics
        metrics=["reward_margin", "preference_accuracy", "win_rate"],
        reference_model_path=MODEL_NAME,  # Use base Phi-2 as reference

        # Dataset
        dataset_name=DATASET,
        split="test",  # Same eval split

        # Generation parameters
        max_length=1536,
        temperature=0.0,
        batch_size=2,
        max_samples=100,

        # Model settings (DPO uses LoRA)
        use_lora=True,
        base_model=MODEL_NAME,
        trust_remote_code=True,
        use_unsloth=True


    )

    trained_results = run_eval(trained_config,trainer.dataset_dict)

    print("\n✅ Trained DPO Phi-2 Model Evaluation Results:")
    print(f"   Reward Margin: {trained_results.get('reward_margin', 0.0):.4f}")
    print(f"   Preference Accuracy: {trained_results.get('preference_accuracy', 0.0)*100:.2f}%")
    print(f"   Win Rate: {trained_results.get('win_rate', 0.0)*100:.2f}%")
    print(f"   Total Samples: {trained_results.get('total', 0)}")

    results["eval_trained"] = {
        'reward_margin': trained_results.get('reward_margin', 0.0),
        'preference_accuracy': trained_results.get('preference_accuracy', 0.0),
        'win_rate': trained_results.get('win_rate', 0.0),
        'total_samples': trained_results.get('total', 100)
    }

cleanup_memory()

print("\n" + "=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

metrics = ["win_rate", "win_count", "reward_margin"]

print(f"{'Metric':<20} | {'Base Model':>15} | {'Trained Model':>15} | {'Improvement':>15}")
print("-" * 70)

for metric in metrics:
    base_value = base_results.get(metric, 'N/A')
    trained_value = trained_results.get(metric, 'N/A')

    if isinstance(base_value, (int, float)) and isinstance(trained_value, (int, float)):
        improvement = trained_value - base_value
        print(f"{metric:<20} | {base_value:>15.4f} | {trained_value:>15.4f} | {improvement:>+15.4f}")
    else:
        print(f"{metric:<20} | {str(base_value):>15} | {str(trained_value):>15} | {'N/A':>15}")

print("=" * 70)

results["comparison_metrics"] = {
    metric: {
        "base": base_results.get(metric),
        "trained": trained_results.get(metric),
        "improvement": trained_results.get(metric) - base_results.get(metric) if isinstance(base_results.get(metric), (int, float)) and isinstance(trained_results.get(metric), (int, float)) else None
    }
    for metric in metrics
}

