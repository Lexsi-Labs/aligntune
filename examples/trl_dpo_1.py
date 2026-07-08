from aligntune.core.backend_factory import create_rl_trainer
from aligntune.eval.runner import EvalConfig, run_eval



model_name = "google/gemma-2-2b-it"
output_dir = "./output/dpo_trl_coherence_safety"

results = {
    "model": model_name,
    "output_dir": output_dir
}



print("=" * 70)
print("STEP 1: TRAINING MODEL WITH DPO")
print("=" * 70)

print("\nCreating DPO trainer with TRL backend for coherence, safety, and length...")

trainer = create_rl_trainer(
    model_name=model_name,
    dataset_name="Anthropic/hh-rlhf",
    algorithm="dpo",
    split="train[:00001%]",
    backend="trl",
    num_epochs=1,
    batch_size=1,
    learning_rate=5e-6,
    max_seq_length=512,
    quantization={"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"},
    output_dir=output_dir,
    loggers=["tensorboard"],
    rewards=[
        {"type": "coherence", "weight": 0.4, "params": {}},
        {"type": "safety", "weight": 0.4, "params": {"strict": True}},
        {
            "type": "length",
            "weight": 0.2,
            "params": {"min_length": 20, "max_length": 400},
        },
    ],
    precision="fp16",
)

print("\nStarting training with coherence, safety, and length rewards...")

try:
    training_results = trainer.train()

    print()
    print("=" * 50)
    print("Training completed successfully!")
    print("=" * 50)
    print(f"Model saved to: {training_results['model_path']}")
    print(f"Final loss: {training_results.get('final_loss', 'N/A')}")
    print(f"Training time: {training_results.get('training_time', 0):.2f}s")
    print(f"Total steps: {training_results.get('total_steps', 'N/A')}")
    print(f"Num reward functions: {training_results.get('num_reward_functions', 'N/A')}")

    trained_model_path = training_results["model_path"]
    results["training"] = training_results

except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    trained_model_path = None


print("\n" + "=" * 70)
print("STEP 2: EVALUATION — TRAINED MODEL")
print("=" * 70)

print("\n>>> Evaluating TRAINED model...")

trained_config = EvalConfig(
    model_path=trained_model_path,
    output_dir=f"{output_dir}/eval_results_trained",
    device="cuda",

    task_type="dpo",
    data_task_type="dpo",
    metrics=["kl_divergence", "win_rate", "reward_margin"],
    reference_model_path=model_name,

    dataset_name="Anthropic/hh-rlhf",
    # dataset_config="main",
    split="test",
    # column_mapping={"chosen": "response"},

    max_length=512,
    temperature=0.0,
    batch_size=8,
    max_samples=200,

    use_lora=False,
)

trained_results = run_eval(trained_config)
print("Trained model Results", trained_results)



print("\n>>> Evaluating BASE model...")

base_config = EvalConfig(
    model_path=model_name,
    output_dir=f"{output_dir}/eval_results_base",
    device="cuda",
    reference_model_path=model_name,

    task_type="dpo",
    data_task_type="dpo",
    metrics=["kl_divergence", "win_rate", "reward_margin"],

    dataset_name="Anthropic/hh-rlhf",
    # dataset_config="full",
    split="test",

    max_length=512,
    temperature=0.0,
    batch_size=8,
    max_samples=100,

    use_lora=False,
)

base_results = run_eval(base_config)

print(f"\n✓ Base Model Results: {base_results}")


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



import json
from pathlib import Path

Path(output_dir).mkdir(parents=True, exist_ok=True)
save_path = Path(output_dir) / "comprehensive_results.json"

with open(save_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n📁 Results saved to:", save_path)