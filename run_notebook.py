#!/usr/bin/env python3
"""Run vanilla GRPO + counterfactual GRPO training (10 steps each) + eval."""
import json, os, subprocess, sys

os.chdir("/home/jovyan/Finetunehub")

MODEL_NAME = "Qwen/Qwen3-1.7B"
STEPS = 10

COMMON = dict(
    model_name=MODEL_NAME,
    dataset_name="openai/gsm8k",
    config_name="main",
    column_mapping={"question": "prompt", "answer": "response"},
    system_prompt="You are a careful math tutor. Solve step by step and provide the final numeric answer.",
    backend="trl",
    num_epochs=1,
    max_steps=STEPS,
    batch_size=8,
    learning_rate=1.5e-5,
    gradient_accumulation_steps=4,
    max_prompt_length=256,
    max_completion_length=768,
    temperature=0.6,
    top_p=0.95,
    num_generations=8,
    beta=0.01,
    use_peft=True,
    lora_r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bf16=True,
    loss_type="dapo",
    max_seq_length=1024,
    rewards=[{"type": "math_correctness", "weight": 1.0, "params": {}}],
    save_steps=STEPS,
    save_total_limit=3,
    logging_steps=1,
    seed=42,
    data_seed=47,
    max_grad_norm=1.0,
)

kwargs_json = json.dumps(COMMON)

# ===== 1. Vanilla GRPO =====
VANILLA_DIR = "./output/vanilla_grpo"
print("=" * 60)
print(f"1. Training Vanilla GRPO ({STEPS} steps)")
print("=" * 60)
subprocess.run(
    [sys.executable, "train_one.py",
     "--algorithm", "grpo",
     "--output_dir", VANILLA_DIR,
     "--kwargs_json", kwargs_json],
    check=True,
)

# ===== 2. Counterfactual GRPO =====
COUNTERFACT_DIR = "./output/counterfact_grpo"
print("\n" + "=" * 60)
print(f"2. Training Counterfactual GRPO ({STEPS} steps)")
print("=" * 60)
subprocess.run(
    [sys.executable, "train_one.py",
     "--algorithm", "counterfact_grpo",
     "--output_dir", COUNTERFACT_DIR,
     "--kwargs_json", kwargs_json],
    check=True,
)

# ===== 3. Eval =====
print("\n" + "=" * 60)
print("3. Evaluating on GSM8K")
print("=" * 60)

results = {}
for label, lora, out in [
    ("Base", None, "nb_base"),
    ("Vanilla GRPO", f"{VANILLA_DIR}/checkpoint-{STEPS}", "nb_vanilla"),
    ("Counterfact GRPO", f"{COUNTERFACT_DIR}/checkpoint-{STEPS}", "nb_counterfact"),
]:
    print(f"\nEvaluating {label}...")
    cmd = [sys.executable, "eval_gsm8k.py", "--model_path", MODEL_NAME,
           "--output_dir", f"./eval_results/{out}", "--temperature", "0.0", "--max_tokens", "2048"]
    if lora:
        cmd += ["--lora_path", lora]
    subprocess.run(cmd, check=True)
    with open(f"./eval_results/{out}/results.json") as f:
        results[label] = json.load(f)["accuracy"]

# ===== 4. Summary =====
base = results["Base"]
print("\n" + "=" * 55)
print(f"{'Model':<30} {'Accuracy':>10} {'vs Base':>10}")
print("-" * 55)
for label, acc in results.items():
    diff = f"{acc - base:+.2%}" if label != "Base" else ""
    print(f"{label:<30} {acc:>10.2%} {diff:>10}")
print("=" * 55)
