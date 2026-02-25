import sys
import json
from pathlib import Path
from aligntune.eval.runner import EvalConfig, run_eval


def run_evaluation(model_path, output_dir, is_pre_training=False, use_lora=False, base_model=None):
    """Run evaluation on GSM8K test set."""
    eval_config = EvalConfig(
        model_path=model_path,
        output_dir=f"{output_dir}/eval_{'pre' if is_pre_training else 'post'}_training",
        task_type="math",
        data_task_type="grpo",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        split="test",
        max_samples=100,
        batch_size=8,
        temperature=0.1,
        max_length=512,
        num_samples=1,
        use_lora=use_lora,
        base_model=base_model,
        use_unsloth=False,
        use_vllm=False,
        metrics=["math_accuracy"],
        column_mapping={"question": "prompt", "answer": "response"},
    )

    result = run_eval(eval_config)

    return {
        'pass_at_1': result.get('math_accuracy', 0.0),
        'total_samples': result.get('total', 0),
        'output_dir': eval_config.output_dir
    }

# Initialize
print("GRPO Training with GSM8K (TRL Backend) - Direct API")
print("="*60 + "\n")

output_dir = "output_direct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
results = {'model': model_name, 'dataset': 'openai/gsm8k', 'output_dir': output_dir}

# Pre-training evaluation
print("\nPRE-TRAINING EVALUATION")
print("="*60)
try:
    pre_eval_results = run_evaluation(model_path=model_name, output_dir=output_dir,
                                     is_pre_training=True, use_lora=False)
    results['pre_training_eval'] = pre_eval_results
    print(f"✅ Pre-training accuracy: {pre_eval_results['pass_at_1']*100:.2f}%")
except Exception as e:
    print(f"Pre-training eval failed: {e}")
    pre_eval_results = None

# CELL 2: TRAINING
from aligntune.core.backend_factory import create_rl_trainer

print("\nTRAINING")
print("="*60)
trainer = create_rl_trainer(
    model_name=model_name,
    dataset_name="openai/gsm8k",
    config_name="main",
    split="train",
    max_samples=1000,
    column_mapping={"question": "prompt", "answer": "response"},
    system_prompt=(
        "You are a helpful math tutor. Solve the problem step by step. "
        "Put your final answer within \\boxed{}."
    ),
    algorithm='grpo',
    backend="trl",
    output_dir=output_dir,

    # --- OPTIMIZED HYPERPARAMETERS FOR TRL 0.23.0 ---
    # Batching: Effective batch = batch_size * grad_accum * num_generations
    # Optimized for L4 GPU: 8 * 2 * 4 = 64 sequences per update
    batch_size=8,                    # Reduced for faster generation
    gradient_accumulation_steps=2,   # Compensate for smaller batch
    num_generations=4,               # Reduced from 8 -> 4 (still good for GRPO variance)

    # Scheduler & Optimizer (TRL 0.23.0 recommendations)
    learning_rate=5e-6,              # Lower LR for RL stability
    weight_decay=0.01,               # Lighter regularization for LoRA
    warmup_ratio=0.03,               # TRL 0.23.0 default
    lr_scheduler="cosine",
    num_epochs=1,
    max_steps=200,                   # More steps with faster iteration

    # GRPO Specifics (TRL 0.23.0)
    beta=0.01,                       # KL coefficient (lower = more exploration)
    loss_type="grpo",                # Valid: grpo, dapo, dr_grpo
    scale_rewards="std",             # TRL 0.23.0: normalize by std dev

    # Generation (Speed optimized - GSM8K solutions ~300 tokens)
    max_prompt_length=384,           # GSM8K prompts are short
    max_completion_length=448,       # Math solutions rarely exceed 400 tokens
    temperature=0.8,                 # Balanced exploration
    top_p=0.95,

    # LoRA Strategy (TRL 0.23.0 + PEFT best practices)
    use_peft=True,
    lora_r=16,                       # 16 sufficient for math reasoning
    lora_alpha=32,                   # alpha = 2 * r
    lora_dropout=0.05,               # Small dropout for regularization
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],

    # Rewards
    rewards=[
        {"type": "math_correctness", "weight": 1.0, "params": {}},
        {
            "type": "math_reasoning",
            "weight": 0.5,
            "params": {
                "check_correctness": True,
                "check_reasoning": True,
                "check_format": True,  # Enforces \boxed{} format
            },
        },
    ],

    # Hardware & Logging
    bf16=True,
    use_gradient_checkpointing=True,
    max_seq_length=832,              # prompt(384) + completion(448)
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    seed=42,
)

# Train
training_results = trainer.train()
results['training'] = {
    'final_loss': training_results.get('final_loss', 0.0),
    'total_steps': training_results.get('total_steps', 0),
    'model_path': output_dir
}

# CELL 3: POST-TRAINING EVALUATION
print("\nPOST-TRAINING EVALUATION")
print("="*60)
try:
    post_eval_results = run_evaluation(model_path=output_dir, output_dir=output_dir,
                                      is_pre_training=False, use_lora=True, base_model=model_name)
    results['post_training_eval'] = post_eval_results
    print(f"✅ Post-training accuracy: {post_eval_results['pass_at_1']*100:.2f}%")
except Exception as e:
    print(f"Post-training eval failed: {e}")
    post_eval_results = None

# Compute improvement
print("\nTRAINING COMPLETE")
print("="*60)
if pre_eval_results and post_eval_results:
    pre_pass1 = pre_eval_results['pass_at_1'] * 100
    post_pass1 = post_eval_results['pass_at_1'] * 100
    improvement = post_pass1 - pre_pass1

    results['improvement'] = {
        'pre_pass_at_1': pre_pass1,
        'post_pass_at_1': post_pass1,
        'absolute_improvement': improvement,
    }

    print(f"📊 Pre-training:   {pre_pass1:.2f}%")
    print(f"📊 Post-training:  {post_pass1:.2f}%")
    print(f"📊 Improvement:    {improvement:+.2f}%\n")

# Save results
Path(output_dir).mkdir(parents=True, exist_ok=True)
with open(Path(output_dir) / 'full_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_dir}/full_results.json")

