#!/usr/bin/env python3
"""Run counterfactual GRPO training with length normalization."""
import os, sys, gc, json, torch
from aligntune.core.backend_factory import create_rl_trainer

output_dir = './output/qwen3_counterfact_grpo_200_lenorm'
model_name = 'Qwen/Qwen3-1.7B'

trainer = create_rl_trainer(
    model_name=model_name,
    dataset_name='openai/gsm8k',
    config_name='main',
    column_mapping={'question': 'prompt', 'answer': 'response'},
    system_prompt='You are a careful math tutor. Solve step by step and provide the final numeric answer.',
    algorithm='counterfact_grpo',
    backend='trl',
    output_dir=output_dir,
    num_epochs=1,
    max_steps=200,
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
    lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    bf16=True,
    loss_type='dapo',
    max_seq_length=1024,
    rewards=[{'type': 'math_correctness', 'weight': 1.0, 'params': {}}],
    save_steps=200,
    save_total_limit=3,
    logging_steps=10,
    seed=42,
    data_seed=47,
    max_grad_norm=1.0,
)

print('Starting counterfact GRPO training with length normalization...')
training_results = trainer.train()
print('Training completed!')
print(f'Model saved to: {output_dir}')
