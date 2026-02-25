#!/usr/bin/env python3
"""Direct training script that properly loads YAML config, bypassing CLI bugs."""
import argparse
import yaml
import os
import sys

os.chdir("/home/jovyan/Finetunehub")

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to YAML config")
args = parser.parse_args()

# Load YAML config
with open(args.config) as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg['model']
dataset_cfg = cfg['datasets'][0]
train_cfg = cfg.get('train', {})
logging_cfg = cfg.get('logging', {})

# Build kwargs for create_rl_trainer, mapping YAML keys to function params
from aligntune.core.backend_factory import create_rl_trainer

# Explicit params that must match function signature
trainer = create_rl_trainer(
    model_name=model_cfg['name_or_path'],
    dataset_name=dataset_cfg['name'],
    algorithm=cfg['algo'],
    backend=cfg.get('backend', 'trl'),
    output_dir=logging_cfg.get('output_dir', './output'),
    num_epochs=train_cfg.get('epochs', 1),
    max_steps=train_cfg.get('max_steps'),
    batch_size=train_cfg.get('per_device_batch_size', 4),
    learning_rate=train_cfg.get('learning_rate', 1.5e-5),
    max_seq_length=model_cfg.get('max_seq_length', 2048),
    max_samples=dataset_cfg.get('max_samples'),
    system_prompt=dataset_cfg.get('system_prompt'),
    # Counterfactual params
    boost_factor=train_cfg.get('boost_factor', 2.0),
    min_weight=train_cfg.get('min_weight', 0.5),
    max_spans=train_cfg.get('max_spans', 10),
    answer_weight=train_cfg.get('answer_weight', 1.5),
    enable_gradient_conservation=train_cfg.get('enable_gradient_conservation', True),
    weight_debug=train_cfg.get('weight_debug', False),
    # Extra kwargs (won't conflict with explicit params since we removed them)
    **{k: v for k, v in {
        **model_cfg,
        **dataset_cfg,
        **train_cfg,
        **logging_cfg,
    }.items() if k not in {
        'name_or_path', 'name', 'split', 'max_samples', 'system_prompt',
        'max_seq_length', 'output_dir', 'epochs', 'per_device_batch_size',
        'learning_rate', 'max_steps', 'boost_factor', 'min_weight',
        'max_spans', 'answer_weight', 'enable_gradient_conservation',
        'weight_debug',
    }},
)

print(f"\nStarting training with config: {args.config}")
print(f"  batch_size={train_cfg.get('per_device_batch_size')}, "
      f"max_steps={train_cfg.get('max_steps')}, "
      f"lr={train_cfg.get('learning_rate')}")
results = trainer.train()

print(f"\n{'='*60}")
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"Training time: {results.get('training_time', 'N/A'):.2f} seconds")
if 'final_loss' in results:
    print(f"Final loss: {results['final_loss']:.4f}")
print(f"Model saved to: {results.get('model_path', logging_cfg.get('output_dir'))}")
print(f"{'='*60}")
