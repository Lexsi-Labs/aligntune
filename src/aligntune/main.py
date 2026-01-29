#!/usr/bin/env python3
"""
Unified Training CLI for FineTuneHub
Supports: SFT, DPO, PPO, GRPO, Classification

Usage Examples:
  # SFT Training
  aligntune train sft --model unsloth/mistral-7b-bnb-4bit --dataset databricks/databricks-dolly-15k --task-type instruction_following
  
  # SFT Evaluation
  aligntune train sft --model ./my_model --dataset databricks/databricks-dolly-15k --task-type instruction_following --mode eval
  
  # RL Training (PPO)
  aligntune train rl --model EleutherAI/pythia-1b-deduped --dataset Anthropic/hh-rlhf --algo ppo
  
  # RL Training (DPO)
  aligntune train rl --model unsloth/Qwen2-1.5B-Instruct-bnb-4bit --dataset Anthropic/hh-rlhf --algo dpo
  
  # Advanced Options
  aligntune train sft --model gpt2 --dataset wikitext --task-type instruction_following \
    --epochs 3 --train-batch 4 --learning-rate 5e-5 \
    --precision bf16 --quantization 4bit \
    --output-dir ./my_model --use-unsloth
"""

import os
import sys
import argparse
import yaml
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# =============================================================================
# CRITICAL: Import Management
# =============================================================================

def disable_unsloth_for_algo(algo: str) -> bool:
    """Determine if Unsloth should be disabled for this algorithm"""
    # PPO has conflicts with Unsloth
    if algo in ['ppo']:
        os.environ['DISABLE_UNSLOTH'] = '1'
        os.environ['USE_UNSLOTH'] = '0'
        return True
    return False

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Training CLI for FineTuneHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # ========== CORE ARGUMENTS ==========
    parser.add_argument('--algo', '--algorithm', type=str, required=True,
                       choices=['sft', 'dpo', 'ppo', 'grpo', 'classification'],
                       help='Training algorithm to use')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path (e.g., unsloth/mistral-7b-bnb-4bit)')
    
    parser.add_argument('--dataset', '--datasets', type=str, 
                       help='Dataset name(s). For GRPO, use comma-separated with configs')
    
    parser.add_argument('--task-type', type=str,
                       choices=['instruction_following', 'supervised_fine_tuning', 
                               'text_classification', 'token_classification',
                               'chat_completion', 'text_generation', 
                               'conversation', 'content_generation', 'helpfulness'],
                       help='Task type for training')
    
    # ========== TRAINING PARAMETERS ==========
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum training steps (overrides epochs)')
    
    parser.add_argument('--batch-size', '--train-batch', type=int, default=None,
                       help='Training batch size per device')
    
    parser.add_argument('--accum', '--gradient-accumulation-steps', type=int, default=None,
                       help='Gradient accumulation steps')
    
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    parser.add_argument('--optimizer', type=str, 
                       choices=['adamw_8bit', 'adamw_torch', 'adamw_bnb_8bit', 'lion_8bit', 'adafactor'],
                       help='Optimizer to use')
    
    parser.add_argument('--warmup-ratio', type=float, default=None,
                       help='Warmup ratio for learning rate scheduler')
    
    # ========== MODEL CONFIGURATION ==========
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'fp32'],
                       help='Training precision')
    
    parser.add_argument('--quantization', type=str, 
                       help='Quantization config (e.g., "4bit", "4bit:quant_type=nf4")')
    
    parser.add_argument('--max-seq-length', type=int, default=None,
                       help='Maximum sequence length')
    
    parser.add_argument('--lora-rank', type=int, default=None,
                       help='LoRA rank (r parameter)')
    
    parser.add_argument('--lora-alpha', type=int, default=None,
                       help='LoRA alpha parameter')
    
    parser.add_argument('--lora-dropout', type=float, default=None,
                       help='LoRA dropout rate')
    
    parser.add_argument('--no-peft', action='store_true',
                       help='Disable PEFT/LoRA (full fine-tuning)')
    
    parser.add_argument('--grad-checkpointing', type=str, 
                       choices=['true', 'false', 'auto'],
                       help='Enable gradient checkpointing')
    
    # ========== DPO SPECIFIC ==========
    parser.add_argument('--scenario', type=str,
                       choices=['general', 'safety', 'helpfulness', 'ultrafeedback', 
                               'stack_exchange', 'custom'],
                       help='DPO scenario configuration')
    
    parser.add_argument('--beta', type=float, default=None,
                       help='DPO beta parameter (preference learning strength)')
    
    parser.add_argument('--loss-type', type=str,
                       choices=['sigmoid', 'hinge', 'ipo'],
                       help='DPO loss type')
    
    # ========== PPO SPECIFIC ==========
    parser.add_argument('--rewards', type=str,
                       help='Reward functions (e.g., "length:weight=0.2:min=50:max=200 numeric_math:weight=0.8")')
    
    parser.add_argument('--kl-coef', type=float, default=None,
                       help='PPO KL divergence coefficient')
    
    parser.add_argument('--ppo-epochs', type=int, default=None,
                       help='Number of PPO epochs per batch')
    
    # ========== GRPO SPECIFIC ==========
    parser.add_argument('--num-generations', type=int, default=None,
                       help='GRPO number of generations per prompt')
    
    parser.add_argument('--use-math', action='store_true',
                       help='Enable GRPO math mode (auto-configures for math tasks)')
    
    # ========== DATASET CONFIGURATION ==========
    parser.add_argument('--dataset-percentage', type=float, default=None,
                       help='Percentage of dataset to use (0.0-1.0)')
    
    parser.add_argument('--validation-split', type=str, default=None,
                       help='Validation split name or ratio')
    
    parser.add_argument('--column-mapping', type=str,
                       help='Column mapping as JSON string or key=value pairs')
    
    # ========== OUTPUT & LOGGING ==========
    parser.add_argument('--output', '--output-dir', type=str, default='./results',
                       help='Output directory for models and logs')
    
    parser.add_argument('--experiment', '--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    parser.add_argument('--save-steps', type=int, default=None,
                       help='Save checkpoint every N steps')
    
    parser.add_argument('--eval-steps', type=int, default=None,
                       help='Evaluate every N steps')
    
    parser.add_argument('--eval-interval', type=int, default=None,
                       help='Alias for --eval-steps')
    
    parser.add_argument('--save-interval', type=int, default=None,
                       help='Alias for --save-steps')
    
    # ========== EXECUTION MODE ==========
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'eval', 'train_and_eval'],
                       default='train_and_eval',
                       help='Execution mode')
    
    parser.add_argument('--eval-only', action='store_true',
                       help='Run evaluation only (sets mode=eval)')
    
    parser.add_argument('--checkpoint', type=str,
                       help='Checkpoint path for evaluation or resuming training')
    
    parser.add_argument('--zero-shot', action='store_true',
                       help='Run zero-shot evaluation')
    
    # ========== BACKEND & SYSTEM ==========
    parser.add_argument('--backend', type=str,
                       help='Training backend config (e.g., "deepspeed:stage=2")')
    
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # ========== UTILITY ==========
    parser.add_argument('--config', type=str,
                       help='Load configuration from YAML file')
    
    parser.add_argument('--save-config', type=str,
                       help='Save generated config to YAML file (without training)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without training')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive configuration mode')
    
    args = parser.parse_args()
    
    # Post-processing
    if args.eval_only:
        args.mode = 'eval'
    
    if args.eval_interval and not args.eval_steps:
        args.eval_steps = args.eval_interval
    
    if args.save_interval and not args.save_steps:
        args.save_steps = args.save_interval
    
    return args

# =============================================================================
# CONFIGURATION BUILDERS
# =============================================================================

def build_base_config(args) -> Dict[str, Any]:
    """Build base configuration common to all algorithms"""
    config = {
        'experiment_name': args.experiment or f'{args.algo}_{args.task_type or "default"}',
        'output_dir': args.output,
        'seed': args.seed,
        'mode': args.mode,
    }
    
    # Logging
    config['logging'] = {
        'use_wandb': args.wandb,
        'use_tensorboard': args.tensorboard,
        'log_level': args.log_level,
    }
    
    return config

def build_model_config(args) -> Dict[str, Any]:
    """Build model configuration"""
    config = {
        'name': args.model,
        'task_type': args.task_type or 'supervised_fine_tuning',
    }
    
    # Precision
    if args.precision:
        if args.precision == 'fp16':
            config['fp16'] = True
            config['bf16'] = False
        elif args.precision == 'bf16':
            config['bf16'] = True
            config['fp16'] = False
        else:  # fp32
            config['fp16'] = False
            config['bf16'] = False
    else:
        config['fp16'] = None
        config['bf16'] = None
    
    # Quantization
    if args.quantization:
        config['load_in_4bit'] = '4bit' in args.quantization.lower()
        config['load_in_8bit'] = '8bit' in args.quantization.lower()
        # Parse additional quant options if provided
        if ':' in args.quantization:
            # e.g., "4bit:quant_type=nf4:compute_dtype=float16"
            parts = args.quantization.split(':')[1:]
            for part in parts:
                if '=' in part:
                    key, val = part.split('=', 1)
                    config[f'quant_{key}'] = val
    
    # LoRA/PEFT
    config['peft_enabled'] = not args.no_peft
    if args.lora_rank:
        config['lora_rank'] = args.lora_rank
    if args.lora_alpha:
        config['lora_alpha'] = args.lora_alpha
    if args.lora_dropout:
        config['lora_dropout'] = args.lora_dropout
    
    # Sequence length
    if args.max_seq_length:
        config['max_seq_length'] = args.max_seq_length
    
    # Gradient checkpointing
    if args.grad_checkpointing:
        config['use_gradient_checkpointing'] = args.grad_checkpointing.lower() == 'true'
    
    # Algorithm-specific: Disable Unsloth for PPO
    if args.algo == 'ppo':
        config['use_unsloth'] = False
    else:
        config['use_unsloth'] = True
    
    return config

def build_dataset_config(args) -> Dict[str, Any]:
    """Build dataset configuration"""
    config = {}
    
    if args.dataset:
        # Parse dataset specification
        # Format: "dataset_name:split#percent=25" or "dataset_name"
        dataset_spec = args.dataset
        
        if '#' in dataset_spec:
            dataset_name, params = dataset_spec.split('#', 1)
            # Parse params like "percent=25"
            for param in params.split(','):
                if '=' in param:
                    key, val = param.split('=', 1)
                    if key == 'percent':
                        config['dataset_percentage'] = float(val) / 100.0
        else:
            dataset_name = dataset_spec
        
        # Handle dataset:split format
        if ':' in dataset_name and not dataset_name.startswith('http'):
            name, split = dataset_name.rsplit(':', 1)
            config['name'] = name
            config['split'] = split
        else:
            config['name'] = dataset_name
    
    # Override with CLI args
    if args.dataset_percentage:
        config['dataset_percentage'] = args.dataset_percentage
    
    if args.validation_split:
        try:
            config['validation_split'] = float(args.validation_split)
        except ValueError:
            config['validation_split'] = args.validation_split
    
    # Batch size
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.accum:
        config['gradient_accumulation_steps'] = args.accum
    
    # Column mapping
    if args.column_mapping:
        try:
            import json
            config['column_mapping'] = json.loads(args.column_mapping)
        except:
            # Parse as key=value pairs
            mapping = {}
            for pair in args.column_mapping.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    mapping[k.strip()] = v.strip()
            if mapping:
                config['column_mapping'] = mapping
    
    return config

def build_training_config(args) -> Dict[str, Any]:
    """Build training configuration"""
    config = {
        'task_type': args.task_type or 'supervised_fine_tuning',
    }
    
    if args.epochs:
        config['epochs'] = args.epochs
    
    if args.max_steps:
        config['max_steps'] = args.max_steps
    
    if args.lr:
        config['learning_rate'] = args.lr
    
    if args.optimizer:
        config['optimizer'] = args.optimizer
    
    if args.warmup_ratio:
        config['warmup_ratio'] = args.warmup_ratio
    
    # Precision
    if args.precision:
        if args.precision == 'fp16':
            config['fp16'] = True
            config['bf16'] = False
        elif args.precision == 'bf16':
            config['bf16'] = True
            config['fp16'] = False
        else:
            config['fp16'] = False
            config['bf16'] = False
    else:
        config['fp16'] = None
        config['bf16'] = None
    
    # Save/eval intervals
    if args.save_steps:
        config['save_steps'] = args.save_steps
    
    if args.eval_steps:
        config['eval_steps'] = args.eval_steps
    
    return config

def build_evaluation_config(args) -> Dict[str, Any]:
    """Build evaluation configuration"""
    config = {}
    
    if args.zero_shot:
        config['run_zero_shot'] = True
    
    if args.checkpoint:
        config['eval_checkpoint'] = args.checkpoint
    
    return config

# =============================================================================
# ALGORITHM-SPECIFIC BUILDERS
# =============================================================================

def build_sft_config(args) -> Dict[str, Any]:
    """Build configuration for SFT training"""
    config = build_base_config(args)
    config['model'] = build_model_config(args)
    config['dataset'] = build_dataset_config(args)
    config['training'] = build_training_config(args)
    config['evaluation'] = build_evaluation_config(args)
    
    # SFT-specific defaults
    if 'epochs' not in config['training']:
        config['training']['epochs'] = 1
    if 'batch_size' not in config['dataset']:
        config['dataset']['batch_size'] = 2
    if 'gradient_accumulation_steps' not in config['dataset']:
        config['dataset']['gradient_accumulation_steps'] = 4
    
    return config

def build_dpo_config(args) -> Dict[str, Any]:
    """Build configuration for DPO training"""
    # Use unified config system - return dict for create_rl_trainer
    scenario = args.scenario or 'general'
    
    # Build base config dict
    config_dict = {
        'model_name': args.model,
        'dataset_name': args.dataset or 'Anthropic/hh-rlhf',
        'algorithm': 'dpo',
        'backend': 'auto',
        'output_dir': args.output,
        'max_seq_length': args.max_seq_length or 1024,
        'learning_rate': args.lr or 5e-6,
        'batch_size': args.batch_size or 1,
        'gradient_accumulation_steps': args.accum or 4,
        'max_steps': args.max_steps or 200,
    }
    
    # DPO-specific parameters
    if args.beta:
        config_dict['beta'] = args.beta
    else:
        # Scenario-based defaults
        beta_map = {'general': 0.1, 'safety': 0.2, 'helpfulness': 0.15}
        config_dict['beta'] = beta_map.get(scenario, 0.1)
    
    if args.loss_type:
        config_dict['loss_type'] = args.loss_type
    
    # PEFT
    config_dict['use_peft'] = not args.no_peft
    if args.lora_rank:
        config_dict['lora_r'] = args.lora_rank
    if args.lora_alpha:
        config_dict['lora_alpha'] = args.lora_alpha
    
    return config_dict

def build_ppo_config(args) -> Dict[str, Any]:
    """Build configuration for PPO training"""
    # Use unified config system instead of legacy YAMLPPOConfig
    from aligntune.core.rl.config_loader import ConfigLoader
    
    task_type = args.task_type or 'conversation'
    
    config_dict = {
        'exp_name': args.experiment or f'ppo_{task_type}',
        'output_dir': os.path.join(args.output, f'ppo_{task_type}'),
        'dataset_name': args.dataset or 'Anthropic/hh-rlhf',
        'dataset_train_split': 'train',
        'dataset_test_split': 'test',
        'dataset_num_proc': 1,
        'max_length': args.max_seq_length or 512,
        'dataset_size_percent': args.dataset_percentage * 100 if args.dataset_percentage else 5.0,
        'model_name_or_path': args.model,
        'sft_model_path': args.model,
        'reward_model_path': args.model,
        'learning_rate': args.lr or 1e-5,
        'total_episodes': args.max_steps or 200,
        'per_device_train_batch_size': args.batch_size or 1,
        'gradient_accumulation_steps': args.accum or 32,
        'num_ppo_epochs': args.ppo_epochs or 4,
        'response_length': 64,
        'eval_steps': args.eval_steps or 50,
        'save_steps': args.save_steps or 100,
        'use_peft': not args.no_peft,
        'seed': args.seed,
    }
    
    # PPO-specific
    if args.kl_coef:
        config_dict['kl_coef'] = args.kl_coef
    
    if args.lora_rank:
        config_dict['lora_r'] = args.lora_rank
    if args.lora_alpha:
        config_dict['lora_alpha'] = args.lora_alpha
    
    # Precision - PPO requires bf16=True, fp16=False
    config_dict['bf16'] = True
    config_dict['fp16'] = False
    
    # Logging
    config_dict['report_to'] = []
    if args.tensorboard:
        config_dict['report_to'].append('tensorboard')
    if args.wandb:
        config_dict['report_to'].append('wandb')
    
    # Return dict for create_rl_trainer - no need for YAMLPPOConfig
    return config_dict

def build_grpo_config(args) -> Dict[str, Any]:
    """Build configuration for GRPO training"""
    # Use unified config system - return dict for create_rl_trainer
    config_dict = {
        'model_name': args.model,
        'dataset_name': args.dataset or 'Anthropic/hh-rlhf',
        'algorithm': 'grpo',
        'backend': 'auto',
        'model_name_or_path': args.model,
        'max_seq_length': args.max_seq_length or 512,
        'learning_rate': args.lr or 5e-6,
        'max_steps': args.max_steps or 10,
        'per_device_train_batch_size': args.batch_size or 1,
        'gradient_accumulation_steps': args.accum or 2,
        'output_dir': os.path.join(args.output, 'grpo'),
        'experiment_name': args.experiment or 'grpo_experiment',
        'use_peft': not args.no_peft,
        'load_in_4bit': '4bit' in (args.quantization or ''),
        'use_unsloth': True,
        'seed': args.seed,
    }
    
    # Parse datasets
    if args.dataset:
        datasets = []
        for ds_spec in args.dataset.split(','):
            ds_config = {'name': ds_spec.strip(), 'split': 'train', 'weight': 1.0}
            
            # Parse dataset config (e.g., "alpaca:weight=1.0:samples=50")
            if ':' in ds_spec:
                parts = ds_spec.split(':')
                ds_config['name'] = parts[0]
                for part in parts[1:]:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        if key == 'weight':
                            ds_config['weight'] = float(val)
                        elif key in ['samples', 'max_samples']:
                            ds_config['max_samples'] = int(val)
                        elif key == 'split':
                            ds_config['split'] = val
            
            datasets.append(ds_config)
        
        config_dict['datasets'] = datasets
    
    # GRPO-specific
    if args.num_generations:
        config_dict['num_generations'] = args.num_generations
    
    if args.use_math:
        config_dict['use_math'] = True
    
    # LoRA
    if args.lora_rank:
        config_dict['lora_r'] = args.lora_rank
    if args.lora_alpha:
        config_dict['lora_alpha'] = args.lora_alpha
    
    # Precision
    if args.precision == 'bf16':
        config_dict['bf16'] = True
    
    # Logging
    config_dict['logging_config'] = {
        'use_wandb': args.wandb,
        'use_tensorboard': args.tensorboard,
    }
    
    # Return dict for create_rl_trainer - no need for EnhancedGRPOConfig
    return config_dict

def build_classification_config(args) -> Dict[str, Any]:
    """Build configuration for classification training"""
    config = build_base_config(args)
    config['model'] = build_model_config(args)
    config['dataset'] = build_dataset_config(args)
    config['training'] = build_training_config(args)
    config['evaluation'] = build_evaluation_config(args)
    
    # Classification-specific
    config['model']['use_unsloth'] = False  # Must be False
    config['training']['use_trl'] = False   # Must be False
    
    if not args.task_type:
        config['model']['task_type'] = 'text_classification'
        config['training']['task_type'] = 'text_classification'
    
    # Defaults
    if 'epochs' not in config['training']:
        config['training']['epochs'] = 3
    if 'batch_size' not in config['dataset']:
        config['dataset']['batch_size'] = 16
    
    return config

# =============================================================================
# TRAINING EXECUTION
# =============================================================================

def run_sft_training(config: Dict[str, Any]):
    """Execute SFT training"""
    from aligntune.core.backend_factory import create_sft_trainer
    
    print("\n" + "="*60)
    print("Starting SFT Training")
    print("="*60)
    
    # Extract config parameters and use create_sft_trainer
    trainer = create_sft_trainer(
        model_name=config.get('model', {}).get('name', config.get('model_name', 'microsoft/DialoGPT-small')),
        dataset_name=config.get('dataset', {}).get('name', config.get('dataset_name', 'tatsu-lab/alpaca')),
        backend=config.get('backend', 'auto'),
        output_dir=config.get('output_dir', './output'),
        num_epochs=config.get('training', {}).get('epochs', config.get('epochs', 1)),
        batch_size=config.get('dataset', {}).get('batch_size', config.get('batch_size', 1)),
        learning_rate=config.get('training', {}).get('learning_rate', config.get('learning_rate', 5e-5)),
        max_seq_length=config.get('model', {}).get('max_seq_length', config.get('max_seq_length', 512)),
        **config
    )
    trainer.train()
    
    print("\n" + "="*60)
    print("SFT Training Complete")
    print("="*60)
    
    return {}

def run_dpo_training(config):
    """Execute DPO training"""
    from aligntune.core.backend_factory import create_rl_trainer
    
    print("\n" + "="*60)
    print("Starting DPO Training")
    print("="*60)
    
    # Extract config parameters and use create_rl_trainer
    trainer = create_rl_trainer(
        model_name=config.get('model_name', 'microsoft/DialoGPT-medium'),
        dataset_name=config.get('dataset_name', 'Anthropic/hh-rlhf'),
        algorithm='dpo',
        backend=config.get('backend', 'auto'),
        **config
    )
    trainer.train()
    
    print("\n" + "="*60)
    print("DPO Training Complete")
    print("="*60)
    
    return {}

def run_ppo_training(config):
    """Execute PPO training"""
    from aligntune.core.backend_factory import create_rl_trainer
    
    print("\n" + "="*60)
    print("Starting PPO Training")
    print("="*60)
    
    # Extract config parameters and use create_rl_trainer
    trainer = create_rl_trainer(
        model_name=config.get('model_name', 'microsoft/DialoGPT-medium'),
        dataset_name=config.get('dataset_name', 'Anthropic/hh-rlhf'),
        algorithm='ppo',
        backend=config.get('backend', 'auto'),
        **config
    )
    result = trainer.train()
    
    print("\n" + "="*60)
    print("PPO Training Complete")
    print(f"Model saved to: {config.output_dir}")
    print("="*60)
    
    return result

def run_grpo_training(config):
    """Execute GRPO training"""
    from aligntune.core.backend_factory import create_rl_trainer
    
    print("\n" + "="*60)
    print("Starting GRPO Training")
    print("="*60)
    
    # Extract config parameters and use create_rl_trainer
    trainer = create_rl_trainer(
        model_name=config.get('model_name', config.get('model_name_or_path', 'microsoft/DialoGPT-medium')),
        dataset_name=config.get('dataset_name', 'Anthropic/hh-rlhf'),
        algorithm='grpo',
        backend=config.get('backend', 'auto'),
        **config
    )
    result = trainer.train()
    
    print("\n" + "="*60)
    print("GRPO Training Complete")
    print("="*60)
    
    return result

def run_classification_training(config: Dict[str, Any]):
    """Execute classification training"""
    from aligntune.core.backend_factory import create_sft_trainer
    
    print("\n" + "="*60)
    print("Starting Classification Training")
    print("="*60)
    
    # Extract config parameters and use create_sft_trainer
    trainer = create_sft_trainer(
        model_name=config.get('model', {}).get('name', config.get('model_name', 'microsoft/DialoGPT-small')),
        dataset_name=config.get('dataset', {}).get('name', config.get('dataset_name', 'tatsu-lab/alpaca')),
        backend='trl',  # Classification tasks use TRL backend
        output_dir=config.get('output_dir', './output'),
        num_epochs=config.get('training', {}).get('epochs', config.get('epochs', 1)),
        batch_size=config.get('dataset', {}).get('batch_size', config.get('batch_size', 1)),
        learning_rate=config.get('training', {}).get('learning_rate', config.get('learning_rate', 5e-5)),
        max_seq_length=config.get('model', {}).get('max_seq_length', config.get('max_seq_length', 512)),
        task_type='text_classification',
        **config
    )
    trainer.train()
    
    print("\n" + "="*60)
    print("Classification Training Complete")
    print("="*60)
    
    return {}

# =============================================================================
# CONFIGURATION DISPLAY & VALIDATION
# =============================================================================

def display_config(config, algo: str):
    """Display configuration summary"""
    print("\n" + "="*60)
    print(f"  CONFIGURATION SUMMARY - {algo.upper()}")
    print("="*60)
    
    if isinstance(config, dict):
        print(f"Experiment:  {config.get('experiment_name', 'N/A')}")
        print(f"Output:      {config.get('output_dir', 'N/A')}")
        print(f"Mode:        {config.get('mode', 'N/A')}")
        
        if 'model' in config:
            print(f"\nModel Configuration:")
            print(f"  Name:        {config['model'].get('name', 'N/A')}")
            print(f"  Task:        {config['model'].get('task_type', 'N/A')}")
            print(f"  PEFT:        {config['model'].get('peft_enabled', 'N/A')}")
            print(f"  LoRA Rank:   {config['model'].get('lora_rank', 'N/A')}")
            print(f"  Precision:   fp16={config['model'].get('fp16')}, bf16={config['model'].get('bf16')}")
        
        if 'dataset' in config:
            print(f"\nDataset Configuration:")
            print(f"  Name:        {config['dataset'].get('name', 'N/A')}")
            print(f"  Batch Size:  {config['dataset'].get('batch_size', 'N/A')}")
            print(f"  Percentage:  {config['dataset'].get('dataset_percentage', 1.0)*100:.1f}%")
        
        if 'training' in config:
            print(f"\nTraining Configuration:")
            print(f"  Epochs:      {config['training'].get('epochs', 'N/A')}")
            print(f"  Max Steps:   {config['training'].get('max_steps', 'N/A')}")
            print(f"  LR:          {config['training'].get('learning_rate', 'N/A')}")
            print(f"  Optimizer:   {config['training'].get('optimizer', 'N/A')}")
    
    else:
        # For dataclass configs (DPO, PPO, GRPO)
        print(f"Experiment:  {getattr(config, 'experiment_name', getattr(config, 'exp_name', 'N/A'))}")
        print(f"Output:      {getattr(config, 'output_dir', 'N/A')}")
        print(f"Model:       {getattr(config, 'model_name_or_path', 'N/A')}")
        
        if hasattr(config, 'dataset_name'):
            print(f"Dataset:     {config.dataset_name}")
        elif hasattr(config, 'datasets'):
            print(f"Datasets:    {len(config.datasets)} datasets")
        
        print(f"Learning Rate: {getattr(config, 'learning_rate', 'N/A')}")
        print(f"Batch Size:  {getattr(config, 'per_device_train_batch_size', 'N/A')}")
        
        if hasattr(config, 'beta'):
            print(f"Beta:        {config.beta}")
        if hasattr(config, 'kl_coef'):
            print(f"KL Coef:     {config.kl_coef}")
    
    print("="*60 + "\n")

def validate_config(config, algo: str) -> bool:
    """Validate configuration before training"""
    if isinstance(config, dict):
        # Basic validation for dict configs
        if 'model' not in config or 'name' not in config['model']:
            print("ERROR: Model name is required")
            return False
        
        if algo in ['sft', 'classification'] and 'dataset' not in config:
            print("ERROR: Dataset configuration is required")
            return False
    
    return True

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode() -> Dict[str, Any]:
    """Interactive configuration builder"""
    print("\n" + "="*60)
    print("  INTERACTIVE CONFIGURATION MODE")
    print("="*60)
    
    # Select algorithm
    print("\nSelect Training Algorithm:")
    print("  1. SFT (Supervised Fine-Tuning)")
    print("  2. DPO (Direct Preference Optimization)")
    print("  3. PPO (Proximal Policy Optimization)")
    print("  4. GRPO (Group Relative Policy Optimization)")
    print("  5. Classification (Text/Token Classification)")
    
    algo_choice = input("\nEnter choice (1-5): ").strip()
    algo_map = {'1': 'sft', '2': 'dpo', '3': 'ppo', '4': 'grpo', '5': 'classification'}
    algo = algo_map.get(algo_choice, 'sft')
    
    print(f"\nSelected: {algo.upper()}")
    
    # Get model
    print("\nCommon models:")
    if algo in ['sft', 'dpo', 'grpo']:
        print("  - unsloth/mistral-7b-bnb-4bit")
        print("  - unsloth/llama-2-7b-bnb-4bit")
        print("  - unsloth/Qwen2-1.5B-Instruct-bnb-4bit")
    elif algo == 'ppo':
        print("  - EleutherAI/pythia-1b-deduped")
        print("  - gpt2")
    elif algo == 'classification':
        print("  - distilbert-base-uncased")
        print("  - bert-base-uncased")
    
    model = input("\nModel name (or press Enter for default): ").strip()
    if not model:
        model = {
            'sft': 'unsloth/mistral-7b-bnb-4bit',
            'dpo': 'unsloth/Qwen2-1.5B-Instruct-bnb-4bit',
            'ppo': 'EleutherAI/pythia-1b-deduped',
            'grpo': 'unsloth/llama-2-7b-bnb-4bit',
            'classification': 'distilbert-base-uncased'
        }[algo]
    
    # Get dataset
    print("\nCommon datasets:")
    if algo in ['sft']:
        print("  - databricks/databricks-dolly-15k (instruction following)")
        print("  - wikitext (text generation)")
    elif algo in ['dpo', 'ppo']:
        print("  - Anthropic/hh-rlhf")
    elif algo == 'grpo':
        print("  - alpaca,gsm8k (comma-separated)")
    elif algo == 'classification':
        print("  - imdb (text classification)")
        print("  - conll2003 (token classification)")
    
    dataset = input("\nDataset name (or press Enter for default): ").strip()
    if not dataset:
        dataset = {
            'sft': 'databricks/databricks-dolly-15k',
            'dpo': 'Anthropic/hh-rlhf',
            'ppo': 'Anthropic/hh-rlhf',
            'grpo': 'alpaca',
            'classification': 'imdb'
        }[algo]
    
    # Quick or detailed config?
    detail = input("\nUse detailed configuration? (y/N): ").strip().lower()
    
    # Build args namespace
    class Args:
        pass
    
    args = Args()
    args.algo = algo
    args.model = model
    args.dataset = dataset
    args.task_type = None
    args.scenario = None
    args.experiment = None
    args.output = './results'
    args.mode = 'train_and_eval'
    args.seed = 3407
    
    # Set defaults based on quick/detailed
    if detail == 'y':
        args.epochs = int(input("Epochs (default 1): ").strip() or "1")
        args.batch_size = int(input("Batch size (default 2): ").strip() or "2")
        args.lr = float(input("Learning rate (default 2e-4): ").strip() or "2e-4")
        args.dataset_percentage = float(input("Dataset percentage (0.0-1.0, default 0.1): ").strip() or "0.1")
    else:
        # Quick defaults
        args.epochs = 1
        args.batch_size = 2 if algo != 'classification' else 16
        args.lr = 2e-4
        args.dataset_percentage = 0.1
    
    # Set other defaults
    args.max_steps = None
    args.accum = 4
    args.optimizer = None
    args.warmup_ratio = None
    args.precision = None
    args.quantization = '4bit' if algo != 'classification' else None
    args.max_seq_length = None
    args.lora_rank = None
    args.lora_alpha = None
    args.lora_dropout = None
    args.no_peft = False
    args.grad_checkpointing = None
    args.beta = None
    args.loss_type = None
    args.rewards = None
    args.kl_coef = None
    args.ppo_epochs = None
    args.num_generations = None
    args.use_math = False
    args.validation_split = None
    args.column_mapping = None
    args.wandb = False
    args.tensorboard = False
    args.log_level = 'INFO'
    args.save_steps = None
    args.eval_steps = None
    args.eval_interval = None
    args.save_interval = None
    args.eval_only = False
    args.checkpoint = None
    args.zero_shot = False
    args.backend = None
    args.num_workers = 0
    args.config = None
    args.save_config = None
    args.dry_run = False
    args.interactive = False
    
    return args

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point - delegates to the unified CLI."""
    try:
        # Import and run the unified CLI
        from .cli import app
        app()
    except ImportError:
        print("ERROR: Could not import CLI.")
        print("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())