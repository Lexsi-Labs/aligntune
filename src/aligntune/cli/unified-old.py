"""
Unified CLI for aligntune.

This module provides a unified command-line interface that integrates with the
backend factory system, supporting all training types and backends.
"""

import logging
import typer
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
from ..core.backend_factory import (
    BackendFactory,
    TrainingType,
    BackendType,
    RLAlgorithm,
    create_sft_trainer,
    create_rl_trainer,
    list_backends,
)

# Evaluation imports are optional; command will error clearly if unavailable
try:
    from ..eval.lm_eval_integration import (
        LMEvalRunner,
        LMEvalConfig,
        get_lm_eval_task,
    )
    EVAL_AVAILABLE = True
except Exception:  # ImportError or runtime issues
    LMEvalRunner = None
    LMEvalConfig = None
    get_lm_eval_task = None
    EVAL_AVAILABLE = False

# Recipes CLI imports
try:
    from .recipes import app as recipes_app
    RECIPES_AVAILABLE = True
except ImportError:
    recipes_app = None
    RECIPES_AVAILABLE = False

# Validate CLI imports
try:
    from .validate import app as validate_app
    VALIDATE_AVAILABLE = True
except ImportError:
    validate_app = None
    VALIDATE_AVAILABLE = False

# Diagnose CLI imports
try:
    from .diagnose import app as diagnose_app
    DIAGNOSE_AVAILABLE = True
except ImportError:
    diagnose_app = None
    DIAGNOSE_AVAILABLE = False

app = typer.Typer(
    name="finetunehub",
    help="FinetuneHub: A comprehensive fine-tuning library for SFT and RL training",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)



def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(yaml_config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Merge YAML config with CLI arguments, CLI takes precedence."""
    merged = yaml_config.copy()
    
    # CLI args override YAML config
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    
    return merged


@app.command()
def train(
    # ========== Core Configuration ==========
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    
    # ========== Model Configuration ==========
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name or path (e.g., 'microsoft/DialoGPT-medium')"
    ),
    precision: Optional[str] = typer.Option(
        None, "--precision", help="Model precision: fp32, fp16, bf16"
    ),
    gradient_checkpointing: Optional[bool] = typer.Option(
        None, "--gradient-checkpointing", help="Enable gradient checkpointing"
    ),
    attn_implementation: Optional[str] = typer.Option(
        None, "--attn-implementation", help="Attention implementation: auto, flash_attention_2, sdpa"
    ),
    use_unsloth: Optional[bool] = typer.Option(
        None, "--use-unsloth", help="Use Unsloth optimizations"
    ),
    
    # ========== Training Type & Backend ==========
    training_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Training type: sft, dpo, ppo, grpo, gspo"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Backend: auto, trl, unsloth"
    ),
    
    # ========== Dataset Configuration ==========
    dataset_name: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Dataset name (e.g., 'tatsu-lab/alpaca')"
    ),
    dataset_split: Optional[str] = typer.Option(
        None, "--split", help="Dataset split to use"
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", help="Maximum number of samples to use"
    ),
    dataset_percent: Optional[float] = typer.Option(
        None, "--percent", help="Percentage of dataset to use (0.0-1.0)"
    ),
    task_type: Optional[str] = typer.Option(
        None, "--task-type", help="Task type: supervised_fine_tuning, instruction_following, chat, text_classification, token_classification"
    ),
    
    # Column Mapping
    column_mapping_prompt: Optional[str] = typer.Option(
        None, "--col-prompt", help="Column name for prompts"
    ),
    column_mapping_chosen: Optional[str] = typer.Option(
        None, "--col-chosen", help="Column name for chosen responses"
    ),
    column_mapping_rejected: Optional[str] = typer.Option(
        None, "--col-rejected", help="Column name for rejected responses"
    ),
    column_mapping_text: Optional[str] = typer.Option(
        None, "--col-text", help="Column name for text"
    ),
    column_mapping_instruction: Optional[str] = typer.Option(
        None, "--col-instruction", help="Column name for instructions"
    ),
    column_mapping_input: Optional[str] = typer.Option(
        None, "--col-input", help="Column name for input"
    ),
    column_mapping_output: Optional[str] = typer.Option(
        None, "--col-output", help="Column name for output"
    ),
    column_mapping_context: Optional[str] = typer.Option(
        None, "--col-context", help="Column name for context"
    ),
    column_mapping_label: Optional[str] = typer.Option(
        None, "--col-label", help="Column name for labels"
    ),
    
    # ========== Training Parameters ==========
    num_epochs: Optional[int] = typer.Option(
        None, "--epochs", "-e", help="Number of training epochs"
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", help="Maximum training steps (overrides epochs)"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Per-device batch size"
    ),
    learning_rate: Optional[float] = typer.Option(
        None, "--lr", help="Learning rate"
    ),
    max_seq_length: Optional[int] = typer.Option(
        None, "--max-length", help="Maximum sequence length"
    ),
    max_prompt_length: Optional[int] = typer.Option(
        None, "--max-prompt-length", help="Maximum prompt length (RL only)"
    ),
    gradient_accumulation_steps: Optional[int] = typer.Option(
        None, "--grad-accum", help="Gradient accumulation steps"
    ),
    warmup_ratio: Optional[float] = typer.Option(
        None, "--warmup", help="Warmup ratio"
    ),
    warmup_steps: Optional[int] = typer.Option(
        None, "--warmup-steps", help="Warmup steps"
    ),
    weight_decay: Optional[float] = typer.Option(
        None, "--weight-decay", help="Weight decay"
    ),
    
    # ========== RL-Specific Parameters ==========
    beta: Optional[float] = typer.Option(
        None, "--beta", help="Beta parameter for DPO/KTO"
    ),
    kl_coef: Optional[float] = typer.Option(
        None, "--kl-coef", help="KL divergence coefficient for PPO/GRPO"
    ),
    response_length: Optional[int] = typer.Option(
        None, "--response-length", help="Maximum response length for generation"
    ),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", help="Temperature for generation"
    ),
    reward_model_name: Optional[str] = typer.Option(
        None, "--reward-model", help="Reward model name or path (for PPO/GRPO)"
    ),
    reward_model_path: Optional[str] = typer.Option(
        None, "--reward-model-path", help="Local path to reward model"
    ),
    loss_type: Optional[str] = typer.Option(
        None, "--loss-type", help="Loss type: sigmoid, hinge, ipo, kto_pair, etc."
    ),
    
    # ========== Output & Logging ==========
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for trained model"
    ),
    run_name: Optional[str] = typer.Option(
        None, "--run-name", help="Name for this training run"
    ),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR"
    ),
    loggers: Optional[str] = typer.Option(
        None, "--loggers", help="Comma-separated list of loggers: tensorboard,wandb,mlflow"
    ),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="Weights & Biases project name"
    ),
    
    # ========== Evaluation & Saving ==========
    eval_interval: Optional[int] = typer.Option(
        None, "--eval-interval", help="Steps between evaluations"
    ),
    save_interval: Optional[int] = typer.Option(
        None, "--save-interval", help="Steps between model saves"
    ),
    eval_steps: Optional[int] = typer.Option(
        None, "--eval-steps", help="Steps between evaluations (legacy, use eval-interval)"
    ),
    save_steps: Optional[int] = typer.Option(
        None, "--save-steps", help="Steps between saves (legacy, use save-interval)"
    ),
    
    # ========== Quantization ==========
    load_in_4bit: Optional[bool] = typer.Option(
        None, "--4bit", help="Load model in 4-bit quantization"
    ),
    load_in_8bit: Optional[bool] = typer.Option(
        None, "--8bit", help="Load model in 8-bit quantization"
    ),
    bnb_4bit_compute_dtype: Optional[str] = typer.Option(
        None, "--bnb-dtype", help="BitsAndBytes 4-bit compute dtype"
    ),
    bnb_4bit_quant_type: Optional[str] = typer.Option(
        None, "--bnb-quant-type", help="BitsAndBytes quantization type"
    ),
    
    # ========== LoRA/PEFT ==========
    use_peft: Optional[bool] = typer.Option(
        None, "--use-peft", help="Enable PEFT/LoRA"
    ),
    lora_r: Optional[int] = typer.Option(
        None, "--lora-r", help="LoRA rank"
    ),
    lora_alpha: Optional[int] = typer.Option(
        None, "--lora-alpha", help="LoRA alpha"
    ),
    lora_dropout: Optional[float] = typer.Option(
        None, "--lora-dropout", help="LoRA dropout"
    ),
    lora_target_modules: Optional[str] = typer.Option(
        None, "--lora-targets", help="Comma-separated LoRA target modules"
    ),
    
    # ========== Advanced Options ==========
    packing: Optional[bool] = typer.Option(
        None, "--packing", help="Enable sequence packing (SFT only)"
    ),
    chat_template: Optional[str] = typer.Option(
        None, "--chat-template", help="Chat template: auto, chatml, llama2, etc."
    ),
    dataset_num_proc: Optional[int] = typer.Option(
        None, "--dataset-num-proc", help="Number of processes for dataset preprocessing"
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Cache directory"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed"
    ),
    
    # ========== Distributed Training ==========
    distributed_backend: Optional[str] = typer.Option(
        None, "--distributed", help="Distributed backend: single, ddp, deepspeed, fsdp"
    ),
):
    """
    Train a model using aligntune.
    
    Configuration can be provided via:
    1. YAML config file (--config)
    2. CLI arguments (override YAML)
    3. Both (CLI takes precedence)
    
    Examples:
    
    # Using YAML config
    finetunehub train --config configs/dpo_simple.yaml
    
    # Using CLI only
    finetunehub train --model microsoft/DialoGPT-small --dataset Anthropic/hh-rlhf --type dpo
    
    # Mixing YAML and CLI (CLI overrides)
    finetunehub train --config configs/dpo_simple.yaml --batch-size 8 --lr 1e-4
    """
    
    # Initialize config
    final_config = {}
    
    # Load YAML config if provided
    if config:
        if not Path(config).exists():
            typer.echo(f"Error: Config file not found: {config}")
            raise typer.Exit(1)
        
        final_config = load_yaml_config(config)
        typer.echo(f"✓ Loaded configuration from: {config}")
    
    # Build CLI config (non-None values)
    cli_config = {}
    
    # Model config
    model_config = {}
    if model_name: model_config['name_or_path'] = model_name
    if precision: model_config['precision'] = precision
    if gradient_checkpointing is not None: model_config['gradient_checkpointing'] = gradient_checkpointing
    if attn_implementation: model_config['attn_implementation'] = attn_implementation
    if use_unsloth is not None: model_config['use_unsloth'] = use_unsloth
    if max_seq_length: model_config['max_seq_length'] = max_seq_length
    if use_peft is not None: model_config['use_peft'] = use_peft
    if lora_r: model_config['lora_r'] = lora_r
    if lora_alpha: model_config['lora_alpha'] = lora_alpha
    if lora_dropout: model_config['lora_dropout'] = lora_dropout
    if lora_target_modules: model_config['lora_target_modules'] = lora_target_modules.split(',')
    if reward_model_name: model_config['reward_model_name'] = reward_model_name
    if reward_model_path: model_config['reward_model_path'] = reward_model_path
    
    # Quantization
    if load_in_4bit or load_in_8bit:
        quantization = {}
        if load_in_4bit: 
            quantization['load_in_4bit'] = True
            if bnb_4bit_compute_dtype: quantization['bnb_4bit_compute_dtype'] = bnb_4bit_compute_dtype
            if bnb_4bit_quant_type: quantization['bnb_4bit_quant_type'] = bnb_4bit_quant_type
        if load_in_8bit: quantization['load_in_8bit'] = True
        model_config['quantization'] = quantization
    
    if model_config:
        cli_config['model'] = model_config
    
    # Dataset config
    dataset_config = {}
    if dataset_name: dataset_config['name'] = dataset_name
    if dataset_split: dataset_config['split'] = dataset_split
    if max_samples: dataset_config['max_samples'] = max_samples
    if dataset_percent: dataset_config['percent'] = dataset_percent
    if task_type: dataset_config['task_type'] = task_type
    if dataset_num_proc: dataset_config['dataset_num_proc'] = dataset_num_proc
    if chat_template: dataset_config['chat_template'] = chat_template
    
    # Column mapping
    column_mapping = {}
    if column_mapping_prompt: column_mapping['prompt'] = column_mapping_prompt
    if column_mapping_chosen: column_mapping['chosen'] = column_mapping_chosen
    if column_mapping_rejected: column_mapping['rejected'] = column_mapping_rejected
    if column_mapping_text: column_mapping['text'] = column_mapping_text
    if column_mapping_instruction: column_mapping['instruction'] = column_mapping_instruction
    if column_mapping_input: column_mapping['input'] = column_mapping_input
    if column_mapping_output: column_mapping['output'] = column_mapping_output
    if column_mapping_context: column_mapping['context'] = column_mapping_context
    if column_mapping_label: column_mapping['label'] = column_mapping_label
    
    if column_mapping:
        dataset_config['column_mapping'] = column_mapping
    
    if dataset_config:
        # Handle both single dataset and list format
        if 'datasets' in final_config and isinstance(final_config['datasets'], list):
            # Merge with first dataset in list
            if final_config['datasets']:
                final_config['datasets'][0].update(dataset_config)
            else:
                final_config['datasets'] = [dataset_config]
        else:
            cli_config['datasets'] = [dataset_config]
    
    # Training config
    train_config = {}
    if num_epochs: train_config['epochs'] = num_epochs
    if max_steps: train_config['max_steps'] = max_steps
    if batch_size: train_config['per_device_batch_size'] = batch_size
    if learning_rate: train_config['learning_rate'] = learning_rate
    if gradient_accumulation_steps: train_config['gradient_accumulation_steps'] = gradient_accumulation_steps
    if warmup_ratio: train_config['warmup_ratio'] = warmup_ratio
    if warmup_steps: train_config['warmup_steps'] = warmup_steps
    if weight_decay: train_config['weight_decay'] = weight_decay
    if max_prompt_length: train_config['max_prompt_length'] = max_prompt_length
    if beta: train_config['beta'] = beta
    if kl_coef: train_config['kl_coef'] = kl_coef
    if response_length: train_config['response_length'] = response_length
    if temperature: train_config['temperature'] = temperature
    if loss_type: train_config['loss_type'] = loss_type
    if packing is not None: train_config['packing'] = packing
    
    # Eval/Save intervals
    if eval_interval: train_config['eval_interval'] = eval_interval
    if save_interval: train_config['save_interval'] = save_interval
    if eval_steps: train_config['eval_interval'] = eval_steps  # Legacy support
    if save_steps: train_config['save_interval'] = save_steps  # Legacy support
    
    if train_config:
        cli_config['train'] = train_config
    
    # Logging config
    logging_config = {}
    if output_dir: logging_config['output_dir'] = output_dir
    if run_name: logging_config['run_name'] = run_name
    if log_level: logging_config['log_level'] = log_level
    if loggers: logging_config['loggers'] = loggers.split(',')
    if wandb_project: 
        # Handle wandb config
        if 'wandb' not in logging_config:
            logging_config['wandb'] = {}
        logging_config['wandb']['project'] = wandb_project
    
    if logging_config:
        cli_config['logging'] = logging_config
    
    # Other top-level configs
    if training_type: cli_config['algo'] = training_type.lower()
    if chat_template: cli_config['chat_template'] = chat_template
    if cache_dir: 
        cli_config['caching'] = {'root': cache_dir, 'enabled': True}
    if distributed_backend:
        cli_config['distributed'] = {'backend': distributed_backend}
    
    # Merge configs (CLI overrides YAML)
    if cli_config:
        # Deep merge for nested dicts
        for key, value in cli_config.items():
            if key in final_config and isinstance(value, dict) and isinstance(final_config[key], dict):
                final_config[key].update(value)
            else:
                final_config[key] = value
    
    # Validate required fields
    if 'model' not in final_config or 'name_or_path' not in final_config['model']:
        typer.echo("Error: Model name is required (--model or in config)")
        raise typer.Exit(1)
    
    # Handle datasets format (ensure it's a list)
    if 'datasets' not in final_config and 'dataset' in final_config:
        final_config['datasets'] = [final_config.pop('dataset')]
    elif 'datasets' in final_config and not isinstance(final_config['datasets'], list):
        final_config['datasets'] = [final_config['datasets']]
    
    if not final_config.get('datasets'):
        typer.echo("Error: Dataset name is required (--dataset or in config)")
        raise typer.Exit(1)
    
    # Determine training type
    algo = final_config.get('algo', 'sft').lower()
    
    # Set up logging
    log_lvl = final_config.get('logging', {}).get('log_level', 'INFO')
    logging.basicConfig(level=getattr(logging, log_lvl.upper()))
    
    try:
        # Determine backend
        backend_str = backend or final_config.get('backend', 'auto')
        if backend_str.lower() == "auto":
            backend_enum = BackendType.TRL  # Default to TRL
        else:
            try:
                backend_enum = BackendType(backend_str.lower())
            except ValueError:
                typer.echo(f"Error: Invalid backend '{backend_str}'. Must be: auto, trl, unsloth")
                raise typer.Exit(1)
        
        # Extract config components for trainer creation
        model_cfg = final_config.get('model', {})
        dataset_cfg = final_config['datasets'][0]  # First dataset
        train_cfg = final_config.get('train', {})
        logging_cfg = final_config.get('logging', {})
        
        # Print training info
        typer.echo("\n" + "="*60)
        typer.echo(f"ALIGNTUNE - TRAINING CONFIGURATION")
        typer.echo("="*60)
        typer.echo(f"Algorithm: {algo.upper()}")
        typer.echo(f"Backend: {backend_enum.value.upper()}")
        typer.echo(f"Model: {model_cfg.get('name_or_path')}")
        typer.echo(f"Dataset: {dataset_cfg.get('name')}")
        typer.echo(f"Output: {logging_cfg.get('output_dir', './output')}")
        typer.echo("="*60 + "\n")
        
        # Create trainer based on algorithm
        if algo == 'sft':
            trainer = create_sft_trainer(
                model_name=model_cfg['name_or_path'],
                dataset_name=dataset_cfg['name'],
                backend=backend_enum,
                output_dir=logging_cfg.get('output_dir', './output'),
                num_epochs=train_cfg.get('epochs', 3),
                max_steps=train_cfg.get('max_steps'),
                batch_size=train_cfg.get('per_device_batch_size', 4),
                learning_rate=train_cfg.get('learning_rate', 2e-4),
                max_seq_length=model_cfg.get('max_seq_length', 512),
                max_samples=dataset_cfg.get('max_samples'),
                split=dataset_cfg.get('split', 'train'),
                percent=dataset_cfg.get('percent'),
                column_mapping=dataset_cfg.get('column_mapping', {}),
                task_type=dataset_cfg.get('task_type'),
                gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
                warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
                warmup_steps=train_cfg.get('warmup_steps', 0),
                weight_decay=train_cfg.get('weight_decay', 0.01),
                quantization=model_cfg.get('quantization', {}),
                use_peft=model_cfg.get('use_peft', False),
                lora_r=model_cfg.get('lora_r', 16),
                lora_alpha=model_cfg.get('lora_alpha', 32),
                lora_dropout=model_cfg.get('lora_dropout', 0.05),
                lora_target_modules=model_cfg.get('lora_target_modules'),
                packing=train_cfg.get('packing', False),
                chat_template=final_config.get('chat_template', 'auto'),
                run_name=logging_cfg.get('run_name'),
                loggers=logging_cfg.get('loggers', ['tensorboard']),
                dataset_num_proc=dataset_cfg.get('dataset_num_proc'),
            )
        else:
            # RL training
            trainer = create_rl_trainer(
                model_name=model_cfg['name_or_path'],
                dataset_name=dataset_cfg['name'],
                algorithm=algo,
                backend=backend_enum,
                output_dir=logging_cfg.get('output_dir', './output'),
                num_epochs=train_cfg.get('epochs', 3),
                max_steps=train_cfg.get('max_steps'),
                batch_size=train_cfg.get('per_device_batch_size', 4),
                learning_rate=train_cfg.get('learning_rate', 2e-4),
                max_seq_length=model_cfg.get('max_seq_length', 512),
                max_samples=dataset_cfg.get('max_samples'),
                split=dataset_cfg.get('split', 'train'),
                percent=dataset_cfg.get('percent'),
                column_mapping=dataset_cfg.get('column_mapping', {}),
                field_mappings=dataset_cfg.get('field_mappings', {}),
                gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
                warmup_ratio=train_cfg.get('warmup_ratio', 0.1),
                warmup_steps=train_cfg.get('warmup_steps', 0),
                weight_decay=train_cfg.get('weight_decay', 0.01),
                quantization=model_cfg.get('quantization', {}),
                use_peft=model_cfg.get('use_peft', True),
                lora_r=model_cfg.get('lora_r', 16),
                lora_alpha=model_cfg.get('lora_alpha', 32),
                lora_dropout=model_cfg.get('lora_dropout', 0.05),
                lora_target_modules=model_cfg.get('lora_target_modules'),
                reward_model_name=model_cfg.get('reward_model_name'),
                reward_model_path=model_cfg.get('reward_model_path'),
                beta=train_cfg.get('beta'),
                kl_coef=train_cfg.get('kl_coef', 0.1),
                response_length=train_cfg.get('response_length', 128),
                temperature=train_cfg.get('temperature', 0.7),
                max_prompt_length=train_cfg.get('max_prompt_length', 512),
                loss_type=train_cfg.get('loss_type'),
                chat_template=final_config.get('chat_template', 'auto'),
                run_name=logging_cfg.get('run_name'),
                loggers=logging_cfg.get('loggers', ['tensorboard']),
                dataset_num_proc=dataset_cfg.get('dataset_num_proc'),
                cache_dir=final_config.get('caching', {}).get('root', 'cache'),
            )
        
        # Start training
        typer.echo("Starting training...")
        results = trainer.train()
        
        typer.echo("\n" + "="*60)
        typer.echo("TRAINING COMPLETED SUCCESSFULLY!")
        typer.echo("="*60)
        typer.echo(f"Training time: {results.get('training_time', 'N/A'):.2f} seconds")
        if 'final_loss' in results:
            typer.echo(f"Final loss: {results['final_loss']:.4f}")
        typer.echo(f"Model saved to: {results.get('model_path', logging_cfg.get('output_dir'))}")
        typer.echo("="*60 + "\n")
        
    except Exception as e:
        typer.echo(f"\n{'='*60}")
        typer.echo("ERROR DURING TRAINING")
        typer.echo("="*60)
        typer.echo(f"{e}")
        typer.echo("="*60 + "\n")
        logger.exception("Training failed")
        raise typer.Exit(1)


@app.command()
def validate_config(
    config: str = typer.Argument(..., help="Path to YAML configuration file")
):
    """Validate a YAML configuration file without running training."""
    if not Path(config).exists():
        typer.echo(f"Error: Config file not found: {config}")
        raise typer.Exit(1)
    
    try:
        cfg = load_yaml_config(config)
        
        typer.echo("✓ Configuration file is valid YAML")
        typer.echo("\nConfiguration summary:")
        typer.echo(f"  Algorithm: {cfg.get('algo', 'not specified')}")
        typer.echo(f"  Model: {cfg.get('model', {}).get('name_or_path', 'not specified')}")
        
        datasets = cfg.get('datasets', cfg.get('dataset'))
        if isinstance(datasets, list):
            typer.echo(f"  Datasets: {len(datasets)} dataset(s)")
            for i, ds in enumerate(datasets):
                typer.echo(f"    {i+1}. {ds.get('name', 'unnamed')}")
        elif isinstance(datasets, dict):
            typer.echo(f"  Dataset: {datasets.get('name', 'unnamed')}")
        else:
            typer.echo("  Dataset: not specified")
        
        typer.echo(f"  Output: {cfg.get('logging', {}).get('output_dir', 'not specified')}")
        
    except Exception as e:
        typer.echo(f"Error validating config: {e}")
        raise typer.Exit(1)

# @app.command()
# def train(
#     # Model configuration
#     model_name: str = typer.Option(
#         ..., "--model", "-m", help="Model name or path (e.g., 'microsoft/DialoGPT-medium')"
#     ),

#     # Training type
#     training_type: str = typer.Option(
#         "sft", "--type", "-t", help="Training type: sft, dpo, ppo, grpo, gspo"
#     ),

#     # Backend selection
#     backend: str = typer.Option(
#         "auto", "--backend", "-b", help="Backend: auto, trl, unsloth"
#     ),

#     # Dataset configuration
#     dataset_name: str = typer.Option(
#         ..., "--dataset", "-d", help="Dataset name (e.g., 'tatsu-lab/alpaca')"
#     ),
#     dataset_split: str = typer.Option(
#         "train", "--split", help="Dataset split to use"
#     ),
#     max_samples: Optional[int] = typer.Option(
#         None, "--max-samples", help="Maximum number of samples to use"
#     ),

#     # Training parameters
#     num_epochs: int = typer.Option(
#         3, "--epochs", "-e", help="Number of training epochs"
#     ),
#     batch_size: int = typer.Option(
#         4, "--batch-size", help="Per-device batch size"
#     ),
#     learning_rate: float = typer.Option(
#         2e-4, "--lr", help="Learning rate"
#     ),
#     max_seq_length: int = typer.Option(
#         512, "--max-length", help="Maximum sequence length"
#     ),

#     # Output configuration
#     output_dir: str = typer.Option(
#         "./output", "--output", "-o", help="Output directory for trained model"
#     ),

#     # Logging
#     log_level: str = typer.Option(
#         "INFO", "--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR"
#     ),
#     wandb_project: Optional[str] = typer.Option(
#         None, "--wandb-project", help="Weights & Biases project name"
#     ),

#     # Advanced options
#     gradient_accumulation_steps: int = typer.Option(
#         1, "--grad-accum", help="Gradient accumulation steps"
#     ),
#     warmup_ratio: float = typer.Option(
#         0.1, "--warmup", help="Warmup ratio"
#     ),
#     save_steps: int = typer.Option(
#         500, "--save-steps", help="Steps between model saves"
#     ),
#     eval_steps: int = typer.Option(
#         500, "--eval-steps", help="Steps between evaluations"
#     ),

#     # Quantization
#     load_in_4bit: bool = typer.Option(
#         False, "--4bit", help="Load model in 4-bit quantization"
#     ),
#     load_in_8bit: bool = typer.Option(
#         False, "--8bit", help="Load model in 8-bit quantization"
#     ),
# ):
#     """Train a model using FinetuneHub with the specified configuration."""

#     # Set up logging
#     logging.basicConfig(level=getattr(logging, log_level.upper()))

#     try:
#         # Validate training type
#         if training_type.lower() == "sft":
#             training_type_enum = TrainingType.SFT
#         elif training_type.lower() in ["dpo", "ppo", "grpo", "gspo"]:
#             training_type_enum = TrainingType.RL
#         else:
#             typer.echo(f"Error: Invalid training type '{training_type}'. Must be one of: sft, dpo, ppo, grpo, gspo")
#             raise typer.Exit(1)

#         # Validate backend
#         if backend.lower() == "auto":
#             backend_enum = BackendType.UNSLOTH  # Default to Unsloth for speed
#         elif backend.lower() == "trl":
#             backend_enum = BackendType.TRL
#         elif backend.lower() == "unsloth":
#             backend_enum = BackendType.UNSLOTH
#         else:
#             typer.echo(f"Error: Invalid backend '{backend}'. Must be one of: auto, trl, unsloth")
#             raise typer.Exit(1)

#         # Create trainer based on training type
#         if training_type_enum == TrainingType.SFT:
#             trainer = create_sft_trainer(
#                 model_name=model_name,
#                 dataset_name=dataset_name,
#                 backend=backend_enum,
#                 output_dir=output_dir,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 learning_rate=learning_rate,
#                 max_seq_length=max_seq_length,
#                 max_samples=max_samples,
#                 gradient_accumulation_steps=gradient_accumulation_steps,
#                 warmup_ratio=warmup_ratio,
#                 save_steps=save_steps,
#                 eval_steps=eval_steps,
#                 load_in_4bit=load_in_4bit,
#                 load_in_8bit=load_in_8bit,
#                 wandb_project=wandb_project,
#             )
#         else:
#             # RL training
#             algorithm_map = {
#                 "dpo": RLAlgorithm.DPO,
#                 "ppo": RLAlgorithm.PPO,
#                 "grpo": RLAlgorithm.GRPO,
#                 "gspo": RLAlgorithm.GSPO,
#             }

#             algorithm = algorithm_map.get(training_type.lower())
#             if not algorithm:
#                 typer.echo(f"Error: Invalid RL algorithm '{training_type}'")
#                 raise typer.Exit(1)

#             trainer = create_rl_trainer(
#                 model_name=model_name,
#                 dataset_name=dataset_name,
#                 # algorithm=algorithm,
#                 algorithm = training_type,
#                 backend=backend_enum,
#                 output_dir=output_dir,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 learning_rate=learning_rate,
#                 max_seq_length=max_seq_length,
#                 max_samples=max_samples,
#                 gradient_accumulation_steps=gradient_accumulation_steps,
#                 warmup_ratio=warmup_ratio,
#                 save_steps=save_steps,
#                 eval_steps=eval_steps,
#                 load_in_4bit=load_in_4bit,
#                 load_in_8bit=load_in_8bit,
#                 wandb_project=wandb_project,
#             )

#         # Start training
#         typer.echo(f"Starting {training_type.upper()} training with {backend.upper()} backend...")
#         typer.echo(f"Model: {model_name}")
#         typer.echo(f"Dataset: {dataset_name}")
#         typer.echo(f"Output: {output_dir}")

#         results = trainer.train()

#         typer.echo("Training completed successfully!")
#         typer.echo(f"Training time: {results.get('training_time', 'N/A'):.2f} seconds")
#         typer.echo(f"Final loss: {results.get('final_loss', 'N/A'):.4f}")
#         typer.echo(f"Model saved to: {results.get('model_path', output_dir)}")

#     except Exception as e:
#         typer.echo(f"Error during training: {e}")
#         logger.exception("Training failed")
#         raise typer.Exit(1)


@app.command()
def list_backends_cmd():
    """List all available backends and their capabilities."""
    try:
        backends = list_backends()
        
        typer.echo("Available FinetuneHub Backends:")
        typer.echo("=" * 50)
        
        for backend_name, capabilities in backends.items():
            typer.echo(f"\n{backend_name.upper()}:")
            if isinstance(capabilities, list):
                # SFT backends are a list
                typer.echo(f"  Available: {', '.join(capabilities)}")
            elif isinstance(capabilities, dict):
                # RL backends are a dict with algorithms
                for training_type, algorithms in capabilities.items():
                    if algorithms:
                        typer.echo(f"  {training_type.upper()}: {', '.join(algorithms)}")
                    else:
                        typer.echo(f"  {training_type.upper()}: Available")
        
        typer.echo("\n" + "=" * 50)
        
    except Exception as e:
        typer.echo(f"Error listing backends: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    model_name: str = typer.Option(
        ..., "--model", "-m", help="Model name or path to validate"
    ),
    backend: str = typer.Option(
        "auto", "--backend", "-b", help="Backend to validate: auto, trl, unsloth"
    ),
):
    """Validate that a model and backend are compatible."""
    
    try:
        typer.echo(f"Validating model: {model_name}")
        typer.echo(f"Backend: {backend}")
        
        # Check model availability
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            typer.echo("✓ Model tokenizer loaded successfully")
        except Exception as e:
            typer.echo(f"✗ Failed to load model tokenizer: {e}")
            raise typer.Exit(1)
        
        # Check backend availability
        if backend.lower() == "auto":
            backend_enum = BackendType.UNSLOTH
        elif backend.lower() == "trl":
            backend_enum = BackendType.TRL
        elif backend.lower() == "unsloth":
            backend_enum = BackendType.UNSLOTH
        else:
            typer.echo(f"Error: Invalid backend '{backend}'. Must be one of: auto, trl, unsloth")
            raise typer.Exit(1)
        
        # Test backend availability
        factory = BackendFactory()
        if factory.is_backend_available(backend_enum):
            typer.echo(f"✓ Backend {backend.upper()} is available")
        else:
            typer.echo(f"✗ Backend {backend.upper()} is not available")
            raise typer.Exit(1)
        
        typer.echo("✓ Validation completed successfully!")
        
    except Exception as e:
        typer.echo(f"Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def info(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostics")
):
    """Show FinetuneHub information and system status."""
    
    try:
        from .. import (
            __version__,
            __author__,
            get_available_trainers,
            check_dependencies,
            UNSLOTH_ERROR_INFO,
        )
        
        typer.echo("FinetuneHub Information")
        typer.echo("=" * 30)
        typer.echo(f"Version: {__version__}")
        typer.echo(f"Author: {__author__}")
        
        typer.echo("\nAvailable Trainers:")
        trainers = get_available_trainers()
        for trainer, available in trainers.items():
            status = "✓" if available else "✗"
            typer.echo(f"  {status} {trainer}")
        
        typer.echo("\nDependencies:")
        deps = check_dependencies()
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            typer.echo(f"  {status} {dep}")
        
        typer.echo("\nBackend Status:")
        backends = list_backends()
        for backend_name, capabilities in backends.items():
            typer.echo(f"  ✓ {backend_name.upper()}")
        
        # Show detailed Unsloth diagnostics if verbose or if Unsloth is unavailable
        if verbose or (UNSLOTH_ERROR_INFO is not None):
            typer.echo("\nUnsloth Diagnostics:")
            if UNSLOTH_ERROR_INFO:
                typer.echo(f"  Status: ✗ Not available")
                typer.echo(f"  Error Type: {UNSLOTH_ERROR_INFO['error_type']}")
                typer.echo(f"  Error: {UNSLOTH_ERROR_INFO['error']}")
                typer.echo(f"  Environment:")
                env = UNSLOTH_ERROR_INFO['environment']
                typer.echo(f"    PyTorch: {env.get('pytorch_version', 'unknown')}")
                typer.echo(f"    CUDA: {env.get('cuda_version', 'unknown')}")
                typer.echo(f"  Suggestions:")
                for suggestion in UNSLOTH_ERROR_INFO['suggestion']:
                    typer.echo(f"    - {suggestion}")
            else:
                typer.echo(f"  Status: ✓ Available")
        
    except Exception as e:
        typer.echo(f"Error getting info: {e}")
        raise typer.Exit(1)


@app.command()
def diagnose():
    """Run comprehensive environment diagnostics for unsloth and dependencies."""
    try:
        from ..utils.diagnostics import run_comprehensive_diagnostics
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        console.print("[green]🔍 Running comprehensive environment diagnostics...[/green]")

        diagnostics = run_comprehensive_diagnostics()

        # System info
        console.print("\n[bold blue]System Information[/bold blue]")
        sys_info = diagnostics["system_info"]
        console.print(f"  CPU Cores: {sys_info['cpu_count']}")
        console.print(".1f")
        console.print(f"  Platform: {sys_info['platform']}")

        # GPU info
        console.print("\n[bold blue]GPU Information[/bold blue]")
        gpu_info = diagnostics["gpu_info"]
        if "error" not in gpu_info:
            console.print(f"  GPU Count: {gpu_info['gpu_count']}")
            for i, gpu in enumerate(gpu_info['gpus']):
                console.print(f"  GPU {i}: {gpu['name']} ({gpu['memory_total_gb']:.1f}GB)")
        else:
            console.print(f"  [red]GPU detection failed: {gpu_info['error']}[/red]")

        # Library versions
        console.print("\n[bold blue]Library Versions[/bold blue]")
        libs = diagnostics["library_versions"]
        for lib, version in libs.items():
            status = "[green]✓[/green]" if version != "not installed" else "[red]✗[/red]"
            console.print(f"  {status} {lib}: {version}")

        # Compatibility
        console.print("\n[bold blue]Compatibility Checks[/bold blue]")
        compat = diagnostics["compatibility_checks"]
        for check, available in compat.items():
            status = "[green]✓[/green]" if available else "[red]✗[/red]"
            console.print(f"  {status} {check.replace('_', ' ').title()}")

    except Exception as e:
        typer.echo(f"Error running diagnostics: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    config: str = typer.Argument(..., help="Path to configuration YAML file"),
    config_type: str = typer.Option("auto", "--type", "-t", help="Configuration type: sft, rl, or auto")
):
    """
    Validate a training configuration file.

    Examples:
        finetunehub validate config.yaml
        finetunehub validate config.yaml --type sft
    """
    from ..utils.validation import validate_config, ConfigValidator
    from ..utils.diagnostics import generate_training_report
    from rich.console import Console
    from pathlib import Path

    console = Console()

    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]❌ Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)

    try:
        # Load and validate config
        if config_type == "sft":
            from ..core.sft.config_loader import SFTConfigLoader
            loaded_config = SFTConfigLoader.load_from_yaml(config_path)
        else:
            from ..core.rl.config_loader import ConfigLoader
            loaded_config = ConfigLoader.load_from_yaml(config_path)

        console.print(f"[green]✅ Configuration loaded from {config_path}[/green]")

        # Run validation
        is_valid, errors = validate_config(loaded_config, config_type)

        if is_valid:
            console.print("[green]✅ Configuration validation passed![/green]")
        else:
            console.print("[red]❌ Configuration validation failed:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            raise typer.Exit(1)

        # Show memory estimate
        console.print("\n[bold blue]Memory Estimate[/bold blue]")
        memory_info = ConfigValidator.estimate_memory_usage(loaded_config)
        console.print(f"  Estimated Memory: {memory_info:.1f} GB")
        console.print(f"  Recommended GPUs: {max(1, int(memory_info / 24))}")
        console.print(f"  Memory efficiency: {'Good' if memory_info < 16 else 'High' if memory_info < 32 else 'Very High'}")
        console.print(f"  Training time estimate: {'Fast' if memory_info < 8 else 'Moderate' if memory_info < 16 else 'Slow'}")
        # Show environment compatibility
        console.print("\n[bold blue]Environment Compatibility[/bold blue]")
        env_compat = ConfigValidator.check_environment_compatibility(loaded_config)
        for check, available in env_compat["compatibility_checks"].items():
            status = "[green]✓[/green]" if available else "[red]✗[/red]"
            console.print(f"  {status} {check.replace('_', ' ').title()}")

        if env_compat["issues"]:
            console.print("\n[red]Compatibility Issues:[/red]")
            for issue in env_compat["issues"]:
                console.print(f"  • {issue}")

    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_name: str = typer.Option(
        ..., "--model", "-m", help="Model name or path (e.g., 'microsoft/DialoGPT-medium')"
    ),
    tasks: List[str] = typer.Option(
        ["hellaswag"], "--task", "-t", help="Evaluation task(s) to run"
    ),
    batch_size: int = typer.Option(1, "--batch-size", help="Evaluation batch size"),
    max_samples: int = typer.Option(0, "--max-samples", help="Limit samples per task (0=all)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """Run evaluation for any HF model independently of training.

    Uses the library's lm-eval integration when available.
    """
    import logging as _logging
    _logging.basicConfig(level=getattr(_logging, log_level.upper()))

    if not EVAL_AVAILABLE or LMEvalRunner is None:
        typer.echo("Evaluation components not available. Install lm-eval and extras.")
        raise typer.Exit(1)

    try:
        cfg = LMEvalConfig(
            model_name=model_name,
            batch_size=batch_size,
            limit=None if max_samples <= 0 else max_samples,
        )
        runner = LMEvalRunner(cfg)
        lm_tasks = [get_lm_eval_task(t) for t in tasks]
        results = runner.evaluate_tasks(lm_tasks)
        typer.echo("Evaluation completed.")
        typer.echo(str([r.to_dict() for r in results]))
    except Exception as e:
        typer.echo(f"Evaluation failed: {e}")
        raise typer.Exit(1)


# Add recipes subcommand if available
if RECIPES_AVAILABLE and recipes_app:
    app.add_typer(recipes_app, name="recipes", help="Manage and run training recipes")

# Add validate subcommand if available
if VALIDATE_AVAILABLE and validate_app:
    app.add_typer(validate_app, name="validate", help="Validate configurations and check system compatibility")

# Add diagnose subcommand if available
if DIAGNOSE_AVAILABLE and diagnose_app:
    app.add_typer(diagnose_app, name="diagnose", help="Diagnose training issues and monitor system health")

if __name__ == "__main__":
    app()
