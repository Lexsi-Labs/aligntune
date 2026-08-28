"""
Shared SFT Configuration Builder.

This utility centralizes the construction of Hugging Face TrainingArguments / SFTConfig
for all SFT backends (TRL, Unsloth, etc.). It resolves all intervals, strategies,
and steps dynamically to prevent configuration fragmentation.
"""

import logging
from typing import Optional, Any
from trl import SFTConfig
from ..precision_handler import PrecisionHandler
from ...utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)

def build_sft_config(
    config: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    output_dir: str = "./output",
    **kwargs
) -> SFTConfig:
    """
    Build a unified SFTConfig for HF Trainer.
    
    Args:
        config: The main SFT configuration object.
        train_dataset: The processed training dataset.
        eval_dataset: The optional evaluation dataset.
        output_dir: Output directory for checkpoints.
        **kwargs: Extra arguments to override or append to SFTConfig.
        
    Returns:
        SFTConfig: Fully populated configuration for SFTTrainer.
    """
    train_cfg = getattr(config, 'train', None)
    log_cfg = getattr(config, 'logging', None)
    
    # 1. Basic Training Parameters
    epochs = getattr(train_cfg, 'epochs', 3) if train_cfg else 3
    num_epochs = getattr(config, 'num_epochs', None) or epochs
    batch_size = getattr(config, 'batch_size', None) or (getattr(train_cfg, 'per_device_batch_size', 2) if train_cfg else 2)
    grad_accum = getattr(train_cfg, 'gradient_accumulation_steps', 1) if train_cfg else 1
    lr = getattr(train_cfg, 'learning_rate', 2e-4) if train_cfg else 2e-4
    
    # 2. Dynamic Max Steps Calculation
    max_steps = getattr(train_cfg, 'max_steps', None) if train_cfg else None
    if max_steps is None:
        dataset_size = len(train_dataset) if train_dataset else 1000
        batch_size_calc = batch_size or 2
        grad_accum_calc = grad_accum or 1
        max_steps = (dataset_size * num_epochs) // (batch_size_calc * grad_accum_calc)
        max_steps = max(max_steps, 10)  # Minimum 10 steps
        logger.info(f"Calculated max_steps={max_steps} from {num_epochs} epochs, {dataset_size} samples")
        
    # 3. Dynamic Evaluation Strategy
    has_eval_ds = eval_dataset is not None
    user_eval_strategy = getattr(log_cfg, 'eval_strategy', None) if log_cfg else None
    
    if not has_eval_ds:
        eval_strategy = "no"
    elif user_eval_strategy:
        eval_strategy = user_eval_strategy
    else:
        eval_strategy = "steps"
        
    eval_steps = getattr(train_cfg, 'eval_interval', 500) if eval_strategy == "steps" and train_cfg else None
    
    # 4. Intervals & Warmup
    save_steps = getattr(train_cfg, 'save_interval', 500) if train_cfg else 500
    logging_steps = getattr(log_cfg, 'log_interval', 10) if log_cfg else 10
    warmup_ratio = getattr(train_cfg, 'warmup_ratio', 0.1) if train_cfg else 0.1
    
    # 5. Precision Handling
    precision = PrecisionHandler.get_precision_from_config(config, default="auto")
    precision_args = PrecisionHandler.get_training_args_precision(precision)
    
    # 6. Logging Providers
    # `logging.report_to` carries the caller's raw `report_to=` kwarg (default
    # sentinel "none"); `logging.loggers` is aligntune's own separate default
    # (["tensorboard"]). Prefer the caller's explicit value when they set one.
    explicit_report_to = getattr(log_cfg, 'report_to', None) if log_cfg else None
    if explicit_report_to and explicit_report_to != "none":
        report_to = explicit_report_to
    elif log_cfg and getattr(log_cfg, 'loggers', None):
        report_to = log_cfg.loggers
    else:
        report_to = []

    # Model's own max_seq_length is the source of truth for TRL's max_length
    # (train.max_length is a separate, RL-only field -- it doesn't exist on
    # SFTTrainingConfig, hence reading it from config.model here instead).
    model_cfg = getattr(config, 'model', None)
    max_length = getattr(model_cfg, 'max_seq_length', 1024) if model_cfg else 1024
    # Build SFTConfig
    config_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": getattr(
            train_cfg, 'per_device_eval_batch_size', batch_size
        ) if train_cfg else batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "max_steps": max_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": getattr(train_cfg, 'warmup_steps', 0) if train_cfg else 0,
        "lr_scheduler_type": getattr(train_cfg, 'lr_scheduler', 'cosine') if train_cfg else "cosine",
        "weight_decay": getattr(train_cfg, 'weight_decay', 0.01) if train_cfg else 0.01,
        "max_grad_norm": getattr(train_cfg, 'max_grad_norm', 1.0) if train_cfg else 1.0,
        "dataloader_num_workers": getattr(train_cfg, 'dataloader_num_workers', 0) if train_cfg else 0,
        "remove_unused_columns": getattr(train_cfg, 'remove_unused_columns', False) if train_cfg else False,
        "report_to": report_to,
        "run_name": getattr(log_cfg, 'run_name', None) if log_cfg else None,
    }
    config_dict["max_length"] = max_length
    
    # (LoRA+ config is handled separately via kwargs when needed)
    
    # Add precision args
    config_dict.update(precision_args)
    
    # 8. TRL Native Multi-turn / Completion Only Loss Flags
    dataset_cfg = getattr(config, 'dataset', None)
    
    # Extract assistant_only_loss or completion_only_loss from config
    assistant_only_loss = getattr(train_cfg, 'assistant_only_loss', False) if train_cfg else False
    if assistant_only_loss:
        config_dict["assistant_only_loss"] = True
        logger.info("Enabled assistant_only_loss in SFTConfig")
        
    completion_only_loss = getattr(train_cfg, 'completion_only_loss', None) if train_cfg else None
    if completion_only_loss is not None:
        config_dict["completion_only_loss"] = completion_only_loss

    # Threaded explicitly (rather than left for the extractor backfill below)
    # because trl.SFTConfig's own __post_init__ resolves its None default into
    # a concrete value ('chunked_nll') before this function ever sees it --
    # which would make a later, unset loss_type look "already set" to anyone
    # comparing against the class's raw declared default.
    loss_type = getattr(train_cfg, 'loss_type', None) if train_cfg else None
    if loss_type is not None:
        config_dict["loss_type"] = loss_type


    dataset_text_field = getattr(dataset_cfg, 'text_column', 'text') if dataset_cfg else 'text'
    config_dict["dataset_text_field"] = dataset_text_field
    
    dataset_kwargs = getattr(dataset_cfg, 'dataset_kwargs', {}) if dataset_cfg else {}
    if dataset_kwargs:
        config_dict["dataset_kwargs"] = dataset_kwargs

    # 9. Padding-free batching
    # Explicitly set padding_free (defaults to False) so it doesn't fall back to
    # None, which Unsloth's SFTTrainer patch treats as "auto-enable". Auto-enabling
    # padding_free while max_length stays at TRL's default (non-None) and packing is
    # off raises `ValueError: When padding_free=True without packing, max_length is
    # not enforced.` Setting it explicitly opts out of that auto-detection.
    config_dict["padding_free"] = getattr(train_cfg, 'padding_free', False) if train_cfg else False

    # Apply kwargs overrides (e.g. optimizer, dataloader_pin_memory, etc)
    config_dict.update(kwargs)

    sft_config = SFTConfig(**config_dict)

    # Backfill anything else the caller passed to create_sft_trainer() that
    # isn't already explicitly threaded above (packing, seed, report_to,
    # save_strategy, load_best_model_at_end, group_by_length, loss_type, ...)
    # instead of hand-listing every trl.SFTConfig field here. Values already
    # set above (e.g. the dynamically computed max_steps, eval_strategy) are
    # protected since they differ from SFTConfig's own class defaults.
    missing = extract_extra_and_missing_params(
        backend_config=sft_config, config=config, algorithm='sft'
    )
    for key, value in missing.items():
        setattr(sft_config, key, value)

    return sft_config
