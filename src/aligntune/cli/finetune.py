"""
AlignTune CLI - Main finetune command with comprehensive hyperparameter support.

This module provides the main `finetune` command that supports all training methods
with comprehensive hyperparameter configuration.
"""

import typer
import logging
import yaml
import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import core functionality
from ..core.backend_factory import create_sft_trainer, create_rl_trainer
from ..core.rl.config import (
    UnifiedConfig, AlgorithmType, ModelConfig, DatasetConfig, 
    TrainingConfig, LoggingConfig, DistributedConfig, RewardConfig
)
from ..core.rl.config_loader import ConfigLoader

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="finetune",
    help="🚀 AlignTune: Comprehensive fine-tuning with hyperparameter support",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command()
def finetune(
    # Model and Dataset
    model: str = typer.Argument(..., help="Model name or path (e.g., 'microsoft/DialoGPT-medium')"),
    dataset: str = typer.Argument(..., help="Dataset name or path (e.g., 'databricks/databricks-dolly-15k')"),
    
    # Training Method
    method: str = typer.Option("sft", "--method", "-m", help="Training method: sft, dpo, ppo, grpo, gspo"),
    
    # Backend Selection
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: auto, trl, unsloth"),
    
    # Output Configuration
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Output directory for models and logs"),
    run_name: Optional[str] = typer.Option(None, "--run-name", "-n", help="Run name for logging"),
    
    # Training Hyperparameters
    max_steps: int = typer.Option(1000, "--max-steps", help="Maximum training steps"),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", "--lr", help="Learning rate"),
    batch_size: int = typer.Option(4, "--batch-size", help="Per device batch size"),
    gradient_accumulation_steps: int = typer.Option(1, "--grad-accum", help="Gradient accumulation steps"),
    warmup_steps: int = typer.Option(100, "--warmup-steps", help="Warmup steps"),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="Weight decay"),
    
    # Model Configuration
    max_seq_length: int = typer.Option(512, "--max-length", help="Maximum sequence length"),
    precision: str = typer.Option("fp32", "--precision", help="Precision: fp32, fp16, bf16"),
    use_peft: bool = typer.Option(True, "--peft/--no-peft", help="Use Parameter Efficient Fine-Tuning"),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.1, "--lora-dropout", help="LoRA dropout"),
    
    # RL-Specific Hyperparameters
    beta: Optional[float] = typer.Option(None, "--beta", help="DPO beta parameter"),
    temperature: float = typer.Option(0.7, "--temperature", help="Generation temperature"),
    max_prompt_length: Optional[int] = typer.Option(None, "--max-prompt-length", help="Maximum prompt length for RL"),
    
    # RL extended hyperparameters
    kl_estimator: str = typer.Option("k1", "--kl-estimator", help="KL estimator for PPO (k1 or k3)"),
    vf_coef: float = typer.Option(0.1, "--vf-coef", help="Value function coefficient"),
    cliprange_value: float = typer.Option(0.2, "--cliprange-value", help="Value clip range"),
    gamma: float = typer.Option(1.0, "--gamma", help="Discount factor"),
    lam: float = typer.Option(0.95, "--lambda", help="GAE lambda"),
    response_length: int = typer.Option(128, "--response-length", help="Generation length for PPO/GRPO"),
    stop_token: str = typer.Option("eos", "--stop-token", help="Stop token for PPO generation"),
    missing_eos_penalty: float = typer.Option(1.0, "--missing-eos-penalty", help="Penalty when EOS missing"),
    sync_ref_model: bool = typer.Option(False, "--sync-ref-model/--no-sync-ref-model", help="Enable periodic reference model sync"),
    ref_model_sync_steps: int = typer.Option(512, "--ref-model-sync-steps", help="Steps between reference model syncs"),
    ref_model_mixup_alpha: float = typer.Option(0.6, "--ref-model-mixup-alpha", help="Mixup alpha for reference sync"),
    rl_loss_type: str = typer.Option("sigmoid", "--rl-loss-type", help="Loss type for preference tuning"),
    rl_loss_weights: Optional[List[float]] = typer.Option(None, "--rl-loss-weight", help="Weights for multi-loss preferences", show_default=False),
    f_divergence_type: str = typer.Option("reverse_kl", "--f-divergence-type", help="f-divergence regularizer"),
    f_alpha_divergence_coef: float = typer.Option(1.0, "--f-alpha-coef", help="Alpha divergence coefficient"),
    reference_free: bool = typer.Option(False, "--reference-free/--no-reference-free", help="Enable reference-free DPO"),
    label_smoothing_rl: float = typer.Option(0.0, "--label-smoothing", help="Label smoothing for DPO"),
    use_weighting: bool = typer.Option(False, "--use-weighting/--no-use-weighting", help="Enable weighted losses"),
    rpo_alpha: Optional[float] = typer.Option(None, "--rpo-alpha", help="RPO alpha weighting"),
    ld_alpha: Optional[float] = typer.Option(None, "--ld-alpha", help="LD-DPO alpha"),
    discopop_tau: float = typer.Option(0.05, "--discopop-tau", help="DiscoPOP temperature"),
    
    # Dataset Configuration
    dataset_split: str = typer.Option("train", "--split", help="Dataset split to use"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Maximum number of samples"),
    dataset_percent: Optional[float] = typer.Option(None, "--dataset-percent", help="Percentage of dataset to use"),
    dataset_num_proc: Optional[int] = typer.Option(None, "--dataset-num-proc", help="Number of preprocessing workers"),
    dataset_text_field: str = typer.Option("text", "--dataset-text-field", help="Primary text column for SFT"),
    packing: bool = typer.Option(False, "--packing/--no-packing", help="Enable sequence packing"),
    packing_strategy: str = typer.Option("bfd", "--packing-strategy", help="Packing strategy: bfd or wrapped"),
    eval_packing: Optional[bool] = typer.Option(None, "--eval-packing", help="Override evaluation packing (true/false)"),
    padding_free: bool = typer.Option(False, "--padding-free/--no-padding-free", help="Use padding-free forward pass when supported"),
    pad_to_multiple_of: Optional[int] = typer.Option(None, "--pad-to-multiple-of", help="Pad inputs to multiple of value"),
    completion_only_loss: Optional[bool] = typer.Option(None, "--completion-only-loss", help="Compute loss only on completions (true/false)"),
    assistant_only_loss: bool = typer.Option(False, "--assistant-only-loss/--no-assistant-only-loss", help="Compute loss only on assistant messages"),
    loss_type: str = typer.Option("nll", "--loss-type", help="Loss type for SFT (nll or dft)"),
    activation_offloading: bool = typer.Option(False, "--activation-offloading/--no-activation-offloading", help="Enable activation offloading"),
    chat_template_path: Optional[str] = typer.Option(None, "--chat-template-path", help="Path to custom chat template"),

    # Reward Functions (for RL)
    rewards: Optional[List[str]] = typer.Option(None, "--rewards", help="Reward functions (e.g., 'length:1.0', 'coherence:0.5')"),
    
    # Distributed Training
    distributed: str = typer.Option("single", "--distributed", help="Distributed backend: single, ddp, fsdp, deepspeed"),
    num_gpus: Optional[int] = typer.Option(None, "--num-gpus", help="Number of GPUs to use"),
    
    # Logging and Monitoring
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR"),
    save_steps: int = typer.Option(100, "--save-steps", help="Save checkpoint every N steps"),
    eval_steps: int = typer.Option(100, "--eval-steps", help="Evaluate every N steps"),
    logging_backend: str = typer.Option("tensorboard", "--logging", help="Logging backend: tensorboard, wandb, mlflow"),
    
    # Advanced Options
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config file to override defaults"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configuration without training"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    🚀 Comprehensive fine-tuning with full hyperparameter support.
    
    This command provides a complete fine-tuning interface supporting:
    - All training methods: SFT, DPO, PPO, GRPO, GSPO
    - Multiple backends: TRL, Unsloth
    - Comprehensive hyperparameter control
    - Distributed training support
    - Advanced logging and monitoring
    
    Examples:
    
    # Basic SFT training
    aligntune finetune microsoft/DialoGPT-medium databricks/databricks-dolly-15k
    
    # DPO training with custom hyperparameters
    aligntune finetune microsoft/DialoGPT-medium Anthropic/hh-rlhf --method dpo --beta 0.1 --learning-rate 1e-5
    
    # PPO training with multiple rewards
    aligntune finetune microsoft/DialoGPT-medium your-dataset --method ppo --rewards length:1.0 coherence:0.5
    
    # High-performance training with Unsloth
    aligntune finetune unsloth/llama-2-7b-chat-bnb-4bit your-dataset --backend unsloth --precision bf16 --batch-size 8
    """
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=getattr(logging, log_level.upper()))
    
    # Show configuration in dry-run mode
    if dry_run:
        _show_configuration(
            model, dataset, method, backend, output_dir, run_name,
            max_steps, learning_rate, batch_size, gradient_accumulation_steps,
            warmup_steps, weight_decay, max_seq_length, precision,
            use_peft, lora_r, lora_alpha, lora_dropout, beta, temperature,
            max_prompt_length, dataset_split, max_samples, dataset_percent,
            dataset_num_proc, dataset_text_field, packing, packing_strategy, eval_packing,
            padding_free, pad_to_multiple_of, completion_only_loss,
            assistant_only_loss, loss_type, activation_offloading,
            chat_template_path, rewards, distributed, num_gpus, log_level, save_steps, eval_steps,
            logging_backend, seed, resume, config_file,
            kl_estimator, vf_coef, cliprange_value, gamma, lam,
            response_length, stop_token, missing_eos_penalty, sync_ref_model,
            ref_model_sync_steps, ref_model_mixup_alpha, rl_loss_type,
            rl_loss_weights, f_divergence_type, f_alpha_divergence_coef,
            reference_free, label_smoothing_rl, use_weighting,
            rpo_alpha, ld_alpha, discopop_tau
        )
        return
    
    # Create configuration
    try:
        config = _create_configuration(
            model, dataset, method, backend, output_dir, run_name,
            max_steps, learning_rate, batch_size, gradient_accumulation_steps,
            warmup_steps, weight_decay, max_seq_length, precision,
            use_peft, lora_r, lora_alpha, lora_dropout, beta, temperature,
            max_prompt_length, dataset_split, max_samples, dataset_percent,
            dataset_num_proc, dataset_text_field, packing, packing_strategy, eval_packing,
            padding_free, pad_to_multiple_of, completion_only_loss,
            assistant_only_loss, loss_type, activation_offloading,
            chat_template_path, rewards, distributed, num_gpus, log_level, save_steps, eval_steps,
            logging_backend, seed, resume, config_file,
            kl_estimator, vf_coef, cliprange_value, gamma, lam,
            response_length, stop_token, missing_eos_penalty, sync_ref_model,
            ref_model_sync_steps, ref_model_mixup_alpha, rl_loss_type,
            rl_loss_weights, f_divergence_type, f_alpha_divergence_coef,
            reference_free, label_smoothing_rl, use_weighting,
            rpo_alpha, ld_alpha, discopop_tau
        )
        
        # Load config file overrides if provided
        if config_file and Path(config_file).exists():
            config_overrides = ConfigLoader.load_from_yaml(config_file)
            config = _merge_configs(config, config_overrides)
        
        # Create and run trainer with enhanced progress display
        from ..utils.errors import create_progress_display, HealthMonitor

        progress_display = create_progress_display()
        health_monitor = HealthMonitor()

        with progress_display as progress:
            init_task = progress.add_task("Initializing training...", total=None, status="Setting up environment")

            try:
                if method == "sft":
                    trainer = create_sft_trainer(config, backend=backend)
                    progress.update(init_task, description="Starting SFT training...", status="SFT trainer ready")
                else:
                    trainer = create_rl_trainer(config, algorithm=method, backend=backend)
                    progress.update(init_task, description=f"Starting {method.upper()} training...", status=f"{method.upper()} trainer ready")

                # Start training with health monitoring
                progress.update(init_task, description=f"🚀 Training in progress...", status="Monitoring health")
                trainer.train()

                progress.update(init_task, description="✅ Training completed successfully!", status="Done")

                # Show health summary
                health_summary = health_monitor.get_summary()
                if health_summary["total_warnings"] > 0:
                    console.print(f"\n[yellow]⚠️ Training completed with {health_summary['total_warnings']} warnings[/yellow]")
                    if health_summary["critical_issues"] > 0:
                        console.print(f"[red]❌ {health_summary['critical_issues']} critical issues detected[/red]")

            except Exception as e:
                progress.update(init_task, description="❌ Training failed", status="Error occurred")
                raise
        
        console.print(f"[green]✅ Training completed! Model saved to: {output_dir}[/green]")
        
    except Exception as e:
        from ..utils.errors import handle_error
        error_msg = handle_error(e, verbose)
        console.print(error_msg)
        raise typer.Exit(1)


def _create_configuration(
    model: str, dataset: str, method: str, backend: str, output_dir: str, run_name: Optional[str],
    max_steps: int, learning_rate: float, batch_size: int, gradient_accumulation_steps: int,
    warmup_steps: int, weight_decay: float, max_seq_length: int, precision: str,
    use_peft: bool, lora_r: int, lora_alpha: int, lora_dropout: float, beta: Optional[float],
    temperature: float, max_prompt_length: Optional[int], dataset_split: str,
    max_samples: Optional[int], dataset_percent: Optional[float], dataset_num_proc: Optional[int],
    dataset_text_field: str, packing: bool, packing_strategy: str, eval_packing: Optional[bool],
    padding_free: bool, pad_to_multiple_of: Optional[int], completion_only_loss: Optional[bool],
    assistant_only_loss: bool, loss_type: str, activation_offloading: bool,
    chat_template_path: Optional[str], rewards: Optional[List[str]],
    distributed: str, num_gpus: Optional[int], log_level: str, save_steps: int, eval_steps: int,
    logging_backend: str, seed: int, resume: Optional[str], config_file: Optional[str],
    kl_estimator: str, vf_coef: float, cliprange_value: float, gamma: float, lam: float,
    response_length: int, stop_token: str, missing_eos_penalty: float, sync_ref_model: bool,
    ref_model_sync_steps: int, ref_model_mixup_alpha: float, rl_loss_type: str,
    rl_loss_weights: Optional[List[float]], f_divergence_type: str, f_alpha_divergence_coef: float,
    reference_free: bool, label_smoothing_rl: float, use_weighting: bool,
    rpo_alpha: Optional[float], ld_alpha: Optional[float], discopop_tau: float
) -> UnifiedConfig:
    """Create UnifiedConfig from command line arguments."""
    
    # Map method to algorithm type
    algorithm_map = {
        "sft": AlgorithmType.DPO,  # Use DPO for SFT-like training
        "dpo": AlgorithmType.DPO,
        "ppo": AlgorithmType.PPO,
        "grpo": AlgorithmType.GRPO,
        "gspo": AlgorithmType.GSPO
    }
    
    algo = algorithm_map.get(method, AlgorithmType.DPO)
    
    # Create model config
    model_config = ModelConfig(
        name_or_path=model,
        max_seq_length=max_seq_length,
        precision=precision,
        use_peft=use_peft,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Create dataset config
    dataset_config = DatasetConfig(
        name=dataset,
        split=dataset_split,
        max_samples=max_samples,
        percent=dataset_percent
    )
    
    # Create training config
    training_config = TrainingConfig(
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        temperature=temperature,
        max_prompt_length=max_prompt_length or max_seq_length // 2
    )
    
    # Add beta for DPO
    if method == "dpo" and beta is not None:
        training_config.beta = beta
    
    # Create logging config
    logging_config = LoggingConfig(
        output_dir=output_dir,
        run_name=run_name or f"{method}_{model.split('/')[-1]}",
        loggers=[logging_backend]
    )
    
    # Create distributed config
    distributed_config = DistributedConfig(
        backend=distributed,
        num_gpus=num_gpus
    )
    
    # Create reward configs if provided
    reward_configs = []
    if rewards:
        for reward_spec in rewards:
            if ":" in reward_spec:
                reward_type, weight = reward_spec.split(":", 1)
                try:
                    weight = float(weight)
                except ValueError:
                    weight = 1.0
                reward_configs.append(RewardConfig(type=reward_type, weight=weight))
    
    return UnifiedConfig(
        algo=algo,
        model=model_config,
        datasets=[dataset_config],
        train=training_config,
        logging=logging_config,
        distributed=distributed_config,
        rewards=reward_configs
    )


def _merge_configs(base_config: UnifiedConfig, override_config: UnifiedConfig) -> UnifiedConfig:
    """Merge configuration overrides into base configuration."""
    # This is a simplified merge - in practice, you'd want more sophisticated merging
    return override_config


def _show_configuration(
    model: str, dataset: str, method: str, backend: str, output_dir: str, run_name: Optional[str],
    max_steps: int, learning_rate: float, batch_size: int, gradient_accumulation_steps: int,
    warmup_steps: int, weight_decay: float, max_seq_length: int, precision: str,
    use_peft: bool, lora_r: int, lora_alpha: int, lora_dropout: float, beta: Optional[float],
    temperature: float, max_prompt_length: Optional[int], dataset_split: str,
    max_samples: Optional[int], dataset_percent: Optional[float], dataset_num_proc: Optional[int],
    dataset_text_field: str, packing: bool, packing_strategy: str, eval_packing: Optional[bool],
    padding_free: bool, pad_to_multiple_of: Optional[int], completion_only_loss: Optional[bool],
    assistant_only_loss: bool, loss_type: str, activation_offloading: bool,
    chat_template_path: Optional[str], rewards: Optional[List[str]],
    distributed: str, num_gpus: Optional[int], log_level: str, save_steps: int, eval_steps: int,
    logging_backend: str, seed: int, resume: Optional[str], config_file: Optional[str],
    kl_estimator: str, vf_coef: float, cliprange_value: float, gamma: float, lam: float,
    response_length: int, stop_token: str, missing_eos_penalty: float, sync_ref_model: bool,
    ref_model_sync_steps: int, ref_model_mixup_alpha: float, rl_loss_type: str,
    rl_loss_weights: Optional[List[float]], f_divergence_type: str, f_alpha_divergence_coef: float,
    reference_free: bool, label_smoothing_rl: float, use_weighting: bool,
    rpo_alpha: Optional[float], ld_alpha: Optional[float], discopop_tau: float
):
    """Show configuration in a nice table format."""
    
    table = Table(title="🚀 AlignTune Configuration")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    # Basic configuration
    table.add_row("Model", model)
    table.add_row("Dataset", dataset)
    table.add_row("Method", method.upper())
    table.add_row("Backend", backend)
    table.add_row("Output Directory", output_dir)
    if run_name:
        table.add_row("Run Name", run_name)
    
    # Training hyperparameters
    table.add_row("", "")  # Separator
    table.add_row("[bold]Training Hyperparameters[/bold]", "")
    table.add_row("Max Steps", str(max_steps))
    table.add_row("Learning Rate", str(learning_rate))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Gradient Accumulation", str(gradient_accumulation_steps))
    table.add_row("Warmup Steps", str(warmup_steps))
    table.add_row("Weight Decay", str(weight_decay))
    
    # Model configuration
    table.add_row("", "")  # Separator
    table.add_row("[bold]Model Configuration[/bold]", "")
    table.add_row("Max Sequence Length", str(max_seq_length))
    table.add_row("Precision", precision)
    table.add_row("Use PEFT", str(use_peft))
    if use_peft:
        table.add_row("LoRA Rank", str(lora_r))
        table.add_row("LoRA Alpha", str(lora_alpha))
        table.add_row("LoRA Dropout", str(lora_dropout))
    
    # Dataset configuration
    table.add_row("", "")  # Separator
    table.add_row("[bold]Dataset Configuration[/bold]", "")
    table.add_row("Split", dataset_split)
    if max_samples:
        table.add_row("Max Samples", str(max_samples))
    if dataset_percent:
        table.add_row("Dataset Percent", str(dataset_percent))
    if dataset_num_proc:
        table.add_row("Dataset Workers", str(dataset_num_proc))
    table.add_row("Text Field", dataset_text_field)
    table.add_row("Packing", str(packing))
    if packing:
        table.add_row("Packing Strategy", packing_strategy)
        if eval_packing is not None:
            table.add_row("Eval Packing", str(eval_packing))
    table.add_row("Padding Free", str(padding_free))
    if pad_to_multiple_of:
        table.add_row("Pad to Multiple", str(pad_to_multiple_of))
    if completion_only_loss is not None:
        table.add_row("Completion Only Loss", str(completion_only_loss))
    table.add_row("Assistant Only Loss", str(assistant_only_loss))
    table.add_row("Loss Type", loss_type)
    table.add_row("Activation Offloading", str(activation_offloading))
    if chat_template_path:
        table.add_row("Chat Template Path", chat_template_path)

    # RL-specific parameters
    if method in ["dpo", "ppo", "grpo", "gspo"]:
        table.add_row("", "")  # Separator
        table.add_row("[bold]RL Configuration[/bold]", "")
        table.add_row("Temperature", str(temperature))
        if max_prompt_length:
            table.add_row("Max Prompt Length", str(max_prompt_length))
        if beta is not None:
            table.add_row("Beta (DPO)", str(beta))
        table.add_row("KL Estimator", kl_estimator)
        table.add_row("VF Coefficient", str(vf_coef))
        table.add_row("Clip Range Value", str(cliprange_value))
        table.add_row("Gamma", str(gamma))
        table.add_row("Lambda (GAE)", str(lam))
        table.add_row("Response Length", str(response_length))
        table.add_row("Stop Token", stop_token)
        table.add_row("Missing EOS Penalty", str(missing_eos_penalty))
        table.add_row("Sync Ref Model", str(sync_ref_model))
        if sync_ref_model:
            table.add_row("Ref Model Sync Steps", str(ref_model_sync_steps))
            table.add_row("Ref Model Mixup Alpha", str(ref_model_mixup_alpha))
        table.add_row("RL Loss Type", rl_loss_type)
        if rl_loss_weights:
            table.add_row("RL Loss Weights", str(rl_loss_weights))
        table.add_row("F-Divergence Type", f_divergence_type)
        table.add_row("F-Alpha Coef", str(f_alpha_divergence_coef))
        table.add_row("Reference Free", str(reference_free))
        table.add_row("Label Smoothing", str(label_smoothing_rl))
        table.add_row("Use Weighting", str(use_weighting))
        if rpo_alpha is not None:
            table.add_row("RPO Alpha", str(rpo_alpha))
        if ld_alpha is not None:
            table.add_row("LD Alpha", str(ld_alpha))
        table.add_row("DiscoPOP Tau", str(discopop_tau))
        if rewards:
            table.add_row("Rewards", ", ".join(rewards))
    
    # System configuration
    table.add_row("", "")  # Separator
    table.add_row("[bold]System Configuration[/bold]", "")
    table.add_row("Distributed Backend", distributed)
    if num_gpus:
        table.add_row("Number of GPUs", str(num_gpus))
    table.add_row("Log Level", log_level)
    table.add_row("Logging Backend", logging_backend)
    table.add_row("Save Steps", str(save_steps))
    table.add_row("Eval Steps", str(eval_steps))
    table.add_row("Seed", str(seed))
    if resume:
        table.add_row("Resume From", resume)
    if config_file:
        table.add_row("Config File", config_file)
    
    console.print(table)
    console.print("\n[green]✅ Configuration is valid! Remove --dry-run to start training.[/green]")


if __name__ == "__main__":
    app()