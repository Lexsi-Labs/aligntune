"""
Unified CLI for AlignTune.


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
    name="aligntune",
    help="AlignTune: A comprehensive fine-tuning library for SFT and RL training",
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

@app.command()
def list_backends_cmd():
    """List all available backends and their capabilities."""
    try:
        backends = list_backends()
        
        typer.echo("Available AlignTune Backends:")
        typer.echo("=" * 50)
        
        for backend_name, capabilities in backends.items():
            typer.echo(f"\n{backend_name.upper()}:")
            
            # Handle different capability formats
            if isinstance(capabilities, dict):
                # RL backends are a dict with algorithms
                for training_type, algorithms in capabilities.items():
                    if algorithms:
                        # Ensure algorithms is iterable
                        if isinstance(algorithms, (list, tuple)):
                            typer.echo(f"  {training_type.upper()}: {', '.join(str(a) for a in algorithms)}")
                        else:
                            typer.echo(f"  {training_type.upper()}: {algorithms}")
                    else:
                        typer.echo(f"  {training_type.upper()}: Available")
            elif isinstance(capabilities, (list, tuple)):
                # SFT backends are a list
                typer.echo(f"  Available: {', '.join(str(c) for c in capabilities)}")
            elif isinstance(capabilities, str):
                # Single string capability
                typer.echo(f"  Available: {capabilities}")
            else:
                # Unknown format - just print it
                typer.echo(f"  {capabilities}")
        
        typer.echo("\n" + "=" * 50)
        
    except Exception as e:
        typer.echo(f"Error listing backends: {e}")
        logger.exception("Detailed error information:")
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
    """Show AlignTune information and system status."""
    
    try:
        from .. import (
            __version__,
            __author__,
            get_available_trainers,
            check_dependencies,
            UNSLOTH_ERROR_INFO,
        )
        
        typer.echo("AlignTune Information")
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
        aligntune validate config.yaml
        aligntune validate config.yaml --type sft
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
        None, "--precision", help="Model precision: fp32, fp16, bf16, auto"
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
    device_map: Optional[str] = typer.Option(
        None, "--device-map", help="Device map for model loading (default: auto)"
    ),
    trust_remote_code: Optional[bool] = typer.Option(
        None, "--trust-remote-code", help="Trust remote code when loading model"
    ),
    
    # ========== Training Type & Backend ==========
    training_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Training type: sft, dpo, ppo, grpo, gspo, etc."
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Backend: auto, trl, unsloth"
    ),
    
    # ========== Dataset Configuration ==========
    dataset_name: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Dataset name (e.g., 'tatsu-lab/alpaca')"
    ),
    dataset_split: Optional[str] = typer.Option(
        None, "--split", help="Dataset split to use (default: train)"
    ),
    dataset_subset: Optional[str] = typer.Option(
        None, "--subset", help="Dataset subset/config name"
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
    system_prompt: Optional[str] = typer.Option(
        None, "--system-prompt", help="System prompt for dataset"
    ),
    dataset_num_proc: Optional[int] = typer.Option(
        None, "--dataset-num-proc", help="Number of processes for dataset preprocessing"
    ),
    dataset_text_field: Optional[str] = typer.Option(
        None, "--dataset-text-field", help="Text field name in dataset (default: text)"
    ),
    text_column: Optional[str] = typer.Option(
        None, "--text-column", help="Text column name (for classification)"
    ),
    label_column: Optional[str] = typer.Option(
        None, "--label-column", help="Label column name (for classification)"
    ),
    pad_token: Optional[str] = typer.Option(
        None, "--pad-token", help="Padding token"
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
    eval_batch_size: Optional[int] = typer.Option(
        None, "--eval-batch-size", help="Per-device evaluation batch size"
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
        None, "--warmup-ratio", help="Warmup ratio"
    ),
    warmup_steps: Optional[int] = typer.Option(
        None, "--warmup-steps", help="Warmup steps"
    ),
    weight_decay: Optional[float] = typer.Option(
        None, "--weight-decay", help="Weight decay"
    ),
    max_grad_norm: Optional[float] = typer.Option(
        None, "--max-grad-norm", help="Maximum gradient norm for clipping"
    ),
    optimizer: Optional[str] = typer.Option(
        None, "--optimizer", help="Optimizer type (default: adamw_torch)"
    ),
    lr_scheduler: Optional[str] = typer.Option(
        None, "--lr-scheduler", help="Learning rate scheduler (default: cosine)"
    ),
    
    # ========== SFT-Specific Parameters ==========
    fp16: Optional[bool] = typer.Option(
        None, "--fp16", help="Use FP16 training"
    ),
    bf16: Optional[bool] = typer.Option(
        None, "--bf16", help="Use BF16 training"
    ),
    dataloader_num_workers: Optional[int] = typer.Option(
        None, "--dataloader-num-workers", help="Number of dataloader workers"
    ),
    remove_unused_columns: Optional[bool] = typer.Option(
        None, "--remove-unused-columns", help="Remove unused columns from dataset"
    ),
    group_by_length: Optional[bool] = typer.Option(
        None, "--group-by-length", help="Group sequences by length"
    ),
    dataloader_drop_last: Optional[bool] = typer.Option(
        None, "--dataloader-drop-last", help="Drop last incomplete batch"
    ),
    eval_accumulation_steps: Optional[int] = typer.Option(
        None, "--eval-accumulation-steps", help="Evaluation accumulation steps"
    ),
    label_smoothing_factor: Optional[float] = typer.Option(
        None, "--label-smoothing", help="Label smoothing factor"
    ),
    early_stopping_patience: Optional[int] = typer.Option(
        None, "--early-stopping-patience", help="Early stopping patience"
    ),
    early_stopping_threshold: Optional[float] = typer.Option(
        None, "--early-stopping-threshold", help="Early stopping threshold"
    ),
    load_best_model_at_end: Optional[bool] = typer.Option(
        None, "--load-best-model-at-end", help="Load best model at end of training"
    ),
    metric_for_best_model: Optional[str] = typer.Option(
        None, "--metric-for-best-model", help="Metric to use for best model selection"
    ),
    greater_is_better: Optional[bool] = typer.Option(
        None, "--greater-is-better", help="Whether higher metric is better"
    ),
    use_trl: Optional[bool] = typer.Option(
        None, "--use-trl", help="Use TRL trainer for SFT"
    ),
    packing: Optional[bool] = typer.Option(
        None, "--packing", help="Enable sequence packing (SFT only)"
    ),
    packing_strategy: Optional[str] = typer.Option(
        None, "--packing-strategy", help="Packing strategy: bfd, ffd (default: bfd)"
    ),
    eval_packing: Optional[bool] = typer.Option(
        None, "--eval-packing", help="Enable packing for evaluation"
    ),
    padding_free: Optional[bool] = typer.Option(
        None, "--padding-free", help="Use padding-free training"
    ),
    pad_to_multiple_of: Optional[int] = typer.Option(
        None, "--pad-to-multiple-of", help="Pad sequences to multiple of this value"
    ),
    completion_only_loss: Optional[bool] = typer.Option(
        None, "--completion-only-loss", help="Compute loss only on completion tokens"
    ),
    assistant_only_loss: Optional[bool] = typer.Option(
        None, "--assistant-only-loss", help="Compute loss only on assistant tokens"
    ),
    activation_offloading: Optional[bool] = typer.Option(
        None, "--activation-offloading", help="Enable activation offloading"
    ),
    use_flash_attention_2: Optional[bool] = typer.Option(
        None, "--use-flash-attention-2", help="Use Flash Attention 2"
    ),
    gradient_checkpointing_kwargs: Optional[str] = typer.Option(
        None, "--gradient-checkpointing-kwargs", help="JSON string of gradient checkpointing kwargs"
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
    top_p: Optional[float] = typer.Option(
        None, "--top-p", help="Top-p sampling parameter"
    ),
    reward_model_name: Optional[str] = typer.Option(
        None, "--reward-model", help="Reward model name or path (for PPO/GRPO)"
    ),
    reward_model_path: Optional[str] = typer.Option(
        None, "--reward-model-path", help="Local path to reward model"
    ),
    loss_type: Optional[str] = typer.Option(
        None, "--loss-type", help="Loss type: sigmoid, hinge, ipo, kto_pair, nll, etc."
    ),
    weighting_mode: Optional[str] = typer.Option(
        None, "--weighting-mode", help="Counterfactual GRPO weighting: counterfactual, random, inverted, vanilla"
    ),
    use_gradient_checkpointing: Optional[bool] = typer.Option(
        None, "--use-gradient-checkpointing", help="Enable gradient checkpointing"
    ),
    fast_inference: Optional[bool] = typer.Option(
        None, "--fast-inference", help="Enable Unsloth's fast inference via vLLM (2-3x faster generation)"
    ),
    vllm_gpu_memory: Optional[float] = typer.Option(
        None, "--vllm-gpu-memory", help="vLLM GPU memory utilization (0.0-1.0, default 0.7, use 0.95 for max speed)"
    ),
    cliprange: Optional[float] = typer.Option(
        None, "--cliprange", help="PPO clip range"
    ),
    max_completion_length: Optional[int] = typer.Option(
        None, "--max-completion-length", help="Maximum completion/generation length"
    ),
    max_target_length: Optional[int] = typer.Option(
        None, "--max-target-length", help="Maximum target length"
    ),
    num_generations: Optional[int] = typer.Option(
        None, "--num-generations", help="Number of generations per prompt (GRPO/GSPO)"
    ),
    generation_batch_size: Optional[int] = typer.Option(
        None, "--generation-batch-size", help="Batch size for generation"
    ),
    use_cache: Optional[bool] = typer.Option(
        None, "--use-cache", help="Use KV cache during generation"
    ),
    rollout_batch_size: Optional[int] = typer.Option(
        None, "--rollout-batch-size", help="Rollout batch size for PPO"
    ),
    num_ppo_epochs: Optional[int] = typer.Option(
        None, "--num-ppo-epochs", help="Number of PPO epochs"
    ),
    whiten_rewards: Optional[bool] = typer.Option(
        None, "--whiten-rewards", help="Whiten rewards"
    ),
    kl_estimator: Optional[str] = typer.Option(
        None, "--kl-estimator", help="KL estimator type (default: k1)"
    ),
    vf_coef: Optional[float] = typer.Option(
        None, "--vf-coef", help="Value function coefficient"
    ),
    cliprange_value: Optional[float] = typer.Option(
        None, "--cliprange-value", help="Value function clip range"
    ),
    gamma: Optional[float] = typer.Option(
        None, "--gamma", help="Discount factor"
    ),
    lam: Optional[float] = typer.Option(
        None, "--lam", help="GAE lambda parameter"
    ),
    stop_token: Optional[str] = typer.Option(
        None, "--stop-token", help="Stop token for generation (default: eos)"
    ),
    missing_eos_penalty: Optional[float] = typer.Option(
        None, "--missing-eos-penalty", help="Penalty for missing EOS token"
    ),
    truncation_mode: Optional[str] = typer.Option(
        None, "--truncation-mode", help="Truncation mode: keep_end, keep_start"
    ),
    f_divergence_type: Optional[str] = typer.Option(
        None, "--f-divergence-type", help="F-divergence type (default: reverse_kl)"
    ),
    f_alpha_divergence_coef: Optional[float] = typer.Option(
        None, "--f-alpha-divergence-coef", help="F-alpha divergence coefficient"
    ),
    reference_free: Optional[bool] = typer.Option(
        None, "--reference-free", help="Use reference-free training"
    ),
    label_smoothing: Optional[float] = typer.Option(
        None, "--label-smoothing", help="Label smoothing for RL"
    ),
    use_weighting: Optional[bool] = typer.Option(
        None, "--use-weighting", help="Use importance weighting"
    ),
    rpo_alpha: Optional[float] = typer.Option(
        None, "--rpo-alpha", help="RPO alpha parameter"
    ),
    ld_alpha: Optional[float] = typer.Option(
        None, "--ld-alpha", help="LD alpha parameter"
    ),
    discopop_tau: Optional[float] = typer.Option(
        None, "--discopop-tau", help="DiscoPOP tau parameter"
    ),
    sync_ref_model: Optional[bool] = typer.Option(
        None, "--sync-ref-model", help="Sync reference model"
    ),
    ref_model_mixup_alpha: Optional[float] = typer.Option(
        None, "--ref-model-mixup-alpha", help="Reference model mixup alpha"
    ),
    ref_model_sync_steps: Optional[int] = typer.Option(
        None, "--ref-model-sync-steps", help="Reference model sync steps"
    ),
    use_liger_kernel: Optional[bool] = typer.Option(
        None, "--use-liger-kernel", help="Use Liger kernel"
    ),
    use_liger_loss: Optional[bool] = typer.Option(
        None, "--use-liger-loss", help="Use Liger loss"
    ),
    scale_rewards: Optional[str] = typer.Option(
        None, "--scale-rewards", help="Reward scaling: group, batch, none"
    ),
    enable_thinking: Optional[bool] = typer.Option(
        None, "--enable-thinking", help="Enable thinking mode"
    ),
    use_rewards_directly: Optional[bool] = typer.Option(
        None, "--use-rewards-directly", help="Use rewards directly without processing"
    ),
    mask_truncated_completions: Optional[bool] = typer.Option(
        None, "--mask-truncated-completions", help="Mask truncated completions"
    ),
    
    # ========== Custom Reward Model Training ==========
    train_custom_reward_model: Optional[bool] = typer.Option(
        None, "--train-custom-reward", help="Train a custom reward model"
    ),
    reward_functions: Optional[str] = typer.Option(
        None, "--reward-functions", help="Comma-separated list of reward functions"
    ),
    reward_function_weights: Optional[str] = typer.Option(
        None, "--reward-function-weights", help="Comma-separated list of reward function weights"
    ),
    reward_training_base_model: Optional[str] = typer.Option(
        None, "--reward-base-model", help="Base model for reward model training"
    ),
    reward_training_output_dir: Optional[str] = typer.Option(
        None, "--reward-output-dir", help="Output directory for reward model"
    ),
    reward_training_epochs: Optional[int] = typer.Option(
        None, "--reward-epochs", help="Epochs for reward model training"
    ),
    reward_training_lr: Optional[float] = typer.Option(
        None, "--reward-lr", help="Learning rate for reward model"
    ),
    reward_training_batch_size: Optional[int] = typer.Option(
        None, "--reward-batch-size", help="Batch size for reward model training"
    ),
    reward_training_texts: Optional[str] = typer.Option(
        None, "--reward-training-texts", help="Path to file with training texts for reward model"
    ),
    
    # ========== Value Model Configuration ==========
    reward_value_model: Optional[str] = typer.Option(
        None, "--value-model", help="Value model name or path"
    ),
    reward_value_loading_type: Optional[str] = typer.Option(
        None, "--value-loading-type", help="Value model loading type: standard, lora, etc."
    ),
    value_model_num_labels: Optional[int] = typer.Option(
        None, "--value-num-labels", help="Number of labels for value model"
    ),
    value_model_quantization: Optional[str] = typer.Option(
        None, "--value-model-quantization", help="Path to JSON file with value model quantization config"
    ),
    reward_model_quantization: Optional[str] = typer.Option(
        None, "--reward-model-quantization", help="Path to JSON file with reward model quantization config"
    ),
    reward_device: Optional[str] = typer.Option(
        None, "--reward-device", help="Device for reward model (default: auto)"
    ),
    
    # ========== Rewards Configuration ==========
    rewards: Optional[str] = typer.Option(
        None, "--rewards", help="Path to JSON file with rewards configuration OR inline format: 'type1:weight1,type2:weight2'"
    ),
    reward_params: Optional[str] = typer.Option(
        None, "--reward-params", help="Path to JSON file with reward parameters for each type"
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
    log_interval: Optional[int] = typer.Option(
        None, "--log-interval", help="Logging interval in steps"
    ),
    logging_steps: Optional[int] = typer.Option(
        None, "--logging-steps", help="Logging steps (same as log-interval)"
    ),
    report_to: Optional[str] = typer.Option(
        None, "--report-to", help="Report metrics to: none, wandb, tensorboard, mlflow"
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
    save_total_limit: Optional[int] = typer.Option(
        None, "--save-total-limit", help="Maximum number of checkpoints to keep"
    ),
    save_strategy: Optional[str] = typer.Option(
        None, "--save-strategy", help="Save strategy: steps, epoch, no"
    ),
    eval_strategy: Optional[str] = typer.Option(
        None, "--eval-strategy", help="Evaluation strategy: steps, epoch, no"
    ),
    logging_strategy: Optional[str] = typer.Option(
        None, "--logging-strategy", help="Logging strategy: steps, epoch"
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
        False, "--use-peft", help="Enable PEFT/LoRA"
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
    lora_bias: Optional[str] = typer.Option(
        None, "--lora-bias", help="LoRA bias: none, all, lora_only"
    ),
    model_adapter_name: Optional[str] = typer.Option(
        None, "--model-adapter-name", help="Model adapter name"
    ),
    ref_adapter_name: Optional[str] = typer.Option(
        None, "--ref-adapter-name", help="Reference adapter name"
    ),
    force_use_ref_model: Optional[bool] = typer.Option(
        None, "--force-use-ref-model", help="Force use of reference model"
    ),
    disable_dropout: Optional[bool] = typer.Option(
        None, "--disable-dropout", help="Disable dropout"
    ),
    
    # ========== Advanced Options ==========
    chat_template: Optional[str] = typer.Option(
        None, "--chat-template", help="Chat template: auto, chatml, llama2, etc."
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Cache directory"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed"
    ),
    data_seed: Optional[int] = typer.Option(
        None, "--data-seed", help="Data shuffling seed"
    ),
    max_memory: Optional[str] = typer.Option(
        None, "--max-memory", help="Maximum memory per device"
    ),
    num_labels: Optional[int] = typer.Option(
        None, "--num-labels", help="Number of labels for classification"
    ),
    config_name: Optional[str] = typer.Option(
        None, "--config-name", help="Dataset config name"
    ),
    
    # ========== Distributed Training ==========
    distributed_backend: Optional[str] = typer.Option(
        None, "--distributed", help="Distributed backend: single, ddp, deepspeed, fsdp"
    ),
    
    # ========== Counterfactual GRPO Parameters ==========
    boost_factor: Optional[float] = typer.Option(
        None, "--boost-factor", help="Boost factor for counterfactual GRPO (default: 2.0)"
    ),
    min_weight: Optional[float] = typer.Option(
        None, "--min-weight", help="Minimum weight for counterfactual GRPO (default: 0.5)"
    ),
    max_spans: Optional[int] = typer.Option(
        None, "--max-spans", help="Maximum spans for counterfactual GRPO (default: 10)"
    ),
    answer_weight: Optional[float] = typer.Option(
        None, "--answer-weight", help="Answer weight for counterfactual GRPO (default: 1.5)"
    ),
    method_name: Optional[str] = typer.Option(
        None, "--method-name", help="Method name for counterfactual GRPO (default: counterfactual)"
    ),
    random_importance: Optional[bool] = typer.Option(
        None, "--random-importance", help="Use random importance (deprecated)"
    ),
    invert_importance: Optional[bool] = typer.Option(
        None, "--invert-importance", help="Invert importance (deprecated)"
    ),
    enable_gradient_conservation: Optional[bool] = typer.Option(
        None, "--enable-gradient-conservation", help="Enable gradient conservation"
    ),
    weight_debug: Optional[bool] = typer.Option(
        None, "--weight-debug", help="Enable weight debugging"
    ),
    
    # ========== GBMPO Parameters ==========
    gbmpo_divergence_type: Optional[str] = typer.Option(
        None, "--gbmpo-divergence-type", help="GBMPO divergence type: l2, l2kl, prob_l2, prob_l2kl"
    ),
    gbmpo_l2_coefficient: Optional[float] = typer.Option(
        None, "--gbmpo-l2-coefficient", help="GBMPO L2 coefficient"
    ),
    gbmpo_epsilon: Optional[float] = typer.Option(
        None, "--gbmpo-epsilon", help="GBMPO epsilon"
    ),
    
    # ========== BOLT Parameters ==========
    curriculum_enabled: Optional[bool] = typer.Option(
        None, "--curriculum-enabled", help="Enable BOLT curriculum learning"
    ),
    curriculum_epsilon: Optional[float] = typer.Option(
        None, "--curriculum-epsilon", help="BOLT curriculum epsilon"
    ),
    curriculum_update_freq: Optional[int] = typer.Option(
        None, "--curriculum-update-freq", help="BOLT curriculum update frequency"
    ),
    baseline_enabled: Optional[bool] = typer.Option(
        None, "--baseline-enabled", help="Enable BOLT baseline"
    ),
    baseline_rho_min: Optional[float] = typer.Option(
        None, "--baseline-rho-min", help="BOLT baseline rho min"
    ),
    baseline_rho_max: Optional[float] = typer.Option(
        None, "--baseline-rho-max", help="BOLT baseline rho max"
    ),
    baseline_D_half: Optional[float] = typer.Option(
        None, "--baseline-d-half", help="BOLT baseline D half"
    ),
    baseline_warm_start: Optional[int] = typer.Option(
        None, "--baseline-warm-start", help="BOLT baseline warm start steps"
    ),
    use_baseline_advantages: Optional[bool] = typer.Option(
        None, "--use-baseline-advantages", help="Use BOLT baseline advantages"
    ),
    
    # ========== Sample Logging ==========
    enable_sample_logging: Optional[bool] = typer.Option(
        None, "--enable-sample-logging", help="Enable sample logging during training"
    ),
    sample_logging_prompts: Optional[str] = typer.Option(
        None, "--sample-logging-prompts", help="Comma-separated prompts for sample logging"
    ),
    sample_logging_interval_steps: Optional[int] = typer.Option(
        None, "--sample-logging-interval", help="Sample logging interval in steps"
    ),
    sample_logging_percent_of_max_steps: Optional[float] = typer.Option(
        None, "--sample-logging-percent", help="Sample logging as percent of max steps"
    ),
    sample_logging_max_new_tokens: Optional[int] = typer.Option(
        None, "--sample-logging-max-tokens", help="Max tokens for sample logging"
    ),
    sample_logging_temperature: Optional[float] = typer.Option(
        None, "--sample-logging-temperature", help="Temperature for sample logging"
    ),
    sample_logging_top_p: Optional[float] = typer.Option(
        None, "--sample-logging-top-p", help="Top-p for sample logging"
    ),
    sample_logging_num_samples: Optional[int] = typer.Option(
        None, "--sample-logging-num-samples", help="Number of samples to log"
    ),
    
    # ========== Evaluation Configuration (SFT) ==========
    compute_perplexity: Optional[bool] = typer.Option(
        None, "--compute-perplexity", help="Compute perplexity metric"
    ),
    compute_rouge: Optional[bool] = typer.Option(
        None, "--compute-rouge", help="Compute ROUGE metric"
    ),
    compute_bleu: Optional[bool] = typer.Option(
        None, "--compute-bleu", help="Compute BLEU metric"
    ),
    compute_meteor: Optional[bool] = typer.Option(
        None, "--compute-meteor", help="Compute METEOR metric"
    ),
    compute_bertscore: Optional[bool] = typer.Option(
        None, "--compute-bertscore", help="Compute BERTScore metric"
    ),
    compute_semantic_similarity: Optional[bool] = typer.Option(
        None, "--compute-semantic-similarity", help="Compute semantic similarity"
    ),
    compute_codebleu: Optional[bool] = typer.Option(
        None, "--compute-codebleu", help="Compute CodeBLEU metric"
    ),
    max_samples_for_quality_metrics: Optional[int] = typer.Option(
        None, "--max-samples-for-quality-metrics", help="Max samples for quality metrics"
    ),
    bertscore_model: Optional[str] = typer.Option(
        None, "--bertscore-model", help="Model for BERTScore computation"
    ),
    semantic_similarity_model: Optional[str] = typer.Option(
        None, "--semantic-similarity-model", help="Model for semantic similarity"
    ),
):
    """
    Train a model using AlignTune.
    
    Configuration can be provided via:
    1. YAML config file (--config)
    2. CLI arguments (override YAML)
    3. Both (CLI takes precedence)
    
    Examples:
    
    # Using YAML config
    aligntune train --config configs/dpo_simple.yaml
    
    # Using CLI only
    aligntune train --model microsoft/DialoGPT-small --dataset Anthropic/hh-rlhf --type dpo
    
    # Mixing YAML and CLI (CLI overrides)
    aligntune train --config configs/dpo_simple.yaml --batch-size 8 --lr 1e-4
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
    
    # Load YAML config if provided
    if config:
        if not Path(config).exists():
            typer.echo(f"Error: Config file not found: {config}")
            raise typer.Exit(1)
        
        final_config = load_yaml_config(config)
        typer.echo(f"✓ Loaded configuration from: {config}")
        
        # If config is provided, pass it directly to trainer
        # Determine training type
        algo = final_config.get('algo', 'sft').lower()
        
        # Set up logging
        log_lvl = final_config.get('logging', {}).get('log_level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_lvl.upper()))
        
        try:
            # Determine backend
            backend_str = backend or final_config.get('backend', 'auto')
            if backend_str.lower() == "auto":
                backend_enum = BackendType.TRL
            else:
                try:
                    backend_enum = BackendType(backend_str.lower())
                except ValueError:
                    typer.echo(f"Error: Invalid backend '{backend_str}'. Must be: auto, trl, unsloth")
                    raise typer.Exit(1)
            
            # Print training info
            model_cfg = final_config.get('model', {})
            dataset_cfg = final_config.get('datasets', [{}])[0]
            logging_cfg = final_config.get('logging', {})
            
            typer.echo("\n" + "="*60)
            typer.echo(f"ALIGNTUNE - TRAINING CONFIGURATION")
            typer.echo("="*60)
            typer.echo(f"Algorithm: {algo.upper()}")
            typer.echo(f"Backend: {backend_enum.value.upper()}")
            typer.echo(f"Model: {model_cfg.get('name_or_path')}")
            typer.echo(f"Dataset: {dataset_cfg.get('name')}")
            typer.echo(f"Output: {logging_cfg.get('output_dir', './output')}")
            typer.echo("="*60 + "\n")
            
            # Create trainer by passing config directly
            if algo == 'sft':
                trainer = create_sft_trainer(config=final_config)
            else:
                trainer = create_rl_trainer(config=final_config)
            
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
            
            return  # Exit after training with config
            
        except Exception as e:
            typer.echo(f"\n{'='*60}")
            typer.echo("ERROR DURING TRAINING")
            typer.echo("="*60)
            typer.echo(f"{e}")
            typer.echo("="*60 + "\n")
            logger.exception("Training failed")
            raise typer.Exit(1)
    
    
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
    if lora_bias: model_config['lora_bias'] = lora_bias
    if reward_model_name: model_config['reward_model_name'] = reward_model_name
    if reward_model_path: model_config['reward_model_path'] = reward_model_path
    if device_map: model_config['device_map'] = device_map
    if trust_remote_code is not None: model_config['trust_remote_code'] = trust_remote_code
    if max_memory: model_config['max_memory'] = max_memory
    if num_labels: model_config['num_labels'] = num_labels
    if model_adapter_name: model_config['model_adapter_name'] = model_adapter_name
    if ref_adapter_name: model_config['ref_adapter_name'] = ref_adapter_name
    if force_use_ref_model is not None: model_config['force_use_ref_model'] = force_use_ref_model
    if disable_dropout is not None: model_config['disable_dropout'] = disable_dropout
    if reward_device: model_config['reward_device'] = reward_device
    if reward_value_model: model_config['reward_value_model'] = reward_value_model
    if reward_value_loading_type: model_config['reward_value_loading_type'] = reward_value_loading_type
    
    # Quantization
    if load_in_4bit or load_in_8bit:
        quantization = {}
        if load_in_4bit: 
            quantization['load_in_4bit'] = True
            if bnb_4bit_compute_dtype: quantization['bnb_4bit_compute_dtype'] = bnb_4bit_compute_dtype
            if bnb_4bit_quant_type: quantization['bnb_4bit_quant_type'] = bnb_4bit_quant_type
        if load_in_8bit: quantization['load_in_8bit'] = True
        model_config['quantization'] = quantization
    
    # Separate quantization for reward and value models
    if reward_model_quantization:
        import json
        with open(reward_model_quantization, 'r') as f:
            model_config['reward_model_quantization'] = json.load(f)
    
    if value_model_quantization:
        import json
        with open(value_model_quantization, 'r') as f:
            model_config['value_model_quantization'] = json.load(f)
    
    if model_config:
        cli_config['model'] = model_config
    
    # Dataset config
    dataset_config = {}
    if dataset_name: dataset_config['name'] = dataset_name
    if dataset_split: dataset_config['split'] = dataset_split
    if dataset_subset: dataset_config['subset'] = dataset_subset
    if max_samples: dataset_config['max_samples'] = max_samples
    if dataset_percent: dataset_config['percent'] = dataset_percent
    if task_type: dataset_config['task_type'] = task_type
    if system_prompt: dataset_config['system_prompt'] = system_prompt
    if dataset_num_proc: dataset_config['dataset_num_proc'] = dataset_num_proc
    if chat_template: dataset_config['chat_template'] = chat_template
    if dataset_text_field: dataset_config['dataset_text_field'] = dataset_text_field
    if text_column: dataset_config['text_column'] = text_column
    if label_column: dataset_config['label_column'] = label_column
    if pad_token: dataset_config['pad_token'] = pad_token
    if config_name: dataset_config['config_name'] = config_name
    
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
    if eval_batch_size: train_config['per_device_eval_batch_size'] = eval_batch_size
    if learning_rate: train_config['learning_rate'] = learning_rate
    if gradient_accumulation_steps: train_config['gradient_accumulation_steps'] = gradient_accumulation_steps
    if warmup_ratio: train_config['warmup_ratio'] = warmup_ratio
    if warmup_steps: train_config['warmup_steps'] = warmup_steps
    if weight_decay: train_config['weight_decay'] = weight_decay
    if max_grad_norm: train_config['max_grad_norm'] = max_grad_norm
    if optimizer: train_config['optimizer'] = optimizer
    if lr_scheduler: train_config['lr_scheduler'] = lr_scheduler
    
    # SFT-specific parameters
    if fp16 is not None: train_config['fp16'] = fp16
    if bf16 is not None: train_config['bf16'] = bf16
    if dataloader_num_workers: train_config['dataloader_num_workers'] = dataloader_num_workers
    if remove_unused_columns is not None: train_config['remove_unused_columns'] = remove_unused_columns
    if group_by_length is not None: train_config['group_by_length'] = group_by_length
    if dataloader_drop_last is not None: train_config['dataloader_drop_last'] = dataloader_drop_last
    if eval_accumulation_steps: train_config['eval_accumulation_steps'] = eval_accumulation_steps
    if label_smoothing_factor: train_config['label_smoothing_factor'] = label_smoothing_factor
    if early_stopping_patience: train_config['early_stopping_patience'] = early_stopping_patience
    if early_stopping_threshold: train_config['early_stopping_threshold'] = early_stopping_threshold
    if load_best_model_at_end is not None: train_config['load_best_model_at_end'] = load_best_model_at_end
    if metric_for_best_model: train_config['metric_for_best_model'] = metric_for_best_model
    if greater_is_better is not None: train_config['greater_is_better'] = greater_is_better
    if use_trl is not None: train_config['use_trl'] = use_trl
    if packing is not None: train_config['packing'] = packing
    if packing_strategy: train_config['packing_strategy'] = packing_strategy
    if eval_packing is not None: train_config['eval_packing'] = eval_packing
    if padding_free is not None: train_config['padding_free'] = padding_free
    if pad_to_multiple_of: train_config['pad_to_multiple_of'] = pad_to_multiple_of
    if completion_only_loss is not None: train_config['completion_only_loss'] = completion_only_loss
    if assistant_only_loss is not None: train_config['assistant_only_loss'] = assistant_only_loss
    if activation_offloading is not None: train_config['activation_offloading'] = activation_offloading
    if use_flash_attention_2 is not None: train_config['use_flash_attention_2'] = use_flash_attention_2
    if gradient_checkpointing_kwargs:
        import json
        train_config['gradient_checkpointing_kwargs'] = json.loads(gradient_checkpointing_kwargs)
    
    # RL-specific parameters
    if max_prompt_length: train_config['max_prompt_length'] = max_prompt_length
    if beta: train_config['beta'] = beta
    if kl_coef: train_config['kl_coef'] = kl_coef
    if response_length: train_config['response_length'] = response_length
    if temperature: train_config['temperature'] = temperature
    if top_p: train_config['top_p'] = top_p
    if loss_type: train_config['loss_type'] = loss_type
    if weighting_mode: train_config['weighting_mode'] = weighting_mode
    if use_gradient_checkpointing is not None: train_config['use_gradient_checkpointing'] = use_gradient_checkpointing
    if fast_inference is not None: train_config['fast_inference'] = fast_inference
    if vllm_gpu_memory is not None: train_config['vllm_gpu_memory_utilization'] = vllm_gpu_memory
    if cliprange: train_config['cliprange'] = cliprange
    if max_completion_length: train_config['max_completion_length'] = max_completion_length
    if max_target_length: train_config['max_target_length'] = max_target_length
    if num_generations: train_config['num_generations'] = num_generations
    if generation_batch_size: train_config['generation_batch_size'] = generation_batch_size
    if use_cache is not None: train_config['use_cache'] = use_cache
    if rollout_batch_size: train_config['rollout_batch_size'] = rollout_batch_size
    if num_ppo_epochs: train_config['num_ppo_epochs'] = num_ppo_epochs
    if whiten_rewards is not None: train_config['whiten_rewards'] = whiten_rewards
    if kl_estimator: train_config['kl_estimator'] = kl_estimator
    if vf_coef: train_config['vf_coef'] = vf_coef
    if cliprange_value: train_config['cliprange_value'] = cliprange_value
    if gamma: train_config['gamma'] = gamma
    if lam: train_config['lam'] = lam
    if stop_token: train_config['stop_token'] = stop_token
    if missing_eos_penalty: train_config['missing_eos_penalty'] = missing_eos_penalty
    if truncation_mode: train_config['truncation_mode'] = truncation_mode
    if f_divergence_type: train_config['f_divergence_type'] = f_divergence_type
    if f_alpha_divergence_coef: train_config['f_alpha_divergence_coef'] = f_alpha_divergence_coef
    if reference_free is not None: train_config['reference_free'] = reference_free
    if label_smoothing: train_config['label_smoothing'] = label_smoothing
    if use_weighting is not None: train_config['use_weighting'] = use_weighting
    if rpo_alpha: train_config['rpo_alpha'] = rpo_alpha
    if ld_alpha: train_config['ld_alpha'] = ld_alpha
    if discopop_tau: train_config['discopop_tau'] = discopop_tau
    if sync_ref_model is not None: train_config['sync_ref_model'] = sync_ref_model
    if ref_model_mixup_alpha: train_config['ref_model_mixup_alpha'] = ref_model_mixup_alpha
    if ref_model_sync_steps: train_config['ref_model_sync_steps'] = ref_model_sync_steps
    if use_liger_kernel is not None: train_config['use_liger_kernel'] = use_liger_kernel
    if use_liger_loss is not None: train_config['use_liger_loss'] = use_liger_loss
    if scale_rewards: train_config['scale_rewards'] = scale_rewards
    if enable_thinking is not None: train_config['enable_thinking'] = enable_thinking
    if use_rewards_directly is not None: train_config['use_rewards_directly'] = use_rewards_directly
    if mask_truncated_completions is not None: train_config['mask_truncated_completions'] = mask_truncated_completions
    
    # Counterfactual GRPO parameters
    if boost_factor: train_config['boost_factor'] = boost_factor
    if min_weight: train_config['min_weight'] = min_weight
    if max_spans: train_config['max_spans'] = max_spans
    if answer_weight: train_config['answer_weight'] = answer_weight
    if method_name: train_config['method_name'] = method_name
    if random_importance is not None: train_config['random_importance'] = random_importance
    if invert_importance is not None: train_config['invert_importance'] = invert_importance
    if enable_gradient_conservation is not None: train_config['enable_gradient_conservation'] = enable_gradient_conservation
    if weight_debug is not None: train_config['weight_debug'] = weight_debug
    
    # GBMPO parameters
    if gbmpo_divergence_type: train_config['gbmpo_divergence_type'] = gbmpo_divergence_type
    if gbmpo_l2_coefficient: train_config['gbmpo_l2_coefficient'] = gbmpo_l2_coefficient
    if gbmpo_epsilon: train_config['gbmpo_epsilon'] = gbmpo_epsilon
    
    # BOLT parameters
    if curriculum_enabled is not None: train_config['curriculum_enabled'] = curriculum_enabled
    if curriculum_epsilon: train_config['curriculum_epsilon'] = curriculum_epsilon
    if curriculum_update_freq: train_config['curriculum_update_freq'] = curriculum_update_freq
    if baseline_enabled is not None: train_config['baseline_enabled'] = baseline_enabled
    if baseline_rho_min: train_config['baseline_rho_min'] = baseline_rho_min
    if baseline_rho_max: train_config['baseline_rho_max'] = baseline_rho_max
    if baseline_D_half: train_config['baseline_D_half'] = baseline_D_half
    if baseline_warm_start: train_config['baseline_warm_start'] = baseline_warm_start
    if use_baseline_advantages is not None: train_config['use_baseline_advantages'] = use_baseline_advantages
    
    # Custom reward model training
    if train_custom_reward_model is not None: train_config['train_custom_reward_model'] = train_custom_reward_model
    if reward_functions: train_config['reward_functions'] = reward_functions.split(',')
    if reward_function_weights: train_config['reward_function_weights'] = [float(w) for w in reward_function_weights.split(',')]
    if reward_training_base_model: train_config['reward_training_base_model'] = reward_training_base_model
    if reward_training_output_dir: train_config['reward_training_output_dir'] = reward_training_output_dir
    if reward_training_epochs: train_config['reward_training_epochs'] = reward_training_epochs
    if reward_training_lr: train_config['reward_training_lr'] = reward_training_lr
    if reward_training_batch_size: train_config['reward_training_batch_size'] = reward_training_batch_size
    if reward_training_texts:
        import json
        with open(reward_training_texts, 'r') as f:
            train_config['reward_training_texts'] = json.load(f)
    
    # Value model config
    if value_model_num_labels: train_config['value_model_num_labels'] = value_model_num_labels
    
    # Rewards config
    if rewards:
        if Path(rewards).exists():
            import json
            with open(rewards, 'r') as f:
                train_config['rewards'] = json.load(f)
        else:
            # Parse inline format
            rewards_list = []
            for reward_spec in rewards.split(','):
                parts = reward_spec.strip().split(':')
                if len(parts) == 2:
                    rtype, weight = parts
                    reward_entry = {
                        "type": rtype.strip(),
                        "weight": float(weight.strip()),
                    }
                    rewards_list.append(reward_entry)
                else:
                    typer.echo(f"Warning: Invalid reward format '{reward_spec}', expected 'type:weight'")
            
            if reward_params and Path(reward_params).exists():
                import json
                with open(reward_params, 'r') as f:
                    params_dict = json.load(f)
                for reward_entry in rewards_list:
                    rtype = reward_entry["type"]
                    if rtype in params_dict:
                        reward_entry["params"] = params_dict[rtype]
            
            train_config['rewards'] = rewards_list
    
    # Eval/Save intervals
    if eval_interval: train_config['eval_interval'] = eval_interval
    if save_interval: train_config['save_interval'] = save_interval
    if eval_steps: train_config['eval_steps'] = eval_steps
    if save_steps: train_config['save_steps'] = save_steps
    if save_total_limit: train_config['save_total_limit'] = save_total_limit
    if save_strategy: train_config['save_strategy'] = save_strategy
    if eval_strategy: train_config['eval_strategy'] = eval_strategy
    if logging_strategy: train_config['logging_strategy'] = logging_strategy
    
    # Seeds
    if seed: train_config['seed'] = seed
    if data_seed: train_config['data_seed'] = data_seed
    
    if train_config:
        cli_config['train'] = train_config
    
    # Logging config
    logging_config = {}
    if output_dir: logging_config['output_dir'] = output_dir
    if run_name: logging_config['run_name'] = run_name
    if log_level: logging_config['log_level'] = log_level
    if loggers: logging_config['loggers'] = loggers.split(',')
    if log_interval: logging_config['log_interval'] = log_interval
    if logging_steps: logging_config['logging_steps'] = logging_steps
    if report_to: logging_config['report_to'] = report_to
    if wandb_project: 
        if 'wandb' not in logging_config:
            logging_config['wandb'] = {}
        logging_config['wandb']['project'] = wandb_project
    
    # Sample logging
    sample_logging_config = {}
    if enable_sample_logging is not None: sample_logging_config['enabled'] = enable_sample_logging
    if sample_logging_prompts: sample_logging_config['prompts'] = sample_logging_prompts.split(',')
    if sample_logging_interval_steps: sample_logging_config['interval_steps'] = sample_logging_interval_steps
    if sample_logging_percent_of_max_steps: sample_logging_config['percent_of_max_steps'] = sample_logging_percent_of_max_steps
    if sample_logging_max_new_tokens: sample_logging_config['max_new_tokens'] = sample_logging_max_new_tokens
    if sample_logging_temperature: sample_logging_config['temperature'] = sample_logging_temperature
    if sample_logging_top_p: sample_logging_config['top_p'] = sample_logging_top_p
    if sample_logging_num_samples: sample_logging_config['num_samples'] = sample_logging_num_samples
    
    if sample_logging_config:
        logging_config['sample_logging'] = sample_logging_config
    
    if logging_config:
        cli_config['logging'] = logging_config
    
    # Evaluation config (SFT)
    evaluation_config = {}
    if compute_perplexity is not None: evaluation_config['compute_perplexity'] = compute_perplexity
    if compute_rouge is not None: evaluation_config['compute_rouge'] = compute_rouge
    if compute_bleu is not None: evaluation_config['compute_bleu'] = compute_bleu
    if compute_meteor is not None: evaluation_config['compute_meteor'] = compute_meteor
    if compute_bertscore is not None: evaluation_config['compute_bertscore'] = compute_bertscore
    if compute_semantic_similarity is not None: evaluation_config['compute_semantic_similarity'] = compute_semantic_similarity
    if compute_codebleu is not None: evaluation_config['compute_codebleu'] = compute_codebleu
    if max_samples_for_quality_metrics: evaluation_config['max_samples_for_quality_metrics'] = max_samples_for_quality_metrics
    if bertscore_model: evaluation_config['bertscore_model'] = bertscore_model
    if semantic_similarity_model: evaluation_config['semantic_similarity_model'] = semantic_similarity_model
    
    if evaluation_config:
        cli_config['evaluation'] = evaluation_config
    
    # Other top-level configs
    if training_type: cli_config['algo'] = training_type.lower()
    if chat_template: cli_config['chat_template'] = chat_template
    if cache_dir: 
        cli_config['caching'] = {'root': cache_dir, 'enabled': True}
    if distributed_backend:
        cli_config['distributed'] = {'backend': distributed_backend}
    
    # Merge configs (CLI overrides YAML)
    if cli_config:
        for key, value in cli_config.items():
            if key in final_config and isinstance(value, dict) and isinstance(final_config[key], dict):
                final_config[key].update(value)
            else:
                final_config[key] = value
    
    # Validate required fields
    if 'model' not in final_config or 'name_or_path' not in final_config['model']:
        typer.echo("Error: Model name is required (--model or in config)")
        raise typer.Exit(1)
    
    # Handle datasets format
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
            backend_enum = BackendType.TRL
        else:
            try:
                backend_enum = BackendType(backend_str.lower())
            except ValueError:
                typer.echo(f"Error: Invalid backend '{backend_str}'. Must be: auto, trl, unsloth")
                raise typer.Exit(1)
        
        # Extract config components
        model_cfg = final_config.get('model', {})
        dataset_cfg = final_config['datasets'][0]
        train_cfg = final_config.get('train', {})
        logging_cfg = final_config.get('logging', {})
        eval_cfg = final_config.get('evaluation', {})
        
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
        
        # Create trainer
        if algo == 'sft':
            trainer = create_sft_trainer(
                model_name=model_cfg['name_or_path'],
                dataset_name=dataset_cfg['name'],
                backend=backend,
                output_dir=logging_cfg.get('output_dir', './output'),
                num_epochs=train_cfg.get('epochs', 3),
                batch_size=train_cfg.get('per_device_batch_size', 4),
                learning_rate=train_cfg.get('learning_rate', 2e-4),
                max_seq_length=model_cfg.get('max_seq_length', 512),
                max_samples=dataset_cfg.get('max_samples'),
                system_prompt=dataset_cfg.get('system_prompt'),
                config=None,  # Already processed
                # Pass ALL parameters from config
                **{**model_cfg, **dataset_cfg, **train_cfg, **logging_cfg, **eval_cfg}
            )
        else:
            trainer = create_rl_trainer(
                model_name=model_cfg['name_or_path'],
                dataset_name=dataset_cfg['name'],
                algorithm=algo,
                backend=backend,
                output_dir=logging_cfg.get('output_dir', './output'),
                num_epochs=train_cfg.get('epochs', 100),
                max_steps=train_cfg.get('max_steps'),
                batch_size=train_cfg.get('per_device_batch_size', 4),
                learning_rate=train_cfg.get('learning_rate', 2e-4),
                max_seq_length=model_cfg.get('max_seq_length', 512),
                max_samples=dataset_cfg.get('max_samples'),
                system_prompt=dataset_cfg.get('system_prompt'),
                config=None,  # Already processed
                # Pass ALL parameters from config
                **{**model_cfg, **dataset_cfg, **train_cfg, **logging_cfg}
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
