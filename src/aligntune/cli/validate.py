"""
Configuration validation CLI commands.

This module provides commands to validate configurations, check system compatibility,
and diagnose potential training issues.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.validation import (
    validate_config, estimate_memory_usage, get_system_info,
    validate_model_access, validate_dataset_access, ConfigValidator
)
from ..utils.diagnostics import run_comprehensive_diagnostics, run_config_validation
from ..utils.errors import handle_error, format_validation_errors
from ..core.sft.config_loader import SFTConfigLoader
from ..core.rl.config_loader import ConfigLoader

console = Console()
app = typer.Typer(
    name="validate",
    help="🔍 Validate configurations and check system compatibility",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command()
def config(
    config_file: str = typer.Argument(..., help="Path to configuration YAML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation output"),
    check_access: bool = typer.Option(True, "--check-access/--no-check-access", help="Check model/dataset accessibility")
):
    """
    Validate a training configuration file.

    Examples:

    # Basic validation
    aligntune validate config my_config.yaml

    # Detailed validation with access checks
    aligntune validate config my_config.yaml --verbose --check-access
    """

    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"[red]❌ Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading configuration...", total=None)

            # Try to load as different config types
            config = None
            config_type = None

            try:
                # First try to load directly (for config-only files)
                config = SFTConfigLoader.load_from_yaml(config_path)
                config_type = "SFT"
            except Exception:
                try:
                    # Try RL config loader
                    config = ConfigLoader.load_from_yaml(config_path)
                    config_type = "RL"
                except Exception:
                    try:
                        # Try loading as recipe file (with recipe and config sections)
                        import yaml
                        with open(config_path, 'r') as f:
                            data = yaml.safe_load(f)

                        if 'config' in data:
                            # This is a recipe file, extract the config section
                            config_data = data['config']
                            # Determine config type from recipe metadata or config content
                            if 'algo' in config_data:
                                config = ConfigLoader.load_from_dict(config_data)
                                config_type = "RL"
                            else:
                                config = SFTConfigLoader.load_from_dict(config_data)
                                config_type = "SFT"
                        else:
                            raise ValueError("No config section found in recipe file")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
                        raise typer.Exit(1)

            progress.update(task, description="Validating configuration...")

            # Run validation
            is_valid, errors, warnings = validate_config(config)

            progress.update(task, description="✅ Validation complete")

        # Display results
        console.print(f"\n[bold]Configuration Validation Results[/bold]")
        console.print(f"Config file: {config_path}")
        console.print(f"Config type: {config_type}")

        if is_valid:
            console.print("[green]✅ Configuration is valid[/green]")
        else:
            console.print("[red]❌ Configuration has errors[/red]")

        # Show errors and warnings
        validation_output = format_validation_errors(errors, warnings)
        console.print(validation_output)

        # Additional checks
        if check_access and config_type == "SFT":
            console.print(f"\n[bold]Accessibility Checks[/bold]")

            # Check model access
            try:
                is_accessible, message = validate_model_access(config.model.name_or_path)
                status = "[green]✅[/green]" if is_accessible else "[red]❌[/red]"
                console.print(f"Model access: {status} {message}")
            except Exception as e:
                console.print(f"Model access: [red]❌ Error checking access: {e}[/red]")

            # Check dataset access
            try:
                is_accessible, message = validate_dataset_access(config.dataset.name)
                status = "[green]✅[/green]" if is_accessible else "[red]❌[/red]"
                console.print(f"Dataset access: {status} {message}")
            except Exception as e:
                console.print(f"Dataset access: [red]❌ Error checking access: {e}[/red]")

        # Memory estimate
        console.print(f"\n[bold]Memory Estimate[/bold]")
        memory_gb = estimate_memory_usage(config)
        system_info = get_system_info()

        console.print(f"Estimated memory usage: {memory_gb:.1f} GB")
        if system_info['gpu_count'] > 0:
            gpu_memory = max(system_info['gpu_memory_gb'])
            if memory_gb > gpu_memory:
                console.print(f"[red]⚠️  Estimated memory ({memory_gb:.1f} GB) exceeds GPU capacity ({gpu_memory:.1f} GB)[/red]")
            else:
                console.print(f"[green]✅ Fits within GPU memory ({gpu_memory:.1f} GB available)[/green]")
        else:
            available_memory = system_info['available_memory_gb']
            if memory_gb > available_memory:
                console.print(f"[red]⚠️  Estimated memory ({memory_gb:.1f} GB) exceeds available RAM ({available_memory:.1f} GB)[/red]")
            else:
                console.print(f"[green]✅ Fits within available RAM ({available_memory:.1f} GB available)[/green]")

        if verbose and warnings:
            console.print(f"\n[bold]Detailed Warnings:[/bold]")
            for warning in warnings:
                console.print(f"  • {warning}")

        if errors:
            raise typer.Exit(1)

    except Exception as e:
        handle_error(e, verbose=verbose)


@app.command()
def system(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed system information")
):
    """
    Check system compatibility and show diagnostics.

    Example:
    aligntune validate system --detailed
    """

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running system diagnostics...", total=None)
            diagnostics = run_comprehensive_diagnostics()
            progress.update(task, description="✅ Diagnostics complete")

        # Display system info
        console.print(f"\n[bold]🖥️  System Information[/bold]")

        sys_info = diagnostics['system_info']

        # CPU info
        console.print(f"CPU Cores: {sys_info['cpu_count']}")

        # Memory info
        console.print(f"Total RAM: {sys_info['memory_total_gb']:.1f} GB")
        console.print(f"Available RAM: {sys_info['memory_available_gb']:.1f} GB")

        # GPU info
        if sys_info['gpu_available']:
            console.print(f"GPU Devices: {sys_info['gpu_count']}")
            for i, gpu in enumerate(sys_info['gpu_memory']):
                console.print(f"  GPU {i}: {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)")
                if detailed:
                    console.print(f"    Compute capability: {gpu['compute_capability']}")

            console.print(f"CUDA version: {sys_info.get('cuda_version', 'N/A')}")
        else:
            console.print("[yellow]GPU: Not available[/yellow]")

        # PyTorch info
        console.print(f"PyTorch version: {diagnostics['libraries']['torch']['version']}")

        # Library status
        console.print(f"\n[bold]📚 Library Status[/bold]")

        libs = diagnostics['libraries']
        for lib_name, lib_info in libs.items():
            if lib_name == 'torch':
                continue
            status = "[green]✅ Available[/green]" if lib_info.get('library_available', False) else "[red]❌ Missing[/red]"
            console.print(f"{lib_name.capitalize()}: {status}")

        # CUDA functionality
        if diagnostics.get('cuda_info'):
            cuda_info = diagnostics['cuda_info']
            console.print(f"\n[bold]🎮 CUDA Status[/bold]")

            functional = cuda_info.get('cuda_functional', False)
            status = "[green]✅ Functional[/green]" if functional else "[red]❌ Issues detected[/red]"
            console.print(f"CUDA functionality: {status}")

            if not functional and cuda_info.get('error'):
                console.print(f"Error: {cuda_info['error']}")

            if cuda_info.get('tensor_ops_working', False):
                console.print("[green]Tensor operations: Working[/green]")
            else:
                console.print("[red]Tensor operations: Failed[/red]")

        # Recommendations
        console.print(f"\n[bold]💡 Recommendations[/bold]")

        recommendations = []

        if not sys_info['gpu_available']:
            recommendations.append("Consider using a GPU-enabled machine for faster training")

        available_memory = sys_info['available_memory_gb']
        if available_memory < 16:
            recommendations.append("Consider upgrading RAM for better performance")

        if not diagnostics['libraries']['transformers'].get('library_available'):
            recommendations.append("Install transformers: pip install transformers")

        if not diagnostics['libraries']['datasets'].get('library_available'):
            recommendations.append("Install datasets: pip install datasets")

        if recommendations:
            for rec in recommendations:
                console.print(f"  • {rec}")
        else:
            console.print("[green]  • System looks good for training![/green]")

        if detailed:
            console.print(f"\n[bold]🔧 Detailed Diagnostics[/bold]")
            import json
            console.print(json.dumps(diagnostics, indent=2, default=str))

    except Exception as e:
        handle_error(e, verbose=detailed)


@app.command()
def model(
    model_name: str = typer.Argument(..., help="Model name to validate"),
    check_download: bool = typer.Option(False, "--check-download", help="Actually attempt to download the model")
):
    """
    Validate model accessibility and compatibility.

    Examples:

    # Quick access check
    aligntune validate model meta-llama/Llama-3-8B-Instruct

    # Full download test
    aligntune validate model meta-llama/Llama-3-8B-Instruct --check-download
    """

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Checking model {model_name}...", total=None)

            is_accessible, message = validate_model_access(model_name)

            if check_download:
                progress.update(task, description="Attempting to download model...")
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                    message += " (Successfully downloaded)"
                    progress.update(task, description="✅ Model downloaded successfully")
                except Exception as e:
                    is_accessible = False
                    message = f"Download failed: {str(e)}"
                    progress.update(task, description="❌ Download failed")

            progress.update(task, description="✅ Validation complete")

        # Display results
        console.print(f"\n[bold]🤖 Model Validation: {model_name}[/bold]")

        if is_accessible:
            console.print("[green]✅ Model is accessible[/green]")
        else:
            console.print("[red]❌ Model is not accessible[/red]")

        console.print(f"Details: {message}")

        # Additional info
        if "requires authentication" in message.lower():
            console.print(f"\n[yellow]💡 To access this model:[/yellow]")
            console.print("  1. Run: huggingface-cli login")
            console.print("  2. Make sure you have access to the model (some are gated)")
            console.print("  3. Check your token permissions")

    except Exception as e:
        handle_error(e)


@app.command()
def dataset(
    dataset_name: str = typer.Argument(..., help="Dataset name to validate"),
    check_download: bool = typer.Option(False, "--check-download", help="Actually attempt to download the dataset")
):
    """
    Validate dataset accessibility and compatibility.

    Examples:

    # Quick access check
    aligntune validate dataset anthropic/hh-rlhf

    # Full download test
    aligntune validate dataset anthropic/hh-rlhf --check-download
    """

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Checking dataset {dataset_name}...", total=None)

            is_accessible, message = validate_dataset_access(dataset_name)

            if check_download:
                progress.update(task, description="Attempting to download dataset...")
                try:
                    from datasets import load_dataset
                    dataset = load_dataset(dataset_name, split="train[:1%]")
                    message += f" (Successfully downloaded {len(dataset)} samples)"
                    progress.update(task, description="✅ Dataset downloaded successfully")
                except Exception as e:
                    is_accessible = False
                    message = f"Download failed: {str(e)}"
                    progress.update(task, description="❌ Download failed")

            progress.update(task, description="✅ Validation complete")

        # Display results
        console.print(f"\n[bold]📊 Dataset Validation: {dataset_name}[/bold]")

        if is_accessible:
            console.print("[green]✅ Dataset is accessible[/green]")
        else:
            console.print("[red]❌ Dataset is not accessible[/red]")

        console.print(f"Details: {message}")

        # Additional info
        if "requires authentication" in message.lower():
            console.print(f"\n[yellow]💡 To access this dataset:[/yellow]")
            console.print("  1. Run: huggingface-cli login")
            console.print("  2. Make sure you have access to the dataset (some are gated)")
            console.print("  3. Check your token permissions")

    except Exception as e:
        handle_error(e)


@app.command()
def memory(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file to analyze"),
    model_name: Optional[str] = typer.Option(None, "--model", help="Model name for estimation"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size for estimation"),
    seq_length: int = typer.Option(2048, "--seq-length", help="Sequence length for estimation")
):
    """
    Estimate memory requirements for training.

    Examples:

    # Estimate from config
    aligntune validate memory --config my_config.yaml

    # Manual estimation
    aligntune validate memory --model meta-llama/Llama-3-8B --batch-size 8 --seq-length 4096
    """

    try:
        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                console.print(f"[red]❌ Configuration file not found: {config_file}[/red]")
                raise typer.Exit(1)

            # Load config
            try:
                config = SFTConfigLoader.load_from_yaml(config_path)
            except Exception:
                config = ConfigLoader.load_from_yaml(config_path)

            memory_gb = estimate_memory_usage(config)

            console.print(f"\n[bold]💾 Memory Estimation from Config[/bold]")
            console.print(f"Config: {config_path}")
            console.print(f"Estimated Memory: {memory_gb:.1f} GB")

        elif model_name:
            # Manual estimation
            validator = ConfigValidator()

            # Rough estimation based on model size
            if 'llama-3' in model_name.lower():
                if '8b' in model_name.lower():
                    model_memory = 16.0
                elif '70b' in model_name.lower():
                    model_memory = 140.0
                else:
                    model_memory = 8.0
            elif 'qwen' in model_name.lower():
                model_memory = 8.0
            else:
                model_memory = 4.0  # Default

            # Add batch and sequence effects
            batch_memory = batch_size * seq_length * 0.000002  # Rough token memory
            gradient_memory = model_memory * 2  # Gradients
            optimizer_memory = model_memory * 2  # Optimizer states

            memory_gb = model_memory + batch_memory + gradient_memory + optimizer_memory

            console.print(f"\n[bold]💾 Manual Memory Estimation[/bold]")
            console.print(f"Model: {model_name}")
            console.print(f"Batch size: {batch_size}")
            console.print(f"Sequence length: {seq_length}")
            console.print(f"Estimated Memory: {memory_gb:.1f} GB")

        else:
            console.print("[red]❌ Either --config or --model must be specified[/red]")
            raise typer.Exit(1)

        # System comparison
        system_info = get_system_info()

        console.print(f"\n[bold]System Resources[/bold]")
        console.print(f"Available RAM: {system_info['available_memory_gb']:.1f} GB")

        if system_info['gpu_count'] > 0:
            gpu_memory = max(system_info['gpu_memory_gb'])
            console.print(f"GPU Memory: {gpu_memory:.1f} GB")

            if memory_gb > gpu_memory:
                console.print("[red]❌ Estimated memory exceeds GPU capacity![/red]")
                console.print("Consider:")
                console.print("  • Using PEFT (LoRA)")
                console.print("  • Reducing batch size")
                console.print("  • Enabling gradient checkpointing")
                console.print("  • Using a smaller model")
            else:
                utilization = (memory_gb / gpu_memory) * 100
                console.print(f"[green]✅ Fits within GPU memory ({utilization:.1f}% utilization)[/green]")
        else:
            console.print("[yellow]⚠️  No GPU detected - CPU training will be very slow[/yellow]")

    except Exception as e:
        handle_error(e)


if __name__ == "__main__":
    app()
