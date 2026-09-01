"""
CLI command for cost estimation and algorithm recommendations.

This module provides the 'advise' subcommand for estimating training resources
and recommending algorithms.
"""

import typer
import logging
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.advisor import (
    estimate_resources,
    recommend_algorithm,
    suggest_optimizations,
    format_estimate_table,
    format_recommendations,
    format_optimizations,
    GPU_PROFILES,
    REGION_CARBON_INTENSITY,
)

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="advise",
    help="Get cost estimates and algorithm recommendations",
    add_completion=False,
    rich_markup_mode="rich",
)


def list_gpus_command():
    """List available GPU profiles."""
    console.print("\n[bold cyan]Available GPU Profiles:[/bold cyan]\n")
    table = Table(title="GPU Profiles", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("VRAM (GB)", style="green", justify="right")
    table.add_column("TFLOPS FP16", style="yellow", justify="right")
    table.add_column("Price/Hour (USD)", style="magenta", justify="right")
    table.add_column("Power (W)", style="red", justify="right")

    for gpu_id in sorted(GPU_PROFILES.keys()):
        gpu = GPU_PROFILES[gpu_id]
        table.add_row(
            gpu.name,
            f"{gpu.vram_gb:.0f}",
            f"{gpu.tflops_fp16:.0f}",
            f"${gpu.price_per_hour_usd:.2f}",
            f"{gpu.power_consumption_watts:.0f}",
        )

    console.print(table)
    console.print("")


@app.command()
def estimate(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., 'Qwen/Qwen2.5-7B')"),
    dataset_size: int = typer.Option(..., "--dataset-size", "-d", help="Number of training samples"),
    algorithm: str = typer.Option("sft", "--algorithm", "-a", help="Training algorithm (sft, dpo, ppo, lora, qlora, etc.)"),
    gpu: str = typer.Option("a100-40gb", "--gpu", "-g", help="GPU hardware profile"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
    seq_len: int = typer.Option(512, "--seq-len", help="Sequence length in tokens"),
    num_epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    gradient_accumulation: int = typer.Option(1, "--grad-accum", help="Gradient accumulation steps"),
    region: str = typer.Option("default", "--region", "-r", help=(
        "Cloud region for carbon intensity estimate "
        "(e.g., us-east-1, us-west-2, eu-west-1). "
        f"Supported: {', '.join(sorted(REGION_CARBON_INTENSITY.keys()))}"
    )),
    num_gpus: int = typer.Option(1, "--num-gpus", "-n", help="Number of GPUs (used for carbon estimation)"),
):
    """
    Estimate training resources (VRAM, time, cost, carbon).

    Examples:

        # Estimate DPO training for Qwen2.5-7B on 10k samples with A100-40GB
        aligntune advise estimate --model "Qwen/Qwen2.5-7B" --dataset-size 10000 --algorithm dpo

        # Estimate LoRA fine-tuning with custom batch size and region
        aligntune advise estimate --model "meta-llama/Llama-2-70b-hf" --dataset-size 50000 \\
            --algorithm lora --batch-size 8 --gpu h100 --region us-west-2

        # List available GPUs
        aligntune advise list-gpus
    """
    try:
        logger.info(f"Estimating resources for {model} with {algorithm}")

        estimate_result = estimate_resources(
            model_name=model,
            dataset_size=dataset_size,
            algorithm=algorithm,
            hardware_profile=gpu,
            batch_size=batch_size,
            seq_len=seq_len,
            num_epochs=num_epochs,
            gradient_accumulation=gradient_accumulation,
            region=region,
            num_gpus=num_gpus,
        )

        # Format and display results
        console.print("")
        console.print(Panel(
            format_estimate_table(model, dataset_size, algorithm, gpu, estimate_result, region=region),
            title="[bold cyan]Resource Estimate[/bold cyan]",
            expand=False,
        ))

        # Carbon summary line
        if estimate_result.carbon is not None:
            c = estimate_result.carbon
            console.print(
                f"[dim]Carbon:[/dim] ~{c.co2_grams:.1f}g CO2 (~{c.kwh:.3f} kWh) "
                f"[dim]({region}, {c.intensity} gCO2eq/kWh)[/dim]\n"
            )

        # Check if estimate fits on GPU
        gpu_profile = GPU_PROFILES.get(gpu.lower())
        if gpu_profile and estimate_result.vram_gb > gpu_profile.vram_gb:
            console.print(f"[bold red]Warning:[/bold red] Estimate exceeds {gpu_profile.name} VRAM ({gpu_profile.vram_gb:.0f}GB)")
            console.print(f"  Consider using a larger GPU or LoRA/QLoRA for memory efficiency\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def recommend(
    task: str = typer.Option(..., "--task", "-t", help="Task description (e.g., 'alignment', 'speed', 'distill')"),
    dataset_size: int = typer.Option(..., "--dataset-size", "-d", help="Number of training samples"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Optional budget in USD"),
    model_size: Optional[str] = typer.Option(None, "--model-size", "-m", help="Model size hint (e.g., '7b', '70b')"),
):
    """
    Recommend algorithms based on task and constraints.

    Examples:

        # Recommend algorithms for alignment task with 10k samples
        aligntune advise recommend --task alignment --dataset-size 10000

        # Recommend algorithms with budget constraint
        aligntune advise recommend --task "speed" --dataset-size 50000 --budget 10.0

        # Recommend for large model
        aligntune advise recommend --task general --dataset-size 100000 --model-size 70b
    """
    try:
        logger.info(f"Recommending algorithms for task: {task}")

        recommendations = recommend_algorithm(
            task_description=task,
            dataset_size=dataset_size,
            budget_usd=budget,
            model_size=model_size,
        )

        # Format and display results
        console.print("")
        console.print(Panel(
            format_recommendations(recommendations),
            title="[bold cyan]Algorithm Recommendations[/bold cyan]",
            expand=False,
        ))

        if budget:
            console.print(f"[dim]Filtered by budget: ${budget:.2f}[/dim]\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def optimize(
    model_size: str = typer.Option(..., "--model-size", "-m", help="Model size (e.g., '7b', '70b')"),
    precision: str = typer.Option("fp32", "--precision", "-p", help="Training precision (fp32, fp16, bf16, int8, int4)"),
    gpu: str = typer.Option("a100-40gb", "--gpu", "-g", help="GPU hardware profile"),
    dataset_size: int = typer.Option(10000, "--dataset-size", "-d", help="Number of training samples"),
    vram_tight: bool = typer.Option(False, "--vram-tight", help="Flag if VRAM is constrained"),
):
    """
    Get optimization suggestions for training.

    Examples:

        # Get suggestions for 7B model on A100
        aligntune advise optimize --model-size 7b --precision fp32 --gpu a100-40gb

        # Get suggestions with tight VRAM constraints
        aligntune advise optimize --model-size 70b --vram-tight --dataset-size 50000

        # Get suggestions for fast training
        aligntune advise optimize --model-size 13b --precision bf16 --gpu h100
    """
    try:
        logger.info(f"Generating optimization suggestions for {model_size} model")

        suggestions = suggest_optimizations(
            model_size=model_size,
            precision=precision,
            hardware=gpu,
            dataset_size=dataset_size,
            vram_tight=vram_tight,
        )

        # Format and display results
        console.print("")
        console.print(Panel(
            format_optimizations(suggestions),
            title="[bold cyan]Optimization Suggestions[/bold cyan]",
            expand=False,
        ))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def list_gpus():
    """List available GPU profiles with specs and pricing."""
    list_gpus_command()
