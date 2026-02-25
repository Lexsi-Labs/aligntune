"""
CLI for evaluation system.

This module provides command-line interface for running evaluations
on trained models using various benchmarks and metrics.
"""

import typer
from typing import List, Optional
import logging
from pathlib import Path
import torch

from .core import EvalConfig, EvalType, TaskCategory, EvalRunner
from .lm_eval_integration import LMEvalConfig, LMEvalRunner, get_available_lm_eval_tasks
from .registry import EvaluationRegistry

app = typer.Typer(name="eval", help="Model evaluation commands")

logger = logging.getLogger(__name__)


@app.command()
def benchmark(
    model_name: str = typer.Argument(..., help="Model name or path to evaluate"),
    tasks: Optional[List[str]] = typer.Option(None, "--task", "-t", help="Specific tasks to run"),
    output_dir: str = typer.Option("./eval_results", "--output-dir", "-o", help="Output directory for results"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size for evaluation"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of samples"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cpu, cuda)"),
    verbose: bool = typer.Option(False, "--verbose/--quiet", help="Verbose output"),
):
    """Run standard benchmark evaluation on a model."""
    typer.echo(f"🔍 Running benchmark evaluation on model: {model_name}")
    
    if tasks is None:
        # Use default benchmark tasks
        tasks = ["hellaswag", "arc_challenge", "mmlu", "gsm8k", "human_eval"]
        typer.echo(f"Using default benchmark tasks: {tasks}")
    
    # Check if tasks are lm-eval tasks
    available_lm_eval = get_available_lm_eval_tasks()
    lm_eval_tasks = [task for task in tasks if task in available_lm_eval]
    custom_tasks = [task for task in tasks if task not in available_lm_eval]
    
    results = []
    
    # Run lm-eval tasks
    if lm_eval_tasks:
        typer.echo(f"Running lm-eval tasks: {lm_eval_tasks}")
        config = LMEvalConfig(
            model_name=model_name,
            batch_size=batch_size,
            limit=limit,
            output_dir=output_dir,
            verbose=verbose
        )
        runner = LMEvalRunner(config)
        
        from .lm_eval_integration import get_lm_eval_task
        lm_eval_task_objects = [get_lm_eval_task(task) for task in lm_eval_tasks]
        lm_eval_results = runner.evaluate_tasks(lm_eval_task_objects)
        results.extend(lm_eval_results)
    
    # Run custom tasks
    if custom_tasks:
        typer.echo(f"Running custom tasks: {custom_tasks}")
        # This would require loading the model and tokenizer
        typer.echo("Custom task evaluation not yet implemented")
    
    # Display results
    typer.echo("\n📊 Evaluation Results:")
    typer.echo("=" * 50)
    
    for result in results:
        typer.echo(f"\nTask: {result.task_name}")
        typer.echo(f"Category: {result.category.value}")
        typer.echo(f"Samples: {result.num_samples}")
        typer.echo(f"Time: {result.eval_time:.2f}s")
        typer.echo("Metrics:")
        for metric, value in result.metrics.items():
            typer.echo(f"  {metric}: {value:.4f}")
    
    typer.echo(f"\n✅ Evaluation completed! Results saved to: {output_dir}")


@app.command()
def task(
    model_name: str = typer.Argument(..., help="Model name or path to evaluate"),
    task_name: str = typer.Argument(..., help="Name of the evaluation task"),
    output_dir: str = typer.Option("./eval_results", "--output-dir", "-o", help="Output directory for results"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for evaluation"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-m", help="Maximum number of samples"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cpu, cuda)"),
    precision: str = typer.Option("bf16", "--precision", "-p", help="Precision (fp16, bf16, fp32)"),
    save_predictions: bool = typer.Option(True, "--save-predictions/--no-save-predictions", help="Save predictions"),
    verbose: bool = typer.Option(False, "--verbose/--quiet", help="Verbose output"),
):
    """Run evaluation on a specific task."""
    typer.echo(f"🔍 Running evaluation on task: {task_name}")
    
    try:
        # Get the task
        task = EvaluationRegistry.get_task(task_name)
        
        # Create evaluation config
        config = EvalConfig(
            eval_type=EvalType.STANDALONE,
            task_categories=[task.category],
            batch_size=batch_size,
            max_samples=max_samples,
            device=device,
            precision=precision,
            output_dir=output_dir,
            save_predictions=save_predictions,
            verbose=verbose
        )
        
        # Load model and tokenizer
        typer.echo("Loading model and tokenizer...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, precision) if hasattr(torch, precision) else torch.float16,
            device_map=device if device != "auto" else None
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Run evaluation
        runner = EvalRunner(config)
        result = runner.evaluate_task(task, model, tokenizer)
        
        # Display results
        typer.echo("\n📊 Evaluation Results:")
        typer.echo("=" * 50)
        typer.echo(f"Task: {result.task_name}")
        typer.echo(f"Category: {result.category.value}")
        typer.echo(f"Samples: {result.num_samples}")
        typer.echo(f"Time: {result.eval_time:.2f}s")
        typer.echo("Metrics:")
        for metric, value in result.metrics.items():
            typer.echo(f"  {metric}: {value:.4f}")
        
        typer.echo(f"\n✅ Evaluation completed! Results saved to: {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_tasks():
    """List all available evaluation tasks."""
    typer.echo("📋 Available Evaluation Tasks:")
    typer.echo("=" * 50)
    
    # Custom tasks
    custom_tasks = EvaluationRegistry.list_tasks()
    if custom_tasks:
        typer.echo("\nCustom Tasks:")
        for task_name in custom_tasks:
            task = EvaluationRegistry.get_task(task_name)
            typer.echo(f"  {task_name}: {task.description}")
    
    # lm-eval tasks
    lm_eval_tasks = get_available_lm_eval_tasks()
    if lm_eval_tasks:
        typer.echo("\nlm-eval Tasks:")
        for task_name in lm_eval_tasks:
            typer.echo(f"  {task_name}")
    
    typer.echo(f"\nTotal: {len(custom_tasks) + len(lm_eval_tasks)} tasks available")


@app.command()
def list_metrics():
    """List all available evaluation metrics."""
    typer.echo("📊 Available Evaluation Metrics:")
    typer.echo("=" * 50)
    
    metrics = EvaluationRegistry.list_metrics()
    for metric in metrics:
        typer.echo(f"  {metric}")
    
    typer.echo(f"\nTotal: {len(metrics)} metrics available")


@app.command()
def compare(
    models: List[str] = typer.Argument(..., help="Models to compare"),
    tasks: Optional[List[str]] = typer.Option(None, "--task", "-t", help="Tasks to run"),
    output_dir: str = typer.Option("./eval_comparison", "--output-dir", "-o", help="Output directory"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit samples"),
):
    """Compare multiple models on the same tasks."""
    typer.echo(f"🔄 Comparing models: {models}")
    
    if tasks is None:
        tasks = ["hellaswag", "arc_challenge", "mmlu"]
    
    all_results = {}
    
    for model in models:
        typer.echo(f"\nEvaluating model: {model}")
        
        # Run evaluation for this model
        config = LMEvalConfig(
            model_name=model,
            batch_size=batch_size,
            limit=limit,
            output_dir=f"{output_dir}/{model.replace('/', '_')}",
            verbose=False
        )
        runner = LMEvalRunner(config)
        
        from .lm_eval_integration import get_lm_eval_task
        task_objects = [get_lm_eval_task(task) for task in tasks if task in get_available_lm_eval_tasks()]
        results = runner.evaluate_tasks(task_objects)
        
        all_results[model] = results
    
    # Display comparison
    typer.echo("\n📊 Model Comparison Results:")
    typer.echo("=" * 80)
    
    for task in tasks:
        if task in get_available_lm_eval_tasks():
            typer.echo(f"\nTask: {task}")
            typer.echo("-" * 40)
            
            for model, results in all_results.items():
                task_result = next((r for r in results if r.task_name == task), None)
                if task_result:
                    typer.echo(f"{model:30} | ", end="")
                    for metric, value in task_result.metrics.items():
                        typer.echo(f"{metric}: {value:.4f} ", end="")
                    typer.echo()
    
    typer.echo(f"\n✅ Comparison completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    app()
