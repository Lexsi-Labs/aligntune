"""
Common utilities for AlignTune example scripts.

This module extracts shared functionality to reduce boilerplate in example scripts.
Includes: argument parsing, model/dataset resolution, config loading, and banners.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


def parse_args(
    description: str = "AlignTune Example Script",
    add_dataset: bool = True,
    add_backend: bool = True,
    add_config: bool = True,
    add_output: bool = True,
) -> argparse.Namespace:
    """
    Parse common command-line arguments for example scripts.

    Args:
        description: Script description for help text
        add_dataset: Include --dataset argument
        add_backend: Include --backend argument (trl, unsloth)
        add_config: Include --config argument for YAML config file
        add_output: Include --output-dir argument

    Returns:
        Parsed arguments as Namespace object
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (e.g., google/gemma-2-2b)",
    )

    if add_dataset:
        parser.add_argument(
            "--dataset",
            type=str,
            default=None,
            help="Dataset name or path (e.g., Anthropic/hh-rlhf)",
        )

    if add_backend:
        parser.add_argument(
            "--backend",
            type=str,
            default="trl",
            choices=["trl", "unsloth"],
            help="Training backend to use",
        )

    if add_config:
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML configuration file",
        )

    if add_output:
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./output",
            help="Output directory for model and logs",
        )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def resolve_model_and_dataset(
    model_arg: Optional[str] = None,
    dataset_arg: Optional[str] = None,
    config_path: Optional[str] = None,
) -> tuple:
    """
    Resolve model and dataset names from arguments or config file.

    Precedence: CLI args > config file > defaults

    Args:
        model_arg: Model name from --model argument
        dataset_arg: Dataset name from --dataset argument
        config_path: Path to YAML config file

    Returns:
        Tuple of (model_name, dataset_name)

    Raises:
        ValueError: If model or dataset cannot be resolved
    """
    model_name = model_arg
    dataset_name = dataset_arg

    # Load from config if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if config:
                if not model_name and "model" in config:
                    model_name = config["model"].get("name_or_path", model_name)
                if not dataset_name and "datasets" in config:
                    datasets = config["datasets"]
                    if isinstance(datasets, list) and len(datasets) > 0:
                        dataset_name = datasets[0].get("name", dataset_name)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    if not model_name:
        raise ValueError(
            "Model name must be provided via --model or config file"
        )
    if not dataset_name:
        raise ValueError(
            "Dataset name must be provided via --dataset or config file"
        )

    return model_name, dataset_name


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def print_banner(
    title: str,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    extra_info: Optional[Dict[str, str]] = None,
) -> None:
    """
    Print a formatted welcome banner with configuration summary.

    Args:
        title: Main title for the banner
        model_name: Model being used
        dataset_name: Dataset being used
        output_dir: Output directory for results
        extra_info: Additional key-value pairs to display
    """
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    if model_name:
        print(f"  Model:      {model_name}")
    if dataset_name:
        print(f"  Dataset:    {dataset_name}")
    if output_dir:
        print(f"  Output:     {output_dir}")

    if extra_info:
        for key, value in extra_info.items():
            print(f"  {key.capitalize():<8} {value}")

    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print("=" * 70 + "\n")


def cleanup_memory() -> None:
    """Clear GPU memory and cache."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: str = "results.json",
) -> None:
    """
    Save results dictionary to JSON file.

    Args:
        results: Dictionary of results to save
        output_dir: Output directory
        filename: JSON filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def safe_train(trainer) -> bool:
    """
    Safely execute trainer.train() with error handling and resource cleanup.

    Args:
        trainer: Trainer instance with train() method

    Returns:
        True if training succeeded, False if an exception occurred
    """
    try:
        trainer.train()
        cleanup_memory()
        return True
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup_memory()
        return False
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False


def run_eval_pair(trained_ckpt, base_model, eval_config) -> Dict[str, Any]:
    """
    Run evaluation on both trained checkpoint and base model, returning combined results.

    Deduplicates the common pattern of evaluating a trained model against its base
    for comparison. Assumes eval_config has a run() method.

    Args:
        trained_ckpt: Path to trained model checkpoint or model object
        base_model: Path to base model or model object
        eval_config: EvalConfig instance with run() method

    Returns:
        Dictionary with 'trained' and 'base' keys containing eval results
    """
    results = {}

    try:
        # Evaluate trained checkpoint
        print(f"\nEvaluating trained model: {trained_ckpt}")
        results['trained'] = eval_config.run(model=trained_ckpt)
    except Exception as e:
        print(f"Error evaluating trained model: {e}")
        results['trained'] = None

    try:
        # Evaluate base model
        print(f"\nEvaluating base model: {base_model}")
        results['base'] = eval_config.run(model=base_model)
    except Exception as e:
        print(f"Error evaluating base model: {e}")
        results['base'] = None

    return results


def print_comparison_table(results_dict: Dict[str, Any]) -> None:
    """
    Print ASCII comparison table for evaluation results.

    Compares metrics across different model versions (e.g., trained vs base).
    Handles nested dicts and common metric types (float, int, bool).

    Args:
        results_dict: Dictionary with model names as keys and result dicts as values.
                     E.g., {'trained': {...metrics...}, 'base': {...metrics...}}
    """
    if not results_dict or all(v is None for v in results_dict.values()):
        print("No evaluation results to compare")
        return

    # Gather all metrics
    all_metrics = set()
    for result in results_dict.values():
        if result and isinstance(result, dict):
            all_metrics.update(result.keys())

    if not all_metrics:
        print("No metrics found in results")
        return

    all_metrics = sorted(all_metrics)

    # Print header
    print("\n" + "=" * 80)
    print(f"{'Metric':<40} {' | '.join(k.ljust(15) for k in sorted(results_dict.keys()))}")
    print("=" * 80)

    # Print each metric row
    for metric in all_metrics:
        row = f"{metric:<40}"
        for model_name in sorted(results_dict.keys()):
            result = results_dict[model_name]
            if result and metric in result:
                value = result[metric]
                if isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
            else:
                formatted = "N/A"
            row += f" | {formatted.ljust(15)}"
        print(row)

    print("=" * 80 + "\n")
