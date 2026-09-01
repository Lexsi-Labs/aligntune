"""
Reward visualization utilities for per-component reward tracking.

This module provides tools to visualize reward trajectories and analyze
correlations between different reward components during RL training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def read_tensorboard_logs(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Read scalar logs from TensorBoard event files.

    Args:
        log_dir: Path to tensorboard directory

    Returns:
        Dict mapping metric name to list of (step, value) tuples
    """
    try:
        from tensorboard.compat.proto import event_pb2
    except ImportError:
        logger.warning("tensorboard not installed, falling back to metrics.json")
        return {}

    metrics_dict = {}

    try:
        for event_file in log_dir.glob("events.out.tfevents.*"):
            for event in event_pb2.Event.FromString(open(event_file, 'rb').read()):
                if event.summary.value:
                    for value in event.summary.value:
                        if value.simple_value:
                            metric_name = value.tag
                            step = event.step
                            value_float = float(value.simple_value)

                            if metric_name not in metrics_dict:
                                metrics_dict[metric_name] = []
                            metrics_dict[metric_name].append((step, value_float))
    except Exception as e:
        logger.warning(f"Failed to read TensorBoard logs: {e}")
        return {}

    return metrics_dict


def read_metrics_json(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Read metrics from metrics_history.json if available.

    Args:
        log_dir: Path to directory containing metrics_history.json

    Returns:
        Dict mapping metric name to list of (step, value) tuples
    """
    metrics_file = log_dir / "metrics_history.json"
    if not metrics_file.exists():
        return {}

    metrics_dict = {}
    try:
        with open(metrics_file) as f:
            history = json.load(f)

        for entry in history:
            step = entry.get("step")
            if step is None:
                continue

            for key, value in entry.items():
                if key not in ("step", "timestamp") and isinstance(value, (int, float)):
                    if key not in metrics_dict:
                        metrics_dict[key] = []
                    metrics_dict[key].append((step, float(value)))

    except Exception as e:
        logger.warning(f"Failed to read metrics.json: {e}")

    return metrics_dict


def extract_reward_metrics(metrics: Dict[str, List[Tuple[int, float]]]) -> Dict[str, List[Tuple[int, float]]]:
    """Extract only reward-related metrics.

    Args:
        metrics: Dict of all metrics

    Returns:
        Dict containing only metrics with "rewards/" prefix
    """
    reward_metrics = {}
    for key, values in metrics.items():
        if "rewards/" in key:
            # Extract the reward name after "rewards/"
            reward_name = key.split("rewards/")[-1]
            reward_metrics[reward_name] = values

    return reward_metrics


def plot_reward_trajectory(run_dir: str) -> Optional["plt.Figure"]:
    """Generate stacked-area chart and correlation heatmap of reward components.

    Args:
        run_dir: Path to the training run directory

    Returns:
        matplotlib Figure object or None if generation fails
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not installed, cannot generate visualization")
        return None

    run_path = Path(run_dir)
    if not run_path.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return None

    # Try to read metrics
    metrics = {}

    # Try TensorBoard logs first
    tensorboard_dir = run_path / "tensorboard"
    if tensorboard_dir.exists():
        metrics = read_tensorboard_logs(tensorboard_dir)

    # Fall back to metrics.json
    if not metrics:
        metrics = read_metrics_json(run_path)

    if not metrics:
        logger.error(f"No metrics found in {run_dir}")
        return None

    # Extract reward metrics
    reward_metrics = extract_reward_metrics(metrics)

    if not reward_metrics:
        logger.warning(f"No reward metrics found in {run_dir}")
        return None

    # Sort by step
    for reward_name in reward_metrics:
        reward_metrics[reward_name].sort(key=lambda x: x[0])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Stacked-area chart of reward trajectories
    ax1 = axes[0]

    # Extract steps and values
    all_steps = set()
    for values in reward_metrics.values():
        all_steps.update(step for step, _ in values)
    all_steps = sorted(all_steps)

    # Create data matrix
    data = {}
    for reward_name, values in reward_metrics.items():
        value_dict = {step: value for step, value in values}
        data[reward_name] = [value_dict.get(step, np.nan) for step in all_steps]

    # Plot stacked area
    reward_names = sorted(data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(reward_names)))

    values_array = np.array([data[name] for name in reward_names])
    ax1.stackplot(all_steps, values_array, labels=reward_names, colors=colors, alpha=0.8)

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Reward Value", fontsize=12)
    ax1.set_title("Reward Component Trajectories (Stacked Area)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correlation heatmap
    ax2 = axes[1]

    # Build correlation matrix
    if len(reward_names) > 1:
        # Create data for correlation
        reward_data = {}
        for reward_name, values in reward_metrics.items():
            value_dict = {step: value for step, value in values}
            # Use all available steps, filling NaN as needed
            reward_data[reward_name] = [value_dict.get(step, np.nan) for step in all_steps]

        # Convert to numpy array for correlation
        data_matrix = np.array([reward_data[name] for name in reward_names])

        # Compute correlation (ignoring NaNs)
        corr_matrix = np.corrcoef(data_matrix)

        # Plot heatmap
        im = ax2.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        # Set ticks and labels
        ax2.set_xticks(range(len(reward_names)))
        ax2.set_yticks(range(len(reward_names)))
        ax2.set_xticklabels(reward_names, rotation=45, ha="right", fontsize=10)
        ax2.set_yticklabels(reward_names, fontsize=10)

        # Add correlation values in cells
        for i in range(len(reward_names)):
            for j in range(len(reward_names)):
                text = ax2.text(j, i, f"{corr_matrix[i, j]:.2f}",
                              ha="center", va="center", color="black", fontsize=9)

        ax2.set_title("Reward Component Correlation Matrix", fontsize=14, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Correlation", fontsize=11)
    else:
        ax2.text(0.5, 0.5, "Need at least 2 reward components for correlation analysis",
                ha="center", va="center", fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

    plt.tight_layout()
    return fig


def save_reward_visualization(run_dir: str, output_path: Optional[str] = None) -> Optional[Path]:
    """Generate and save reward visualization as PNG.

    Args:
        run_dir: Path to the training run directory
        output_path: Optional path to save the figure (defaults to run_dir/reward_visualization.png)

    Returns:
        Path to saved figure or None if generation fails
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not installed, cannot save visualization")
        return None

    fig = plot_reward_trajectory(run_dir)
    if fig is None:
        return None

    if output_path is None:
        output_path = str(Path(run_dir) / "reward_visualization.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reward visualization to {output_path}")
        plt.close(fig)
        return output_path
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")
        plt.close(fig)
        return None
