"""
AlignTune Aligner CLI: Interactive training inspector.

Provides command-line interface for live training inspection with
Python API and optional dashboard.
"""

import logging
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

from ..core.aligner import AlignerSession, AlignerCallback, AlignerDashboard, create_dashboard
try:
    from ..core.backend_factory import BackendFactory
except ImportError:
    BackendFactory = None
from ..utils.config_utils import load_config, parse_config_to_unified

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="aligner",
    help="Interactive training inspector with live metric inspection and hyperparameter adjustment",
    no_args_is_help=True,
)


@app.command()
def aligner(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to training config YAML",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name/path (overrides config)",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset name/path (overrides config)",
    ),
    no_dashboard: bool = typer.Option(
        False,
        "--no-dashboard",
        help="Disable dashboard, use Python REPL instead",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (overrides config)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        "-n",
        help="Run name for logging",
    ),
) -> None:
    """
    Start interactive training session.

    Examples:
        aligntune aligner --config config.yaml
        aligntune aligner --config config.yaml --model gpt2
        aligntune aligner --config config.yaml --no-dashboard
    """
    try:
        # Load configuration
        console.print("[cyan]Loading configuration...[/cyan]")
        config_dict = load_config(config)

        # Override with CLI arguments
        if model:
            if "model" not in config_dict:
                config_dict["model"] = {}
            config_dict["model"]["name"] = model

        if dataset:
            if "dataset" not in config_dict:
                config_dict["dataset"] = {}
            config_dict["dataset"]["name"] = dataset

        if output_dir:
            if "train" not in config_dict:
                config_dict["train"] = {}
            config_dict["train"]["output_dir"] = output_dir

        if run_name:
            if "train" not in config_dict:
                config_dict["train"] = {}
            config_dict["train"]["run_name"] = run_name

        # Parse unified config
        unified_config = parse_config_to_unified(config_dict)

        # Detect training type from config
        from ..core.rl.config import AlgorithmType

        is_rl = hasattr(unified_config, "algo") and unified_config.algo in [
            AlgorithmType.DPO,
            AlgorithmType.PPO,
            AlgorithmType.GRPO,
            AlgorithmType.GSPO,
            AlgorithmType.ORPO,
        ]

        # Create trainer
        console.print("[cyan]Creating trainer...[/cyan]")
        if is_rl:
            trainer = BackendFactory.create_rl_trainer(unified_config)
        else:
            trainer = BackendFactory.create_sft_trainer(unified_config)

        console.print(f"[green]✓ Trainer created: {trainer.__class__.__name__}[/green]")

        # Create aligner session
        console.print("[cyan]Initializing aligner session...[/cyan]")
        aligner_session = AlignerSession(trainer)

        # Add aligner callback to trainer
        aligner_callback = AlignerCallback(aligner_session)
        if hasattr(trainer, "callbacks"):
            trainer.callbacks.append(aligner_callback)
        else:
            trainer.callbacks = [aligner_callback]

        console.print("[green]✓ Aligner session initialized[/green]")

        # Start training and optional dashboard
        if no_dashboard:
            _run_repl_mode(aligner_session)
        else:
            _run_dashboard_mode(aligner_session)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Aligner command failed")
        raise typer.Exit(code=1)


def _run_dashboard_mode(aligner_session: AlignerSession) -> None:
    """Run with dashboard."""
    console.print("[cyan]Starting training with dashboard...[/cyan]")

    # Create dashboard
    dashboard = create_dashboard(aligner_session)
    if dashboard is None:
        console.print("[yellow]Warning: Dashboard unavailable, switching to REPL mode[/yellow]")
        _run_repl_mode(aligner_session)
        return

    # Start training and dashboard
    aligner_session.start()
    try:
        dashboard.interactive_mode()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping training...[/yellow]")
        aligner_session.stop()


def _run_repl_mode(aligner_session: AlignerSession) -> None:
    """Run in Python REPL mode."""
    console.print("[cyan]Starting training in REPL mode[/cyan]")
    console.print("[cyan]Use 'aligner_session' variable to control training[/cyan]\n")

    # Print help
    console.print("[bold cyan]Available commands:[/bold cyan]")
    console.print("  aligner_session.start()       - Start training")
    console.print("  aligner_session.pause()       - Pause training")
    console.print("  aligner_session.resume()      - Resume training")
    console.print("  aligner_session.stop()        - Stop training")
    console.print("  aligner_session.peek()        - Get current state")
    console.print("  aligner_session.sample(prompt) - Generate samples")
    console.print("  aligner_session.worst_examples(n) - Get worst examples")
    console.print("  aligner_session.set(lr=...)   - Update hyperparameters")
    console.print("  aligner_session.rollback(step) - Rollback to step")
    console.print("  aligner_session.history()     - Get metrics history\n")

    # Start training
    aligner_session.start()

    # Drop into interactive Python shell
    import code
    import readline  # noqa: F401

    code.interact(
        banner="[bold cyan]AlignTune Aligner REPL[/bold cyan]",
        local={"aligner_session": aligner_session},
        exitmsg="[cyan]Exiting Aligner[/cyan]",
    )

    # Cleanup
    aligner_session.stop()
