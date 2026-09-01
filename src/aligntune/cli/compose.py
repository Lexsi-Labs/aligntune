"""
CLI commands for composition-based multi-stage training.

This module provides the `aligntune run-composition` command for executing
multi-stage training pipelines defined in YAML composition files.
"""

import logging
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.composition import (
    CompositionLoader,
    CompositionRunner,
    CompositionExecutor,
)

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="compose",
    help="🔗 Run multi-stage training compositions",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command(name="run")
def run_composition(
    composition: str = typer.Argument(
        ...,
        help="Path to composition YAML file (e.g., recipes/configs/compositions/full_stack.yaml)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Base output directory for all stages (default: ./output/<composition-name>)"
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to use (cpu, cuda, cuda:0, etc.)"
    ),
    skip_failed: bool = typer.Option(
        False,
        "--skip-failed",
        help="Continue to next stage even if current stage fails"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
):
    """Run a multi-stage training composition.

    Executes a composition pipeline that chains multiple training stages together.
    Stages are executed in order, with checkpoints threaded from one stage to the next.

    Examples:

    # Run default full stack composition
    aligntune compose run recipes/configs/compositions/full_stack.yaml

    # Run with custom output directory
    aligntune compose run recipes/configs/compositions/full_stack.yaml --output-dir ./my_output

    # Run on GPU and continue even if a stage fails
    aligntune compose run recipes/configs/compositions/full_stack.yaml --device cuda --skip-failed

    # Enable debug logging
    aligntune compose run recipes/configs/compositions/full_stack.yaml --log-level DEBUG
    """
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Validate composition file exists
        composition_path = Path(composition)
        if not composition_path.exists():
            console.print(f"[red]❌ Composition file not found: {composition_path}[/red]")
            raise typer.Exit(1)

        # Load composition
        console.print(f"\n[bold blue]Loading Composition[/bold blue]")
        console.print(f"  File: {composition_path.absolute()}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading composition...", total=None)
            composition_spec = CompositionLoader.load_composition(composition_path)
            progress.update(task, completed=True)

        console.print(f"  Name: {composition_spec.name}")
        console.print(f"  Description: {composition_spec.description}")
        console.print(f"  Stages: {len(composition_spec.stages)}")

        # Determine output directory
        if output_dir is None:
            output_dir = f"./output/{composition_spec.name}"

        output_path = Path(output_dir)
        console.print(f"  Output dir: {output_path.absolute()}")

        # Show stage list
        console.print(f"\n[bold blue]Pipeline Stages[/bold blue]")
        table = Table()
        table.add_column("Stage", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Config", style="yellow")
        table.add_column("Init From", style="magenta")

        for stage in composition_spec.stages:
            init_from = stage.init_from or "-"
            table.add_row(stage.name, stage.algo, stage.config_path, init_from)

        console.print(table)

        # Create runner and execute
        console.print(f"\n[bold blue]Executing Composition[/bold blue]")

        runner = CompositionRunner(
            composition=composition_spec,
            base_output_dir=output_dir,
            device=device
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running composition...", total=len(composition_spec.stages))

            for idx, stage in enumerate(composition_spec.stages):
                task_desc = f"[{idx+1}/{len(composition_spec.stages)}] Executing {stage.name}..."
                progress.update(task, description=task_desc)
                progress.advance(task)

        results = runner.run(skip_failed=skip_failed, stop_on_failure=not skip_failed)

        # Print results summary
        console.print(f"\n[bold blue]Execution Results[/bold blue]")

        results_table = Table()
        results_table.add_column("Stage", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Duration (s)", style="yellow")
        results_table.add_column("Checkpoint", style="magenta")

        for result in results:
            status_icon = "✅" if result.is_success() else ("❌" if result.is_failed() else "⊘")
            status_text = f"{status_icon} {result.status.upper()}"

            checkpoint = result.checkpoint_dir or "-"
            if checkpoint != "-":
                checkpoint = Path(checkpoint).name

            results_table.add_row(
                result.stage_name,
                status_text,
                f"{result.duration_seconds:.1f}",
                checkpoint
            )

        console.print(results_table)

        # Print summary
        summary = runner.get_results_summary()
        successful = summary['successful_stages']
        failed = summary['failed_stages']
        total_duration = summary['total_duration_seconds']

        console.print(f"\n[bold blue]Summary[/bold blue]")
        console.print(f"  Total stages: {summary['total_stages']}")
        console.print(f"  Successful: {successful}")
        if failed > 0:
            console.print(f"  [red]Failed: {failed}[/red]")
        console.print(f"  Total duration: {total_duration:.1f}s")

        # Check for failures
        if failed > 0:
            console.print(f"\n[red]❌ Composition failed ({failed} stage(s) failed)[/red]")
            console.print("\nFailed stage details:")
            for result in results:
                if result.is_failed():
                    console.print(f"  • {result.stage_name}: {result.error_msg}")
            raise typer.Exit(1)
        else:
            console.print(f"\n[green]✅ Composition completed successfully![/green]")
            console.print(f"  Checkpoints saved to: {output_path.absolute()}")

    except FileNotFoundError as e:
        console.print(f"[red]❌ File not found: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]❌ Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logger.exception("Composition execution failed")
        raise typer.Exit(1)


@app.command(name="list")
def list_compositions(
    search: Optional[str] = typer.Option(
        None,
        "--search",
        "-s",
        help="Search for compositions by name or description"
    ),
):
    """List available composition templates.

    Examples:

    # List all available compositions
    aligntune compose list

    # Search for compositions
    aligntune compose list --search "full"
    """
    from pathlib import Path

    # Find all composition YAML files
    compositions_dir = Path(__file__).parent.parent.parent.parent / "recipes" / "configs" / "compositions"

    if not compositions_dir.exists():
        console.print("[yellow]No compositions directory found[/yellow]")
        return

    composition_files = list(compositions_dir.glob("*.yaml"))

    if not composition_files:
        console.print("[yellow]No composition files found[/yellow]")
        return

    # Load and display compositions
    table = Table(title=f"🔗 Available Compositions ({len(composition_files)} found)")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Stages", style="green")
    table.add_column("Path", style="yellow")

    found = 0
    for comp_file in sorted(composition_files):
        try:
            comp = CompositionLoader.load_composition(comp_file)

            # Apply search filter
            if search and (search.lower() not in comp.name.lower() and
                          search.lower() not in comp.description.lower()):
                continue

            stages_str = " → ".join(s.name for s in comp.stages)
            table.add_row(
                comp.name,
                comp.description[:50] + "..." if len(comp.description) > 50 else comp.description,
                stages_str,
                comp_file.name
            )
            found += 1
        except Exception as e:
            logger.warning(f"Failed to load {comp_file}: {e}")

    if found > 0:
        console.print(table)
    else:
        console.print("[yellow]No compositions matched your search[/yellow]")


@app.command(name="inspect")
def inspect_composition(
    composition: str = typer.Argument(
        ...,
        help="Path to composition YAML file"
    ),
):
    """Inspect a composition file in detail.

    Shows detailed information about a composition including all stages and their configurations.

    Examples:

    # Inspect a composition
    aligntune compose inspect recipes/configs/compositions/full_stack.yaml
    """
    try:
        composition_path = Path(composition)
        if not composition_path.exists():
            console.print(f"[red]❌ Composition file not found: {composition_path}[/red]")
            raise typer.Exit(1)

        comp = CompositionLoader.load_composition(composition_path)

        # Display composition info
        console.print(f"\n[bold blue]Composition: {comp.name}[/bold blue]")
        console.print(f"Description: {comp.description}\n")

        if comp.metadata:
            console.print("[bold]Metadata:[/bold]")
            for key, value in comp.metadata.items():
                console.print(f"  {key}: {value}")
            console.print()

        # Display stages
        console.print("[bold blue]Stages:[/bold blue]")
        for idx, stage in enumerate(comp.stages, 1):
            console.print(f"\n  [{idx}] {stage.name}")
            console.print(f"      Algorithm: {stage.algo}")
            console.print(f"      Config: {stage.config_path}")
            if stage.init_from:
                console.print(f"      Initialize from: {stage.init_from}")
            if stage.target_params:
                console.print(f"      Target params: {stage.target_params}")

    except ValueError as e:
        console.print(f"[red]❌ Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)
