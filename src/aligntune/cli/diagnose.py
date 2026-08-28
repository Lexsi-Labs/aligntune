"""
Training diagnostics CLI commands.

This module provides commands for diagnosing training issues,
monitoring system health, and generating diagnostic reports.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from ..utils.diagnostics import TrainingDiagnostics, run_comprehensive_diagnostics, run_config_validation
from ..utils.errors import handle_error
from ..core.sft.config_loader import SFTConfigLoader
from ..core.rl.config_loader import ConfigLoader

console = Console()
app = typer.Typer(
    name="diagnose",
    help="🩺 Diagnose training issues and monitor system health",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command()
def system(
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save diagnostics to file"),
    interval: float = typer.Option(5.0, "--interval", help="Monitoring interval in seconds"),
    duration: float = typer.Option(30.0, "--duration", help="Monitoring duration in seconds")
):
    """
    Run comprehensive system diagnostics with optional monitoring.

    Examples:

    # Quick system check
    aligntune diagnose system

    # Extended monitoring
    aligntune diagnose system --duration 60 --interval 2

    # Save to file
    aligntune diagnose system --output diagnostics.json
    """

    try:
        console.print("[green]🔍 Running system diagnostics...[/green]")

        # Initial diagnostics
        diagnostics = run_comprehensive_diagnostics()

        # Display key information
        sys_info = diagnostics['system_info']

        console.print(f"\n[bold]🖥️  System Overview[/bold]")
        console.print(f"CPU Cores: {sys_info['cpu_count']}")
        console.print(f"Total RAM: {sys_info['memory_total_gb']:.1f} GB")
        console.print(f"Available RAM: {sys_info['memory_available_gb']:.1f} GB")

        if sys_info['gpu_available']:
            console.print(f"GPU Devices: {sys_info['gpu_count']}")
            for i, gpu in enumerate(sys_info['gpu_memory']):
                console.print(f"  GPU {i}: {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)")
        else:
            console.print("[yellow]GPU: Not available[/yellow]")

        # Monitor for a period if requested
        if duration > 0:
            console.print(f"\n[bold]📊 Monitoring system for {duration} seconds...[/bold]")

            with Progress() as progress:
                task = progress.add_task("Monitoring...", total=int(duration))

                start_time = time.time()
                while time.time() - start_time < duration:
                    time.sleep(interval)
                    progress.update(task, advance=int(interval))

            console.print("[green]✅ Monitoring complete[/green]")

        # Save if requested
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)

            console.print(f"[green]💾 Diagnostics saved to {output_path}[/green]")

        # Show detailed diagnostics
        console.print(f"\n[bold]🔧 Detailed Diagnostics[/bold]")

        # Library status
        libs = diagnostics['libraries']
        lib_table = Table(title="📚 Libraries")
        lib_table.add_column("Library", style="cyan")
        lib_table.add_column("Status", style="green")
        lib_table.add_column("Version", style="magenta")

        for lib_name, lib_info in libs.items():
            status = "✅ Available" if lib_info.get('library_available', False) else "❌ Missing"
            version = lib_info.get('version', 'N/A')
            lib_table.add_row(lib_name, status, version)

        console.print(lib_table)

        # CUDA status
        if diagnostics.get('cuda_info'):
            cuda_info = diagnostics['cuda_info']
            cuda_table = Table(title="🎮 CUDA Status")
            cuda_table.add_column("Component", style="cyan")
            cuda_table.add_column("Status", style="green")

            cuda_table.add_row("CUDA Available", "✅ Yes" if cuda_info.get('cuda_functional') else "❌ No")
            cuda_table.add_row("Tensor Operations", "✅ Working" if cuda_info.get('tensor_ops_working') else "❌ Failed")

            if cuda_info.get('error'):
                cuda_table.add_row("Error", cuda_info['error'])

            console.print(cuda_table)

    except Exception as e:
        handle_error(e, verbose=False)


@app.command()
def config(
    config_file: str = typer.Argument(..., help="Path to configuration YAML file"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save diagnostics to file")
):
    """
    Diagnose a configuration for potential issues.

    Example:
    aligntune diagnose config my_config.yaml --output config_diagnostics.json
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

            # Load config
            try:
                config = SFTConfigLoader.load_from_yaml(config_path)
                config_type = "SFT"
            except Exception:
                config = ConfigLoader.load_from_yaml(config_path)
                config_type = "RL"

            progress.update(task, description="Running diagnostics...")
            diagnostics = run_config_validation(config_path, "auto")

            progress.update(task, description="✅ Diagnostics complete")

        # Display results
        console.print(f"\n[bold]🔍 Configuration Diagnostics[/bold]")
        console.print(f"Config file: {config_path}")
        console.print(f"Config type: {config_type}")

        # Validation status
        if diagnostics['is_valid']:
            console.print("[green]✅ Configuration is valid[/green]")
        else:
            console.print("[red]❌ Configuration has issues[/red]")

        # Errors
        if diagnostics['errors']:
            console.print(f"\n[bold red]❌ Errors ({len(diagnostics['errors'])}):[/bold red]")
            for error in diagnostics['errors']:
                console.print(f"  • {error}")

        # Warnings
        if diagnostics['warnings']:
            console.print(f"\n[bold yellow]⚠️  Warnings ({len(diagnostics['warnings'])}):[/bold yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        # Memory estimate
        memory_gb = diagnostics['memory_estimate_gb']
        console.print(f"\n[bold]💾 Memory Estimate: {memory_gb:.1f} GB[/bold]")

        system_info = diagnostics['system_info']
        if system_info['gpu_count'] > 0:
            gpu_memory = max(system_info['gpu_memory_gb'])
            if memory_gb > gpu_memory:
                console.print(f"[red]⚠️  Estimated memory ({memory_gb:.1f} GB) exceeds GPU capacity ({gpu_memory:.1f} GB)[/red]")
                console.print("Consider:")
                console.print("  • Using PEFT (LoRA)")
                console.print("  • Reducing batch size")
                console.print("  • Enabling gradient checkpointing")
                console.print("  • Using a smaller model")
            else:
                utilization = (memory_gb / gpu_memory) * 100
                console.print(f"[green]✅ Fits within GPU memory ({utilization:.1f}% utilization)[/green]")
        else:
            available_memory = system_info['available_memory_gb']
            if memory_gb > available_memory:
                console.print(f"[red]⚠️  Estimated memory ({memory_gb:.1f} GB) exceeds available RAM ({available_memory:.1f} GB)[/red]")
            else:
                console.print(f"[green]✅ Fits within available RAM ({available_memory:.1f} GB available)[/green]")

        # Recommendations
        recommendations = [r for r in diagnostics.get('recommendations', []) if r]
        if recommendations:
            console.print(f"\n[bold]💡 Recommendations:[/bold]")
            for rec in recommendations:
                console.print(f"  • {rec}")

        # Save if requested
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)

            console.print(f"[green]💾 Diagnostics saved to {output_path}[/green]")

    except Exception as e:
        handle_error(e, verbose=False)


@app.command()
def training(
    config_file: str = typer.Argument(..., help="Path to configuration YAML file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory to save diagnostic reports"),
    monitor: bool = typer.Option(False, "--monitor", help="Start monitoring during training")
):
    """
    Set up training diagnostics and monitoring.

    This command initializes diagnostic monitoring for training sessions.
    Use this before starting training to enable detailed diagnostics.

    Example:
    aligntune diagnose training my_config.yaml --monitor --output-dir ./diagnostics
    """

    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"[red]❌ Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        # Load config
        try:
            config = SFTConfigLoader.load_from_yaml(config_path)
        except Exception:
            config = ConfigLoader.load_from_yaml(config_path)

        # Initialize diagnostics
        diagnostics = TrainingDiagnostics(config, output_dir)

        console.print("[green]✅ Training diagnostics initialized[/green]")
        console.print(f"Output directory: {diagnostics.output_dir}")

        if monitor:
            console.print("[yellow]📊 Starting system monitoring...[/yellow]")
            console.print("Press Ctrl+C to stop monitoring")

            diagnostics.start_monitoring()

            try:
                # Keep running until interrupted
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 Stopping monitoring...[/yellow]")

            diagnostics.stop_monitoring()

            # Generate final report
            report_path = diagnostics.save_report()
            console.print(f"[green]📋 Final diagnostic report saved to {report_path}[/green]")

        else:
            console.print("\n[bold]Available diagnostic commands:[/bold]")
            console.print("  • Run training with diagnostics enabled")
            console.print("  • Use 'aligntune diagnose system' for system checks")
            console.print("  • Use 'aligntune diagnose config' for configuration analysis")

    except Exception as e:
        handle_error(e, verbose=False)


@app.command()
def info():
    """
    Show information about available diagnostic tools.

    Example:
    aligntune diagnose info
    """

    console.print("[bold]🩺 AlignTune Diagnostic Tools[/bold]")
    console.print()

    tools_table = Table()
    tools_table.add_column("Command", style="cyan", no_wrap=True)
    tools_table.add_column("Description", style="white")
    tools_table.add_column("Use Case", style="green")

    tools_table.add_row(
        "diagnose system",
        "Check system compatibility and hardware",
        "Before starting training"
    )
    tools_table.add_row(
        "diagnose config <config>",
        "Validate configuration and estimate resources",
        "After creating/modifying configs"
    )
    tools_table.add_row(
        "diagnose training <config>",
        "Monitor training progress and system health",
        "During training sessions"
    )
    tools_table.add_row(
        "validate config <config>",
        "Quick configuration validation",
        "Quick checks"
    )
    tools_table.add_row(
        "validate model <model>",
        "Check model accessibility",
        "When having access issues"
    )
    tools_table.add_row(
        "validate dataset <dataset>",
        "Check dataset accessibility",
        "When having access issues"
    )
    tools_table.add_row(
        "validate memory",
        "Estimate memory requirements",
        "Memory planning"
    )

    console.print(tools_table)

    console.print(f"\n[bold]💡 Tips:[/bold]")
    console.print("  • Run diagnostics before long training sessions")
    console.print("  • Use --output to save reports for later analysis")
    console.print("  • Combine with --verbose for detailed information")
    console.print("  • Check the docs for advanced diagnostic features")


if __name__ == "__main__":
    app()
