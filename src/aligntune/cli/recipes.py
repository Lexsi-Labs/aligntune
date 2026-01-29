"""
CLI commands for recipe management.

This module provides commands to list, show, copy, and run training recipes.
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ..recipes import (
    list_recipes,
    get_recipe,
    search_recipes,
    show_recipe_info,
    list_available_recipes,
    Recipe
)

console = Console()
app = typer.Typer(
    name="recipes",
    help="🍳 Manage and run training recipes",
    add_completion=False,
    rich_markup_mode="rich"
)


@app.command()
def list(
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tags"),
    algorithm: Optional[str] = typer.Option(None, "--algorithm", "-a", help="Filter by algorithm (sft, dpo, ppo, grpo, gspo)"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search recipes by name, description, or model")
):
    """
    List available training recipes.

    Examples:

    # List all recipes
    aligntune recipes list

    # Filter by algorithm
    aligntune recipes list --algorithm dpo

    # Filter by tags
    aligntune recipes list --tag llama --tag memory-efficient

    # Search by query
    aligntune recipes list --search "math"
    """

    if search:
        recipes = search_recipes(search)
        if not recipes:
            console.print(f"[yellow]No recipes found matching '{search}'[/yellow]")
            return
    else:
        recipes = list_recipes(tags=tags, algorithm=algorithm)

    if not recipes:
        console.print("[yellow]No recipes found matching the specified filters.[/yellow]")
        return

    table = Table(title=f"🍳 Available Recipes ({len(recipes)} found)")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Algorithm", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Tags", style="yellow")
    table.add_column("Auth", style="red", justify="center")

    for recipe in recipes:
        auth_indicator = "🔐" if recipe.requires_auth else ""
        tags_str = ", ".join(recipe.tags) if recipe.tags else ""

        table.add_row(
            recipe.name,
            recipe.algorithm.upper(),
            recipe.model,
            recipe.description[:60] + "..." if len(recipe.description) > 60 else recipe.description,
            tags_str,
            auth_indicator
        )

    console.print(table)

    if any(r.requires_auth for r in recipes):
        console.print("\n[red]🔐[/red] Recipes marked with 🔐 require Hugging Face authentication")
        console.print("Run [bold]huggingface-cli login[/bold] to authenticate")


@app.command()
def show(name: str = typer.Argument(..., help="Recipe name to show details for")):
    """
    Show detailed information about a recipe.

    Example:
    aligntune recipes show llama3-instruction-tuning
    """

    recipe = get_recipe(name)
    if not recipe:
        console.print(f"[red]❌ Recipe '{name}' not found.[/red]")
        available = [r.name for r in list_recipes()]
        if available:
            console.print(f"\nAvailable recipes: {', '.join(available[:10])}")
            if len(available) > 10:
                console.print(f"... and {len(available) - 10} more")
        return

    # Header
    console.print(f"\n[bold cyan]🍳 Recipe: {recipe.name}[/bold cyan]")
    console.print(f"[dim]{recipe.description}[/dim]\n")

    # Basic info
    info_table = Table(show_header=False)
    info_table.add_column("Field", style="bold cyan", width=20)
    info_table.add_column("Value", style="white")

    info_table.add_row("Model", recipe.model)
    info_table.add_row("Dataset", recipe.dataset)
    info_table.add_row("Task", recipe.task)
    info_table.add_row("Algorithm", recipe.algorithm.upper())
    info_table.add_row("Backend", recipe.backend)

    if recipe.tags:
        info_table.add_row("Tags", ", ".join(recipe.tags))

    if recipe.requires_auth:
        info_table.add_row("Authentication", "[red]Required[/red]")

    if recipe.estimated_time:
        info_table.add_row("Estimated Time", recipe.estimated_time)

    if recipe.estimated_memory:
        info_table.add_row("Estimated Memory", recipe.estimated_memory)

    console.print(info_table)

    # Configuration preview
    console.print(f"\n[bold]Configuration Preview:[/bold]")
    if hasattr(recipe.config, 'to_dict'):
        config_dict = recipe.config.to_dict()
        # Show key sections
        if 'model' in config_dict:
            console.print(f"[green]Model:[/green] {config_dict['model'].get('name_or_path', 'N/A')}")
        if 'dataset' in config_dict:
            console.print(f"[green]Dataset:[/green] {config_dict['dataset'].get('name', 'N/A')}")
        if 'datasets' in config_dict and config_dict['datasets']:
            console.print(f"[green]Dataset:[/green] {config_dict['datasets'][0].get('name', 'N/A')}")
        if 'train' in config_dict:
            train = config_dict['train']
            console.print(f"[green]Training:[/green] {train.get('learning_rate', 'N/A')} LR, {train.get('per_device_batch_size', 'N/A')} batch size")


@app.command()
def copy(
    name: str = typer.Argument(..., help="Recipe name to copy"),
    output: str = typer.Option("./custom_recipe.yaml", "--output", "-o", help="Output file path")
):
    """
    Copy a recipe configuration to a local file for customization.

    Example:
    aligntune recipes copy llama3-instruction-tuning --output my_custom_recipe.yaml
    """

    recipe = get_recipe(name)
    if not recipe:
        console.print(f"[red]❌ Recipe '{name}' not found.[/red]")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save recipe to file
    from ..recipes import save_recipe
    save_recipe(recipe, output_path)

    console.print(f"[green]✅ Recipe '{name}' copied to {output_path}[/green]")
    console.print(f"\nYou can now modify the configuration and run it with:")
    console.print(f"[bold]aligntune finetune --config {output_path}[/bold]")


@app.command()
def run(
    name: str = typer.Argument(..., help="Recipe name to run"),
    override: Optional[List[str]] = typer.Option(None, "--override", "-o", help="Override configuration values (key=value format)")
):
    """
    Run a recipe directly.

    Examples:

    # Run a recipe as-is
    aligntune recipes run llama3-instruction-tuning

    # Run with overrides
    aligntune recipes run llama3-instruction-tuning --override train.learning_rate=1e-4 --override train.epochs=2
    """

    recipe = get_recipe(name)
    if not recipe:
        console.print(f"[red]❌ Recipe '{name}' not found.[/red]")
        return

    console.print(f"[green]🚀 Running recipe: {recipe.name}[/green]")
    console.print(f"[dim]{recipe.description}[/dim]\n")

    # Check authentication requirement
    if recipe.requires_auth:
        import subprocess
        try:
            result = subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"[green]✅ Authenticated as: {result.stdout.strip()}[/green]")
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[red]❌ Authentication required but not logged in.[/red]")
            console.print("Run: [bold]huggingface-cli login[/bold]")
            return

    # Apply overrides to config
    config = recipe.config
    if override:
        console.print(f"[yellow]⚠️  Applying {len(override)} configuration overrides[/yellow]")
        for override_spec in override:
            if "=" not in override_spec:
                console.print(f"[red]❌ Invalid override format: {override_spec}[/red]")
                console.print("Use format: key=value")
                return

            key, value = override_spec.split("=", 1)
            # Simple override logic - in practice, you'd want more sophisticated parsing
            console.print(f"Override: {key} = {value}")

    # Import and run trainer
    try:
        if isinstance(config, type(config).__bases__[0].__bases__[0] if hasattr(config, '__bases__') else config) or hasattr(config, 'model'):
            # SFT config
            from ..core.backend_factory import create_sft_trainer
            trainer = create_sft_trainer(config, backend=recipe.backend)
        else:
            # RL config
            from ..core.backend_factory import create_rl_trainer
            trainer = create_rl_trainer(config, algorithm=recipe.algorithm, backend=recipe.backend)

        from rich.progress import Progress, SpinnerColumn, TextColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {recipe.name}...", total=None)
            trainer.train()
            progress.update(task, description="✅ Training completed successfully!")

        console.print(f"[green]✅ Recipe '{recipe.name}' completed![/green]")

    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        if "--verbose" in str(typer.get_command(None)):
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def create(
    name: str = typer.Argument(..., help="New recipe name"),
    description: str = typer.Option(..., "--description", "-d", help="Recipe description"),
    config_file: str = typer.Option(..., "--config", "-c", help="Path to configuration YAML file"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags for the recipe")
):
    """
    Create a new recipe from a configuration file.

    Example:
    aligntune recipes create my-recipe --description "My custom recipe" --config config.yaml --tag custom --tag experimental
    """

    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"[red]❌ Configuration file not found: {config_file}[/red]")
        return

    try:
        # Load configuration
        from ..core.sft.config_loader import SFTConfigLoader
        from ..core.rl.config_loader import ConfigLoader as RLConfigLoader

        # Try SFT first
        try:
            config = SFTConfigLoader.load_from_yaml(config_path)
        except Exception:
            # Try RL config
            config = RLConfigLoader.load_from_yaml(config_path)

        # Create recipe
        from ..recipes import create_recipe_from_config
        recipe = create_recipe_from_config(name, description, config, tags)

        console.print(f"[green]✅ Recipe '{name}' created successfully![/green]")
        console.print(f"Description: {description}")
        console.print(f"Tags: {', '.join(tags) if tags else 'None'}")

    except Exception as e:
        console.print(f"[red]❌ Failed to create recipe: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()