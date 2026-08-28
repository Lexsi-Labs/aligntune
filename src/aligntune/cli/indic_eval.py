"""
Indic evaluation CLI for AlignTune.

Provides standardized evaluation of models on Indic language benchmarks
(Hindi, Tamil, Bengali) with support for multiple evaluation suites.
"""

import logging
import json
import typer
from typing import Optional, List
from pathlib import Path
from datetime import datetime

from ..eval.lm_eval_integration import (
    run_indic_benchmark,
    get_available_indic_tasks_by_language,
    get_available_indic_tasks,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="indic-eval",
    help="Evaluate models on Indic language benchmarks (Hindi, Tamil, Bengali)",
)


@app.command("run")
def run_indic_eval(
    model: str = typer.Option(
        ...,
        "--model",
        help="HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b')",
    ),
    languages: str = typer.Option(
        "hi,ta,bn",
        "--languages",
        help="Comma-separated language codes: hi (Hindi), ta (Tamil), bn (Bengali)",
    ),
    benchmarks: str = typer.Option(
        "milu,indicxtreme,genbench,sarvam",
        "--benchmarks",
        help="Comma-separated benchmarks: milu, indicxtreme, genbench, sarvam",
    ),
    output_dir: str = typer.Option(
        "./indic_eval_results",
        "--output-dir",
        help="Output directory for results",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Batch size for evaluation",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Limit number of samples per task (for testing)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Run Indic benchmark evaluation on a specified model.

    Examples:
        # Evaluate on all languages and benchmarks
        aligntune indic-eval run --model meta-llama/Llama-2-7b

        # Evaluate Hindi and Tamil on MILU only
        aligntune indic-eval run \\
            --model meta-llama/Llama-2-7b \\
            --languages hi,ta \\
            --benchmarks milu

        # Test with limit
        aligntune indic-eval run \\
            --model meta-llama/Llama-2-7b \\
            --limit 10
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)

    # Parse languages
    try:
        lang_list = [l.strip() for l in languages.split(",")]
        valid_langs = {"hi", "ta", "bn"}
        for lang in lang_list:
            if lang not in valid_langs:
                typer.echo(f"Error: Invalid language '{lang}'. Supported: hi, ta, bn", err=True)
                raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error parsing languages: {e}", err=True)
        raise typer.Exit(1)

    # Parse benchmarks
    try:
        benchmark_list = [b.strip() for b in benchmarks.split(",")]
        valid_benchmarks = {"milu", "indicxtreme", "genbench", "sarvam"}
        for bench in benchmark_list:
            if bench not in valid_benchmarks:
                typer.echo(f"Error: Invalid benchmark '{bench}'. Supported: milu, indicxtreme, genbench, sarvam", err=True)
                raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error parsing benchmarks: {e}", err=True)
        raise typer.Exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"\nStarting Indic evaluation...")
    typer.echo(f"Model: {model}")
    typer.echo(f"Languages: {', '.join(lang_list)}")
    typer.echo(f"Benchmarks: {', '.join(benchmark_list)}")
    typer.echo(f"Output directory: {output_path}")
    typer.echo()

    try:
        # Run benchmarks
        results = run_indic_benchmark(
            model_name=model,
            languages=lang_list,
            benchmarks=benchmark_list,
            output_dir=str(output_path),
            batch_size=batch_size,
            limit=limit,
            save_results=True,
        )

        if not results:
            typer.echo("Warning: No results obtained from evaluation", err=True)
            raise typer.Exit(1)

        # Organize results by language and benchmark
        results_by_lang = {}
        for result in results:
            # Infer language from task name
            task_name = result.task_name
            lang = None
            if "_hi" in task_name or "Hindi" in task_name:
                lang = "hi"
            elif "_ta" in task_name or "Tamil" in task_name:
                lang = "ta"
            elif "_bn" in task_name or "Bengali" in task_name:
                lang = "bn"

            if lang:
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(result.to_dict())

        # Display results
        typer.echo("\n" + "=" * 70)
        typer.echo("INDIC EVALUATION RESULTS")
        typer.echo("=" * 70)

        for lang in ["hi", "ta", "bn"]:
            if lang in results_by_lang:
                lang_name = {"hi": "Hindi", "ta": "Tamil", "bn": "Bengali"}[lang]
                typer.echo(f"\n{lang_name} ({lang}):")
                typer.echo("-" * 70)

                for result in results_by_lang[lang]:
                    typer.echo(f"  {result['task_name']}: {result['metrics']}")

        typer.echo("\n" + "=" * 70)

        # Save detailed results
        detailed_results = {
            "model": model,
            "evaluation_time": datetime.utcnow().isoformat(),
            "languages": lang_list,
            "benchmarks": benchmark_list,
            "batch_size": batch_size,
            "results_by_language": results_by_lang,
            "all_results": [r.to_dict() for r in results],
        }

        results_file = output_path / "indic_eval_detailed.json"
        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        typer.echo(f"\nDetailed results saved to: {results_file}")

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        typer.echo(f"Error: Evaluation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("list")
def list_indic_tasks(
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="Filter by language (hi, ta, bn)",
    ),
) -> None:
    """
    List available Indic evaluation tasks.

    Examples:
        # List all tasks
        aligntune indic-eval list

        # List Hindi tasks only
        aligntune indic-eval list --language hi
    """
    typer.echo("\nAvailable Indic Evaluation Tasks")
    typer.echo("=" * 70)

    if language:
        try:
            tasks = get_available_indic_tasks_by_language(language)
            lang_name = {"hi": "Hindi", "ta": "Tamil", "bn": "Bengali"}.get(language)
            typer.echo(f"\n{lang_name} ({language}):")
            for task in sorted(tasks):
                typer.echo(f"  - {task}")
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
    else:
        all_tasks = get_available_indic_tasks()

        # Group by language
        for lang_code, lang_name in [("hi", "Hindi"), ("ta", "Tamil"), ("bn", "Bengali")]:
            lang_tasks = [t for t in all_tasks if f"_{lang_code}" in t]
            if lang_tasks:
                typer.echo(f"\n{lang_name} ({lang_code}):")
                for task in sorted(lang_tasks):
                    typer.echo(f"  - {task}")

    typer.echo("\nBenchmark Categories:")
    typer.echo("  - milu: IIT-KGP Indic MMLU (multiple choice)")
    typer.echo("  - indicxtreme: IndicCOPA, IndicSentiment, IndicXNLI, IndicQA")
    typer.echo("  - genbench: FloresIN, CrossSumIN, XQuAD-IN")
    typer.echo("  - sarvam: MMLU-IN, GSM8K-IN, TriviaQA-IN")
    typer.echo()


if __name__ == "__main__":
    app()
