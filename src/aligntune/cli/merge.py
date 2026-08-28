"""
CLI command for merging models.

Provides a unified interface for model merging via mergekit (linear,
task_arithmetic) and PEFT (LoRA adapter merge).

Examples
--------
Linear merge:
    aligntune merge --method linear \\
        --models org/model-a org/model-b \\
        --output ./merged_linear \\
        --weights 0.5 0.5

LoRA adapter merge (no mergekit needed):
    aligntune merge --method lora-merge \\
        --models org/base-model \\
        --adapter ./my_lora_adapter \\
        --output ./merged_full
"""

import logging
from typing import List, Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="merge",
    help="Merge multiple models via linear, task_arithmetic, or LoRA merge.",
    no_args_is_help=True,
)

# Merge methods dispatched to MergekitMerger
_MERGEKIT_METHODS = {"linear", "task_arithmetic"}
# Merge methods dispatched to PEFTMerger
_PEFT_METHODS = {"lora-merge"}
_ALL_METHODS = _MERGEKIT_METHODS | _PEFT_METHODS


@app.command()
def run(
    method: str = typer.Option(
        ...,
        "--method",
        "-m",
        help=(
            "Merge method. Mergekit methods: linear, task_arithmetic. "
            "PEFT method: lora-merge."
        ),
    ),
    models: Optional[List[str]] = typer.Option(
        None,
        "--models",
        help=(
            "One or more model paths or HuggingFace IDs to merge. "
            "For lora-merge, may be omitted when --base is supplied."
        ),
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for the merged model.",
    ),
    # LoRA-specific
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter",
        help="[lora-merge only] Path to the LoRA adapter directory.",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        help=(
            "[lora-merge] Explicit base model path/ID. "
            "If set, prepended to --models list."
        ),
    ),
    # mergekit-shared
    weights: Optional[List[float]] = typer.Option(
        None,
        "--weights",
        help=(
            "Per-model weights for linear, task_arithmetic. "
            "Must match number of --models."
        ),
    ),
    density: Optional[float] = typer.Option(
        None,
        "--density",
        help="Unused by the currently supported merge methods.",
    ),
    t: Optional[float] = typer.Option(
        None,
        "--t",
        help="Unused by the currently supported merge methods.",
    ),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        help="Output dtype: bfloat16, float16, float32.",
    ),
):
    """
    Merge models using the specified method.

    Dispatches to MergekitMerger (linear/task_arithmetic)
    or PEFTMerger (lora-merge) based on --method.
    """
    method = method.lower().strip()

    if method not in _ALL_METHODS:
        typer.echo(
            f"Error: Unknown merge method '{method}'. "
            f"Supported: {', '.join(sorted(_ALL_METHODS))}",
            err=True,
        )
        raise typer.Exit(1)

    # Handle --base convenience flag for lora-merge
    effective_models: List[str] = list(models) if models else []
    if base:
        effective_models = [base] + effective_models

    if not effective_models:
        typer.echo("Error: At least one model must be specified via --models.", err=True)
        raise typer.Exit(1)

    typer.echo(f"AlignTune Model Merge")
    typer.echo(f"  Method  : {method}")
    typer.echo(f"  Models  : {effective_models}")
    typer.echo(f"  Output  : {output}")
    if weights:
        typer.echo(f"  Weights : {weights}")
    if density is not None:
        typer.echo(f"  Density : {density}")
    if t is not None:
        typer.echo(f"  t       : {t}")
    if adapter:
        typer.echo(f"  Adapter : {adapter}")
    typer.echo("")

    try:
        if method in _MERGEKIT_METHODS:
            _run_mergekit(
                models=effective_models,
                method=method,
                output=output,
                weights=weights,
                density=density,
                t=t,
                dtype=dtype,
            )
        else:
            _run_peft(
                models=effective_models,
                output=output,
                adapter_path=adapter,
                dtype=dtype,
            )
    except ImportError as exc:
        typer.echo(f"Error (missing dependency): {exc}", err=True)
        raise typer.Exit(1)
    except ValueError as exc:
        typer.echo(f"Error (invalid arguments): {exc}", err=True)
        raise typer.Exit(1)
    except RuntimeError as exc:
        typer.echo(f"Error (merge failed): {exc}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during merge")
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)


def _run_mergekit(
    models: List[str],
    method: str,
    output: str,
    weights: Optional[List[float]],
    density: Optional[float],
    t: Optional[float],
    dtype: str,
) -> None:
    from ..core.merge import MergekitMerger

    merger = MergekitMerger()

    typer.echo("Generating mergekit config...")
    yaml_preview = merger.generate_yaml(
        models=models,
        method=method,
        weights=weights,
        density=density,
        t=t,
        dtype=dtype,
    )
    typer.echo("--- mergekit YAML ---")
    typer.echo(yaml_preview)
    typer.echo("---------------------")

    typer.echo("Running mergekit merge (this may take several minutes)...")
    result_path = merger.merge(
        models=models,
        output_path=output,
        method=method,
        weights=weights,
        density=density,
        t=t,
        dtype=dtype,
    )

    typer.echo(f"\nMerge complete!")
    typer.echo(f"  Merged model saved to: {result_path}")
    typer.echo(f"\nUsage:")
    typer.echo(f"  from transformers import AutoModelForCausalLM")
    typer.echo(f"  model = AutoModelForCausalLM.from_pretrained('{result_path}')")


def _run_peft(
    models: List[str],
    output: str,
    adapter_path: Optional[str],
    dtype: str,
) -> None:
    from ..core.merge import PEFTMerger

    merger = PEFTMerger()

    typer.echo("Running PEFT LoRA adapter merge...")
    result_path = merger.merge(
        models=models,
        output_path=output,
        method="lora-merge",
        adapter_path=adapter_path,
        torch_dtype=dtype,
    )

    typer.echo(f"\nMerge complete!")
    typer.echo(f"  Merged model saved to: {result_path}")
    typer.echo(f"\nUsage:")
    typer.echo(f"  from transformers import AutoModelForCausalLM")
    typer.echo(f"  model = AutoModelForCausalLM.from_pretrained('{result_path}')")
