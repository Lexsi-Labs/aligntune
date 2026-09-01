"""
CLI command for exporting fine-tuned models.

Provides unified interface for exporting models to GGUF, Ollama, and HuggingFace Hub.
"""

import logging
import typer
from pathlib import Path
from typing import Optional, Literal
from ..core.export import (
    GGUFExporter,
    OllamaExporter,
    HFHubExporter,
    MergeAdapterExporter,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="export",
    help="Export fine-tuned models to various formats",
    no_args_is_help=True,
)


@app.command()
def gguf(
    checkpoint: str = typer.Argument(
        ..., help="Path to checkpoint directory"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory (default: ./exports/gguf)"
    ),
    quantization: Optional[str] = typer.Option(
        None,
        "--quantization",
        "-q",
        help="Quantization preset: Q2_K, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0",
    ),
    converter: Optional[str] = typer.Option(
        None,
        "--converter",
        "-c",
        help="Converter to use: llama-cpp or unsloth (auto-detected by default)",
    ),
):
    """
    Export model to GGUF format for llama.cpp/Ollama.

    GGUF format enables efficient inference with quantization support.

    Examples:
        aligntune export gguf ./checkpoint --output ./models
        aligntune export gguf ./checkpoint -q Q4_K_M -c llama-cpp
    """
    try:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        output_dir = Path(output) if output else Path("./exports/gguf")
        output_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Exporting checkpoint to GGUF...")
        typer.echo(f"  Checkpoint: {checkpoint_path}")
        typer.echo(f"  Output: {output_dir}")
        if quantization:
            typer.echo(f"  Quantization: {quantization}")
        if converter:
            typer.echo(f"  Converter: {converter}")

        exporter = GGUFExporter(
            output_dir=output_dir,
            converter=converter,
            quantization=quantization,
        )

        artifact_path = exporter.export(checkpoint_path)

        typer.echo(f"\nSuccess! GGUF exported to:")
        typer.echo(f"  {artifact_path}")
        typer.echo(f"\nUsage:")
        typer.echo(f"  llama-cli -m {artifact_path} --prompt 'Hello'")
        typer.echo(f"  Or copy to Ollama models directory for use with Ollama")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("GGUF export failed")
        raise typer.Exit(1)


@app.command()
def ollama(
    checkpoint: str = typer.Argument(
        ..., help="Path to checkpoint directory or GGUF file"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for Modelfile (default: ./exports/ollama)"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Ollama model name (default: custom-model:latest)"
    ),
    quantization: Optional[str] = typer.Option(
        None, "-q", help="Quantization for GGUF conversion if needed"
    ),
    create: bool = typer.Option(
        False, "--create", help="Run 'ollama create' to load model into Ollama"
    ),
):
    """
    Export model to Ollama format.

    Creates a Modelfile for use with Ollama runtime. If input is a checkpoint,
    first converts to GGUF.

    Examples:
        aligntune export ollama ./checkpoint --create
        aligntune export ollama ./model.gguf --name my-model:latest --create
    """
    try:
        input_path = Path(checkpoint)
        if not input_path.exists():
            typer.echo(f"Error: Path not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        output_dir = Path(output) if output else Path("./exports/ollama")
        output_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Exporting to Ollama format...")
        typer.echo(f"  Input: {input_path}")
        typer.echo(f"  Output: {output_dir}")
        if model_name:
            typer.echo(f"  Model name: {model_name}")

        # Check if input is GGUF or checkpoint
        if input_path.suffix.lower() == ".gguf":
            typer.echo("Input is GGUF file, skipping conversion")
            gguf_path = input_path
        else:
            typer.echo("Input is checkpoint, converting to GGUF first...")
            gguf_exporter = GGUFExporter(output_dir=output_dir, quantization=quantization)
            gguf_path = gguf_exporter.export(input_path)
            typer.echo(f"  GGUF created: {gguf_path}")

        # Create Ollama Modelfile
        exporter = OllamaExporter(
            output_dir=output_dir,
            gguf_path=gguf_path,
            model_name=model_name,
            create_model=create,
        )

        modelfile_path = exporter.export(gguf_path=gguf_path)

        typer.echo(f"\nSuccess! Ollama model configured at:")
        typer.echo(f"  {modelfile_path}")

        if create:
            typer.echo(f"\nModel loaded into Ollama as: {model_name or 'custom-model:latest'}")
            typer.echo(f"Usage: ollama run {model_name or 'custom-model:latest'}")
        else:
            typer.echo(f"\nTo load into Ollama, run:")
            typer.echo(f"  ollama create {model_name or 'custom-model:latest'} -f {modelfile_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Ollama export failed")
        raise typer.Exit(1)


@app.command()
def hf_hub(
    checkpoint: str = typer.Argument(
        ..., help="Path to checkpoint directory"
    ),
    repo_id: str = typer.Option(
        ..., "--repo", "-r", help="HuggingFace Hub repo ID (username/model-name)"
    ),
    adapter_only: bool = typer.Option(
        False, "--adapter-only", help="Upload only LoRA adapters"
    ),
    private: bool = typer.Option(
        False, "--private", help="Make repository private"
    ),
    token: Optional[str] = typer.Option(
        None, "--token", "-t", help="HuggingFace API token (uses HF_TOKEN env var if not provided)"
    ),
):
    """
    Upload model to HuggingFace Hub.

    Uploads fine-tuned weights or just LoRA adapters to a HF Hub repository.

    Examples:
        aligntune export hf_hub ./checkpoint --repo username/my-model
        aligntune export hf_hub ./checkpoint --repo username/my-lora --adapter-only
    """
    try:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Uploading to HuggingFace Hub...")
        typer.echo(f"  Checkpoint: {checkpoint_path}")
        typer.echo(f"  Repository: {repo_id}")
        typer.echo(f"  Adapter only: {adapter_only}")
        typer.echo(f"  Private: {private}")

        exporter = HFHubExporter(
            repo_id=repo_id,
            adapter_only=adapter_only,
            private=private,
            token=token,
        )

        hub_url = exporter.export(checkpoint_path)

        typer.echo(f"\nSuccess! Model uploaded to:")
        typer.echo(f"  {hub_url}")
        typer.echo(f"\nUsage:")
        typer.echo(f"  from transformers import AutoModelForCausalLM")
        typer.echo(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("HF Hub export failed")
        raise typer.Exit(1)


@app.command()
def merge_adapter(
    checkpoint: str = typer.Argument(
        ..., help="Path to checkpoint with LoRA adapters"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory (default: ./exports/merged)"
    ),
):
    """
    Merge LoRA adapters into base model weights.

    Converts a LoRA-trained checkpoint into a full model with merged weights.
    This is a pre-processing step before other exports if needed.

    Examples:
        aligntune export merge_adapter ./checkpoint --output ./merged_model
    """
    try:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        output_dir = Path(output) if output else Path("./exports/merged")
        output_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Merging LoRA adapters...")
        typer.echo(f"  Checkpoint: {checkpoint_path}")
        typer.echo(f"  Output: {output_dir}")

        exporter = MergeAdapterExporter(output_dir=output_dir)

        merged_path = exporter.export(checkpoint_path)

        typer.echo(f"\nSuccess! Merged model saved to:")
        typer.echo(f"  {merged_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Adapter merge failed")
        raise typer.Exit(1)
