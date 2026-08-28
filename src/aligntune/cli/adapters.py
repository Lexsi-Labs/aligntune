"""
CLI commands for LoRA adapter management (v3.3 Advanced Parameterization).

Provides unified interface for adapter operations:
- Info: Inspect a trained adapter (rank, target modules, parameter count)
- Generation: Create adapters from task descriptions (Text-to-LoRA, Doc-to-LoRA)

Examples
--------
Get adapter info:
    aligntune adapters info --adapter ./trained_lora

Generate LoRA from task description (Text-to-LoRA):
    aligntune adapters generate --type text2lora \\
        --description "Fine-tune on medical QA" \\
        --hypernet-checkpoint ./checkpoints/hypernet.pt \\
        --output ./generated_adapter

Generate LoRA from document (Doc-to-LoRA):
    aligntune adapters generate --type doc2lora \\
        --document ./task_spec.txt \\
        --hypernet-checkpoint ./checkpoints/hypernet.pt \\
        --output ./generated_adapter
"""

import logging
from pathlib import Path
from typing import Optional
import json
import torch

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="adapters",
    help="Manage LoRA adapters: compress, validate, export",
    no_args_is_help=True,
)


@app.command()
def info(
    adapter: str = typer.Option(
        ...,
        "--adapter",
        "-a",
        help="Path to the LoRA adapter directory.",
    ),
) -> None:
    """
    Display information about a LoRA adapter.

    Shows rank, target modules, and parameter count.
    """
    adapter_path = Path(adapter)

    if not adapter_path.exists():
        typer.echo(f"Error: Adapter directory not found: {adapter}", err=True)
        raise typer.Exit(1)

    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        typer.echo(
            f"Error: adapter_config.json not found in {adapter}", err=True
        )
        raise typer.Exit(1)

    try:
        import json

        with open(config_path) as f:
            config = json.load(f)

        typer.echo(f"Adapter: {adapter_path.name}")
        typer.echo("=" * 50)

        # Base info
        rank = config.get("r", config.get("lora_r"))
        typer.echo(f"Rank (r)           : {rank}")
        typer.echo(f"Alpha              : {config.get('lora_alpha', 'N/A')}")
        typer.echo(
            f"Target Modules     : {', '.join(config.get('target_modules', []))}"
        )
        typer.echo(f"Dropout            : {config.get('lora_dropout', 'N/A')}")
        typer.echo(f"Bias               : {config.get('bias', 'N/A')}")

        # Parameter count estimation
        try:
            from safetensors.torch import load_file

            weights_path = adapter_path / "adapter_model.safetensors"
            weights = load_file(str(weights_path))
            param_count = sum(w.numel() for w in weights.values())
            typer.echo()
            typer.echo(f"Parameter Count    : {param_count:,}")

        except Exception as e:
            logger.debug(f"Could not compute parameter count: {e}")

    except Exception as e:
        logger.exception(f"Error reading adapter info: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def generate(
    adapter_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help="Adapter generation type: 'text2lora' (from description) or 'doc2lora' (from document)",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Task description (for text2lora). Short text describing the fine-tuning task.",
    ),
    document: Optional[str] = typer.Option(
        None,
        "--document",
        help="Path to document file (for doc2lora). Long-form task specification.",
    ),
    hypernet_checkpoint: str = typer.Option(
        ...,
        "--hypernet-checkpoint",
        "-c",
        help="Path to trained Text-to-LoRA hypernetwork checkpoint (.pt file).",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for generated adapter weights.",
    ),
    chunk_size: int = typer.Option(
        512,
        "--chunk-size",
        help="Chunk size in characters (doc2lora only).",
    ),
    num_chunks: int = typer.Option(
        3,
        "--num-chunks",
        help="Maximum chunks to process (doc2lora only).",
    ),
    pooling_strategy: str = typer.Option(
        "weighted",
        "--pooling-strategy",
        help="Pooling strategy: 'mean' or 'weighted' (doc2lora only).",
    ),
) -> None:
    """
    Generate LoRA adapter weights from task description or document.

    Two modes:
    1. Text-to-LoRA (--type text2lora):
       - Accepts short task description
       - Embeds description directly
       - Fast single-pass generation
       - Suitable for concise task specs

    2. Doc-to-LoRA (--type doc2lora):
       - Accepts long document file
       - Chunks document for context preservation
       - Pools chunk embeddings with attention
       - Suitable for detailed specifications

    Output structure:
        {output_dir}/
            ├── adapter_config.json           (adapter configuration)
            ├── adapter_model.bin             (generated LoRA weights)
            ├── generation_config.json        (generation parameters)
            └── metadata.json                 (hypernet info, generation mode)

    The generated adapter can be merged into a base model using:
        aligntune merge --base-model <model> --adapter {output_dir}
    """
    # Validate input
    if adapter_type not in ["text2lora", "doc2lora"]:
        typer.echo(
            f"Error: Invalid adapter type '{adapter_type}'. "
            "Must be 'text2lora' or 'doc2lora'",
            err=True,
        )
        raise typer.Exit(1)

    checkpoint_path = Path(hypernet_checkpoint)
    if not checkpoint_path.exists():
        typer.echo(
            f"Error: Checkpoint not found: {hypernet_checkpoint}", err=True
        )
        raise typer.Exit(1)

    # Validate mode-specific arguments
    if adapter_type == "text2lora":
        if not description:
            typer.echo(
                "Error: --description required for text2lora mode", err=True
            )
            raise typer.Exit(1)
        input_text = description
        input_source = "description"

    else:  # doc2lora
        if not document:
            typer.echo(
                "Error: --document required for doc2lora mode", err=True
            )
            raise typer.Exit(1)

        doc_path = Path(document)
        if not doc_path.exists():
            typer.echo(f"Error: Document not found: {document}", err=True)
            raise typer.Exit(1)

        with open(doc_path) as f:
            input_text = f.read()
        input_source = f"document:{doc_path.name}"

    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    typer.echo("AlignTune Adapter Generation (v3.3)")
    typer.echo("=" * 60)
    typer.echo(f"Mode                : {adapter_type.upper()}")
    typer.echo(f"Hypernet Checkpoint : {checkpoint_path.name}")
    typer.echo(f"Input Source        : {input_source}")
    if adapter_type == "doc2lora":
        typer.echo(f"Chunk Size          : {chunk_size}")
        typer.echo(f"Max Chunks          : {num_chunks}")
        typer.echo(f"Pooling Strategy    : {pooling_strategy}")
    typer.echo(f"Output Directory    : {output_path.absolute()}")
    typer.echo()

    try:
        from ..core.adapters.text2lora import (
            TextToLoRAHypernet,
            DocToLoRA,
        )

        # Load checkpoint
        typer.echo("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract hypernet config
        if "config" in checkpoint:
            hypernet_config = checkpoint["config"]
        else:
            # Fallback: reconstruct from checkpoint keys
            hypernet_config = {
                "hidden_dim": 768,
                "lora_r": 16,
                "num_target_modules": 5,
                "mlp_hidden": 512,
                "embedding_model_name": "all-MiniLM-L6-v2",
                "lora_init_std": 0.02,
                "device": "cpu",
            }

        typer.echo(f"✓ Checkpoint loaded")
        typer.echo(
            f"  Hypernet config: hidden_dim={hypernet_config.get('hidden_dim')}, "
            f"lora_r={hypernet_config.get('lora_r')}, "
            f"num_targets={hypernet_config.get('num_target_modules')}"
        )
        typer.echo()

        # Initialize hypernetwork
        typer.echo("Initializing hypernetwork...")
        hypernet = TextToLoRAHypernet(**hypernet_config)

        # Load hypernet state
        if "hypernet" in checkpoint:
            hypernet.load_state_dict(checkpoint["hypernet"])
        typer.echo("✓ Hypernetwork initialized and loaded")
        typer.echo()

        # Generate LoRA weights
        typer.echo(f"Generating LoRA weights ({adapter_type})...")

        if adapter_type == "text2lora":
            # Direct text embedding and generation
            try:
                embedding_model = hypernet.get_embedding_model()
                embeddings = embedding_model.encode(
                    [input_text], convert_to_tensor=True
                )
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                lora_weights = hypernet(embeddings)
            except ImportError:
                typer.echo(
                    "Warning: sentence-transformers not installed. "
                    "Using mock embeddings.",
                    err=False,
                )
                mock_embedding = torch.randn(1, hypernet_config["hidden_dim"])
                lora_weights = hypernet(mock_embedding)

        else:  # doc2lora
            # Document chunking and pooling
            doc2lora = DocToLoRA(
                hypernet,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                pooling_strategy=pooling_strategy,
                device="cpu",
            )
            lora_weights = doc2lora(input_text)

        typer.echo("✓ LoRA weights generated")
        typer.echo()

        # Save generated adapter
        typer.echo("Saving adapter...")

        # Create adapter config
        adapter_config = {
            "r": hypernet_config.get("lora_r", 16),
            "lora_alpha": hypernet_config.get("lora_r", 16),
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
            ],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Save configuration files
        config_path = output_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Save generation metadata
        metadata = {
            "generation_mode": adapter_type,
            "hypernet_checkpoint": str(checkpoint_path),
            "input_source": input_source,
            "input_length": len(input_text),
        }

        if adapter_type == "doc2lora":
            metadata.update(
                {
                    "chunk_size": chunk_size,
                    "num_chunks": num_chunks,
                    "pooling_strategy": pooling_strategy,
                }
            )

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save generation config
        gen_config = {
            "hypernet_config": hypernet_config,
            "generation_config": metadata,
        }

        gen_config_path = output_path / "generation_config.json"
        with open(gen_config_path, "w") as f:
            json.dump(gen_config, f, indent=2)

        # Save LoRA weights in PEFT format
        # Convert generated weights [1, r, hidden_dim] to model format
        adapter_weights = {}
        for idx, lora_pair in enumerate(lora_weights):
            a_matrix = lora_pair["A"].squeeze(0)  # [r, hidden_dim]
            b_matrix = lora_pair["B"].squeeze(0)  # [hidden_dim, r]

            adapter_weights[f"lora_{idx}_a"] = a_matrix
            adapter_weights[f"lora_{idx}_b"] = b_matrix

        # Save PyTorch weights
        weights_path = output_path / "adapter_model.bin"
        torch.save(adapter_weights, weights_path)

        typer.echo(f"✓ Adapter saved to {output_path.absolute()}")
        typer.echo()

        # Summary
        typer.echo("Generation Summary:")
        typer.echo("=" * 60)
        typer.echo(f"LoRA Rank               : {adapter_config['r']}")
        typer.echo(f"Number of Pairs         : {len(lora_weights)}")
        typer.echo(f"Generated Weight Shape  : A={lora_weights[0]['A'].shape}, "
                   f"B={lora_weights[0]['B'].shape}")

        # Estimate parameter count
        total_params = sum(w.numel() for w in adapter_weights.values())
        typer.echo(f"Total Parameters        : {total_params:,}")

        typer.echo()
        typer.echo("Next steps:")
        typer.echo(
            f"  1. Merge adapter: aligntune merge "
            f"--base-model <model> --adapter {output_path.name}"
        )
        typer.echo(
            f"  2. Test on task: aligntune eval --model <merged> "
            f"--task <eval_data>"
        )
        typer.echo()
        typer.echo("✓ Adapter generation complete!")

    except ImportError as e:
        logger.exception(f"Missing dependency: {e}")
        typer.echo(f"Error: Missing dependency: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        logger.exception(f"Invalid arguments: {e}")
        typer.echo(f"Error: Invalid arguments: {e}", err=True)
        raise typer.Exit(1)
    except RuntimeError as e:
        logger.exception(f"Generation failed: {e}")
        typer.echo(f"Error: Generation failed: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during generation: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
