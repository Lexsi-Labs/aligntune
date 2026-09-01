"""
CLI command for verifying exported artifacts against baseline.

Provides `aligntune verify-export` command for quantization regression testing.
Verifies that quantized/exported artifacts haven't degraded in quality or
alignment compared to a baseline fp16 checkpoint.
"""

import logging
import typer
from pathlib import Path
from typing import Optional, List
import json
import yaml

from ..eval.quant_regression import (
    ExportedArtifact,
    RegressionThresholds,
    QuantRegressionRunner,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="verify-export",
    help="Verify exported artifacts haven't degraded vs. baseline.",
    no_args_is_help=True,
)


def _parse_artifact_spec(spec: str) -> ExportedArtifact:
    """
    Parse artifact spec string: 'name:format:path'.

    Example:
        Q4_K_M:gguf:./exports/gguf/model.gguf
        int8_hf:hf_8bit:./exports/hf/int8

    Args:
        spec: Artifact specification string

    Returns:
        ExportedArtifact instance

    Raises:
        ValueError: If spec format is invalid
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid artifact spec: '{spec}'. Expected 'name:format:path'"
        )

    name, format_type, path = parts
    name = name.strip()
    format_type = format_type.strip()
    path = path.strip()

    if not name or not format_type or not path:
        raise ValueError(
            f"Invalid artifact spec: '{spec}'. All fields must be non-empty"
        )

    # Validate format
    valid_formats = ["hf", "hf_4bit", "hf_8bit", "gguf", "ollama"]
    if format_type not in valid_formats:
        raise ValueError(
            f"Invalid format: '{format_type}'. Must be one of: {', '.join(valid_formats)}"
        )

    return ExportedArtifact(
        name=name,
        path=path,
        format=format_type,
    )


def _load_thresholds(path: Optional[Path]) -> Optional[RegressionThresholds]:
    """Load thresholds from YAML file, or None to use defaults."""
    if path is None:
        return None

    if not path.exists():
        typer.echo(f"Error: Threshold file not found: {path}", err=True)
        raise typer.Exit(1)

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Convert dict to RegressionThresholds if provided
        if config and "thresholds" in config:
            thresh_dict = config["thresholds"]
            return RegressionThresholds(**thresh_dict)
        elif config:
            return RegressionThresholds(**config)
        else:
            return None
    except Exception as e:
        typer.echo(f"Error loading thresholds from {path}: {e}", err=True)
        raise typer.Exit(1)


def _load_eval_config(path: Path):
    """Load eval config from YAML file."""
    if not path.exists():
        typer.echo(f"Error: Eval config file not found: {path}", err=True)
        raise typer.Exit(1)

    try:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        from ..eval.runner import EvalConfig
        return EvalConfig(**config_dict)
    except Exception as e:
        typer.echo(f"Error loading eval config from {path}: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def run(
    checkpoint: Path = typer.Argument(
        ..., help="Path to fp16 baseline checkpoint dir"
    ),
    artifact: List[str] = typer.Option(
        ...,
        "--artifact",
        "-a",
        help="Artifact spec: 'name:format:path' (repeatable). "
        "Example: -a Q4_K_M:gguf:./exports/gguf/model.gguf",
    ),
    probe_set: Path = typer.Option(..., help="JSON/JSONL probe set for auditor"),
    eval_config: Path = typer.Option(..., help="YAML EvalConfig"),
    thresholds: Optional[Path] = typer.Option(
        None, help="Optional threshold YAML"
    ),
    output_dir: Path = typer.Option("./reports/quant_regression"),
    format: str = typer.Option(
        "both", help="Output format: json|markdown|both"
    ),
):
    """
    Verify exported artifacts against baseline.

    Runs alignment audit and evaluation on each artifact, comparing metrics
    against baseline with configurable thresholds.

    Example:
        aligntune verify-export run ./checkpoint \\
          -a Q4_K_M:gguf:./exports/gguf/model.gguf \\
          -a int8:hf_8bit:./exports/hf_int8 \\
          --probe-set probes.json \\
          --eval-config eval.yaml

    Exit codes:
        0: All artifacts PASS regression tests
        1: Any artifact FAIL or config error
    """
    try:
        # Validate checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        # Parse artifact specs
        try:
            artifacts = [_parse_artifact_spec(spec) for spec in artifact]
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

        if not artifacts:
            typer.echo("Error: At least one artifact must be specified", err=True)
            raise typer.Exit(1)

        # Validate probe set
        probe_set = probe_set.resolve()
        if not probe_set.exists():
            typer.echo(f"Error: Probe set not found: {probe_set}", err=True)
            raise typer.Exit(1)

        # Load eval config
        eval_config_path = eval_config.resolve()
        eval_config_obj = _load_eval_config(eval_config_path)

        # Load thresholds
        threshold_obj = None
        if thresholds:
            threshold_obj = _load_thresholds(thresholds.resolve())

        # Validate output format
        if format not in ["json", "markdown", "both"]:
            typer.echo(
                f"Error: Invalid format '{format}'. Must be json|markdown|both",
                err=True,
            )
            raise typer.Exit(1)

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Print summary
        typer.echo("\n" + "=" * 70)
        typer.echo("Quantization Regression Verification")
        typer.echo("=" * 70)
        typer.echo(f"Baseline checkpoint: {checkpoint}")
        typer.echo(f"Artifacts to verify: {len(artifacts)}")
        for art in artifacts:
            typer.echo(f"  - {art.name} ({art.format}): {art.path}")
        typer.echo(f"Probe set: {probe_set}")
        typer.echo(f"Output directory: {output_dir}")
        typer.echo("=" * 70 + "\n")

        # Run regression test
        try:
            runner = QuantRegressionRunner(
                baseline_path=str(checkpoint),
                artifacts=artifacts,
                probe_set_path=str(probe_set),
                eval_config=eval_config_obj,
                thresholds=threshold_obj,
            )
            report = runner.run()
        except Exception as e:
            typer.echo(f"Error running regression test: {e}", err=True)
            logger.exception("Regression test failed")
            raise typer.Exit(1)

        # Output results
        if format in ["json", "both"]:
            json_path = output_dir / "regression_report.json"
            report.to_json(str(json_path))
            typer.echo(f"Saved JSON report: {json_path}")

        if format in ["markdown", "both"]:
            md_path = output_dir / "regression_report.md"
            report.to_markdown(str(md_path))
            typer.echo(f"Saved markdown report: {md_path}")

        # Print summary table
        typer.echo()
        report.print_table()

        # Determine exit code based on verdicts
        has_fail = any(v == "FAIL" for v in report.verdicts.values())
        exit_code = 1 if has_fail else 0

        typer.echo()
        if exit_code == 0:
            typer.echo("✓ All artifacts PASSED regression tests")
        else:
            typer.echo("✗ Some artifacts FAILED regression tests")

        raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Verification failed")
        raise typer.Exit(1)


@app.command()
def auto_discover(
    checkpoint: Path = typer.Argument(
        ..., help="Path to fp16 baseline checkpoint dir"
    ),
    exports_root: Path = typer.Argument(
        ..., help="Directory containing export subdirs (gguf/, hf/, ollama/, etc.)"
    ),
    probe_set: Path = typer.Option(..., help="JSON/JSONL probe set for auditor"),
    eval_config: Path = typer.Option(..., help="YAML EvalConfig"),
    thresholds: Optional[Path] = typer.Option(
        None, help="Optional threshold YAML"
    ),
    output_dir: Path = typer.Option("./reports/quant_regression"),
    format: str = typer.Option(
        "both", help="Output format: json|markdown|both"
    ),
):
    """
    Auto-discover exported artifacts and run regression verification.

    Walks exports_root for GGUF files and HF checkpoints, auto-detects format,
    and verifies against baseline.

    Example:
        aligntune verify-export auto-discover ./checkpoint ./exports \\
          --probe-set probes.json \\
          --eval-config eval.yaml

    Expected directory structure:
        exports/
          gguf/
            model.gguf
            model_Q4_K_M.gguf
            model_Q8_0.gguf
          hf/
            int8/
            int4/
          ollama/
    """
    try:
        # Validate inputs
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        exports_root = exports_root.resolve()
        if not exports_root.exists():
            typer.echo(f"Error: Exports root not found: {exports_root}", err=True)
            raise typer.Exit(1)

        probe_set = probe_set.resolve()
        if not probe_set.exists():
            typer.echo(f"Error: Probe set not found: {probe_set}", err=True)
            raise typer.Exit(1)

        eval_config_path = eval_config.resolve()
        eval_config_obj = _load_eval_config(eval_config_path)

        threshold_obj = None
        if thresholds:
            threshold_obj = _load_thresholds(thresholds.resolve())

        # Auto-discover artifacts
        artifacts = []

        # Look for GGUF files
        gguf_dir = exports_root / "gguf"
        if gguf_dir.exists():
            for gguf_file in gguf_dir.glob("*.gguf"):
                name = gguf_file.stem
                artifacts.append(
                    ExportedArtifact(
                        name=name,
                        path=str(gguf_file),
                        format="gguf",
                    )
                )
                typer.echo(f"Discovered GGUF: {name}")

        # Look for HF checkpoints
        hf_dir = exports_root / "hf"
        if hf_dir.exists():
            for quant_type in ["int8", "int4", "fp16"]:
                quant_dir = hf_dir / quant_type
                if quant_dir.exists() and (
                    quant_dir / "config.json"
                ).exists():  # Check if valid HF dir
                    format_type = f"hf_{quant_type}" if quant_type != "fp16" else "hf"
                    artifacts.append(
                        ExportedArtifact(
                            name=f"hf_{quant_type}",
                            path=str(quant_dir),
                            format=format_type,
                        )
                    )
                    typer.echo(f"Discovered HF checkpoint: {quant_type}")

        # Look for Ollama models
        ollama_dir = exports_root / "ollama"
        if ollama_dir.exists():
            for modelfile in ollama_dir.glob("Modelfile*"):
                name = modelfile.stem
                artifacts.append(
                    ExportedArtifact(
                        name=f"ollama_{name}",
                        path=str(modelfile),
                        format="ollama",
                    )
                )
                typer.echo(f"Discovered Ollama model: {name}")

        if not artifacts:
            typer.echo(
                f"Warning: No artifacts found in {exports_root}. Searched in: gguf/, hf/, ollama/",
                err=True,
            )
            typer.echo("Continuing with empty artifact list...", err=True)

        # Build artifact specs and call run command logic
        artifact_specs = [
            f"{art.name}:{art.format}:{art.path}" for art in artifacts
        ]

        # Call run with auto-discovered artifacts
        if artifact_specs:
            run(
                checkpoint=checkpoint,
                artifact=artifact_specs,
                probe_set=probe_set,
                eval_config=eval_config,
                thresholds=thresholds,
                output_dir=output_dir,
                format=format,
            )
        else:
            typer.echo("No artifacts to verify.", err=True)
            raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error in auto-discovery: {e}", err=True)
        logger.exception("Auto-discovery failed")
        raise typer.Exit(1)
