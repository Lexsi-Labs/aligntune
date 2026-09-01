"""
MergekitMerger — thin wrapper around the `mergekit` library.

Mergekit handles the actual merge algorithms. This build supports:
- Basic: linear
- Task vectors: task_arithmetic
- RL-optimized: ram

This class only generates a mergekit YAML config from AlignTune
parameters and invokes mergekit via subprocess.

Install mergekit with:
    pip install mergekit
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, Any

import yaml

from .base import BaseMerger

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = [
    # Basic methods
    "linear",           # Simple weighted average

    # Task vector methods
    "task_arithmetic",  # Basic task vector merging

    # RL-optimized methods (for PPO/DPO agents)
    "ram",              # Reinforced Agent Merging (sparse RL task vectors)
]


def _require_mergekit() -> None:
    """Raise a helpful ImportError if mergekit is not installed."""
    try:
        import mergekit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mergekit is required for this merge method.\n"
            "Install it with:  pip install mergekit\n"
            "See https://github.com/arcee-ai/mergekit for details."
        ) from exc


class MergekitMerger(BaseMerger):
    """
    Wraps the mergekit library to perform model merges.

    Supported methods:
    - Basic: linear
    - Task vectors: task_arithmetic
    - RL-optimized: ram

    mergekit is NOT imported at module level so the rest of AlignTune continues
    to work even when mergekit is not installed.
    """

    def supports_method(self) -> list[str]:
        return list(SUPPORTED_METHODS)

    # ------------------------------------------------------------------
    # YAML config generation
    # ------------------------------------------------------------------

    def build_config(
        self,
        models: list[str],
        method: str,
        base_model: Optional[str] = None,
        weights: Optional[list[float]] = None,
        density: Optional[float] = None,
        epsilon: Optional[float] = None,
        t: Optional[float] = None,
        dtype: str = "bfloat16",
        global_params: Optional[Dict[str, Any]] = None,
        lora_adapters: Optional[list[Optional[str]]] = None,
    ) -> dict:
        """
        Build a mergekit config dict from AlignTune parameters.

        Args:
            models: List of model paths / HF IDs to merge. If lora_adapters provided, these are base models.
            method: One of SUPPORTED_METHODS.
            base_model: Base model path (required for task_arithmetic / ram).
            weights: Per-model weight list (must align with *models*).
            density: Unused by the supported methods (kept for API compatibility).
            epsilon: Unused by the supported methods (kept for API compatibility, except
                RAM's global_params, see below).
            t: Unused by the supported methods (kept for API compatibility).
            dtype: Target dtype for the merged model.
            global_params: Additional global parameters:
                - For RAM: {"epsilon": 1e-5} - threshold for unchanged parameters
            lora_adapters: Optional list of LoRA adapter paths, one per model. None = no adapter for that model.

        Returns:
            dict suitable for yaml.dump() → mergekit YAML.
        """
        method = method.lower()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported merge method '{method}'. "
                f"Supported: {SUPPORTED_METHODS}"
            )

        if weights is None:
            n = len(models)
            weights = [round(1.0 / n, 6)] * n

        if len(weights) != len(models):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({len(models)})."
            )

        # Validate base_model requirements
        methods_requiring_base = ("task_arithmetic", "ram")
        if method in methods_requiring_base:
            if base_model is None:
                logger.warning(
                    f"base_model not provided for {method}. Using first model as base. "
                    f"For task vector and RL methods, "
                    f"base should be the original pretrained model."
                )
                base_model = models[0]
            elif base_model in models:
                logger.warning(
                    f"base_model '{base_model}' is also in models list. "
                    f"For task vector and RL methods, base should typically be a separate pretrained model."
                )

        # Validate lora_adapters if provided
        if lora_adapters is not None:
            if len(lora_adapters) != len(models):
                raise ValueError(
                    f"Number of lora_adapters ({len(lora_adapters)}) must match "
                    f"number of models ({len(models)})."
                )

        # Build model list entries
        model_entries = []
        for i, (model, weight) in enumerate(zip(models, weights)):
            # Determine if we need nested model format (for LoRA support)
            lora_path = lora_adapters[i] if lora_adapters else None

            if lora_path is not None:
                # Use nested format for LoRA
                entry: dict = {
                    "model": {
                        "model": model,
                        "lora": lora_path,
                    }
                }
            else:
                # Use simple format (backward compatible)
                entry: dict = {"model": model}

            # Build parameters
            params: dict = {}

            # DELLA: weight + density + epsilon (all per-model)
            if method == "della":
                params["weight"] = weight
                if density is not None:
                    params["density"] = density
                if epsilon is not None:
                    params["epsilon"] = epsilon

            # TIES/DARE: weight + density (per-model)
            elif method in ("ties", "dare_ties"):
                params["weight"] = weight
                if density is not None:
                    params["density"] = density

            # Linear/Task Arithmetic: weight only (per-model)
            elif method in ("linear", "task_arithmetic"):
                params["weight"] = weight

            # RAM methods don't use per-model parameters (use global_params instead)
            # SLERP doesn't use per-model parameters (uses global t parameter)

            if params:
                entry["parameters"] = params
            model_entries.append(entry)

        config: dict = {
            "models": model_entries,
            "merge_method": method,
            "dtype": dtype,
        }

        # Add base_model for methods that require it
        if base_model is not None:
            config["base_model"] = base_model

        # Add global parameters
        if global_params:
            config["parameters"] = global_params
        elif method == "slerp":
            # SLERP uses global 't' parameter (default 0.5 if not provided)
            t_value = t if t is not None else 0.5
            config["parameters"] = {"t": t_value}

        return config

    def generate_yaml(
        self,
        models: list[str],
        method: str,
        **kwargs,
    ) -> str:
        """
        Generate a mergekit YAML string from AlignTune parameters.

        Args:
            models: Model paths / HF IDs.
            method: Merge method.
            **kwargs: Forwarded to build_config().

        Returns:
            YAML string.
        """
        config = self.build_config(models, method, **kwargs)
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Merge execution
    # ------------------------------------------------------------------

    def merge(
        self,
        models: list[str],
        output_path: str,
        method: str = "linear",
        base_model: Optional[str] = None,
        weights: Optional[list[float]] = None,
        density: Optional[float] = None,
        epsilon: Optional[float] = None,
        t: Optional[float] = None,
        dtype: str = "bfloat16",
        global_params: Optional[Dict[str, Any]] = None,
        lora_adapters: Optional[list[Optional[str]]] = None,
        extra_mergekit_args: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Merge models using mergekit.

        Generates a temporary mergekit YAML config, then invokes
        ``mergekit-yaml config.yaml <output_path>`` via subprocess.

        Args:
            models: Local paths or HuggingFace model IDs to merge. If lora_adapters provided, these are base models.
            output_path: Directory where the merged model will be saved.
            method: Merge method. Options:
                - "linear": Simple weighted average
                - "task_arithmetic": Basic task vector merging
                - "ram": Reinforced Agent Merging (for RL tasks)
            base_model: Base model (required for all methods except linear).
            weights: Per-model weights.
            density: Unused by the supported methods (kept for API compatibility).
            epsilon: Unused by the supported methods (kept for API compatibility, except
                RAM's global_params, see below).
            t: Unused by the supported methods (kept for API compatibility).
            dtype: Output dtype.
            global_params: Method-specific global parameters:
                - RAM: {"epsilon": 1e-5} - threshold for unchanged parameters
            lora_adapters: Optional list of LoRA adapter paths to merge. Must match length of models.
            extra_mergekit_args: Additional CLI args passed to mergekit-yaml.

        Returns:
            Absolute path to merged model directory.

        Raises:
            ImportError: If mergekit is not installed.
            RuntimeError: If mergekit-yaml subprocess exits non-zero.

        Examples:
            # Multilingual LoRA merge via task arithmetic
            >>> merger.merge(
            ...     models=["base_model", "base_model", "base_model"],
            ...     lora_adapters=["lora_en", "lora_hi", "lora_zh"],
            ...     method="task_arithmetic",
            ...     base_model="base_model",
            ...     weights=[0.33, 0.33, 0.34],
            ...     output_path="./multilingual_merged"
            ... )

            # RL agents merge with RAM
            >>> merger.merge(
            ...     models=["base"] * 3,
            ...     lora_adapters=["rl_task1", "rl_task2", "rl_task3"],
            ...     method="ram",
            ...     base_model="base",
            ...     global_params={"epsilon": 1e-5},
            ...     output_path="./rl_merged"
            ... )
        """
        _require_mergekit()

        self.validate_models(models)

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        yaml_str = self.generate_yaml(
            models=models,
            method=method,
            base_model=base_model,
            weights=weights,
            density=density,
            epsilon=epsilon,
            t=t,
            dtype=dtype,
            global_params=global_params,
            lora_adapters=lora_adapters,
        )

        logger.info("Generated mergekit YAML config:\n%s", yaml_str)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            prefix="aligntune_merge_",
        ) as tmp:
            tmp.write(yaml_str)
            config_path = tmp.name

        try:
            cmd = ["mergekit-yaml", config_path, str(output_path)]
            if extra_mergekit_args:
                cmd.extend(extra_mergekit_args)

            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                logger.info("mergekit stdout:\n%s", result.stdout)
            if result.stderr:
                logger.warning("mergekit stderr:\n%s", result.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    f"mergekit-yaml exited with code {result.returncode}.\n"
                    f"stderr: {result.stderr}"
                )
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass

        logger.info("Merge complete. Output: %s", output_path)
        return str(output_path.resolve())

    def merge_from_yaml(
        self,
        yaml_path: str,
        output_path: str,
        extra_mergekit_args: Optional[list[str]] = None,
    ) -> str:
        """
        Merge models using an existing mergekit YAML config file.

        This enables advanced features not supported by the basic API:
        - Weight gradients: weight: [0.3, 0.5, 0.7, 0.9, 1.0]
        - Filters: filter-specific parameters for mlp, self_attn, etc.
        - Layer ranges: layer_range: [0, 40]

        Only methods in SUPPORTED_METHODS are accepted; the YAML's
        `merge_method` is validated before mergekit is invoked.

        Args:
            yaml_path: Path to mergekit YAML config file.
            output_path: Directory where the merged model will be saved.
            extra_mergekit_args: Additional CLI args passed to mergekit-yaml.

        Returns:
            Absolute path to merged model directory.

        Raises:
            ImportError: If mergekit is not installed.
            FileNotFoundError: If YAML config file not found.
            RuntimeError: If mergekit-yaml subprocess exits non-zero.

        Example:
            >>> merger = MergekitMerger()
            >>> merger.merge_from_yaml(
            ...     yaml_path="./my_advanced_merge.yaml",
            ...     output_path="./merged_model"
            ... )
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found: {yaml_path}")

        # Read and log the config
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()

        parsed_config = yaml.safe_load(yaml_content) or {}
        yaml_method = parsed_config.get("merge_method")
        if yaml_method is not None and yaml_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Merge method '{yaml_method}' (from {yaml_path}) is not supported. "
                f"Supported: {SUPPORTED_METHODS}"
            )

        _require_mergekit()

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Using mergekit YAML config:\n%s", yaml_content)

        cmd = ["mergekit-yaml", str(yaml_path), str(output_path)]
        if extra_mergekit_args:
            cmd.extend(extra_mergekit_args)

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            logger.info("mergekit stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("mergekit stderr:\n%s", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"mergekit-yaml exited with code {result.returncode}.\n"
                f"stderr: {result.stderr}"
            )

        logger.info("Merge complete. Output: %s", output_path)
        return str(output_path.resolve())
