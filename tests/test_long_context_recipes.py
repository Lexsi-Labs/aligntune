"""
Tests for Long Context SFT Recipes (v3.9).

Validates the three long-context YAML recipe files without executing any
model training or downloading model weights.  All tests are CPU-only and
only require PyYAML (already a transitive dependency of the project).

Recipes tested
--------------
- qwen25_7b_128k_longcontext.yaml   (YaRN scaling, 128k)
- llama3_1_8b_128k_longcontext.yaml (dynamic scaling, 128k)
- mistral_7b_32k_sliding_window.yaml (sliding-window, 32k, no RoPE extension)

Test coverage
-------------
- File existence and valid YAML syntax
- Top-level structure (``recipe`` and ``config`` sections present)
- Required recipe metadata fields (``name``, ``model``)
- ``config.model`` fields:
    - ``name_or_path``          (str)
    - ``attn_implementation``   (str, must be ``"flash_attention_2"``)
    - ``target_context_length`` (int, positive)
- RoPE scaling fields when applicable:
    - ``rope_scaling_type``  in {linear, dynamic, yarn, ntk}
    - ``rope_scaling_factor`` (float > 0)
- Sliding-window field for the Mistral recipe
- Long-context packing fields for the Qwen/Llama recipes
- ``config.train.max_seq_length`` matches ``target_context_length``
- Dataset lists are non-empty
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Absolute path to the recipes directory, resolved relative to this file.
RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes" / "configs" / "sft"

#: Valid RoPE scaling type identifiers.
VALID_ROPE_SCALING_TYPES = {"linear", "dynamic", "yarn", "ntk"}


def _load_recipe(filename: str) -> Dict[str, Any]:
    """Load and parse a YAML recipe file.

    Args:
        filename: Bare filename (e.g. ``"qwen25_7b_128k_longcontext.yaml"``).

    Returns:
        Parsed YAML as a Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist in :data:`RECIPES_DIR`.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = RECIPES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Recipe file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Parametrised fixtures
# ---------------------------------------------------------------------------

RECIPE_FILES = [
    "qwen25_7b_128k_longcontext.yaml",
    "llama3_1_8b_128k_longcontext.yaml",
    "mistral_7b_32k_sliding_window.yaml",
]


@pytest.fixture(params=RECIPE_FILES, ids=lambda f: f.replace(".yaml", ""))
def recipe(request: pytest.FixtureRequest) -> Dict[str, Any]:
    """Parametrised fixture that yields each parsed recipe dict."""
    return _load_recipe(request.param)


# ---------------------------------------------------------------------------
# Generic schema tests  (run for every recipe)
# ---------------------------------------------------------------------------


class TestRecipeTopLevelStructure:
    """All recipes must have the expected top-level keys."""

    def test_has_recipe_section(self, recipe: Dict[str, Any]) -> None:
        """Top-level 'recipe' key must be present."""
        assert "recipe" in recipe, "Missing top-level 'recipe' section"

    def test_has_config_section(self, recipe: Dict[str, Any]) -> None:
        """Top-level 'config' key must be present."""
        assert "config" in recipe, "Missing top-level 'config' section"

    def test_recipe_name_is_string(self, recipe: Dict[str, Any]) -> None:
        """recipe.name must be a non-empty string."""
        name = recipe["recipe"].get("name")
        assert isinstance(name, str) and name.strip(), (
            f"recipe.name must be a non-empty string, got: {name!r}"
        )

    def test_recipe_model_is_string(self, recipe: Dict[str, Any]) -> None:
        """recipe.model must be a non-empty string (HF model identifier)."""
        model = recipe["recipe"].get("model")
        assert isinstance(model, str) and model.strip(), (
            f"recipe.model must be a non-empty string, got: {model!r}"
        )

    def test_config_algo_is_sft(self, recipe: Dict[str, Any]) -> None:
        """config.algo must be 'sft'."""
        algo = recipe["config"].get("algo")
        assert algo == "sft", f"config.algo must be 'sft', got: {algo!r}"

    def test_config_model_section_present(self, recipe: Dict[str, Any]) -> None:
        """config.model section must exist."""
        assert "model" in recipe["config"], "Missing config.model section"

    def test_config_train_section_present(self, recipe: Dict[str, Any]) -> None:
        """config.train section must exist."""
        assert "train" in recipe["config"], "Missing config.train section"

    def test_config_datasets_section_present(self, recipe: Dict[str, Any]) -> None:
        """config.datasets section must exist and be a non-empty list."""
        datasets = recipe["config"].get("datasets")
        assert isinstance(datasets, list) and len(datasets) > 0, (
            "config.datasets must be a non-empty list"
        )


class TestModelConfigFields:
    """Validate required fields inside config.model."""

    def test_name_or_path_present(self, recipe: Dict[str, Any]) -> None:
        """config.model.name_or_path must be a non-empty string."""
        nop = recipe["config"]["model"].get("name_or_path")
        assert isinstance(nop, str) and nop.strip(), (
            f"config.model.name_or_path must be a non-empty string, got: {nop!r}"
        )

    def test_attn_implementation_flash_attention_2(self, recipe: Dict[str, Any]) -> None:
        """config.model.attn_implementation must be 'flash_attention_2'."""
        impl = recipe["config"]["model"].get("attn_implementation")
        assert impl == "flash_attention_2", (
            f"config.model.attn_implementation must be 'flash_attention_2', got: {impl!r}"
        )

    def test_target_context_length_present_and_positive(self, recipe: Dict[str, Any]) -> None:
        """config.model.target_context_length must be a positive integer."""
        tcl = recipe["config"]["model"].get("target_context_length")
        assert isinstance(tcl, int) and tcl > 0, (
            f"config.model.target_context_length must be a positive int, got: {tcl!r}"
        )

    def test_precision_is_bf16(self, recipe: Dict[str, Any]) -> None:
        """config.model.precision must be 'bf16'."""
        precision = recipe["config"]["model"].get("precision")
        assert precision == "bf16", (
            f"config.model.precision must be 'bf16', got: {precision!r}"
        )


class TestTrainConfigFields:
    """config.train must include required training hyperparameters."""

    def test_max_seq_length_present_and_positive(self, recipe: Dict[str, Any]) -> None:
        """config.train.max_seq_length must be a positive integer."""
        msl = recipe["config"]["train"].get("max_seq_length")
        assert isinstance(msl, int) and msl > 0, (
            f"config.train.max_seq_length must be a positive int, got: {msl!r}"
        )

    def test_max_seq_length_matches_target_context_length(
        self, recipe: Dict[str, Any]
    ) -> None:
        """config.train.max_seq_length must equal config.model.target_context_length."""
        tcl = recipe["config"]["model"].get("target_context_length")
        msl = recipe["config"]["train"].get("max_seq_length")
        assert msl == tcl, (
            f"config.train.max_seq_length ({msl}) must equal "
            f"config.model.target_context_length ({tcl})"
        )

    def test_learning_rate_positive(self, recipe: Dict[str, Any]) -> None:
        """config.train.learning_rate must be a positive number."""
        lr = recipe["config"]["train"].get("learning_rate")
        assert isinstance(lr, (int, float)) and lr > 0, (
            f"config.train.learning_rate must be a positive number, got: {lr!r}"
        )


# ---------------------------------------------------------------------------
# Recipe-specific tests
# ---------------------------------------------------------------------------


class TestQwen25LongContext:
    """Specific validations for qwen25_7b_128k_longcontext.yaml."""

    FILENAME = "qwen25_7b_128k_longcontext.yaml"

    @pytest.fixture
    def qwen_recipe(self) -> Dict[str, Any]:
        return _load_recipe(self.FILENAME)

    def test_model_identifier(self, qwen_recipe: Dict[str, Any]) -> None:
        """Model must be Qwen2.5-7B-Instruct."""
        nop = qwen_recipe["config"]["model"]["name_or_path"]
        assert "Qwen2.5-7B" in nop, f"Unexpected model: {nop!r}"

    def test_target_context_length_128k(self, qwen_recipe: Dict[str, Any]) -> None:
        """target_context_length must be 131072 (128 × 1024)."""
        tcl = qwen_recipe["config"]["model"]["target_context_length"]
        assert tcl == 131072, f"Expected 131072, got {tcl}"

    def test_rope_scaling_type_yarn(self, qwen_recipe: Dict[str, Any]) -> None:
        """rope_scaling_type must be 'yarn'."""
        rst = qwen_recipe["config"]["model"].get("rope_scaling_type")
        assert rst == "yarn", f"Expected 'yarn', got {rst!r}"

    def test_rope_scaling_type_in_valid_set(self, qwen_recipe: Dict[str, Any]) -> None:
        """rope_scaling_type must belong to the valid set."""
        rst = qwen_recipe["config"]["model"].get("rope_scaling_type")
        assert rst in VALID_ROPE_SCALING_TYPES, (
            f"rope_scaling_type '{rst}' not in {VALID_ROPE_SCALING_TYPES}"
        )

    def test_rope_scaling_factor(self, qwen_recipe: Dict[str, Any]) -> None:
        """rope_scaling_factor must be 4.0."""
        rsf = qwen_recipe["config"]["model"].get("rope_scaling_factor")
        assert isinstance(rsf, (int, float)) and rsf == pytest.approx(4.0), (
            f"Expected 4.0, got {rsf!r}"
        )

    def test_long_context_packing_enabled(self, qwen_recipe: Dict[str, Any]) -> None:
        """long_context_packing must be True."""
        packing = qwen_recipe["config"]["model"].get("long_context_packing")
        assert packing is True, f"Expected True, got {packing!r}"

    def test_packing_stride_zero(self, qwen_recipe: Dict[str, Any]) -> None:
        """packing_stride must be 0 (contiguous packing)."""
        stride = qwen_recipe["config"]["model"].get("packing_stride")
        assert stride == 0, f"Expected 0, got {stride!r}"

    def test_datasets_include_longalpaca(self, qwen_recipe: Dict[str, Any]) -> None:
        """Dataset list must contain longalpaca."""
        dataset_names = [d["name"] for d in qwen_recipe["config"]["datasets"]]
        assert "longalpaca" in dataset_names, (
            f"'longalpaca' not in datasets: {dataset_names}"
        )

    def test_datasets_include_books3(self, qwen_recipe: Dict[str, Any]) -> None:
        """Dataset list must contain books3_chunks."""
        dataset_names = [d["name"] for d in qwen_recipe["config"]["datasets"]]
        assert "books3_chunks" in dataset_names, (
            f"'books3_chunks' not in datasets: {dataset_names}"
        )

    def test_datasets_include_arxiv(self, qwen_recipe: Dict[str, Any]) -> None:
        """Dataset list must contain arxiv_chunks."""
        dataset_names = [d["name"] for d in qwen_recipe["config"]["datasets"]]
        assert "arxiv_chunks" in dataset_names, (
            f"'arxiv_chunks' not in datasets: {dataset_names}"
        )


class TestLlama31LongContext:
    """Specific validations for llama3_1_8b_128k_longcontext.yaml."""

    FILENAME = "llama3_1_8b_128k_longcontext.yaml"

    @pytest.fixture
    def llama_recipe(self) -> Dict[str, Any]:
        return _load_recipe(self.FILENAME)

    def test_model_identifier(self, llama_recipe: Dict[str, Any]) -> None:
        """Model must be Meta-Llama-3.1-8B-Instruct."""
        nop = llama_recipe["config"]["model"]["name_or_path"]
        assert "Llama-3.1-8B" in nop, f"Unexpected model: {nop!r}"

    def test_target_context_length_128k(self, llama_recipe: Dict[str, Any]) -> None:
        """target_context_length must be 131072."""
        tcl = llama_recipe["config"]["model"]["target_context_length"]
        assert tcl == 131072, f"Expected 131072, got {tcl}"

    def test_rope_scaling_type_dynamic(self, llama_recipe: Dict[str, Any]) -> None:
        """rope_scaling_type must be 'dynamic'."""
        rst = llama_recipe["config"]["model"].get("rope_scaling_type")
        assert rst == "dynamic", f"Expected 'dynamic', got {rst!r}"

    def test_rope_scaling_type_in_valid_set(self, llama_recipe: Dict[str, Any]) -> None:
        """rope_scaling_type must belong to the valid set."""
        rst = llama_recipe["config"]["model"].get("rope_scaling_type")
        assert rst in VALID_ROPE_SCALING_TYPES, (
            f"rope_scaling_type '{rst}' not in {VALID_ROPE_SCALING_TYPES}"
        )

    def test_rope_scaling_factor(self, llama_recipe: Dict[str, Any]) -> None:
        """rope_scaling_factor must be 4.0."""
        rsf = llama_recipe["config"]["model"].get("rope_scaling_factor")
        assert isinstance(rsf, (int, float)) and rsf == pytest.approx(4.0), (
            f"Expected 4.0, got {rsf!r}"
        )

    def test_long_context_packing_enabled(self, llama_recipe: Dict[str, Any]) -> None:
        """long_context_packing must be True."""
        packing = llama_recipe["config"]["model"].get("long_context_packing")
        assert packing is True, f"Expected True, got {packing!r}"


class TestMistralSlidingWindow:
    """Specific validations for mistral_7b_32k_sliding_window.yaml."""

    FILENAME = "mistral_7b_32k_sliding_window.yaml"

    @pytest.fixture
    def mistral_recipe(self) -> Dict[str, Any]:
        return _load_recipe(self.FILENAME)

    def test_model_identifier(self, mistral_recipe: Dict[str, Any]) -> None:
        """Model must be Mistral-7B-Instruct-v0.3."""
        nop = mistral_recipe["config"]["model"]["name_or_path"]
        assert "Mistral-7B" in nop, f"Unexpected model: {nop!r}"

    def test_target_context_length_32k(self, mistral_recipe: Dict[str, Any]) -> None:
        """target_context_length must be 32768 (32k)."""
        tcl = mistral_recipe["config"]["model"]["target_context_length"]
        assert tcl == 32768, f"Expected 32768, got {tcl}"

    def test_sliding_window_size(self, mistral_recipe: Dict[str, Any]) -> None:
        """sliding_window_size must be 4096."""
        sws = mistral_recipe["config"]["model"].get("sliding_window_size")
        assert isinstance(sws, int) and sws == 4096, (
            f"Expected 4096, got {sws!r}"
        )

    def test_no_rope_scaling_type(self, mistral_recipe: Dict[str, Any]) -> None:
        """Mistral recipe must NOT set rope_scaling_type (uses built-in SWA)."""
        rst = mistral_recipe["config"]["model"].get("rope_scaling_type")
        assert rst is None, (
            "Mistral uses built-in sliding-window attention and must not set "
            f"rope_scaling_type, got: {rst!r}"
        )

    def test_no_rope_scaling_factor(self, mistral_recipe: Dict[str, Any]) -> None:
        """Mistral recipe must NOT set rope_scaling_factor."""
        rsf = mistral_recipe["config"]["model"].get("rope_scaling_factor")
        assert rsf is None, (
            "Mistral recipe must not set rope_scaling_factor, "
            f"got: {rsf!r}"
        )

    def test_datasets_include_longalpaca(self, mistral_recipe: Dict[str, Any]) -> None:
        """Dataset list must contain longalpaca."""
        dataset_names = [d["name"] for d in mistral_recipe["config"]["datasets"]]
        assert "longalpaca" in dataset_names, (
            f"'longalpaca' not in datasets: {dataset_names}"
        )


# ---------------------------------------------------------------------------
# File-level smoke tests
# ---------------------------------------------------------------------------


class TestRecipeFilesExist:
    """Verify all recipe YAML files are present on disk."""

    @pytest.mark.parametrize("filename", RECIPE_FILES)
    def test_file_exists(self, filename: str) -> None:
        """Recipe file must exist at the expected path."""
        path = RECIPES_DIR / filename
        assert path.exists(), f"Recipe file not found: {path}"

    @pytest.mark.parametrize("filename", RECIPE_FILES)
    def test_file_is_valid_yaml(self, filename: str) -> None:
        """Recipe file must be parseable as valid YAML."""
        path = RECIPES_DIR / filename
        with open(path, "r", encoding="utf-8") as fh:
            content = yaml.safe_load(fh)
        assert isinstance(content, dict), (
            f"{filename} did not parse to a dict (got {type(content).__name__})"
        )

    @pytest.mark.parametrize("filename", RECIPE_FILES)
    def test_file_is_non_empty(self, filename: str) -> None:
        """Recipe file must have a non-zero file size."""
        path = RECIPES_DIR / filename
        assert path.stat().st_size > 0, f"Recipe file is empty: {path}"
