"""
Test cases for model merging functionality using mergekit.

Tests all merge methods: linear, task_arithmetic

Run with: pytest tests/test_mergekit_integration.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml

from aligntune.core.merge.mergekit_merger import MergekitMerger


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def merger():
    """Create MergekitMerger instance."""
    return MergekitMerger()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


# ============================================================
# TEST: LINEAR MERGE CONFIG GENERATION
# ============================================================

def test_linear_merge_config_basic(merger):
    """Test linear merge YAML config generation with default weights."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="linear"
    )

    assert config["merge_method"] == "linear"
    assert config["dtype"] == "bfloat16"
    assert len(config["models"]) == 2
    assert config["models"][0]["model"] == "model1"
    assert config["models"][1]["model"] == "model2"
    assert config["models"][0]["parameters"]["weight"] == 0.5
    assert config["models"][1]["parameters"]["weight"] == 0.5


def test_linear_merge_config_custom_weights(merger):
    """Test linear merge with custom weights."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="linear",
        weights=[0.7, 0.3]
    )

    assert config["models"][0]["parameters"]["weight"] == 0.7
    assert config["models"][1]["parameters"]["weight"] == 0.3


def test_linear_merge_config_three_models(merger):
    """Test linear merge with three models."""
    config = merger.build_config(
        models=["model1", "model2", "model3"],
        method="linear",
        weights=[0.5, 0.3, 0.2]
    )

    assert len(config["models"]) == 3
    assert config["models"][0]["parameters"]["weight"] == 0.5
    assert config["models"][1]["parameters"]["weight"] == 0.3
    assert config["models"][2]["parameters"]["weight"] == 0.2


# ============================================================
# TEST: TASK ARITHMETIC MERGE CONFIG GENERATION
# ============================================================

def test_task_arithmetic_merge_config(merger):
    """Test task arithmetic merge YAML config generation."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="task_arithmetic",
        base_model="base_model",
        weights=[1.0, -0.5]
    )

    assert config["merge_method"] == "task_arithmetic"
    assert config["base_model"] == "base_model"
    assert config["models"][0]["parameters"]["weight"] == 1.0
    assert config["models"][1]["parameters"]["weight"] == -0.5


# ============================================================
# TEST: YAML STRING GENERATION
# ============================================================

def test_generate_yaml_linear(merger):
    """Test YAML string generation for linear merge."""
    yaml_str = merger.generate_yaml(
        models=["model1", "model2"],
        method="linear",
        weights=[0.7, 0.3]
    )

    assert "merge_method: linear" in yaml_str
    assert "model: model1" in yaml_str
    assert "model: model2" in yaml_str
    assert "weight: 0.7" in yaml_str
    assert "weight: 0.3" in yaml_str
    assert "dtype: bfloat16" in yaml_str


# ============================================================
# TEST: YAML FILE READING
# ============================================================

def test_yaml_file_reading(temp_dir):
    """Test reading and validating YAML config file."""
    yaml_config = {
        "models": [
            {"model": "model1", "parameters": {"weight": 0.7}},
            {"model": "model2", "parameters": {"weight": 0.3}}
        ],
        "merge_method": "linear",
        "dtype": "bfloat16"
    }

    yaml_path = Path(temp_dir) / "merge_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f)

    with open(yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)

    assert loaded_config["merge_method"] == "linear"
    assert len(loaded_config["models"]) == 2
    assert loaded_config["models"][0]["parameters"]["weight"] == 0.7


def test_yaml_advanced_config(temp_dir):
    """Test advanced YAML config with gradients and filters."""
    yaml_config = {
        "models": [
            {
                "model": "model1",
                "parameters": {
                    "density": [1, 0.7, 0.1],
                    "weight": 0.5
                }
            },
            {
                "model": "model2",
                "parameters": {
                    "density": 0.5,
                    "weight": [0, 0.3, 0.7, 1]
                }
            }
        ],
        "merge_method": "task_arithmetic",
        "base_model": "base_model",
        "parameters": {
            "normalize": True
        },
        "dtype": "bfloat16"
    }

    yaml_path = Path(temp_dir) / "merge_advanced.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f)

    with open(yaml_path, 'r') as f:
        loaded_config = yaml.safe_load(f)

    assert loaded_config["merge_method"] == "task_arithmetic"
    assert loaded_config["parameters"]["normalize"] is True
    assert loaded_config["models"][0]["parameters"]["density"] == [1, 0.7, 0.1]
    assert loaded_config["models"][1]["parameters"]["weight"] == [0, 0.3, 0.7, 1]


# ============================================================
# TEST: ERROR HANDLING
# ============================================================

def test_invalid_method_raises_error(merger):
    """Test that invalid merge method raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported merge method"):
        merger.build_config(
            models=["model1", "model2"],
            method="invalid_method"
        )


def test_weight_mismatch_raises_error(merger):
    """Test that mismatched weights raises ValueError."""
    with pytest.raises(ValueError, match="Number of weights"):
        merger.build_config(
            models=["model1", "model2"],
            method="linear",
            weights=[0.7]
        )


def test_yaml_file_not_found(merger):
    """Test that non-existent YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        merger.merge_from_yaml(
            yaml_path="/nonexistent/path/config.yaml",
            output_path="./output"
        )


# ============================================================
# TEST: DTYPE OPTIONS
# ============================================================

@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_different_dtypes(merger, dtype):
    """Test config generation with different dtype options."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="linear",
        dtype=dtype
    )
    assert config["dtype"] == dtype


# ============================================================
# TEST: EDGE CASES
# ============================================================

def test_single_model(merger):
    """Test config generation with single model."""
    config = merger.build_config(
        models=["model1"],
        method="linear"
    )

    assert len(config["models"]) == 1
    assert config["models"][0]["parameters"]["weight"] == 1.0


def test_many_models(merger):
    """Test config generation with many models."""
    many_models = [f"model{i}" for i in range(10)]

    config = merger.build_config(
        models=many_models,
        method="linear"
    )

    assert len(config["models"]) == 10
    expected_weight = round(1.0 / 10, 6)
    assert config["models"][0]["parameters"]["weight"] == expected_weight


def test_zero_weights(merger):
    """Test config generation with zero weights."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="linear",
        weights=[0.0, 1.0]
    )

    assert config["models"][0]["parameters"]["weight"] == 0.0
    assert config["models"][1]["parameters"]["weight"] == 1.0


def test_negative_weights_task_arithmetic(merger):
    """Test task arithmetic with negative weights."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="task_arithmetic",
        base_model="base_model",
        weights=[1.0, -0.5]
    )

    assert config["models"][0]["parameters"]["weight"] == 1.0
    assert config["models"][1]["parameters"]["weight"] == -0.5


# ============================================================
# TEST: BASE MODEL HANDLING
# ============================================================

@pytest.mark.parametrize("method", ["task_arithmetic"])
def test_base_model_required_methods(merger, method):
    """Test that base_model is added for methods that need it."""
    config = merger.build_config(
        models=["model1", "model2"],
        method=method,
        base_model="base_model"
    )
    assert "base_model" in config
    assert config["base_model"] == "base_model"


def test_base_model_not_required_for_linear(merger):
    """Test that base_model can be omitted for linear merge."""
    config = merger.build_config(
        models=["model1", "model2"],
        method="linear"
    )

    if "base_model" in config:
        assert config["base_model"] is None or isinstance(config["base_model"], str)


# ============================================================
# RUN TESTS
# ============================================================

# ============================================================
# TEST: ACTUAL MODEL MERGING (INTEGRATION TESTS)
# ============================================================

@pytest.mark.integration
@pytest.mark.slow
def test_actual_merge_linear(temp_dir):
    """Test actual linear merge execution with GPT-2 models."""
    pytest.importorskip("mergekit", reason="mergekit not installed")

    from aligntune.core.backend_factory import merge_models

    output_path = Path(temp_dir) / "merged_linear"

    # Use same model twice (valid for testing merge pipeline)
    # Note: In production, you'd use different fine-tuned versions of same base
    merge_models(
        models=["gpt2", "gpt2"],
        output_path=str(output_path),
        method="linear",
        weights=[0.5, 0.5],
        dtype="float32",  # Smaller dtype for speed
    )

    # Verify merged model was created
    assert output_path.exists()
    assert (output_path / "config.json").exists()
    assert (output_path / "model.safetensors").exists() or (output_path / "pytorch_model.bin").exists()

    # Verify we can load the merged model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(str(output_path))
    tokenizer = AutoTokenizer.from_pretrained(str(output_path))

    assert model is not None
    assert tokenizer is not None


@pytest.mark.integration
@pytest.mark.slow
def test_actual_merge_task_arithmetic(temp_dir):
    """Test actual task arithmetic merge execution."""
    pytest.importorskip("mergekit", reason="mergekit not installed")

    from aligntune.core.backend_factory import merge_models

    output_path = Path(temp_dir) / "merged_task_arithmetic"

    merge_models(
        models=["gpt2", "gpt2"],
        output_path=str(output_path),
        method="task_arithmetic",
        base_model="gpt2",
        weights=[1.0, -0.5],  # Task arithmetic allows negative weights
        dtype="float32",
    )

    assert output_path.exists()
    assert (output_path / "config.json").exists()

    # Verify we can load the merged model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(output_path))
    assert model is not None


@pytest.mark.integration
@pytest.mark.slow
def test_actual_merge_three_models(temp_dir):
    """Test merging three models with linear method."""
    pytest.importorskip("mergekit", reason="mergekit not installed")

    from aligntune.core.backend_factory import merge_models

    output_path = Path(temp_dir) / "merged_three"

    merge_models(
        models=["gpt2", "gpt2", "gpt2"],
        output_path=str(output_path),
        method="linear",
        weights=[0.33, 0.33, 0.34],
        dtype="float32",
    )

    assert output_path.exists()
    assert (output_path / "config.json").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_actual_merge_with_inference(temp_dir):
    """Test that merged model can actually perform inference."""
    pytest.importorskip("mergekit", reason="mergekit not installed")

    from aligntune.core.backend_factory import merge_models

    output_path = Path(temp_dir) / "merged_inference"

    # Merge gpt2 models
    merge_models(
        models=["gpt2", "gpt2"],
        output_path=str(output_path),
        method="linear",
        weights=[0.5, 0.5],
        dtype="float32",
    )

    # Load and test inference
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(str(output_path))
    tokenizer = AutoTokenizer.from_pretrained(str(output_path))

    # Generate text
    input_text = "Hello, world!"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Should at least contain the input text
    assert input_text in generated_text
    # Should have generated something
    assert len(generated_text) > len(input_text)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
