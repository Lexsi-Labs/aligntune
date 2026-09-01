"""
Comprehensive tests for model export pipeline.

Tests GGUF export, Ollama integration, HF Hub metadata, adapter merging,
and round-trip inference.
"""

import pytest
import tempfile
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aligntune.core.export import (
    BaseExporter,
    GGUFExporter,
    OllamaExporter,
    HFHubExporter,
    MergeAdapterExporter,
)
from aligntune.core.callbacks import ExportOnSaveCallback

logger = logging.getLogger(__name__)

# Fixtures

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_checkpoint(temp_dir):
    """Create a mock checkpoint with minimal model files."""
    checkpoint_dir = temp_dir / "checkpoint"
    checkpoint_dir.mkdir()

    model_dir = checkpoint_dir / "model"
    model_dir.mkdir()

    # Create minimal config
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create tokenizer files
    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_config = {"model_type": "llama", "vocab_size": 32000}
    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    # Create checkpoint metadata
    metadata = {
        "checkpoint_name": "test_checkpoint",
        "step": 100,
        "epoch": 1,
        "timestamp": "20240101_120000",
        "backend": "trl",
    }
    with open(checkpoint_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create training config
    training_config = {
        "backend": "trl",
        "output_dir": str(checkpoint_dir),
    }
    with open(checkpoint_dir / "training_config.json", "w") as f:
        json.dump(training_config, f)

    return checkpoint_dir


@pytest.fixture
def mock_gguf_file(temp_dir):
    """Create a mock GGUF file."""
    gguf_path = temp_dir / "model.gguf"
    gguf_path.write_text("GGUF_MOCK_CONTENT")
    return gguf_path


# Tests for BaseExporter

class _ConcreteExporter(BaseExporter):
    """Minimal concrete stub for testing BaseExporter's shared logic.

    BaseExporter is an ABC (prepare_model/export_model are @abstractmethod),
    so it can no longer be instantiated directly. None of the tests below
    exercise prepare_model/export_model - they test validate_checkpoint,
    get_checkpoint_metadata, _detect_backend, and cleanup, which are all
    concrete methods on BaseExporter itself - so trivial stub bodies are
    enough to make it instantiable.
    """

    def prepare_model(self, checkpoint_path, **kwargs):
        raise NotImplementedError

    def export_model(self, model, output_path, **kwargs):
        raise NotImplementedError


class TestBaseExporter:
    """Test base exporter functionality."""

    def test_validate_checkpoint_valid(self, mock_checkpoint):
        """Test checkpoint validation with valid checkpoint."""
        exporter = _ConcreteExporter(output_dir="/tmp")
        assert exporter.validate_checkpoint(mock_checkpoint)

    def test_validate_checkpoint_missing_files(self, temp_dir):
        """Test checkpoint validation with missing config."""
        invalid_checkpoint = temp_dir / "invalid"
        invalid_checkpoint.mkdir()
        exporter = _ConcreteExporter(output_dir="/tmp")
        assert not exporter.validate_checkpoint(invalid_checkpoint)

    def test_validate_checkpoint_nonexistent(self, temp_dir):
        """Test checkpoint validation with nonexistent path."""
        exporter = _ConcreteExporter(output_dir="/tmp")
        assert not exporter.validate_checkpoint(temp_dir / "nonexistent")

    def test_get_checkpoint_metadata(self, mock_checkpoint):
        """Test loading checkpoint metadata."""
        exporter = _ConcreteExporter(output_dir="/tmp")
        metadata = exporter.get_checkpoint_metadata(mock_checkpoint)

        assert metadata["checkpoint_name"] == "test_checkpoint"
        assert metadata["backend"] == "trl"
        assert metadata["step"] == 100

    def test_detect_backend_trl(self, mock_checkpoint):
        """Test TRL backend detection."""
        exporter = _ConcreteExporter(output_dir="/tmp")
        backend = exporter._detect_backend(mock_checkpoint)
        assert backend == "trl"

    def test_detect_backend_unsloth(self, temp_dir):
        """Test Unsloth backend detection."""
        checkpoint_dir = temp_dir / "checkpoint"
        checkpoint_dir.mkdir()
        model_dir = checkpoint_dir / "model"
        model_dir.mkdir()

        # Create adapter config to indicate Unsloth
        adapter_config = {"peft_type": "LORA"}
        with open(model_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)

        with open(model_dir / "config.json", "w") as f:
            json.dump({"vocab_size": 32000}, f)

        exporter = _ConcreteExporter(output_dir="/tmp")
        backend = exporter._detect_backend(checkpoint_dir)
        assert backend == "unsloth"

    def test_cleanup(self, temp_dir):
        """Test temporary file cleanup."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        exporter = _ConcreteExporter(output_dir="/tmp")
        exporter.cleanup(test_file, test_dir)

        assert not test_file.exists()
        assert not test_dir.exists()


# Tests for GGUFExporter

class TestGGUFExporter:
    """Test GGUF exporter functionality."""

    def test_init_default(self):
        """Test GGUF exporter initialization."""
        exporter = GGUFExporter()
        assert exporter.converter is None
        assert exporter.quantization is None

    def test_init_with_options(self):
        """Test GGUF exporter initialization with options."""
        exporter = GGUFExporter(
            converter="llama-cpp",
            quantization="Q4_K_M",
        )
        assert exporter.converter == "llama-cpp"
        assert exporter.quantization == "Q4_K_M"

    def test_check_unsloth_available(self):
        """Test Unsloth availability check."""
        exporter = GGUFExporter()
        # _check_unsloth_available() does `import unsloth` locally inside the
        # method (src/aligntune/core/export/gguf.py:115), not at module scope,
        # so there's no `aligntune.core.export.gguf.unsloth` attribute to
        # patch. Force the import itself to fail instead, via sys.modules.
        with patch.dict("sys.modules", {"unsloth": None}):
            assert not exporter._check_unsloth_available()

    def test_quantization_presets(self):
        """Test quantization preset validation."""
        from aligntune.core.export.gguf import GGUF_QUANT_PRESETS

        assert "Q4_K_M" in GGUF_QUANT_PRESETS
        assert "Q5_K_M" in GGUF_QUANT_PRESETS
        assert "Q8_0" in GGUF_QUANT_PRESETS

    def test_export_validation(self, mock_checkpoint, temp_dir):
        """Test export validation."""
        exporter = GGUFExporter(output_dir=temp_dir)

        # Should validate checkpoint
        assert exporter.validate_checkpoint(mock_checkpoint)

    @patch("aligntune.core.export.gguf.AutoModelForCausalLM")
    def test_prepare_model(self, mock_auto_model, mock_checkpoint, temp_dir):
        """Test model preparation."""
        exporter = GGUFExporter(output_dir=temp_dir)

        mock_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        # Note: This would require actual model files to work
        # For now we test that it attempts to load
        model_dir = mock_checkpoint / "model"
        with open(model_dir / "pytorch_model.bin", "wb") as f:
            f.write(b"mock_model_data")

        with patch.object(exporter, "prepare_model") as mock_prepare:
            mock_prepare.return_value = mock_model
            result = exporter.prepare_model(mock_checkpoint)
            mock_prepare.assert_called_once()


# Tests for OllamaExporter

class TestOllamaExporter:
    """Test Ollama exporter functionality."""

    def test_init_default(self):
        """Test Ollama exporter initialization."""
        exporter = OllamaExporter()
        assert exporter.model_name == "custom-model:latest"
        assert exporter.create_model is False

    def test_init_with_options(self):
        """Test Ollama exporter initialization with options."""
        exporter = OllamaExporter(
            gguf_path="/path/to/model.gguf",
            model_name="my-model:latest",
            create_model=True,
        )
        assert exporter.gguf_path == "/path/to/model.gguf"
        assert exporter.model_name == "my-model:latest"
        assert exporter.create_model is True

    def test_check_ollama_available(self):
        """Test Ollama availability check."""
        exporter = OllamaExporter()
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/ollama"
            assert exporter._check_ollama_available()
            mock_which.assert_called_with("ollama")

    def test_check_ollama_not_available(self):
        """Test Ollama availability check when not installed."""
        exporter = OllamaExporter()
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            assert not exporter._check_ollama_available()

    def test_generate_modelfile(self, mock_gguf_file, temp_dir):
        """Test Modelfile generation."""
        exporter = OllamaExporter(output_dir=temp_dir)
        modelfile_path = exporter._generate_modelfile(mock_gguf_file, temp_dir)

        assert Path(modelfile_path).exists()
        modelfile_content = Path(modelfile_path).read_text()
        assert "FROM" in modelfile_content
        assert "PARAMETER" in modelfile_content

    def test_generate_modelfile_missing_gguf(self, temp_dir):
        """Test Modelfile generation with missing GGUF."""
        exporter = OllamaExporter(output_dir=temp_dir)
        nonexistent_gguf = temp_dir / "nonexistent.gguf"

        with pytest.raises(FileNotFoundError):
            exporter._generate_modelfile(nonexistent_gguf, temp_dir)

    @patch("subprocess.run")
    def test_create_ollama_model(self, mock_run, mock_gguf_file, temp_dir):
        """Test creating Ollama model."""
        exporter = OllamaExporter(
            output_dir=temp_dir,
            model_name="test-model:latest",
        )

        modelfile_path = temp_dir / "Modelfile"
        modelfile_path.write_text("FROM ./model.gguf")

        mock_run.return_value = Mock(returncode=0, stdout="success")

        with patch.object(exporter, "_check_ollama_available", return_value=True):
            result = exporter._create_ollama_model(modelfile_path)

            # Note: subprocess.run is mocked, so we check the call
            mock_run.assert_called_once()


# Tests for HFHubExporter

class TestHFHubExporter:
    """Test HuggingFace Hub exporter functionality."""

    def test_init_default(self):
        """Test HF Hub exporter initialization."""
        exporter = HFHubExporter()
        assert exporter.adapter_only is False
        assert exporter.private is False

    def test_init_with_options(self):
        """Test HF Hub exporter initialization with options."""
        exporter = HFHubExporter(
            repo_id="username/model",
            adapter_only=True,
            private=True,
        )
        assert exporter.repo_id == "username/model"
        assert exporter.adapter_only is True
        assert exporter.private is True

    def test_validate_checkpoint(self, mock_checkpoint):
        """Test checkpoint validation for HF Hub."""
        exporter = HFHubExporter()
        assert exporter.validate_checkpoint(mock_checkpoint)

    def test_get_adapter_files_missing(self, mock_checkpoint):
        """Test adapter file detection when missing."""
        exporter = HFHubExporter(adapter_only=True)
        adapter_files = exporter._get_adapter_files(mock_checkpoint)
        assert len(adapter_files) == 0


# Tests for MergeAdapterExporter

class TestMergeAdapterExporter:
    """Test adapter merge exporter functionality."""

    def test_init_default(self):
        """Test merge adapter exporter initialization."""
        exporter = MergeAdapterExporter()
        assert exporter.output_dir is not None

    def test_validate_checkpoint(self, mock_checkpoint):
        """Test checkpoint validation for merge."""
        exporter = MergeAdapterExporter()
        assert exporter.validate_checkpoint(mock_checkpoint)


# Tests for ExportOnSaveCallback

class TestExportOnSaveCallback:
    """Test export callback functionality."""

    def test_init_disabled(self):
        """Test callback initialization disabled."""
        callback = ExportOnSaveCallback(export_config={"enabled": False})
        assert callback.enabled is False

    def test_init_enabled(self):
        """Test callback initialization enabled."""
        config = {
            "enabled": True,
            "format": "gguf",
            "quantization": "Q4_K_M",
        }
        callback = ExportOnSaveCallback(export_config=config)
        assert callback.enabled is True

    def test_on_save_disabled(self):
        """Test on_save when disabled."""
        callback = ExportOnSaveCallback(export_config={"enabled": False})
        # Should not raise
        callback.on_save(None, None, None)

    def test_on_save_missing_checkpoint_path(self):
        """Test on_save with missing checkpoint path."""
        callback = ExportOnSaveCallback(export_config={"enabled": True})
        # Should log warning but not crash
        callback.on_save(None, None, None)

    def test_on_save_with_checkpoint(self, mock_checkpoint):
        """Test on_save with valid checkpoint."""
        callback = ExportOnSaveCallback(
            export_config={"enabled": True, "format": "gguf"}
        )

        args = Mock(output_dir="/tmp")
        state = Mock(metrics={})
        control = Mock()

        # Mock the export to avoid actual export
        with patch.object(callback, "_export_checkpoint") as mock_export:
            mock_export.return_value = Path("/tmp/model.gguf")
            callback.on_save(args, state, control, checkpoint_path=str(mock_checkpoint))
            mock_export.assert_called_once()


# Integration tests

class TestExportIntegration:
    """Integration tests for export pipeline."""

    def test_gguf_export_with_mock(self, mock_checkpoint, temp_dir):
        """Test GGUF export flow with mocked model loading."""
        exporter = GGUFExporter(output_dir=temp_dir)

        # Mock the converter to avoid needing actual llama.cpp
        with patch.object(exporter, "_export_via_llama_cpp") as mock_convert:
            mock_convert.return_value = str(temp_dir / "model.gguf")

            result = exporter.export(mock_checkpoint)
            mock_convert.assert_called_once()
            assert str(result) == str(temp_dir / "model.gguf")

    def test_ollama_export_from_gguf(self, mock_gguf_file, temp_dir):
        """Test Ollama export from GGUF file."""
        exporter = OllamaExporter(output_dir=temp_dir)
        modelfile_path = exporter.export(gguf_path=mock_gguf_file)

        assert Path(modelfile_path).exists()
        content = Path(modelfile_path).read_text()
        assert "FROM" in content

    def test_export_callback_integration(self, mock_checkpoint, temp_dir):
        """Test export callback integration with training loop."""
        callback = ExportOnSaveCallback(
            export_config={"enabled": True, "format": "gguf"},
            export_dir=temp_dir,
        )

        args = Mock(output_dir=str(temp_dir))
        state = Mock(metrics={})
        control = Mock()

        # Mock export to test callback behavior
        with patch.object(callback, "_export_checkpoint") as mock_export:
            mock_export.return_value = Path(temp_dir) / "model.gguf"
            callback.on_save(args, state, control, checkpoint_path=str(mock_checkpoint))

            # Verify metrics were updated
            assert "export_gguf_path" in state.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
