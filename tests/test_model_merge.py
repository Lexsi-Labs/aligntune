"""
Tests for the AlignTune model merging subsystem.

Tests cover:
- MergekitMerger YAML config generation (no actual mergekit execution)
- PEFTMerger with a tiny model + LoRA adapter (mocked)
- CLI argument parsing
- Method validation
- Model path validation
- Mocked subprocess for mergekit
"""

import os
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_dir(tmp_path: Path, name: str) -> str:
    """Create a minimal fake model directory and return its path."""
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
    return str(model_dir)


# ---------------------------------------------------------------------------
# BaseMerger — validate_models
# ---------------------------------------------------------------------------


class TestValidateModels:
    """Tests for BaseMerger.validate_models()."""

    def _merger(self):
        from aligntune.core.merge import MergekitMerger

        return MergekitMerger()

    def test_valid_local_path(self, tmp_path):
        path = _make_model_dir(tmp_path, "model_a")
        merger = self._merger()
        assert merger.validate_models([path]) is True

    def test_valid_hf_id(self):
        merger = self._merger()
        # HF IDs look like "org/model-name" — no network call needed
        assert merger.validate_models(["meta-llama/Llama-3-8B"]) is True

    def test_invalid_path_raises(self):
        merger = self._merger()
        with pytest.raises(ValueError, match="does not exist"):
            merger.validate_models(["/nonexistent/path/model"])

    def test_empty_list_raises(self):
        merger = self._merger()
        with pytest.raises(ValueError, match="At least one model"):
            merger.validate_models([])

    def test_mixed_valid_invalid_raises(self, tmp_path):
        valid = _make_model_dir(tmp_path, "model_ok")
        merger = self._merger()
        with pytest.raises(ValueError):
            merger.validate_models([valid, "/bad/path/model"])


# ---------------------------------------------------------------------------
# MergekitMerger — config generation
# ---------------------------------------------------------------------------


class TestMergekitMergerConfigGeneration:
    """Tests for MergekitMerger.build_config() and generate_yaml()."""

    @pytest.fixture()
    def merger(self):
        from aligntune.core.merge import MergekitMerger

        return MergekitMerger()

    def test_linear_config_structure(self, merger):
        config = merger.build_config(
            models=["org/a", "org/b"],
            method="linear",
            weights=[0.5, 0.5],
        )
        assert config["merge_method"] == "linear"
        for entry in config["models"]:
            assert entry["parameters"]["weight"] == pytest.approx(0.5)

    def test_task_arithmetic_config(self, merger):
        config = merger.build_config(
            models=["org/base", "org/finetuned"],
            method="task_arithmetic",
            weights=[1.0, 0.5],
        )
        assert config["merge_method"] == "task_arithmetic"

    def test_default_equal_weights(self, merger):
        config = merger.build_config(
            models=["org/a", "org/b", "org/c"],
            method="linear",
        )
        weights = [e["parameters"]["weight"] for e in config["models"]]
        total = sum(weights)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_invalid_method_raises(self, merger):
        with pytest.raises(ValueError, match="Unsupported merge method"):
            merger.build_config(models=["org/a"], method="magic_merge")

    def test_weight_mismatch_raises(self, merger):
        with pytest.raises(ValueError, match="Number of weights"):
            merger.build_config(
                models=["org/a", "org/b"],
                method="linear",
                weights=[0.5],
            )

    def test_generate_yaml_is_valid_yaml(self, merger):
        yaml_str = merger.generate_yaml(
            models=["org/a", "org/b"],
            method="linear",
            weights=[0.5, 0.5],
        )
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert "merge_method" in parsed
        assert "models" in parsed

    def test_custom_dtype(self, merger):
        config = merger.build_config(
            models=["org/a", "org/b"],
            method="linear",
            dtype="float16",
        )
        assert config["dtype"] == "float16"


# ---------------------------------------------------------------------------
# MergekitMerger — subprocess mocking
# ---------------------------------------------------------------------------


class TestMergekitMergerSubprocess:
    """Tests for MergekitMerger.merge() with mocked subprocess and mergekit."""

    @pytest.fixture()
    def merger(self):
        from aligntune.core.merge import MergekitMerger

        return MergekitMerger()

    @pytest.fixture()
    def model_dirs(self, tmp_path):
        return [
            _make_model_dir(tmp_path, "model_a"),
            _make_model_dir(tmp_path, "model_b"),
        ]

    def test_merge_calls_subprocess(self, merger, tmp_path, model_dirs):
        output_dir = str(tmp_path / "merged")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "mergekit done"
        mock_result.stderr = ""

        with patch("aligntune.core.merge.mergekit_merger._require_mergekit"):
            with patch(
                "aligntune.core.merge.mergekit_merger.subprocess.run",
                return_value=mock_result,
            ) as mock_run:
                result = merger.merge(
                    models=model_dirs,
                    output_path=output_dir,
                    method="linear",
                    weights=[0.5, 0.5],
                )

        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "mergekit-yaml"
        assert output_dir in cmd_args[-1] or Path(output_dir).resolve().as_posix() in cmd_args[-1]
        assert Path(result).resolve() == Path(output_dir).resolve()

    def test_merge_raises_on_nonzero_exit(self, merger, tmp_path, model_dirs):
        output_dir = str(tmp_path / "merged")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: model not found"

        with patch("aligntune.core.merge.mergekit_merger._require_mergekit"):
            with patch(
                "aligntune.core.merge.mergekit_merger.subprocess.run",
                return_value=mock_result,
            ):
                with pytest.raises(RuntimeError, match="mergekit-yaml exited"):
                    merger.merge(
                        models=model_dirs,
                        output_path=output_dir,
                        method="linear",
                        weights=[0.5, 0.5],
                    )

    def test_merge_raises_when_mergekit_missing(self, merger, tmp_path, model_dirs):
        output_dir = str(tmp_path / "merged")

        with patch(
            "aligntune.core.merge.mergekit_merger._require_mergekit",
            side_effect=ImportError("mergekit is required"),
        ):
            with pytest.raises(ImportError, match="mergekit"):
                merger.merge(
                    models=model_dirs,
                    output_path=output_dir,
                    method="linear",
                    weights=[0.5, 0.5],
                )

    def test_mergekit_yaml_tempfile_cleaned_up(self, merger, tmp_path, model_dirs):
        """Verify the temporary YAML config file is deleted after merge."""
        output_dir = str(tmp_path / "merged")
        created_tmp_files: List[str] = []

        original_run = __import__("subprocess").run

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        def capture_run(cmd, **kwargs):
            # The config file path is cmd[1]
            created_tmp_files.append(cmd[1])
            return mock_result

        with patch("aligntune.core.merge.mergekit_merger._require_mergekit"):
            with patch(
                "aligntune.core.merge.mergekit_merger.subprocess.run",
                side_effect=capture_run,
            ):
                merger.merge(
                    models=model_dirs,
                    output_path=output_dir,
                    method="linear",
                    weights=[0.5, 0.5],
                )

        assert created_tmp_files, "No subprocess call was recorded"
        for tmp_file in created_tmp_files:
            assert not Path(tmp_file).exists(), (
                f"Temporary config file was not cleaned up: {tmp_file}"
            )


# ---------------------------------------------------------------------------
# MergekitMerger — supports_method
# ---------------------------------------------------------------------------


class TestMergekitSupportsMethod:
    def test_supported_methods(self):
        from aligntune.core.merge import MergekitMerger

        merger = MergekitMerger()
        methods = merger.supports_method()
        assert "linear" in methods
        assert "task_arithmetic" in methods

    def test_lora_not_supported_by_mergekit(self):
        from aligntune.core.merge import MergekitMerger

        merger = MergekitMerger()
        assert "lora-merge" not in merger.supports_method()


# ---------------------------------------------------------------------------
# PEFTMerger
# ---------------------------------------------------------------------------


class TestPEFTMerger:
    """Tests for PEFTMerger using mocked peft / transformers."""

    @pytest.fixture()
    def merger(self):
        from aligntune.core.merge import PEFTMerger

        return PEFTMerger()

    def test_supports_only_lora_merge(self, merger):
        assert merger.supports_method() == ["lora-merge"]

    def test_wrong_method_raises(self, merger, tmp_path):
        with pytest.raises(ValueError, match="lora-merge"):
            merger.merge(
                models=["org/base"],
                output_path=str(tmp_path / "out"),
                method="linear",
            )

    def test_merge_with_adapter(self, merger, tmp_path):
        base_dir = _make_model_dir(tmp_path, "base")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = str(tmp_path / "merged")

        # Build mock objects. peft_merger.py does `isinstance(model, peft.PeftModel)`,
        # which requires peft.PeftModel to be an actual type - a plain MagicMock()
        # instance can't be used as an isinstance() target. Use a MagicMock
        # subclass instead so it's a real class, and make mock_peft_model an
        # instance of it.
        class _FakePeftModel(MagicMock):
            pass

        mock_merged_model = MagicMock()
        mock_peft_model = _FakePeftModel()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model

        mock_base_model = MagicMock()

        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel = _FakePeftModel
        mock_peft_module.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model)

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = (
            mock_base_model
        )
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch(
            "aligntune.core.merge.peft_merger._require_peft",
            return_value=mock_peft_module,
        ):
            with patch(
                "aligntune.core.merge.peft_merger._require_transformers",
                return_value=mock_transformers,
            ):
                result = merger.merge(
                    models=[base_dir],
                    output_path=output_dir,
                    adapter_path=str(adapter_dir),
                )

        mock_peft_module.PeftModel.from_pretrained.assert_called_once()
        mock_peft_model.merge_and_unload.assert_called_once()
        mock_merged_model.save_pretrained.assert_called_once()
        assert Path(result).resolve() == Path(output_dir).resolve()

    def test_merge_raises_when_peft_missing(self, merger, tmp_path):
        base_dir = _make_model_dir(tmp_path, "base")
        with patch(
            "aligntune.core.merge.peft_merger._require_peft",
            side_effect=ImportError("peft required"),
        ):
            with pytest.raises(ImportError, match="peft"):
                merger.merge(
                    models=[base_dir],
                    output_path=str(tmp_path / "out"),
                )


# ---------------------------------------------------------------------------
# CLI — argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Tests for src/aligntune/cli/merge.py via typer test client."""

    @pytest.fixture()
    def cli_runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    @pytest.fixture()
    def merge_app(self):
        from aligntune.cli.merge import app

        return app

    def test_help_shows_methods(self, cli_runner, merge_app):
        result = cli_runner.invoke(merge_app, ["--help"])
        assert result.exit_code == 0
        assert "linear" in result.output or "method" in result.output.lower()

    def test_missing_required_args_exits_nonzero(self, cli_runner, merge_app):
        # Missing --output and --models; should fail with an error exit code.
        result = cli_runner.invoke(merge_app, ["--method", "linear"])
        assert result.exit_code != 0

    def test_invalid_method_exits_nonzero(self, cli_runner, merge_app):
        result = cli_runner.invoke(
            merge_app,
            [
                "--method", "nonexistent_method",
                "--models", "org/a",
                "--output", "./out",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown merge method" in result.output or "Error" in result.output

    def test_valid_linear_invocation_dispatches(self, cli_runner, merge_app, tmp_path):
        model_a = _make_model_dir(tmp_path, "a")
        model_b = _make_model_dir(tmp_path, "b")
        output_dir = str(tmp_path / "merged")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("aligntune.core.merge.mergekit_merger._require_mergekit"):
            with patch(
                "aligntune.core.merge.mergekit_merger.subprocess.run",
                return_value=mock_result,
            ):
                result = cli_runner.invoke(
                    merge_app,
                    [
                        "--method", "linear",
                        "--models", model_a,
                        "--models", model_b,
                        "--output", output_dir,
                        "--weights", "0.5",
                        "--weights", "0.5",
                    ],
                )

        assert result.exit_code == 0, result.output
        assert "Merge complete" in result.output

    def test_lora_merge_invocation(self, cli_runner, merge_app, tmp_path):
        base_dir = _make_model_dir(tmp_path, "base")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = str(tmp_path / "merged")

        # peft_merger.py does `isinstance(model, peft.PeftModel)`, which
        # requires peft.PeftModel to be an actual type - a plain MagicMock()
        # instance can't serve as an isinstance() target. Use a MagicMock
        # subclass instead so it's a real class.
        class _FakePeftModel(MagicMock):
            pass

        mock_merged_model = MagicMock()
        mock_peft_model = _FakePeftModel()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model

        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel = _FakePeftModel
        mock_peft_module.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model)

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch(
            "aligntune.core.merge.peft_merger._require_peft",
            return_value=mock_peft_module,
        ):
            with patch(
                "aligntune.core.merge.peft_merger._require_transformers",
                return_value=mock_transformers,
            ):
                result = cli_runner.invoke(
                    merge_app,
                    [
                        "--method", "lora-merge",
                        "--models", base_dir,
                        "--adapter", str(adapter_dir),
                        "--output", output_dir,
                    ],
                )

        assert result.exit_code == 0, result.output
        assert "Merge complete" in result.output

    def test_base_flag_prepended_to_models(self, cli_runner, merge_app, tmp_path):
        """--base should be prepended to --models for lora-merge."""
        base_dir = _make_model_dir(tmp_path, "base")
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        output_dir = str(tmp_path / "out")

        class _FakePeftModel(MagicMock):
            pass

        mock_merged = MagicMock()
        mock_peft_model = _FakePeftModel()
        mock_peft_model.merge_and_unload.return_value = mock_merged

        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel = _FakePeftModel
        mock_peft_module.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model)

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch(
            "aligntune.core.merge.peft_merger._require_peft",
            return_value=mock_peft_module,
        ):
            with patch(
                "aligntune.core.merge.peft_merger._require_transformers",
                return_value=mock_transformers,
            ):
                result = cli_runner.invoke(
                    merge_app,
                    [
                        "--method", "lora-merge",
                        "--base", base_dir,
                        "--adapter", str(adapter_dir),
                        "--output", output_dir,
                    ],
                )

        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_core_merge_exports(self):
        from aligntune.core.merge import BaseMerger, MergekitMerger, PEFTMerger

        assert issubclass(MergekitMerger, BaseMerger)
        assert issubclass(PEFTMerger, BaseMerger)
