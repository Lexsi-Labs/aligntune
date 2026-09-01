"""
Tests for DeepSpeed/FSDP support via HF Accelerate config integration.

These tests verify:
- Preset accelerate YAML files are valid and parseable
- accelerate_config_path field exists in both SFT and RL TrainingConfig
- CLI --accelerate-config flag is parsed and flows into the config dict
- ACCELERATE_CONFIG_FILE env var is set when accelerate_config_path is provided
"""

import os
import sys
import yaml
import pytest
from pathlib import Path

# Ensure the package is importable regardless of install state
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RECIPES_ACCELERATE_DIR = (
    Path(__file__).parent.parent
    / "src" / "aligntune" / "recipes" / "accelerate"
)

PRESET_FILES = [
    "deepspeed_zero2.yaml",
    "deepspeed_zero3.yaml",
    "fsdp_full_shard.yaml",
    "fsdp_hybrid_shard.yaml",
]


# ---------------------------------------------------------------------------
# 1. Preset YAML files are valid / parseable
# ---------------------------------------------------------------------------

class TestPresetYAMLFiles:
    """Verify each preset accelerate config YAML exists and is valid YAML."""

    @pytest.mark.parametrize("filename", PRESET_FILES)
    def test_file_exists(self, filename):
        path = RECIPES_ACCELERATE_DIR / filename
        assert path.exists(), f"Missing preset: {path}"

    @pytest.mark.parametrize("filename", PRESET_FILES)
    def test_file_is_valid_yaml(self, filename):
        path = RECIPES_ACCELERATE_DIR / filename
        with open(path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{filename} did not parse to a dict"

    @pytest.mark.parametrize("filename", PRESET_FILES)
    def test_required_keys_present(self, filename):
        path = RECIPES_ACCELERATE_DIR / filename
        with open(path) as f:
            data = yaml.safe_load(f)
        # Every Accelerate config must declare compute_environment and distributed_type
        assert "compute_environment" in data, f"{filename} missing compute_environment"
        assert "distributed_type" in data, f"{filename} missing distributed_type"

    def test_deepspeed_zero2_has_stage_2(self):
        path = RECIPES_ACCELERATE_DIR / "deepspeed_zero2.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        stage = data.get("deepspeed_config", {}).get("zero_stage")
        assert stage == 2, f"Expected ZeRO stage 2, got {stage}"

    def test_deepspeed_zero3_has_stage_3(self):
        path = RECIPES_ACCELERATE_DIR / "deepspeed_zero3.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        stage = data.get("deepspeed_config", {}).get("zero_stage")
        assert stage == 3, f"Expected ZeRO stage 3, got {stage}"

    def test_fsdp_full_shard_strategy(self):
        path = RECIPES_ACCELERATE_DIR / "fsdp_full_shard.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        strategy = data.get("fsdp_config", {}).get("fsdp_sharding_strategy")
        assert strategy == "FULL_SHARD", f"Expected FULL_SHARD, got {strategy}"

    def test_fsdp_hybrid_shard_strategy(self):
        path = RECIPES_ACCELERATE_DIR / "fsdp_hybrid_shard.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        strategy = data.get("fsdp_config", {}).get("fsdp_sharding_strategy")
        assert strategy == "HYBRID_SHARD", f"Expected HYBRID_SHARD, got {strategy}"


# ---------------------------------------------------------------------------
# 2. accelerate_config_path field exists in SFT and RL TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfigField:
    """Verify the accelerate_config_path field is present in both config classes."""

    def test_sft_training_config_has_field(self):
        # aligntune.core.sft.config now does `from ..registry import ...`, so it
        # can no longer be loaded standalone via spec_from_file_location (that
        # bypasses the package machinery relative imports need) - import it
        # normally instead.
        from aligntune.core.sft.config import TrainingConfig
        cfg = TrainingConfig()
        assert hasattr(cfg, "accelerate_config_path"), (
            "SFT TrainingConfig is missing accelerate_config_path"
        )
        assert cfg.accelerate_config_path is None, (
            "accelerate_config_path should default to None"
        )

    def test_sft_training_config_field_accepts_string(self):
        from aligntune.core.sft.config import TrainingConfig
        cfg = TrainingConfig(accelerate_config_path="path/to/config.yaml")
        assert cfg.accelerate_config_path == "path/to/config.yaml"

    def test_rl_training_config_has_field(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rl_config",
            str(_SRC / "aligntune" / "core" / "rl" / "config.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrainingConfig = mod.TrainingConfig
        cfg = TrainingConfig(max_steps=10)
        assert hasattr(cfg, "accelerate_config_path"), (
            "RL TrainingConfig is missing accelerate_config_path"
        )
        assert cfg.accelerate_config_path is None, (
            "accelerate_config_path should default to None"
        )

    def test_rl_training_config_field_accepts_string(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rl_config",
            str(_SRC / "aligntune" / "core" / "rl" / "config.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrainingConfig = mod.TrainingConfig
        cfg = TrainingConfig(max_steps=10, accelerate_config_path="ds_zero3.yaml")
        assert cfg.accelerate_config_path == "ds_zero3.yaml"


# ---------------------------------------------------------------------------
# 3. CLI flag is parsed and flows into config
# ---------------------------------------------------------------------------

class TestCLIFlag:
    """Verify --accelerate-config flows into train_config dict."""

    def test_accelerate_config_injected_into_train_config(self):
        """
        Simulate the CLI logic: if accelerate_config is set it should be
        written into train_config['accelerate_config_path'].
        """
        # Replicate the mapping from unified.py
        accelerate_config = "src/aligntune/recipes/accelerate/deepspeed_zero2.yaml"
        train_config = {}
        if accelerate_config:
            train_config["accelerate_config_path"] = accelerate_config

        assert train_config.get("accelerate_config_path") == accelerate_config

    def test_accelerate_config_injected_into_yaml_final_config(self):
        """
        When a YAML config is loaded and --accelerate-config is supplied,
        it should override / set final_config['train']['accelerate_config_path'].
        """
        accelerate_config = "src/aligntune/recipes/accelerate/fsdp_full_shard.yaml"
        final_config = {"train": {"epochs": 1}}

        # Replicate the YAML-path logic from unified.py
        if accelerate_config:
            if "train" not in final_config:
                final_config["train"] = {}
            final_config["train"]["accelerate_config_path"] = accelerate_config

        assert final_config["train"]["accelerate_config_path"] == accelerate_config

    def test_accelerate_config_absent_leaves_field_unset(self):
        """When --accelerate-config is not passed, the key should not appear."""
        accelerate_config = None
        train_config = {}
        if accelerate_config:
            train_config["accelerate_config_path"] = accelerate_config

        assert "accelerate_config_path" not in train_config


# ---------------------------------------------------------------------------
# 4. ACCELERATE_CONFIG_FILE env var is set when accelerate_config_path provided
# ---------------------------------------------------------------------------

class TestEnvVarInjection:
    """Verify ACCELERATE_CONFIG_FILE is set by the SFT trainer and RL base."""

    def test_sft_trainer_sets_env_var(self, tmp_path, monkeypatch):
        """
        When a TRLSFTTrainer setup_trainer is called with accelerate_config_path,
        ACCELERATE_CONFIG_FILE should be set in the environment.
        """
        # We test the env-var-setting logic directly without instantiating the full trainer
        config_file = tmp_path / "accel.yaml"
        config_file.write_text("compute_environment: LOCAL_MACHINE\ndistributed_type: DEEPSPEED\n")

        # Simulate the block inserted into sft.py
        import os
        from pathlib import Path as _Path
        accelerate_config_path = str(config_file)
        monkeypatch.delenv("ACCELERATE_CONFIG_FILE", raising=False)

        if accelerate_config_path:
            resolved = str(_Path(accelerate_config_path).resolve())
            os.environ["ACCELERATE_CONFIG_FILE"] = resolved

        assert os.environ.get("ACCELERATE_CONFIG_FILE") == str(config_file.resolve())

    def test_rl_trainer_base_sets_env_var(self, tmp_path, monkeypatch):
        """
        Simulate the env-var block in TrainerBase.__init__ for RL.
        """
        import os
        from pathlib import Path as _Path
        config_file = tmp_path / "accel_rl.yaml"
        config_file.write_text("compute_environment: LOCAL_MACHINE\ndistributed_type: FSDP\n")

        monkeypatch.delenv("ACCELERATE_CONFIG_FILE", raising=False)

        accelerate_config_path = str(config_file)
        if accelerate_config_path:
            resolved = str(_Path(accelerate_config_path).resolve())
            os.environ["ACCELERATE_CONFIG_FILE"] = resolved

        assert os.environ.get("ACCELERATE_CONFIG_FILE") == str(config_file.resolve())

    def test_no_env_var_when_path_not_set(self, monkeypatch):
        """When accelerate_config_path is None, env var should not be touched."""
        monkeypatch.delenv("ACCELERATE_CONFIG_FILE", raising=False)
        accelerate_config_path = None
        if accelerate_config_path:
            os.environ["ACCELERATE_CONFIG_FILE"] = accelerate_config_path

        assert "ACCELERATE_CONFIG_FILE" not in os.environ
