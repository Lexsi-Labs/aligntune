"""
Regression tests for issue #36: build_sft_config must not invent a step count
(previously clamped to a minimum of 10) when the caller did not pass max_steps.
"""

import pytest
from datasets import Dataset

from aligntune.core.sft.config import DatasetConfig, ModelConfig, SFTConfig, TrainingConfig
from aligntune.core.sft.training_args_builder import build_sft_config


def _config(**train_kwargs):
    return SFTConfig(
        model=ModelConfig(name_or_path="tiny-model"),
        dataset=DatasetConfig(name="rows.jsonl"),
        train=TrainingConfig(per_device_batch_size=8, **train_kwargs),
    )


@pytest.fixture
def small_dataset():
    # 16 rows, batch 8, 1 epoch -> HF Trainer should run 2 steps.
    return Dataset.from_dict({"text": [f"row {i}" for i in range(16)]})


def test_unset_max_steps_defers_to_epochs(small_dataset, tmp_path):
    cfg = build_sft_config(_config(epochs=1), small_dataset, output_dir=str(tmp_path))
    assert cfg.max_steps == -1
    assert cfg.num_train_epochs == 1


def test_unset_max_steps_large_dataset_also_defers_to_epochs(tmp_path):
    ds = Dataset.from_dict({"text": ["x"] * 1000})
    cfg = build_sft_config(_config(epochs=2), ds, output_dir=str(tmp_path))
    assert cfg.max_steps == -1
    assert cfg.num_train_epochs == 2


@pytest.mark.parametrize("max_steps", [3, 50])
def test_explicit_max_steps_is_respected(small_dataset, tmp_path, max_steps):
    cfg = build_sft_config(_config(epochs=1, max_steps=max_steps), small_dataset, output_dir=str(tmp_path))
    assert cfg.max_steps == max_steps


def test_explicit_minus_one_is_respected(small_dataset, tmp_path):
    cfg = build_sft_config(_config(epochs=1, max_steps=-1), small_dataset, output_dir=str(tmp_path))
    assert cfg.max_steps == -1
