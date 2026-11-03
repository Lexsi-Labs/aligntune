"""
Regression test: build_sft_config must stay picklable after trl's SFTConfig is
replaced at runtime (Unsloth does this when it patches its trainers), otherwise
torch.save(args) fails at the first checkpoint with
"Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the same object".
"""

import pickle

import pytest
import trl
import trl.trainer.sft_config as sft_config_module
from datasets import Dataset

from aligntune.core.sft.config import DatasetConfig, ModelConfig, SFTConfig, TrainingConfig
from aligntune.core.sft.training_args_builder import build_sft_config

_ORIGINAL_SFT_CONFIG = sft_config_module.SFTConfig


class PatchedSFTConfig(_ORIGINAL_SFT_CONFIG):
    """Stand-in for Unsloth's UnslothSFTConfig."""


@pytest.fixture
def patched_trl(monkeypatch):
    # Mirror what Unsloth rebinds.
    monkeypatch.setattr(sft_config_module, "SFTConfig", PatchedSFTConfig)
    monkeypatch.setattr(trl, "SFTConfig", PatchedSFTConfig)


def _build(tmp_path):
    config = SFTConfig(
        model=ModelConfig(name_or_path="tiny-model"),
        dataset=DatasetConfig(name="rows.jsonl"),
        train=TrainingConfig(per_device_batch_size=2, epochs=1),
    )
    ds = Dataset.from_dict({"text": ["a", "b", "c", "d"]})
    return build_sft_config(config, ds, output_dir=str(tmp_path))


def test_config_is_picklable_after_trl_sft_config_is_replaced(patched_trl, tmp_path):
    cfg = _build(tmp_path)
    assert isinstance(cfg, PatchedSFTConfig)
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.max_steps == cfg.max_steps


def test_config_is_picklable_without_patching(tmp_path):
    cfg = _build(tmp_path)
    assert type(cfg) is _ORIGINAL_SFT_CONFIG
    pickle.loads(pickle.dumps(cfg))
