"""Regression tests: ``train:`` keys in a YAML/dict config must not be dropped.

``parse_config_to_unified`` copied a fixed list of train keys, so
``gradient_checkpointing``, ``optimizer``, ``lr_scheduler``,
``dataloader_num_workers`` and ``dataset_num_proc`` silently fell back to
defaults. ``dataset_num_proc`` also never reached ``trl.SFTConfig``, so
Unsloth picked ``cpu_count + 4`` tokenizer processes.
``BackendFactory.create_trainer`` is stubbed, so no model is loaded.
"""

from unittest import mock

import pytest

from aligntune.core.backend_factory import BackendFactory, create_sft_trainer
from aligntune.core.sft import training_args_builder
from aligntune.utils.config_utils import parse_config_to_unified

TRAIN = {
    "max_steps": 2,
    "gradient_checkpointing": True,
    "optimizer": "adamw_8bit",
    "lr_scheduler": "linear",
    "dataloader_num_workers": 1,
    "dataset_num_proc": 1,
}

CONFIG = {
    "algo": "sft",
    "backend": "trl",
    "model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"},
    "datasets": [{"name": "toy/sft"}],
    "train": TRAIN,
}


def test_parser_keeps_unlisted_train_keys():
    parsed = parse_config_to_unified(CONFIG, training_type="sft")

    for key in (
        "gradient_checkpointing",
        "optimizer",
        "lr_scheduler",
        "dataloader_num_workers",
        "dataset_num_proc",
    ):
        assert parsed[key] == TRAIN[key], key


def test_parser_keeps_unlisted_train_keys_for_rl():
    parsed = parse_config_to_unified({**CONFIG, "algo": "dpo"}, training_type="rl")

    assert parsed["gradient_checkpointing"] is True
    assert parsed["dataloader_num_workers"] == 1


def test_train_keys_reach_sft_trainer_config():
    with mock.patch.object(
        BackendFactory, "create_trainer", new=lambda config, backend_config: config
    ):
        config = create_sft_trainer(config=CONFIG)

    assert config.train.gradient_checkpointing is True
    assert config.train.optimizer == "adamw_8bit"
    assert config.train.lr_scheduler == "linear"
    assert config.train.dataloader_num_workers == 1
    assert config.dataset.dataset_num_proc == 1


def test_dataset_num_proc_is_passed_to_sftconfig_constructor():
    datasets = pytest.importorskip("datasets")
    with mock.patch.object(
        BackendFactory, "create_trainer", new=lambda config, backend_config: config
    ):
        config = create_sft_trainer(config=CONFIG)

    import trl.trainer.sft_config as sft_config_module

    # build_sft_config looks SFTConfig up on trl's module at call time, so patch it there.
    with mock.patch.object(
        sft_config_module, "SFTConfig", wraps=sft_config_module.SFTConfig
    ) as sft_config_cls:
        sft_config = training_args_builder.build_sft_config(
            config, datasets.Dataset.from_list([{"text": "x"}] * 4), output_dir="unused"
        )

    # Unsloth resolves a None dataset_num_proc inside SFTConfig.__init__, so it
    # has to be a constructor argument, not set afterwards.
    assert sft_config_cls.call_args.kwargs["dataset_num_proc"] == 1
    assert sft_config.dataset_num_proc == 1
