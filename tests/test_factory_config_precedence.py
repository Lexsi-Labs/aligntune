"""Regression tests: values from ``config=`` must reach the trainer config.

``create_sft_trainer`` / ``create_rl_trainer`` used to merge the parsed config
into ``**kwargs`` only, so named parameters (``max_seq_length=512``,
``batch_size=4``, ``output_dir="./output"``, RL ``max_steps=-1``, ...) kept
their signature defaults and silently overrode the config.
``BackendFactory.create_trainer`` is stubbed to return the assembled config,
so no model or dataset is loaded.
"""

from unittest import mock

import pytest

from aligntune.core.backend_factory import (
    BackendFactory,
    create_rl_trainer,
    create_sft_trainer,
)


@pytest.fixture(autouse=True)
def return_config_instead_of_trainer():
    with mock.patch.object(
        BackendFactory, "create_trainer", new=lambda config, backend_config: config
    ):
        yield


def _sft_config(output_dir):
    return {
        "algo": "sft",
        "backend": "trl",
        "model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "max_seq_length": 4096},
        "datasets": [{"name": "toy/sft", "max_samples": 3, "system_prompt": "Be brief."}],
        "train": {"epochs": 1, "per_device_batch_size": 1, "learning_rate": 1e-4},
        "logging": {"output_dir": str(output_dir)},
    }


def test_sft_config_values_override_factory_defaults(tmp_path):
    config = create_sft_trainer(config=_sft_config(tmp_path / "run"))

    assert config.model.max_seq_length == 4096
    assert config.train.per_device_batch_size == 1
    assert config.train.epochs == 1
    assert config.train.learning_rate == 1e-4
    assert config.logging.output_dir == str(tmp_path / "run")
    assert config.dataset.max_samples == 3
    assert config.dataset.system_prompt == "Be brief."


def test_sft_explicit_arguments_override_config(tmp_path):
    config = create_sft_trainer(
        config=_sft_config(tmp_path / "run"), max_seq_length=256, output_dir="explicit"
    )

    assert config.model.max_seq_length == 256
    assert config.logging.output_dir == "explicit"
    assert config.train.per_device_batch_size == 1


def test_sft_resume_from_checkpoint_reaches_train_config(tmp_path):
    config = create_sft_trainer(
        config=_sft_config(tmp_path / "run"), resume_from_checkpoint="run/checkpoint-500"
    )

    assert config.train.resume_from_checkpoint == "run/checkpoint-500"


def test_sft_resume_from_checkpoint_defaults_to_none(tmp_path):
    config = create_sft_trainer(config=_sft_config(tmp_path / "run"))

    assert config.train.resume_from_checkpoint is None


def test_sft_config_file_path_values_override_factory_defaults(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "sft.yaml"
    path.write_text(yaml.safe_dump(_sft_config(tmp_path / "run")))

    config = create_sft_trainer(config=str(path))

    assert config.model.max_seq_length == 4096
    assert config.logging.output_dir == str(tmp_path / "run")


def test_rl_config_values_override_factory_defaults(tmp_path):
    config = create_rl_trainer(
        config={
            "algo": "dpo",
            "backend": "trl",
            "model": {
                "name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_seq_length": 1024,
            },
            "datasets": [{"name": "toy/prefs", "max_samples": 3}],
            "train": {
                "epochs": 1,
                "max_steps": 7,
                "per_device_batch_size": 1,
                "learning_rate": 5e-6,
            },
            "logging": {"output_dir": str(tmp_path / "run")},
        }
    )

    assert config.train.max_length == 1024
    assert config.train.max_steps == 7
    assert config.train.per_device_batch_size == 1
    assert config.train.learning_rate == 5e-6
    assert config.logging.output_dir == str(tmp_path / "run")
    assert config.datasets[0].max_samples == 3


def _backend_chosen(factory, **kwargs):
    with mock.patch.object(
        BackendFactory,
        "create_trainer",
        new=lambda config, backend_config: backend_config,
    ):
        return factory(**kwargs).backend.value


def test_sft_explicit_backend_overrides_config(tmp_path):
    config = _sft_config(tmp_path / "run")
    config["backend"] = "unsloth"

    assert _backend_chosen(create_sft_trainer, config=config, backend="trl") == "trl"


def test_sft_config_without_backend_keeps_explicit_backend(tmp_path):
    # The parsed config fills in backend="auto"; that default must not win either.
    config = _sft_config(tmp_path / "run")
    del config["backend"]

    assert _backend_chosen(create_sft_trainer, config=config, backend="trl") == "trl"


def test_rl_explicit_backend_overrides_config(tmp_path):
    config = {
        "algo": "dpo",
        "backend": "unsloth",
        "model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"},
        "datasets": [{"name": "toy/prefs"}],
        "logging": {"output_dir": str(tmp_path / "run")},
    }

    assert _backend_chosen(create_rl_trainer, config=config, backend="trl") == "trl"
