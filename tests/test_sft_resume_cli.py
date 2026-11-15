"""``aligntune train --resume`` reaches ``train.resume_from_checkpoint``.

``create_sft_trainer`` is replaced with a stub that records its arguments, so
no model or dataset is loaded.
"""

from unittest import mock

import pytest

typer_testing = pytest.importorskip("typer.testing")

from aligntune.cli import unified


class _Trainer:
    def train(self):
        return {"training_time": 0.0, "model_path": "out"}


@pytest.fixture
def calls():
    recorded = []

    def fake(**kwargs):
        recorded.append(kwargs)
        return _Trainer()

    with mock.patch.object(unified, "create_sft_trainer", side_effect=fake), \
            mock.patch.object(unified, "create_rl_trainer", side_effect=AssertionError("RL trainer built")):
        yield recorded


def _run(*args):
    return typer_testing.CliRunner().invoke(unified.app, ["train", *args])


def _write_config(tmp_path, algo="sft"):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "algo": algo,
        "model": {"name_or_path": "m"},
        "datasets": [{"name": "d"}],
        "train": {"max_steps": 2},
        "logging": {"output_dir": str(tmp_path / "out")},
    }))
    return str(path)


@pytest.mark.parametrize("flag, expected", [("latest", True), ("out/checkpoint-4", "out/checkpoint-4")])
def test_resume_with_config_file(tmp_path, calls, flag, expected):
    result = _run("--config", _write_config(tmp_path), "--resume", flag)

    assert result.exit_code == 0, result.output
    assert calls[0]["config"]["train"]["resume_from_checkpoint"] == expected


def test_resume_with_cli_arguments(calls):
    result = _run("--model", "m", "--dataset", "d", "--type", "sft", "--resume", "out/checkpoint-4")

    assert result.exit_code == 0, result.output
    assert calls[0]["resume_from_checkpoint"] == "out/checkpoint-4"


def test_no_resume_flag_leaves_it_unset(tmp_path, calls):
    result = _run("--config", _write_config(tmp_path))

    assert result.exit_code == 0, result.output
    assert "resume_from_checkpoint" not in calls[0]["config"]["train"]


@pytest.mark.parametrize("use_config", [True, False])
def test_resume_is_refused_for_rl(tmp_path, calls, use_config):
    args = ["--config", _write_config(tmp_path, algo="dpo")] if use_config else [
        "--model", "m", "--dataset", "d", "--type", "dpo"]
    result = _run(*args, "--resume", "latest")

    assert result.exit_code != 0
    assert "--resume" in result.output and "SFT" in result.output


# ``python -m aligntune.cli.finetune`` has its own ``--resume`` option.
finetune_cli = pytest.importorskip("aligntune.cli.finetune")


@pytest.fixture
def finetune_calls():
    """The real factories run; only the trainer construction is stubbed."""
    from aligntune.core.backend_factory import BackendFactory

    built = []

    def fake_create_trainer(config, backend_config):
        built.append(config)
        return _Trainer()

    with mock.patch.object(BackendFactory, "create_trainer", new=fake_create_trainer):
        yield built


@pytest.mark.parametrize("flag, expected", [("latest", True), ("out/checkpoint-4", "out/checkpoint-4")])
def test_finetune_sft_resume(finetune_calls, flag, expected):
    result = typer_testing.CliRunner().invoke(
        finetune_cli.app, ["m", "d", "--backend", "trl", "--max-steps", "7", "--resume", flag]
    )

    assert result.exit_code == 0, result.output
    config = finetune_calls[0]
    assert config.model.name_or_path == "m"
    assert config.dataset.name == "d"
    assert config.train.max_steps == 7
    assert config.train.resume_from_checkpoint == expected


def test_finetune_resume_is_refused_for_rl(finetune_calls):
    result = typer_testing.CliRunner().invoke(
        finetune_cli.app, ["m", "d", "--method", "dpo", "--resume", "latest"]
    )

    assert result.exit_code != 0
    assert "--resume" in result.output and "SFT" in result.output


@pytest.mark.parametrize("method", ["dpo", "grpo"])
def test_finetune_rl_builds_a_trainer(finetune_calls, method):
    # It used to stop with "DistributedConfig.__init__() got an unexpected keyword
    # argument 'num_gpus'", then passed its config to create_rl_trainer as model_name.
    result = typer_testing.CliRunner().invoke(finetune_cli.app, [
        "m", "d", "--method", method, "--backend", "trl", "--max-steps", "3",
        "--beta", "0.2", "--rewards", "length:0.5",
    ])

    assert result.exit_code == 0, result.output
    config = finetune_calls[0]
    assert config.algo.value == method
    assert config.model.name_or_path == "m"
    assert config.datasets[0].name == "d"
    assert config.train.max_steps == 3
    assert config.train.beta == 0.2
    assert [(r["type"], r["weight"]) for r in config.rewards] == [("length", 0.5)]
