"""TRL trainer configs train on a machine with no GPU.

TRL's configs default ``bf16`` to ``True``, and ``TrainingArguments`` rejects
that without a GPU ("Your setup doesn't support bf16/gpu. You need to assign
use_cpu ..."). The DPO, distillation and SDFT trainers passed no precision, so
they failed on every CPU-only machine. ``PrecisionHandler.cpu_only_training_args``
returns ``bf16=False, fp16=False, use_cpu=True`` only when there is no GPU, so
GPU runs keep their settings.

GPU availability is mocked; no model or dataset is loaded.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("trl")

from aligntune.core.precision_handler import PrecisionHandler


class _Built(Exception):
    """Raised by the stubbed TRL trainer to hand back its ``args``."""

    def __init__(self, training_args):
        super().__init__("trainer built")
        self.training_args = training_args


def _stub_trainer(*_, args=None, **__):
    raise _Built(args)


@pytest.fixture(params=[False, True], ids=["no_gpu", "gpu"])
def gpu(request):
    # transformers' own bf16 check is what TrainingArguments uses to refuse bf16.
    with mock.patch("torch.cuda.is_available", return_value=request.param), \
            mock.patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=request.param):
        yield request.param


def _check(args, gpu):
    if gpu:
        assert args.use_cpu is False and args.bf16 is True  # TRL's default, unchanged
    else:
        assert args.use_cpu is True and args.bf16 is False and args.fp16 is False


def test_helper(gpu):
    expected = {} if gpu else {"bf16": False, "fp16": False, "use_cpu": True}
    assert PrecisionHandler.cpu_only_training_args() == expected


def test_helper_keeps_a_cuda_gpu_without_bf16():
    # A CUDA card without bf16 (e.g. T4) is still a GPU run: leave it alone.
    with mock.patch("torch.cuda.is_available", return_value=True), \
            mock.patch("transformers.training_args.is_torch_bf16_gpu_available", return_value=False):
        assert PrecisionHandler.cpu_only_training_args() == {}


_OPTIM = {
    "max_steps": 2, "num_epochs": 1, "learning_rate": 1e-5, "lr_scheduler_type": "linear",
    "warmup_steps": 0, "warmup_ratio": None, "optimizer_type": "adamw_torch",
    "optimizer_name": "adamw_torch", "scheduler_name": "linear",
}


def test_dpo(tmp_path, gpu):
    from aligntune.backends.trl.rl.dpo.dpo import TRLDPOTrainer
    from aligntune.core.backend_factory import BackendFactory, create_rl_trainer

    with mock.patch.object(BackendFactory, "create_trainer", new=lambda config, backend_config: config):
        config = create_rl_trainer(
            model_name="m", dataset_name="d", algorithm="dpo", backend="trl",
            output_dir=str(tmp_path), max_steps=2,
        )
    trainer = TRLDPOTrainer(config)
    trainer.model = trainer.reference_model = trainer.tokenizer = None
    trainer.train_dataset = trainer.eval_dataset = None
    with mock.patch.object(TRLDPOTrainer, "_setup_optimizer_scheduler", return_value=_OPTIM), \
            mock.patch("trl.DPOTrainer", new=_stub_trainer), pytest.raises(_Built) as built:
        trainer.setup_trainer()
    _check(built.value.training_args, gpu)


def _distill_trainer(cls, tmp_path):
    trainer = cls.__new__(cls)
    trainer.config = SimpleNamespace(
        logging=SimpleNamespace(output_dir=str(tmp_path)),
        train=SimpleNamespace(
            epochs=1, per_device_batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-5,
            weight_decay=0.0, logging_steps=1, save_steps=10, seed=0, lmbda=None, on_policy=False,
            beta=None, distillation_mode=None, distillation_alpha=0.5,
            num_generations=1, max_completion_length=8,
        ),
        model=SimpleNamespace(teacher_model_kind=None),
    )
    trainer.student_model = trainer.teacher_model = trainer.tokenizer = object()
    trainer.train_dataset, trainer.eval_dataset = [1], None
    trainer._setup_optimizer_scheduler = lambda **_: _OPTIM
    return trainer


def test_distillation(tmp_path, gpu):
    pytest.importorskip("trl.experimental.distillation")
    from aligntune.backends.trl.distill.distillation import distillation as mod

    real_config, _ = mod._distillation_trainer_classes()
    trainer = _distill_trainer(mod.TRLDistillationTrainer, tmp_path)
    with mock.patch.object(mod, "_distillation_trainer_classes", return_value=(real_config, _stub_trainer)), \
            mock.patch.object(mod, "extract_extra_and_missing_params", return_value={}), \
            pytest.raises(_Built) as built:
        trainer.train()
    _check(built.value.training_args, gpu)


def test_sdft(tmp_path, gpu):
    sdft_module = pytest.importorskip("trl.experimental.sdft")
    from aligntune.backends.trl.distill.sdft import sdft as mod

    trainer = _distill_trainer(mod.TRLSDFTTrainer, tmp_path)
    with mock.patch.object(sdft_module, "SDFTTrainer", new=_stub_trainer), \
            mock.patch.object(mod, "extract_extra_and_missing_params", return_value={}), \
            pytest.raises(_Built) as built:
        trainer.train()
    _check(built.value.training_args, gpu)
