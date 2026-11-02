"""
End-to-end test for Standard Distillation in offline mode (lmbda=0.0).

Uses BackendFactory through distillation trainer factory:
- Models: Qwen2.5-1.5B (student), Qwen2.5-7B (teacher)
- Dataset: alpaca (offline with completions)
- Mode: offline (lmbda=0.0)
"""

import os
import pytest
import tempfile
import logging
from datasets import load_dataset

# report_to="none" isn't honored consistently by every backend/trainer path,
# so force wandb off at the environment level to avoid a hard failure in
# CI/offline environments without a wandb API key.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestOfflineDistillationConfig:
    """Test offline distillation configuration."""

    def test_create_offline_config(self):
        """Create offline distillation config."""
        from aligntune.core.distill.config import (
            UnifiedDistillConfig,
            DistillModelConfig,
            DistillDatasetConfig,
            DistillTrainingConfig,
            DistillLoggingConfig,
            DistillationType
        )

        config = UnifiedDistillConfig(
            model=DistillModelConfig(
                student_model="Qwen/Qwen2.5-1.5B-Instruct",
                teacher_model="Qwen/Qwen2.5-7B-Instruct",
                max_seq_length=512,
                use_peft=False,
            ),
            dataset=DistillDatasetConfig(
                name="tatsu-lab/alpaca",
                split="train",
                max_samples=100,
            ),
            train=DistillTrainingConfig(
                per_device_batch_size=4,
                epochs=1,
                lmbda=0.0,      # Offline: use dataset completions
                beta=1.0,       # Reverse KL
                learning_rate=5e-5,
            ),
            logging=DistillLoggingConfig(
                output_dir=tempfile.gettempdir(),
                loggers=["none"],
            )
        )

        # Verify offline mode
        assert config.train.lmbda == 0.0

        # Verify distillation type detection
        distill_type = config.get_distillation_type()
        assert distill_type == DistillationType.STANDARD

        logger.info(f"✓ Config created (offline mode, lmbda={config.train.lmbda})")


class TestOfflineTrainerCreation:
    """Test trainer creation using BackendFactory."""

    def test_create_trainer_via_factory(self):
        """Create trainer using BackendFactory through trainer factory."""
        from aligntune.core.distill.config import (
            UnifiedDistillConfig,
            DistillModelConfig,
            DistillDatasetConfig,
            DistillTrainingConfig,
            DistillLoggingConfig,
        )
        from aligntune.core.distill.trainer_factory import create_trainer_from_config

        config = UnifiedDistillConfig(
            model=DistillModelConfig(
                student_model="Qwen/Qwen2.5-1.5B-Instruct",
                teacher_model="Qwen/Qwen2.5-7B-Instruct",
                max_seq_length=512,
                use_peft=False,
            ),
            dataset=DistillDatasetConfig(
                name="wikitext",
                split="train",
                max_samples=10,
            ),
            train=DistillTrainingConfig(
                per_device_batch_size=2,
                epochs=1,
                lmbda=0.0,
                learning_rate=5e-5,
            ),
            logging=DistillLoggingConfig(
                output_dir=tempfile.gettempdir(),
                loggers=["none"],
            )
        )

        # Create trainer using BackendFactory
        trainer = create_trainer_from_config(config)

        assert trainer is not None
        assert trainer.TASK_TYPE == "distillation"
        assert trainer.config.train.lmbda == 0.0

        logger.info(f"✓ Trainer created via BackendFactory")
        logger.info(f"  Class: {trainer.__class__.__name__}")
        logger.info(f"  Task type: {trainer.TASK_TYPE}")


class TestOfflineDataset:
    """Test offline distillation dataset format."""

    @staticmethod
    def prepare_alpaca_offline(max_samples: int = 100) -> dict:
        """Prepare Alpaca in offline format with completions."""
        logger.info(f"Loading Alpaca dataset...")

        dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=False)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        def format_offline(example):
            user_content = example["instruction"]
            if example.get("input"):
                user_content += f"\n{example['input']}"

            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": example["output"]}
                ]
            }

        dataset = dataset.map(
            format_offline,
            remove_columns=dataset.column_names,
        )

        logger.info(f"✓ Dataset prepared: {len(dataset)} samples")
        return {"train": dataset}

    def test_offline_format_validation(self):
        """Verify offline dataset has both user and assistant messages."""
        dataset_dict = self.prepare_alpaca_offline(5)
        dataset = dataset_dict["train"]

        assert len(dataset) == 5
        assert "messages" in dataset.column_names

        sample = dataset[0]
        assert isinstance(sample["messages"], list)
        assert len(sample["messages"]) == 2
        assert sample["messages"][0]["role"] == "user"
        assert sample["messages"][1]["role"] == "assistant"

        logger.info(f"✓ Dataset format valid")


class TestOfflineParameters:
    """Test offline distillation parameters."""

    def test_offline_mode_params(self):
        """Test offline mode parameters."""
        from aligntune.core.distill.config import DistillTrainingConfig

        config = DistillTrainingConfig(
            lmbda=0.0,      # Offline
            beta=1.0,       # Reverse KL
            temperature=1.0,
            alpha=0.5,
            learning_rate=5e-5,
        )

        assert config.lmbda == 0.0
        assert config.beta == 1.0
        assert config.temperature == 1.0

        logger.info(f"✓ Offline params valid:")
        logger.info(f"  lmbda={config.lmbda} (use dataset completions)")
        logger.info(f"  beta={config.beta} (reverse KL)")
        logger.info(f"  temperature={config.temperature}")


class TestRegistryTaskTypes:
    """Test distillation task types in registry."""

    def test_distillation_tasks_registered(self):
        """Verify distillation tasks in TaskType enum."""
        from aligntune.core.registry import TaskType

        tasks = ['DISTILLATION', 'GOLD', 'SDFT', 'SDPO']
        for task in tasks:
            assert hasattr(TaskType, task)
            value = getattr(TaskType, task).value
            logger.info(f"  ✓ TaskType.{task} = {value}")


class TestOfflineTrainingWithBackendFactory:
    """Test offline distillation training using BackendFactory."""

    def test_offline_training_e2e(self):
        """Test complete offline distillation training via BackendFactory."""
        from aligntune.core.backend_factory import create_distill_trainer

        # Create trainer via BackendFactory
        # Teacher downsized from Qwen2.5-7B-Instruct to fit student+teacher on a
        # single GPU without OOM - this is a test-only sizing choice (same as
        # already applied to the online-distillation test below).
        trainer = create_distill_trainer(
            student_model="Qwen/Qwen2.5-1.5B-Instruct",
            teacher_model="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_name="tatsu-lab/alpaca",
            # DataManager.load_dataset() now keeps the literal `split` string
            # as the DatasetDict key instead of normalizing it to "train" (see
            # src/aligntune/data/manager.py), and trainer_base.py's setup_data()
            # looks up "train" specifically, so "train[:100]" no longer resolves.
            # Match the working notebook pattern (notebooks/19_distillation.ipynb):
            # plain split="train" + max_samples for truncation.
            split="train",
            max_samples=100,
            backend="trl",
            batch_size=2,
            num_epochs=1,
            learning_rate=5e-5,
            temperature=1.0,
            alpha=0.5,
            loss_type="kl",
            seed=42,
        )

        assert trainer is not None
        assert trainer.TASK_TYPE == "distillation"
        logger.info(f"✓ Trainer created via BackendFactory: {trainer.__class__.__name__}")

        # Actual training happens here
        logger.info("Starting offline distillation training...")
        train_result = trainer.train()

        assert train_result is not None
        logger.info(f"✓ Training completed")


class TestOnlineDistillationTraining:
    """Test online distillation training (lmbda=1.0)."""

    def test_online_training_e2e(self):
        """Test complete online distillation training via BackendFactory."""
        from aligntune.core.backend_factory import create_distill_trainer

        # Create trainer via BackendFactory with lmbda=1.0 for online mode
        # Teacher downsized from Qwen2.5-7B-Instruct to fit student+teacher on a
        # single GPU without OOM - this is a test-only sizing choice.
        trainer = create_distill_trainer(
            student_model="Qwen/Qwen2.5-1.5B-Instruct",
            teacher_model="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_name="tatsu-lab/alpaca",
            # DataManager.load_dataset() now keeps the literal `split` string
            # as the DatasetDict key instead of normalizing it to "train" (see
            # src/aligntune/data/manager.py), and trainer_base.py's setup_data()
            # looks up "train" specifically, so "train[:100]" no longer resolves.
            # Match the working notebook pattern (notebooks/19_distillation.ipynb):
            # plain split="train" + max_samples for truncation.
            split="train",
            max_samples=100,
            backend="trl",
            batch_size=2,
            num_epochs=1,
            learning_rate=5e-5,
            temperature=1.0,
            alpha=0.5,
            loss_type="kl",
            lmbda=1.0,  # Online mode: student generates
            beta=1.0,   # Reverse KL
            seed=42,
        )

        assert trainer is not None
        assert trainer.TASK_TYPE == "distillation"
        assert trainer.config.train.lmbda == 1.0
        logger.info(f"✓ Trainer created for online mode (lmbda={trainer.config.train.lmbda})")

        # Actual training happens here
        logger.info("Starting online distillation training...")
        train_result = trainer.train()

        assert train_result is not None
        logger.info(f"✓ Online training completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
