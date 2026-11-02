"""
Standard Distillation Trainer - TRL Backend Implementation.

Simple wrapper: delegates model loading to build_model() and training to TRL.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from aligntune.core.distill.config import UnifiedDistillConfig
from aligntune.core.distill.trainer_base import DistillTrainerBase
from aligntune.core.callbacks import TrainerCallback
from aligntune.core.registry import TaskType
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


def _distillation_trainer_classes():
    """Import TRL DistillationTrainer without requiring a working vLLM install.

    DistillationTrainer always imports VLLMGeneration. If a CUDA-13 vLLM wheel
    is present, that import fails and create_distill_trainer reports
    ``No available backend for TrainingType.DISTILL distillation``.
    """
    if os.environ.get("ALIGNTUNE_ENABLE_VLLM", "0") != "1":
        try:
            import trl.import_utils as _iu
            _iu.is_vllm_available = lambda *a, **k: False
        except Exception:
            pass
    from trl.experimental.distillation import DistillationConfig, DistillationTrainer
    return DistillationConfig, DistillationTrainer


class TRLDistillationTrainer(DistillTrainerBase):
    """Standard Distillation Trainer for TRL."""

    TASK_TYPE = "distillation"
    KEEP_COLUMNS = False

    def __init__(self, config: UnifiedDistillConfig, callbacks: Optional[List[TrainerCallback]] = None):
        super().__init__(config, callbacks)
        self.config = config
        self.student_model = None
        self.teacher_model = None
        self.tokenizer = None
        self.trainer = None

        logger.info("=" * 80)
        logger.info("Standard Distillation Trainer (TRL)")
        logger.info(f"  Student: {config.model.student_model}")
        logger.info(f"  Teacher: {config.model.teacher_model}")
        logger.info("=" * 80)

    @classmethod
    def is_available(cls) -> bool:
        try:
            _distillation_trainer_classes()
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup models using build_model()."""
        from aligntune.core.model_loader import build_model

        logger.info("Setting up models...")

        # Wrapper for build_model - convert student_model to name_or_path
        class StudentConfig:
            def __init__(self, orig):
                self.name_or_path = orig.student_model
                self.max_seq_length = orig.max_seq_length
                self.quantization = orig.quantization
                self.precision = orig.precision
                self.attn_implementation = orig.attn_implementation
                self.gradient_checkpointing = orig.gradient_checkpointing
                self.device_map = orig.device_map
                self.trust_remote_code = orig.trust_remote_code
                self.model_init_kwargs = orig.model_init_kwargs or {}
                self.use_unsloth = orig.use_unsloth
                self.peft = type('obj', (object,), {'enabled': orig.use_peft})()

        class MinimalConfig:
            def __init__(self, model_config):
                self.model = model_config
                self.dataset = type('obj', (object,), {'chat_template': None})()

        # Student
        logger.info(f"Loading student: {self.config.model.student_model}")
        apply_peft = self.config.model.use_peft
        self.student_model, self.tokenizer = build_model(
            config=MinimalConfig(StudentConfig(self.config.model)),
            task_type=TaskType.DISTILLATION,
            use_unsloth=self.config.model.use_unsloth,
            apply_peft=apply_peft,
            is_reference=False
        )
        logger.info("✓ Student loaded")

        # Teacher
        logger.info(f"Loading teacher: {self.config.model.teacher_model}")

        class TeacherConfig:
            def __init__(self, orig):
                self.name_or_path = orig.model.teacher_model
                self.max_seq_length = orig.model.max_seq_length
                self.quantization = orig.model.quantization
                self.precision = orig.model.precision
                self.attn_implementation = orig.model.attn_implementation
                self.gradient_checkpointing = False
                self.device_map = orig.model.device_map
                self.trust_remote_code = orig.model.trust_remote_code
                self.model_init_kwargs = orig.model.teacher_model_init_kwargs or {}
                self.use_unsloth = False
                self.peft = type('obj', (object,), {'enabled': False})()

        self.teacher_model, _ = build_model(
            config=MinimalConfig(TeacherConfig(self.config)),
            task_type=TaskType.DISTILLATION,
            use_unsloth=False,
            apply_peft=False,
            is_reference=True
        )

        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        logger.info("✓ Teacher loaded and frozen")

    def train(self) -> Dict[str, Any]:
        """Train using TRL DistillationTrainer."""
        DistillationConfig, DistillationTrainer = _distillation_trainer_classes()

        logger.info("Starting training...")

        # Setup
        if not self.student_model:
            self.setup_model()
        if not self.train_dataset:
            self.setup_data()

        # Optimizer/scheduler
        optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

        # Resolve the public on_policy switch to TRL's lmbda. An explicit
        # lmbda always wins; otherwise offline is 0.0 and online is 1.0.
        configured_lmbda = self.config.train.lmbda
        if configured_lmbda is None:
            on_policy = self.config.train.on_policy
            if isinstance(on_policy, str):
                on_policy = on_policy.strip().lower() in {"online", "true", "1", "yes"}
            configured_lmbda = 1.0 if on_policy else 0.0

        # TRL config
        trl_config = DistillationConfig(
            output_dir=self.config.logging.output_dir,
            num_train_epochs=self.config.train.epochs or 1,
            max_steps=optim_scheduler['max_steps'],
            per_device_train_batch_size=self.config.train.per_device_batch_size,
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            learning_rate=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
            warmup_steps=optim_scheduler['warmup_steps'],
            logging_steps=self.config.train.logging_steps,
            save_steps=self.config.train.save_steps,
            seed=self.config.train.seed,
            optim=optim_scheduler['optimizer_name'],
            lr_scheduler_type=optim_scheduler['scheduler_name'],
            # Distillation
            lmbda=configured_lmbda,
            beta=self.config.train.beta if self.config.train.beta is not None else 1.0,
        )
        for key, value in extract_extra_and_missing_params(
            backend_config=trl_config, config=self.config, algorithm='distillation'
        ).items():
            setattr(trl_config, key, value)

        if self.eval_dataset is None:
            trl_config.eval_strategy = "no"
            trl_config.do_eval = False

        # TRL trainer
        self.trainer = DistillationTrainer(
            model=self.student_model,
            teacher_model=self.teacher_model,
            args=trl_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        train_result = self.trainer.train()
        logger.info("✓ Training completed")
        return train_result

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        return self.trainer.evaluate(*args, **kwargs) if self.trainer else {}
