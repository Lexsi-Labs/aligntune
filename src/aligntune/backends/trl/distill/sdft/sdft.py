"""
SDFT (Self-Distillation Fine-Tuning) Trainer - TRL Backend Implementation.

Wrapper for TRL's SDFTTrainer: delegates model loading to build_model() and training to TRL.
Supports self-distillation with base/live/ema teacher models.
"""

import logging
from typing import Dict, Any, Optional, List

from aligntune.core.distill.config import UnifiedDistillConfig
from aligntune.core.distill.trainer_base import DistillTrainerBase
from aligntune.core.callbacks import TrainerCallback
from aligntune.core.registry import TaskType
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class TRLSDFTTrainer(DistillTrainerBase):
    """SDFT (Self-Distillation) Trainer for TRL."""

    TASK_TYPE = "distillation_sdft"
    KEEP_COLUMNS = False

    def __init__(self, config: UnifiedDistillConfig, callbacks: Optional[List[TrainerCallback]] = None):
        super().__init__(config, callbacks)
        self.config = config
        self.student_model = None
        self.tokenizer = None
        self.trainer = None

        logger.info("=" * 80)
        logger.info("SDFT (Self-Distillation) Trainer (TRL)")
        logger.info(f"  Model: {config.model.student_model}")
        logger.info(f"  Teacher kind: {config.model.teacher_model_kind or 'base'}")
        logger.info(f"  Distillation mode: {config.train.distillation_mode or 'offline'}")
        logger.info("=" * 80)

    @classmethod
    def is_available(cls) -> bool:
        try:
            from trl.experimental.sdft import SDFTTrainer
            return True
        except ImportError:
            return False

    def setup_model(self) -> None:
        """Setup model using build_model()."""
        from aligntune.core.model_loader import build_model

        logger.info("Setting up model...")

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

        # Load student (teacher is derived from student in SDFT)
        logger.info(f"Loading model: {self.config.model.student_model}")
        apply_peft = self.config.model.use_peft
        self.student_model, self.tokenizer = build_model(
            config=MinimalConfig(StudentConfig(self.config.model)),
            task_type=TaskType.DISTILLATION,
            use_unsloth=self.config.model.use_unsloth,
            apply_peft=apply_peft,
            is_reference=False
        )
        logger.info("✓ Model loaded")

    def train(self) -> Dict[str, Any]:
        """Train using TRL SDFTTrainer."""
        from trl.experimental.sdft import SDFTConfig, SDFTTrainer

        logger.info("Starting training...")

        # Setup
        if not self.student_model:
            self.setup_model()
        if not self.train_dataset:
            self.setup_data()

        # Optimizer/scheduler
        optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

        # TRL config
        trl_config = SDFTConfig(
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
            # SDFT-specific
            teacher_model_kind=self.config.model.teacher_model_kind or "base",
            distillation_mode=self.config.train.distillation_mode or "topk_logits",  # topk_logits, full_logits, sampled_token
            distillation_alpha=self.config.train.distillation_alpha,
            # SDFT generation parameters
            num_generations=self.config.train.num_generations if hasattr(self.config.train, 'num_generations') else 8,
            max_completion_length=self.config.train.max_completion_length if hasattr(self.config.train, 'max_completion_length') else 256,
        )
        for key, value in extract_extra_and_missing_params(
            backend_config=trl_config, config=self.config, algorithm='sdft'
        ).items():
            setattr(trl_config, key, value)

        # SDFTConfig defaults to no evaluation. Do not let the shared
        # distillation config request evaluation when no validation dataset
        # was supplied.
        if self.eval_dataset is None:
            trl_config.eval_strategy = "no"
            trl_config.do_eval = False

        # TRL trainer
        self.trainer = SDFTTrainer(
            model=self.student_model,
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
