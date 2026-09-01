"""
Standard Distillation Trainer - Unsloth Backend Implementation.

Simple wrapper: delegates model loading to build_model() (with Unsloth acceleration
for the student model) and training to TRL's DistillationTrainer.
"""

import logging
from typing import Dict, Any, Optional, List

import torch

from aligntune.core.distill.config import UnifiedDistillConfig
from aligntune.core.distill.trainer_base import DistillTrainerBase
from aligntune.core.callbacks import TrainerCallback
from aligntune.core.registry import TaskType
from aligntune.utils.config_extractor import extract_extra_and_missing_params

logger = logging.getLogger(__name__)


class UnslothDistillationTrainer(DistillTrainerBase):
    """Standard Distillation Trainer for Unsloth."""

    TASK_TYPE = "distillation"

    def __init__(self, config: UnifiedDistillConfig, callbacks: Optional[List[TrainerCallback]] = None):
        super().__init__(config, callbacks)
        self.config = config
        self.student_model = None
        self.teacher_model = None
        self.tokenizer = None
        self.trainer = None

        logger.info("=" * 80)
        logger.info("Standard Distillation Trainer (Unsloth)")
        logger.info(f"  Student: {config.model.student_model}")
        logger.info(f"  Teacher: {config.model.teacher_model}")
        logger.info("=" * 80)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import unsloth
            from trl.experimental.distillation import DistillationTrainer
            return True
        except ImportError:
            return False

    @staticmethod
    def _same_architecture_family(student_name: str, teacher_name: str) -> bool:
        """Cheap same-vs-different-architecture check (config only, no weights)."""
        try:
            from transformers import AutoConfig
            student_type = AutoConfig.from_pretrained(student_name, trust_remote_code=True).model_type
            teacher_type = AutoConfig.from_pretrained(teacher_name, trust_remote_code=True).model_type
            return student_type == teacher_type
        except Exception as e:
            logger.warning(
                f"Could not determine student/teacher architecture family ({e}); "
                "defaulting teacher to plain HF loading (safer default)."
            )
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
                # core/peft/lora.py's apply_to_unsloth() reads
                # self.config.train.gradient_checkpointing / .seed - this
                # minimal config stub needs a matching .train shape or
                # get_peft_model() crashes with AttributeError.
                self.train = type('obj', (object,), {'gradient_checkpointing': 'unsloth', 'seed': 3407})()

        # Student - Unsloth-accelerated
        logger.info(f"Loading student: {self.config.model.student_model}")
        apply_peft = self.config.model.use_peft
        self.student_model, self.tokenizer = build_model(
            config=MinimalConfig(StudentConfig(self.config.model)),
            task_type=TaskType.DISTILLATION,
            use_unsloth=True,
            apply_peft=apply_peft,
            is_reference=False
        )
        logger.info("✓ Student loaded (Unsloth)")

        # Decide whether the teacher loads through Unsloth too.
        #
        # If student and teacher share an architecture family (e.g. this
        # notebook's Qwen2.5-0.5B student / Qwen2.5-1.5B teacher - the common
        # "standard distillation" case), loading the student via Unsloth
        # first monkey-patches that architecture's attention/norm classes
        # process-wide, so a plain-HF-loaded teacher of the same family
        # inherits those patches without the internal buffers Unsloth's own
        # loader sets up, crashing with "'Qwen2Attention' object has no
        # attribute 'apply_qkv'". Loading the teacher via Unsloth too avoids
        # that collision. Different-architecture-family pairs (e.g. GOLD's
        # phi-2 teacher / Qwen student) don't hit it and stay on plain HF,
        # matching prior behavior - explicit config.model.teacher_use_unsloth
        # overrides this auto-detection either way.
        teacher_use_unsloth = getattr(self.config.model, 'teacher_use_unsloth', None)
        if teacher_use_unsloth is None:
            teacher_use_unsloth = self._same_architecture_family(
                self.config.model.student_model, self.config.model.teacher_model
            )
            logger.info(
                f"Auto-detected teacher_use_unsloth={teacher_use_unsloth} "
                f"(student/teacher {'share' if teacher_use_unsloth else 'do not share'} an architecture family)"
            )

        logger.info(f"Loading teacher: {self.config.model.teacher_model} (use_unsloth={teacher_use_unsloth})")

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
                self.use_unsloth = teacher_use_unsloth
                self.peft = type('obj', (object,), {'enabled': False})()

        self.teacher_model, _ = build_model(
            config=MinimalConfig(TeacherConfig(self.config)),
            task_type=TaskType.DISTILLATION,
            use_unsloth=teacher_use_unsloth,
            apply_peft=False,
            is_reference=True
        )

        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        logger.info("✓ Teacher loaded and frozen")

    def train(self) -> Dict[str, Any]:
        """Train using TRL DistillationTrainer with the Unsloth-accelerated student."""
        from trl.experimental.distillation import DistillationConfig, DistillationTrainer

        logger.info("Starting training...")

        # Setup
        if not self.student_model:
            self.setup_model()
        if not self.train_dataset:
            self.setup_data()

        # Optimizer/scheduler
        optim_scheduler = self._setup_optimizer_scheduler(dataset_for_estimation=self.train_dataset)

        # DistillationConfig defaults bf16=True unless fp16 is set, regardless
        # of hardware support (same TrainingArguments-level default as
        # OnlineDPOConfig) - resolve the actual fp16/bf16 flags for the
        # detected GPU instead, so this doesn't crash on pre-Ampere GPUs.
        from aligntune.core.precision_handler import PrecisionHandler
        precision = PrecisionHandler.get_precision_from_config(self.config, default='auto')
        precision_flags = PrecisionHandler.get_training_args_precision(precision)

        # TRL config
        # Resolve the public on_policy switch to TRL's lmbda. An explicit
        # lmbda always wins; otherwise offline is 0.0 and online is 1.0.
        configured_lmbda = self.config.train.lmbda
        if configured_lmbda is None:
            on_policy = self.config.train.on_policy
            if isinstance(on_policy, str):
                on_policy = on_policy.strip().lower() in {"online", "true", "1", "yes"}
            configured_lmbda = 1.0 if on_policy else 0.0

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
            fp16=precision_flags['fp16'],
            bf16=precision_flags['bf16'],
            # DistillationConfig's own max_length (default 1024) governs how
            # long TRL's collator lets an example get - decoupled from
            # config.model.max_seq_length (what we pass to
            # FastLanguageModel.from_pretrained on the Unsloth backend)
            # unless set explicitly. Any example landing between the two
            # sails through the collator untouched, then gets silently
            # truncated by Unsloth inside the model forward, desyncing the
            # logits length from the labels/completion_tokens length
            # DistillationTrainer computed from the untruncated input -
            # "Size does not match at dimension 1" deep inside its loss
            # computation. Keeping both bounds equal prevents that class of
            # desync regardless of which backend or teacher/student loading
            # combination is used.
            max_length=self.config.model.max_seq_length,
            # DistillationConfig requires max_completion_length < max_length
            # (room for the prompt) - its own default (512) can easily
            # exceed a small max_seq_length, so cap it relative to max_length
            # rather than leaving it at a value that may not fit.
            max_completion_length=min(
                self.config.train.max_completion_length or 256,
                max(1, self.config.model.max_seq_length // 2),
            ),
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

        if trl_config.lmbda > 0.0:
            # On-policy/mixed distillation (lmbda>0) makes DistillationTrainer call
            # student_model.generate() mid-training. Unsloth's patched generate()
            # forces torch.inference_mode() internally for speed (see
            # unsloth.models.llama's _get_inference_mode_context_manager), so the
            # returned sequences come out as inference tensors - any later autograd
            # op that touches them (the student log-probs gather in TRL's on-policy
            # divergence loss) crashes with "Inference tensors cannot be saved for
            # backward". Unsloth's own RL integration (unsloth/models/rl.py) works
            # around this by cloning the generate() output before use;
            # DistillationTrainer isn't one of Unsloth's patched RL trainers, so
            # replicate the same fix here.
            original_generate = self.student_model.generate

            def _generate_with_clone(*args, **kwargs):
                out = original_generate(*args, **kwargs)
                if hasattr(out, "sequences"):
                    out.sequences = out.sequences.clone()
                elif isinstance(out, torch.Tensor):
                    out = out.clone()
                return out

            self.student_model.generate = _generate_with_clone

        train_result = self.trainer.train()
        logger.info("✓ Training completed")
        return train_result

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        return self.trainer.evaluate(*args, **kwargs) if self.trainer else {}
