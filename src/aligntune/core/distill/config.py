"""
Unified configuration system for Knowledge Distillation training.

Follows the same pattern as core/rl/config.py with separate config classes
for model, dataset, training, logging, and distributed setup.

The trainer extracts params based on distillation method detected.
Supports: Standard Distillation, SDFT
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class PrecisionType(Enum):
    """Model precision types."""
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    AUTO = "auto"


class BackendType(Enum):
    """Distributed training backends."""
    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"


class DistillationType(Enum):
    """Distillation method types."""
    STANDARD = "distillation"      # Standard distillation with external teacher
    GOLD = "gold"                  # Cross-tokenizer distillation
    SDFT = "sdft"                  # Self-distillation fine-tuning
    SDPO = "sdpo"                  # Self-distillation + RL rewards


@dataclass
class DistillModelConfig:
    """Model configuration for distillation with student and teacher models."""

    # Student model (trainable) - REQUIRED
    student_model: str

    # Teacher model - required for Standard/GOLD, optional for SDFT/SDPO
    teacher_model: Optional[str] = None

    # Teacher tokenizer (for cross-tokenizer GOLD distillation)
    teacher_tokenizer_name_or_path: Optional[str] = None

    # Whether the teacher model backbone loads through Unsloth's
    # FastLanguageModel (backend="unsloth" only; ignored on "trl"). None
    # (default) auto-detects: same-architecture-family student/teacher pairs
    # (the common "standard distillation" case, e.g. Qwen2.5-0.5B student /
    # Qwen2.5-1.5B teacher) need the teacher on Unsloth too, since loading
    # the student via Unsloth first monkey-patches that architecture's
    # attention/norm classes process-wide - a plain-HF-loaded teacher of the
    # same family inherits those patches without the internal buffers
    # Unsloth's own loader sets up, crashing with "'Qwen2Attention' object
    # has no attribute 'apply_qkv'". Different-architecture-family pairs
    # (e.g. GOLD's phi-2 teacher / Qwen student) don't hit that collision and
    # default to plain HF loading, matching prior behavior. Set explicitly to
    # override either way.
    teacher_use_unsloth: Optional[bool] = None

    # Student precision
    student_dtype: str = "bf16"
    teacher_dtype: str = "bf16"

    # Self-distillation teacher type (for SDFT/SDPO)
    # "base": use base model checkpoint as teacher
    # "live": use EMA of current model as teacher
    # "ema": same as "live"
    teacher_model_kind: Optional[str] = None

    # Model precision and quantization
    precision: PrecisionType = PrecisionType.AUTO
    quantization: Dict[str, Any] = field(default_factory=dict)
    attn_implementation: str = "auto"
    gradient_checkpointing: bool = False
    max_memory: Optional[Dict[str, str]] = None
    device_map: Optional[Union[str, Dict[str, int]]] = None
    max_seq_length: int = 2048

    # PEFT (LoRA) configuration
    use_peft: bool = True
    lora_variant: str = "standard"  # standard, kron, moa, text2lora, doc2lora, squeezed
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Advanced PEFT methods
    dora_enabled: bool = False
    rslora_enabled: bool = False
    loftq_init: bool = False
    pissa_init: bool = False

    # Model initialization
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    teacher_model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = True
    # tokenizer_name_or_path: Optional[str] = None  # TODO: Add in next sprint
    # embedding_init_method: str = "random"  # TODO: Add in next sprint

    backend: str = "trl"
    use_unsloth: bool = False

    def __post_init__(self):
        """Validate model configuration."""
        if not self.student_model:
            raise ValueError("student_model is required")

        # Validate teacher model setup
        if self.teacher_model_kind and self.teacher_model:
            raise ValueError("Cannot specify both teacher_model_kind (self-distill) and teacher_model (external)")

        if not self.teacher_model_kind and not self.teacher_model:
            raise ValueError("Must specify either teacher_model (external) or teacher_model_kind (self-distill)")

        # Validate precision
        if not isinstance(self.precision, PrecisionType):
            if isinstance(self.precision, str):
                self.precision = PrecisionType(self.precision)
            else:
                raise ValueError(f"Invalid precision type: {self.precision}")

        if self.device_map is None:
            self.device_map = "auto"


@dataclass
class DistillDatasetConfig:
    """Dataset configuration for distillation."""
    name: str
    split: Optional[str] = None
    percent: Optional[float] = None
    max_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Field mapping
    field_mappings: Dict[str, str] = field(default_factory=dict)
    format_type: Optional[str] = None
    auto_detect_fields: bool = True
    column_mapping: Dict[str, str] = field(default_factory=dict)

    # Optional user override for the trainer's default DataManager task type.
    task_type: Optional[str] = None
    weight: float = 1.0
    chat_template: Optional[str] = None
    system_prompt: Optional[str] = None
    enable_thinking: bool = False

    # Processing parameters
    dataset_num_proc: Optional[int] = None
    pad_token: Optional[str] = None
    label_pad_token_id: int = -100
    truncation_mode: str = 'keep_end'
    padding_free: bool = False

    # Custom processing
    preserve_columns: Optional[List[str]] = None
    keep_columns: Optional[bool] = None
    processing_fn: Optional[Any] = None
    processing_batched: bool = False
    processing_fn_kwargs: Dict[str, Any] = field(default_factory=dict)

    config_name: Optional[str] = None

    # DataManager split and CuratorKIT controls.
    val_split_ratio: Optional[float] = None
    test_split_ratio: Optional[float] = None
    split_seed: int = 42
    curator_schema_gate: bool = True
    curator_clean: bool = False
    curator_dedup: str = "none"
    curator_use_tiktoken: bool = False
    curator_max_tokens: int = 1_000_000

    # Only used by self-distillation methods (SDFT/SDPO).
    privileged_context_column: Optional[str] = None

    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.name:
            raise ValueError("Dataset name is required")

        if self.percent is not None and (self.percent <= 0 or self.percent > 100):
            raise ValueError("Dataset percent must be between 0 and 100")

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")

        if self.weight <= 0:
            raise ValueError("Dataset weight must be positive")


@dataclass
class DistillTrainingConfig:
    """Training configuration for distillation.

    Contains ALL parameters across all distillation methods.
    Trainer extracts relevant params based on detected method.
    """

    # ============================================
    # BASIC TRAINING PARAMETERS
    # ============================================
    per_device_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    epochs: Optional[int] = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer and scheduler
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.1

    # ============================================
    # STANDARD DISTILLATION + GOLD PARAMETERS
    # ============================================

    # Temperature for softmax in KD loss
    temperature: float = 3.0

    # Alpha for blending CE + KD loss
    alpha: float = 0.5

    # Standard/GOLD source of completions: dataset completions when False,
    # student-generated completions when True.
    # Accept a boolean for backward compatibility or the explicit strings
    # "online"/"offline" for a clearer user-facing API.
    on_policy: Union[bool, str] = False

    # On-policy vs Offline control (lmbda parameter)
    # 0.0 = fully offline (teacher only)
    # 1.0 = fully online (student samples)
    # 0.0-1.0 = mixed
    lmbda: Optional[float] = None

    # Divergence type control (beta parameter)
    # 0.0 = forward KL
    # 0.5 = JSD
    # 1.0 = reverse KL
    beta: Optional[float] = None

    # ============================================
    # GOLD-SPECIFIC PARAMETERS
    # ============================================
    use_uld_loss: bool = False
    uld_use_hybrid_loss: bool = False

    # ============================================
    # SELF-DISTILLATION PARAMETERS (SDFT/SDPO)
    # ============================================

    # Distillation mode
    # "sampled_token", "topk_logits", "full_logits"
    distillation_mode: Optional[str] = None

    # Weight for distillation loss vs CE loss
    distillation_alpha: float = 0.5

    # Top-k for topk_logits mode
    distillation_topk: int = 5

    # Weight for distillation in loss computation
    distillation_weight: float = 1.0

    # Generate completions from teacher model
    generate_from_teacher: bool = False

    # ============================================
    # SDPO-SPECIFIC PARAMETERS
    # ============================================
    use_successful_as_teacher: bool = True
    include_environment_feedback: bool = False

    # ============================================
    # EVALUATION & CHECKPOINTING
    # ============================================
    eval_steps: int = 100
    eval_strategy: str = "steps"
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = False

    # Logging
    logging_steps: int = 10
    logging_strategy: str = "steps"

    # ============================================
    # GENERATION PARAMETERS
    # ============================================
    top_p: float = 0.95
    top_k: int = 0
    temperature_generation: float = 0.6
    max_new_tokens: int = 256

    # SDFT-specific generation parameters
    num_generations: int = 8  # Number of generations to sample (for SDFT)
    max_completion_length: Optional[int] = 256  # Max completion length (for SDFT)

    # ============================================
    # VLLM CONFIGURATION (for rollout/generation)
    # ============================================
    use_vllm: bool = False
    vllm_mode: str = "colocate"  # colocate or server
    vllm_model_impl: str = "vllm"
    vllm_enable_sleep_mode: bool = False
    vllm_structured_outputs_regex: Optional[str] = None
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    vllm_group_port: int = 51216
    vllm_gpu_memory_utilization: float = 0.3
    vllm_max_model_length: Optional[int] = None
    vllm_tensor_parallel_size: int = 1

    # ============================================
    # SEED & MISC
    # ============================================
    seed: Optional[int] = 42
    data_seed: Optional[int] = 47
    torch_compile: bool = False
    use_cache: bool = False

    # Distributed training config path
    accelerate_config_path: Optional[str] = None

    # Extra parameters for flexibility - passed to TRL trainer
    # User can add any params here that TRL trainer supports
    # TRL will validate and error if unsupported
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate training configuration."""
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")

        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.max_steps is None and self.epochs is None:
            raise ValueError("Either max_steps or epochs must be specified")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")

        if self.lmbda is not None:
            if not (0.0 <= self.lmbda <= 1.0):
                raise ValueError(f"lmbda must be between 0.0 and 1.0, got {self.lmbda}")

        if isinstance(self.on_policy, str):
            policy = self.on_policy.strip().lower()
            if policy not in {"online", "offline"}:
                raise ValueError("on_policy must be True/False or 'online'/'offline'")
            requested_online = policy == "online"
            if requested_online and self.lmbda not in {None, 1.0}:
                raise ValueError("on_policy='online' requires lmbda=1.0")
            if not requested_online and self.lmbda not in {None, 0.0}:
                raise ValueError("on_policy='offline' requires lmbda=0.0")
            self.on_policy = requested_online

        if self.on_policy:
            if self.lmbda not in {None, 1.0}:
                raise ValueError("on_policy=True requires lmbda=1.0")
            self.lmbda = 1.0
        elif self.lmbda is None:
            self.lmbda = 0.0

        if self.beta is not None:
            if self.beta < 0.0 or self.beta > 1.0:
                raise ValueError(f"beta must be between 0.0 and 1.0, got {self.beta}")

        if not (0 < self.distillation_alpha < 1):
            raise ValueError(f"distillation_alpha must be between 0 and 1, got {self.distillation_alpha}")

        if self.distillation_topk <= 0:
            raise ValueError(f"distillation_topk must be positive")

        if self.distillation_weight <= 0:
            raise ValueError(f"distillation_weight must be positive")


@dataclass
class DistillLoggingConfig:
    """Logging configuration for distillation."""
    loggers: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    output_dir: str = "./distill_output"
    log_level: str = "INFO"
    report_to: str = "none"

    # Weights & Biases integration
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_name: Optional[str] = None

    def __post_init__(self):
        """Validate logging configuration."""
        valid_loggers = {
            "azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub",
            "dvclive", "flyte", "mlflow", "neptune", "swanlab",
            "tensorboard", "trackio", "wandb", "all", "none"
        }
        for logger in self.loggers:
            if logger not in valid_loggers:
                raise ValueError(f"Invalid logger: {logger}")

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    backend: BackendType = BackendType.SINGLE
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    deepspeed_config: Dict[str, Any] = field(default_factory=dict)
    nodes: int = 1
    gpus_per_node: int = 1
    seed: int = 42

    def __post_init__(self):
        """Validate distributed configuration."""
        if not isinstance(self.backend, BackendType):
            if isinstance(self.backend, str):
                self.backend = BackendType(self.backend)
            else:
                raise ValueError(f"Invalid backend type: {self.backend}")

        if self.nodes <= 0:
            raise ValueError("nodes must be positive")

        if self.gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")


@dataclass
class UnifiedDistillConfig:
    """Unified configuration for Knowledge Distillation training.

    Auto-detects distillation method based on config:
    1. GOLD: if use_uld_loss=True OR teacher has different tokenizer
    2. SDPO: if teacher_model_kind is set AND rewards are configured
    3. SDFT: if teacher_model_kind is set (no rewards)
    4. Standard Distillation: default (external teacher)

    The trainer extracts relevant params based on detected method.
    """

    model: DistillModelConfig
    dataset: DistillDatasetConfig
    train: DistillTrainingConfig = field(default_factory=DistillTrainingConfig)
    logging: DistillLoggingConfig = field(default_factory=DistillLoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Optional reward configuration for SDPO
    rewards: List[Dict[str, Any]] = field(default_factory=list)

    # Chat template and caching
    chat_template: Optional[str] = None
    caching: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate unified configuration."""
        pass

    def get_distillation_type(self) -> DistillationType:
        """
        Auto-detect which distillation method to use based on config.

        Returns:
            DistillationType: One of STANDARD, GOLD, SDFT, SDPO
        """
        # Check for GOLD indicators
        if self.train.use_uld_loss:
            return DistillationType.GOLD

        # Check for self-distillation
        if self.model.teacher_model_kind is not None:
            # SDPO if rewards configured
            if self.rewards and len(self.rewards) > 0:
                return DistillationType.SDPO
            # SDFT otherwise
            return DistillationType.SDFT

        # Default to Standard Distillation
        return DistillationType.STANDARD

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def _serialize(value):
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "to_dict"):
                return value.to_dict()
            if isinstance(value, (list, tuple)):
                return [_serialize(item) for item in value]
            if isinstance(value, dict):
                return {k: _serialize(v) for k, v in value.items()}
            if hasattr(value, "__dict__"):
                return {k: _serialize(v) for k, v in value.__dict__.items()}
            return value

        return {
            field_name: _serialize(field_value)
            for field_name, field_value in self.__dict__.items()
        }


# Alias for backwards compatibility
DistillConfig = UnifiedDistillConfig
