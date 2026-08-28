"""
Enhanced SFT Configuration with Complete Task Type Support.

This module extends the original SFT config with:
- All task types (instruction following, classification, chat, etc.)
- Task-specific dataset field mappings
- Enhanced validation
- Task-aware defaults
- Seed support for reproducibility
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum

# Import TaskType and TrainingType from unified registry
from ..registry import TaskType, TrainingType
from ..long_context.rope_config import RopeConfig


class PrecisionType(Enum):
    """Model precision types."""
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    AUTO = "auto"  # ADD THIS


class BackendType(Enum):
    """Distributed training backend types."""
    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"


@dataclass
class PeftConfigData:
    """Dedicated configuration for PEFT settings."""
    enabled: bool = False
    variant: str = "standard"  # "standard", "moa", etc.
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    
    # Advanced LoRA variants
    dora_enabled: bool = False
    rslora_enabled: bool = False
    
    # Advanced initialization
    loftq_init: bool = False
    pissa_init: bool = False
    
    # Legacy flags for backward compatibility
    rslora: bool = False
    init_weights: Union[str, bool] = True
    
    loraplus_lr_ratio: Optional[float] = None
    loraplus_lr_embedding: Optional[float] = None

@dataclass
class ModelConfig:
    """Model configuration with task-aware defaults."""
    name_or_path: str
    tokenizer_name_or_path: Optional[str] = None  # If None, use name_or_path for tokenizer
    precision: PrecisionType = PrecisionType.AUTO
    quantization: Dict[str, Any] = field(default_factory=dict)
    attn_implementation: str = "auto"
    sliding_window: Optional[int] = None
    s2_group_size_ratio: float = 0.25
    s2_min_seq_length: int = 64
    s2_shift_ratio: float = 0.5
    gradient_checkpointing: bool = True
    max_memory: Optional[Dict[str, str]] = None
    use_unsloth: bool = False
    max_seq_length: int = 2048
    use_gradient_checkpointing: bool = True
    peft: PeftConfigData = field(default_factory=PeftConfigData)
    rope: RopeConfig = field(default_factory=RopeConfig)
    # Classification-specific
    num_labels: Optional[int] = None
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    device_map: Optional[Union[str, Dict]] = "auto"
    trust_remote_code: bool = True
    embedding_pad_to_multiple_of: Optional[int] = None
    train_embeddings: bool = False
    embedding_init_method: str = "random"  # Options: random, mean, mean_of_constituents (FVT)
    # VLM fields
    is_vlm: bool = False
    vision_tower_mode: str = "freeze"
    image_column: str = "image"
    max_image_tokens: int = 4096
    # PEFT convenience fields
    peft_enabled: bool = True
    lora_rank: int = 16
    dora_enabled: bool = False
    rslora_enabled: bool = False
    loftq_init: bool = False
    pissa_init: bool = False
    use_liger_kernel: bool = False


    def __post_init__(self):
        """Validate model configuration."""
        if not self.name_or_path:
            raise ValueError("Model name_or_path is required and cannot be empty")
        
        if not isinstance(self.precision, PrecisionType):
            if self.precision is None:
                self.precision = PrecisionType.AUTO
            elif isinstance(self.precision, str):
                self.precision = PrecisionType(self.precision)
            else:
                raise ValueError(f"Invalid precision type: {self.precision}")


@dataclass
class VLMModelConfig:
    """Vision-Language Model configuration."""
    is_vlm: bool = True
    vision_tower_mode: str = "freeze"
    image_column: str = "image"
    max_image_tokens: int = 4096


@dataclass
class DatasetConfig:
    """Dataset configuration with task-specific field support."""
    name: str
    train_split: str = "train"
    test_split: Optional[str] = None
    split: Optional[str] = None  # Legacy split field
    config_name: Optional[str] = None  # Hugging Face dataset configuration
    subset: Optional[str] = None  # Add this line for dataset config/subset
    config: Optional[str] = None  # Alternative name for subset
    percent: Optional[float] = None
    max_samples: Optional[int] = None
    column_mapping: Dict[str, str] = field(default_factory=dict)
    task_type: TaskType = TaskType.SFT
    system_prompt: Optional[str] = None
    
    # Auto-detection
    auto_detect_fields: bool = False
    format_type: Optional[str] = None
    
    # Common fields for all tasks
    text_column: str = "text"
    
    # Instruction/SFT specific
    instruction_column: str = "instruction"
    response_column: str = "response"
    output_column: str = "output"
    input_column: str = "input"
    context_column: str = "context"
    
    # Classification specific
    label_column: str = "label"
    
    # Token classification specific
    tokens_column: str = "tokens"
    tags_column: str = "ner_tags"
    
    # Chat specific
    messages_column: str = "messages"
    
    # Dataset text field for trainers
    dataset_text_field: str = "text"

    # VLM-specific
    image_column: str = "image"
    
    # Chat template
    chat_template: Optional[str] = None

    dataset_num_proc: Optional[int] = None
    pad_token: Optional[str] = None

    # Processing and Filtering (Added to match RLDatasetConfig and fix factory error)
    preserve_columns: Optional[List[str]] = None
    keep_columns: Optional[bool] = None
    processing_fn: Optional[Callable] = None
    processing_batched: bool = False
    processing_fn_kwargs: Dict[str, Any] = field(default_factory=dict)

    # CuratorKIT data processing
    val_split_ratio: Optional[float] = None
    test_split_ratio: Optional[float] = None
    split_seed: int = 42
    curator_schema_gate: bool = True
    curator_clean: bool = False
    curator_dedup: str = "none"
    curator_use_tiktoken: bool = False
    curator_max_tokens: int = 1_000_000
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.name:
            raise ValueError("Dataset name is required and cannot be empty")
        
        if self.percent is not None and (self.percent <= 0 or self.percent > 100):
            raise ValueError("Dataset percent must be between 0 and 100")
        
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        
        if not isinstance(self.task_type, TaskType):
            if isinstance(self.task_type, str):
                self.task_type = TaskType(self.task_type)
            else:
                raise ValueError(f"Invalid task type: {self.task_type}")
        
        # Normalize subset/config (they're the same thing)
        if self.config and not self.subset:
            self.subset = self.config
        elif self.subset and not self.config:
            self.config = self.subset
        
        # Set task-specific defaults
        self._set_task_defaults()
    
    def _set_task_defaults(self):
        if not self.column_mapping:
            self.column_mapping = {}


@dataclass
class TrainingConfig:
    """Training configuration with task-aware defaults."""
    per_device_batch_size: int = 1
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    eval_interval: int = 100
    save_interval: int = 500
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    
    # Optimizer and scheduler
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    
    # Advanced training settings
    group_by_length: bool = False
    dataloader_drop_last: bool = False
    eval_accumulation_steps: Optional[int] = None
    label_smoothing_factor: float = 0.0
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # Evaluation
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # TRL specific
    use_trl: bool = False

    dataset_num_proc: Optional[int] = None
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Sequence packing and padding
    packing: bool = False
    packing_strategy: str = "bfd"
    eval_packing: Optional[bool] = None
    padding_free: bool = False
    pad_to_multiple_of: Optional[int] = None

    # Loss and masking controls
    completion_only_loss: Optional[bool] = None
    assistant_only_loss: bool = False
    loss_type: str = "nll"
    activation_offloading: bool = False
    use_flash_attention_2: Optional[bool] = None
    enable_thinking: bool = False
    gradient_checkpointing: bool = False # Added
    gradient_checkpointing_kwargs: Dict[str, Any] = field(default_factory=dict) # Added
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Speed optimization flags
    torch_compile: bool = False
    use_liger_kernel: bool = False
    fp8_adamw: bool = False

    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None

    # Distributed training via HF Accelerate
    # Point to a valid `accelerate config`-generated YAML to enable DeepSpeed/FSDP.
    # AlignTune surfaces this path; TRL + Accelerate handle the actual distributed setup.
    accelerate_config_path: Optional[str] = None


    def __post_init__(self):
        """Validate training configuration."""
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Set default epochs if max_steps not specified
        if self.epochs is None and self.max_steps is None:
            self.epochs = 3

        if self.dataset_num_proc is not None and self.dataset_num_proc <= 0:
            raise ValueError("dataset_num_proc must be positive")

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive when set")

        if self.packing_strategy not in {"bfd", "wrapped"}:
            raise ValueError("packing_strategy must be 'bfd' or 'wrapped'")

        if self.loss_type not in {"nll", "dft"}:
            raise ValueError("loss_type must be 'nll' or 'dft'")


@dataclass
class EvaluationConfig:
    """Evaluation configuration for SFT models."""
    # Which metrics to compute
    compute_perplexity: bool = True
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_meteor: bool = False  # Optional: requires nltk
    compute_bertscore: bool = False  # Optional: requires bert-score
    compute_semantic_similarity: bool = False  # Optional: requires sentence-transformers
    compute_codebleu: bool = False  # Optional: for code tasks, requires codebleu
    
    # Custom metrics: list of callables taking (prediction, reference, instruction) -> Dict[str, float]
    custom_metrics: Optional[List[Callable[[str, str, str], Dict[str, float]]]] = None
    
    # Evaluation parameters
    max_samples_for_quality_metrics: int = 50  # Limit samples for faster evaluation
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"  # Model for BERTScore
    semantic_similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # For semantic similarity
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.max_samples_for_quality_metrics < 1:
            raise ValueError("max_samples_for_quality_metrics must be >= 1")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    output_dir: str = "./output"
    run_name: Optional[str] = None
    loggers: List[str] = field(default_factory=lambda: ["tensorboard"])
    log_level: str = "INFO"
    log_interval: int = 10
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    report_to: str = "none"
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_loggers = {"azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "swanlab", "tensorboard", "trackio", "wandb", "all", "none"}
        for logger in self.loggers:
            if logger not in valid_loggers:
                raise ValueError(f"Invalid logger: {logger}. Must be one of {valid_loggers}")

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}. Must be one of {valid_levels}")


@dataclass
class SFTConfig:
    """Unified configuration for SFT training. Supports: SFT, Text Classification, Token Classification."""
    model: ModelConfig
    dataset: DatasetConfig
    eval_dataset: Optional[DatasetConfig] = None
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def __post_init__(self):
        """Validate unified configuration and apply task-specific settings."""
        # Validate required fields
        if not self.model.name_or_path:
            raise ValueError("Model name_or_path is required")
        
        if not self.dataset.name:
            raise ValueError("Dataset name is required")
        
        # Apply task-specific validation and defaults
        self._apply_task_specific_settings()
    
    def _apply_task_specific_settings(self):
        import logging
        logger = logging.getLogger(__name__)
        task_type = self.dataset.task_type

        if task_type == TaskType.TEXT_CLASSIFICATION:
            if self.model.use_unsloth:
                logger.warning("Unsloth is not recommended for text classification.")
            if self.model.num_labels is None:
                self.model.num_labels = 2

        elif task_type == TaskType.TOKEN_CLASSIFICATION:
            if self.model.use_unsloth:
                logger.warning("Unsloth is not recommended for token classification.")
            if self.model.num_labels is None:
                self.model.num_labels = 9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif hasattr(field_value, '__dict__'):
                nested_dict = {}
                for nested_name, nested_value in field_value.__dict__.items():
                    if isinstance(nested_value, Enum):
                        nested_dict[nested_name] = nested_value.value
                    else:
                        nested_dict[nested_name] = nested_value
                result[field_name] = nested_dict
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SFTConfig':
        """Create configuration from dictionary."""
        # Extract nested configs
        model_dict = config_dict.get('model', {})
        dataset_dict = config_dict.get('dataset', {})
        train_dict = config_dict.get('train', {})
        logging_dict = config_dict.get('logging', {})
        
        # Convert enums
        if 'precision' in model_dict and isinstance(model_dict['precision'], str):
            model_dict['precision'] = PrecisionType(model_dict['precision'])
        
        if 'task_type' in dataset_dict and isinstance(dataset_dict['task_type'], str):
            dataset_dict['task_type'] = TaskType(dataset_dict['task_type'])
        
        # Create config objects
        model_config = ModelConfig(**model_dict)
        dataset_config = DatasetConfig(**dataset_dict)
        train_config = TrainingConfig(**train_dict)
        logging_config = LoggingConfig(**logging_dict)
        
        return cls(
            model=model_config,
            dataset=dataset_config,
            train=train_config,
            logging=logging_config
        )
    
    def get_task_type(self) -> TaskType:
        """Get the task type for this configuration."""
        return self.dataset.task_type
    
    def is_classification_task(self) -> bool:
        """Check if this is a classification task."""
        return self.dataset.task_type in [
            TaskType.TEXT_CLASSIFICATION,
            TaskType.TOKEN_CLASSIFICATION
        ]
    
    def is_generation_task(self) -> bool:
        return not self.is_classification_task()
    
    def supports_unsloth(self) -> bool:
        """Check if Unsloth backend is suitable for this task."""
        # Unsloth works best with generation tasks
        return self.is_generation_task()
    
    def get_recommended_backend(self) -> str:
        """Get recommended backend for this task type."""
        if self.is_classification_task():
            return "trl"  # TRL for classification
        else:
            return "unsloth"  # Unsloth for generation tasks


# Helper functions for creating configs
def create_text_classification_config(
    model_name: str,
    dataset_name: str,
    num_labels: int,
    output_dir: str = "./output/classification",
    **kwargs
) -> SFTConfig:
    """Create configuration for text classification tasks."""
    return SFTConfig(
        model=ModelConfig(
            name_or_path=model_name,
            max_seq_length=kwargs.get('max_seq_length', 512),
            use_unsloth=False,  # Not recommended for classification
            peft=PeftConfigData(enabled=kwargs.get('peft_enabled', True)),
            num_labels=num_labels,
            quantization=kwargs.get('quantization', {})
        ),
        dataset=DatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples'),
            task_type=TaskType.TEXT_CLASSIFICATION,
            text_column=kwargs.get('text_column', 'text'),
            label_column=kwargs.get('label_column', 'label')
        ),
        train=TrainingConfig(
            epochs=kwargs.get('epochs', 3),
            per_device_batch_size=kwargs.get('batch_size', 8),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1)
        ),
        logging=LoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name', 'text_classification')
        )
    )


def create_token_classification_config(
    model_name: str,
    dataset_name: str,
    num_labels: int,
    output_dir: str = "./output/token_classification",
    **kwargs
) -> SFTConfig:
    """Create configuration for token classification (NER) tasks."""
    return SFTConfig(
        model=ModelConfig(
            name_or_path=model_name,
            max_seq_length=kwargs.get('max_seq_length', 512),
            use_unsloth=False,  # Not recommended for token classification
            peft=PeftConfigData(enabled=kwargs.get('peft_enabled', True)),
            num_labels=num_labels,
            quantization=kwargs.get('quantization', {})
        ),
        dataset=DatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples'),
            task_type=TaskType.TOKEN_CLASSIFICATION,
            tokens_column=kwargs.get('tokens_column', 'tokens'),
            tags_column=kwargs.get('tags_column', 'ner_tags')
        ),
        train=TrainingConfig(
            epochs=kwargs.get('epochs', 3),
            per_device_batch_size=kwargs.get('batch_size', 8),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1)
        ),
        logging=LoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name', 'token_classification')
        )
    )


def create_supervised_finetuning_config(
    model_name: str,
    dataset_name: str,
    output_dir: str = "./output/sft",
    **kwargs
) -> SFTConfig:
    """Create configuration for standard supervised fine-tuning."""
    return SFTConfig(
        model=ModelConfig(
            name_or_path=model_name,
            max_seq_length=kwargs.get('max_seq_length', 512),
            use_unsloth=kwargs.get('use_unsloth', True),
            peft=PeftConfigData(enabled=kwargs.get('peft_enabled', True)),
            quantization=kwargs.get('quantization', {"load_in_4bit": True})
        ),
        dataset=DatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples'),
            task_type=TaskType.SUPERVISED_FINE_TUNING,
            instruction_column=kwargs.get('instruction_column', 'instruction'),
            response_column=kwargs.get('response_column', 'response'),
            auto_detect_fields=kwargs.get('auto_detect_fields', True)
        ),
        train=TrainingConfig(
            epochs=kwargs.get('epochs', 3),
            per_device_batch_size=kwargs.get('batch_size', 4),
            learning_rate=kwargs.get('learning_rate', 2e-4),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 1)
        ),
        logging=LoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name', 'supervised_finetuning')
        )
    )


# Example usage and documentation
__all__ = [
    'TaskType',
    'PrecisionType',
    'BackendType',
    'ModelConfig',
    'VLMModelConfig',

    'DatasetConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'LoggingConfig',
    'SFTConfig',
    'create_text_classification_config',
    'create_token_classification_config',
    'create_supervised_finetuning_config',
]
