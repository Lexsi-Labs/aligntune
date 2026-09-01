"""
Unified configuration system for Tokenization Training.

Supports multilingual LLM adaptation through:
- Continued BPE training for vocabulary extension
- Multi-stage parameter freezing and unfreezing
- Subword-based embedding initialization
- Vocabulary pruning (leaf-based, frequency-based)
- Tokenization evaluation (fertility, fairness metrics)

Based on research from:
- "Teaching Old Tokenizers New Words" (Purason et al., 2026)
- "EEVE: Efficient and Effective Vocabulary Expansion" (Kim et al., 2024)
- "Chinese LLaMA and Alpaca" (Cui et al., 2024)
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


class TokenizationOperation(Enum):
    """Types of tokenization operations."""
    VOCAB_EXTENSION = "vocab_extension"  # Extend vocab with continued BPE
    VOCAB_PRUNING = "vocab_pruning"      # Remove unused tokens
    EMBEDDING_INIT = "embedding_init"     # Initialize embeddings for new tokens
    STAGED_TRAINING = "staged_training"   # Multi-stage parameter freezing
    EVALUATION = "evaluation"             # Evaluate tokenization efficiency


class VocabExtensionMethod(Enum):
    """Methods for vocabulary extension."""
    CONTINUED_BPE = "continued_bpe"      # Continue BPE merge training (recommended)
    NAIVE_EXTENSION = "naive_extension"  # Simple frequency-based addition
    FAST_VOCAB_TRANSFER = "fvt"         # Fast vocabulary transfer


class EmbeddingInitMethod(Enum):
    """Methods for initializing new token embeddings."""
    SUBWORD_AVERAGE_INPUT = "subword_average_input"   # Average of subword embeddings (input)
    FIRST_SUBWORD_OUTPUT = "first_subword_output"     # First subword (output)
    RANDOM = "random"                                 # Random initialization
    XAVIER = "xavier"                                 # Xavier/Glorot initialization
    ZERO = "zero"                                     # Zero initialization


class PruningMethod(Enum):
    """Methods for vocabulary pruning."""
    LEAF_FREQUENCY = "leaf_frequency"    # Leaf-based + frequency (structure-aware)
    FREQUENCY_ONLY = "frequency_only"    # Frequency-based only
    REACHABILITY = "reachability"        # Remove unreachable tokens (STT)
    NONE = "none"                        # No pruning


@dataclass
class StageConfig:
    """Configuration for a single training stage."""

    # Which parameters to train in this stage
    train_input_embeddings: bool = False
    train_output_embeddings: bool = False
    train_new_embeddings_only: bool = False  # Train only newly added embeddings
    train_transformer: bool = False

    # LoRA configuration for this stage
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training parameters for this stage
    learning_rate: float = 2e-4
    num_train_steps: Optional[int] = None
    num_train_tokens: Optional[int] = None  # Train for N tokens (e.g., 200M)

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500

    def __post_init__(self):
        """Validate stage configuration."""
        if not any([
            self.train_input_embeddings,
            self.train_output_embeddings,
            self.train_transformer
        ]):
            raise ValueError("At least one parameter group must be trainable")


@dataclass
class TokenizationModelConfig:
    """Model configuration for tokenization training."""

    # Base model to adapt
    base_model: str

    # Tokenizer configuration
    base_tokenizer: Optional[str] = None  # If None, use base_model tokenizer
    target_vocab_size: Optional[int] = None  # If None, auto-calculate
    new_tokens_count: int = 20000  # Number of new tokens to add

    # Model precision and quantization
    precision: PrecisionType = PrecisionType.AUTO
    quantization: Dict[str, Any] = field(default_factory=dict)
    attn_implementation: str = "auto"
    gradient_checkpointing: bool = True
    max_memory: Optional[Dict[str, str]] = None
    device_map: Optional[Union[str, Dict[str, int]]] = None
    max_seq_length: int = 2048

    # Model initialization
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = True

    backend: str = "trl"
    use_unsloth: bool = False

    def __post_init__(self):
        """Validate model configuration."""
        if not self.base_model:
            raise ValueError("base_model is required")

        if self.new_tokens_count <= 0:
            raise ValueError("new_tokens_count must be positive")

        if not isinstance(self.precision, PrecisionType):
            if isinstance(self.precision, str):
                self.precision = PrecisionType(self.precision)
            else:
                raise ValueError(f"Invalid precision type: {self.precision}")

        if self.device_map is None:
            self.device_map = "auto"


@dataclass
class VocabExtensionConfig:
    """Configuration for vocabulary extension operation."""

    # Target languages for extension
    target_languages: List[str] = field(default_factory=list)

    # Extension method
    method: VocabExtensionMethod = VocabExtensionMethod.CONTINUED_BPE

    # Continued BPE parameters
    bpe_corpus_size: int = 1_000_000  # Number of samples for BPE training
    min_token_frequency: int = 6000   # Minimum frequency threshold

    # Training data for BPE
    bpe_corpus_dataset: Optional[str] = None
    bpe_corpus_split: str = "train"
    bpe_corpus_text_column: str = "text"

    def __post_init__(self):
        """Validate vocab extension configuration."""
        if not self.target_languages:
            raise ValueError("target_languages cannot be empty")

        if not isinstance(self.method, VocabExtensionMethod):
            if isinstance(self.method, str):
                self.method = VocabExtensionMethod(self.method)
            else:
                raise ValueError(f"Invalid extension method: {self.method}")


@dataclass
class EmbeddingInitConfig:
    """Configuration for embedding initialization."""

    # Input embedding initialization
    input_init_method: EmbeddingInitMethod = EmbeddingInitMethod.SUBWORD_AVERAGE_INPUT

    # Output embedding initialization
    output_init_method: EmbeddingInitMethod = EmbeddingInitMethod.FIRST_SUBWORD_OUTPUT

    # Whether to use dual tokenizer approach
    use_dual_tokenizer: bool = False

    def __post_init__(self):
        """Validate embedding init configuration."""
        if not isinstance(self.input_init_method, EmbeddingInitMethod):
            if isinstance(self.input_init_method, str):
                self.input_init_method = EmbeddingInitMethod(self.input_init_method)
            else:
                raise ValueError(f"Invalid input init method: {self.input_init_method}")

        if not isinstance(self.output_init_method, EmbeddingInitMethod):
            if isinstance(self.output_init_method, str):
                self.output_init_method = EmbeddingInitMethod(self.output_init_method)
            else:
                raise ValueError(f"Invalid output init method: {self.output_init_method}")


@dataclass
class PruningConfig:
    """Configuration for vocabulary pruning."""

    # Whether to enable pruning
    enabled: bool = False

    # Pruning method
    method: PruningMethod = PruningMethod.LEAF_FREQUENCY

    # Pruning ratio (fraction of tokens to remove)
    pruning_ratio: float = 0.5

    # Minimum frequency threshold for keeping tokens
    min_frequency: int = 100

    # Evaluation corpus for frequency analysis
    eval_corpus_dataset: Optional[str] = None
    eval_corpus_split: str = "validation"
    eval_corpus_samples: int = 10000

    def __post_init__(self):
        """Validate pruning configuration."""
        if not isinstance(self.method, PruningMethod):
            if isinstance(self.method, str):
                self.method = PruningMethod(self.method)
            else:
                raise ValueError(f"Invalid pruning method: {self.method}")

        if self.enabled:
            if not (0 < self.pruning_ratio < 1):
                raise ValueError(f"pruning_ratio must be between 0 and 1, got {self.pruning_ratio}")

            if self.min_frequency < 0:
                raise ValueError("min_frequency must be non-negative")


@dataclass
class TokenizationDatasetConfig:
    """Dataset configuration for tokenization training."""

    name: str
    split: str = "train"
    percent: Optional[float] = None
    max_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Field mapping
    text_column: str = "text"
    field_mappings: Dict[str, str] = field(default_factory=dict)

    # Processing parameters
    dataset_num_proc: Optional[int] = None
    streaming: bool = False

    # Chat formatting
    chat_template: Optional[str] = None
    system_prompt: Optional[str] = None

    config_name: Optional[str] = None
    eval_split: Optional[str] = None

    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.name:
            raise ValueError("Dataset name is required")

        if self.percent is not None and (self.percent <= 0 or self.percent > 100):
            raise ValueError("Dataset percent must be between 0 and 100")

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")


@dataclass
class StagedTrainingConfig:
    """Configuration for multi-stage training."""

    # Whether to use staged training
    enabled: bool = True

    # Number of stages (default: 7-stage as per EEVE paper)
    num_stages: int = 7

    # Stage configurations (can override defaults)
    stage_configs: Dict[int, StageConfig] = field(default_factory=dict)

    # Total training budget
    total_train_tokens: int = 2_000_000_000  # 2B tokens total

    # Stage-specific token budgets (if None, auto-distribute)
    stage_token_budgets: Optional[Dict[int, int]] = None

    def __post_init__(self):
        """Validate and set up default stage configs."""
        if self.num_stages < 1:
            raise ValueError("num_stages must be at least 1")

        if self.total_train_tokens <= 0:
            raise ValueError("total_train_tokens must be positive")

        # Set up default 7-stage configuration if not provided
        if not self.stage_configs and self.enabled and self.num_stages == 7:
            self.stage_configs = self._create_default_7_stage_config()

    def _create_default_7_stage_config(self) -> Dict[int, StageConfig]:
        """Create default 7-stage configuration from EEVE paper."""
        return {
            1: StageConfig(
                train_input_embeddings=True,
                train_new_embeddings_only=True,
                learning_rate=1e-3,
                num_train_tokens=200_000_000,  # 200M tokens
            ),
            2: StageConfig(
                train_output_embeddings=True,
                train_new_embeddings_only=True,
                learning_rate=1e-3,
                num_train_tokens=200_000_000,
            ),
            3: StageConfig(
                train_input_embeddings=True,
                train_output_embeddings=True,
                train_new_embeddings_only=True,
                learning_rate=5e-4,
                num_train_tokens=200_000_000,
            ),
            4: StageConfig(
                train_output_embeddings=True,
                train_new_embeddings_only=False,  # All output embeddings
                learning_rate=5e-4,
                num_train_tokens=200_000_000,
            ),
            5: StageConfig(
                train_input_embeddings=True,
                train_output_embeddings=True,
                train_new_embeddings_only=False,
                learning_rate=2e-4,
                num_train_tokens=400_000_000,
            ),
            6: StageConfig(
                train_transformer=True,
                use_lora=True,
                lora_rank=8,
                lora_alpha=32,
                learning_rate=2e-4,
                num_train_tokens=400_000_000,
            ),
            7: StageConfig(
                train_transformer=True,
                learning_rate=1e-4,
                num_train_tokens=400_000_000,
            ),
        }


@dataclass
class EvaluationConfig:
    """Configuration for tokenization evaluation."""

    # Evaluation metrics to compute
    compute_fertility: bool = True          # Avg tokens per word
    compute_compression_ratio: bool = True  # Bytes per token
    compute_fairness: bool = True           # Cross-lingual fairness
    compute_coverage: bool = True           # Script character coverage
    compute_reachability: bool = True       # Self-tokenization test (STT)

    # Evaluation datasets
    eval_datasets: List[str] = field(default_factory=list)
    eval_languages: List[str] = field(default_factory=list)

    # Sample sizes
    samples_per_dataset: int = 1000

    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.samples_per_dataset <= 0:
            raise ValueError("samples_per_dataset must be positive")


@dataclass
class TokenizationTrainingConfig:
    """Training configuration for tokenization."""

    # Basic training parameters
    per_device_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    epochs: Optional[int] = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer and scheduler
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.1

    # Evaluation and checkpointing
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

    # Seed and misc
    seed: Optional[int] = 42
    data_seed: Optional[int] = 47
    torch_compile: bool = False
    use_cache: bool = False

    # Distributed training config path
    accelerate_config_path: Optional[str] = None

    # Extra parameters - passed to trainer
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


@dataclass
class TokenizationLoggingConfig:
    """Logging configuration for tokenization training."""

    loggers: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    output_dir: str = "./tokenization_output"
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
class UnifiedTokenizationConfig:
    """
    Unified configuration for tokenization training.

    Supports:
    - Vocabulary extension via continued BPE training
    - Multi-stage parameter freezing/unfreezing
    - Subword-based embedding initialization
    - Vocabulary pruning
    - Tokenization evaluation metrics

    Example:
        >>> config = UnifiedTokenizationConfig(
        ...     model=TokenizationModelConfig(
        ...         base_model="meta-llama/Llama-2-7b-hf",
        ...         new_tokens_count=20000,
        ...     ),
        ...     vocab_extension=VocabExtensionConfig(
        ...         target_languages=["ko", "ja", "zh"],
        ...         method=VocabExtensionMethod.CONTINUED_BPE,
        ...     ),
        ...     dataset=TokenizationDatasetConfig(
        ...         name="wikimedia/wikipedia",
        ...         config_name="20231101.ko",
        ...     ),
        ...     staged_training=StagedTrainingConfig(enabled=True),
        ... )
    """

    model: TokenizationModelConfig
    vocab_extension: VocabExtensionConfig
    dataset: TokenizationDatasetConfig

    embedding_init: EmbeddingInitConfig = field(default_factory=EmbeddingInitConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    staged_training: StagedTrainingConfig = field(default_factory=StagedTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    train: TokenizationTrainingConfig = field(default_factory=TokenizationTrainingConfig)
    logging: TokenizationLoggingConfig = field(default_factory=TokenizationLoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Operations to perform (can perform multiple in sequence)
    operations: List[TokenizationOperation] = field(default_factory=lambda: [
        TokenizationOperation.VOCAB_EXTENSION,
        TokenizationOperation.EMBEDDING_INIT,
        TokenizationOperation.STAGED_TRAINING,
        TokenizationOperation.EVALUATION,
    ])

    def __post_init__(self):
        """Validate unified configuration."""
        # Validate operations
        for op in self.operations:
            if not isinstance(op, TokenizationOperation):
                raise ValueError(f"Invalid operation: {op}")

        # Ensure vocab extension is configured if it's in operations
        if TokenizationOperation.VOCAB_EXTENSION in self.operations:
            if not self.vocab_extension.target_languages:
                raise ValueError("target_languages required for vocab extension")

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
TokenizationConfig = UnifiedTokenizationConfig


# Helper functions for creating configs
def create_korean_adaptation_config(
    base_model: str,
    dataset_name: str,
    output_dir: str = "./output/tokenization/korean",
    **kwargs
) -> UnifiedTokenizationConfig:
    """Create configuration for Korean language adaptation (EEVE-style)."""
    return UnifiedTokenizationConfig(
        model=TokenizationModelConfig(
            base_model=base_model,
            new_tokens_count=kwargs.get('new_tokens_count', 20000),
            precision=PrecisionType.BF16,
        ),
        vocab_extension=VocabExtensionConfig(
            target_languages=["ko"],
            method=VocabExtensionMethod.CONTINUED_BPE,
            bpe_corpus_dataset=dataset_name,
        ),
        dataset=TokenizationDatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples'),
        ),
        staged_training=StagedTrainingConfig(
            enabled=True,
            num_stages=7,
            total_train_tokens=2_000_000_000,
        ),
        logging=TokenizationLoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name', 'korean_adaptation'),
        ),
    )


def create_chinese_adaptation_config(
    base_model: str,
    dataset_name: str,
    output_dir: str = "./output/tokenization/chinese",
    **kwargs
) -> UnifiedTokenizationConfig:
    """Create configuration for Chinese language adaptation."""
    return UnifiedTokenizationConfig(
        model=TokenizationModelConfig(
            base_model=base_model,
            new_tokens_count=kwargs.get('new_tokens_count', 20000),
            precision=PrecisionType.BF16,
        ),
        vocab_extension=VocabExtensionConfig(
            target_languages=["zh"],
            method=VocabExtensionMethod.CONTINUED_BPE,
            bpe_corpus_dataset=dataset_name,
        ),
        dataset=TokenizationDatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples'),
        ),
        staged_training=StagedTrainingConfig(
            enabled=True,
            num_stages=7,
            total_train_tokens=2_000_000_000,
        ),
        logging=TokenizationLoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name', 'chinese_adaptation'),
        ),
    )


def create_multilingual_adaptation_config(
    base_model: str,
    target_languages: List[str],
    dataset_name: str,
    output_dir: str = "./output/tokenization/multilingual",
    **kwargs
) -> UnifiedTokenizationConfig:
    """Create configuration for multilingual adaptation (70 languages like in paper)."""
    return UnifiedTokenizationConfig(
        model=TokenizationModelConfig(
            base_model=base_model,
            new_tokens_count=kwargs.get('new_tokens_count', 40000),
            precision=PrecisionType.BF16,
        ),
        vocab_extension=VocabExtensionConfig(
            target_languages=target_languages,
            method=VocabExtensionMethod.CONTINUED_BPE,
            bpe_corpus_dataset=dataset_name,
        ),
        dataset=TokenizationDatasetConfig(
            name=dataset_name,
            split=kwargs.get('split', 'train'),
            max_samples=kwargs.get('max_samples'),
        ),
        staged_training=StagedTrainingConfig(
            enabled=True,
            num_stages=7,
            total_train_tokens=kwargs.get('total_train_tokens', 5_000_000_000),  # 5B for multilingual
        ),
        pruning=PruningConfig(
            enabled=kwargs.get('enable_pruning', True),
            method=PruningMethod.LEAF_FREQUENCY,
            pruning_ratio=kwargs.get('pruning_ratio', 0.5),
        ),
        logging=TokenizationLoggingConfig(
            output_dir=output_dir,
            run_name=kwargs.get('run_name', 'multilingual_adaptation'),
        ),
    )


__all__ = [
    # Main config
    'UnifiedTokenizationConfig',
    'TokenizationConfig',

    # Sub-configs
    'TokenizationModelConfig',
    'VocabExtensionConfig',
    'EmbeddingInitConfig',
    'PruningConfig',
    'StagedTrainingConfig',
    'StageConfig',
    'TokenizationDatasetConfig',
    'TokenizationTrainingConfig',
    'TokenizationLoggingConfig',
    'EvaluationConfig',
    'DistributedConfig',

    # Enums
    'PrecisionType',
    'BackendType',
    'TokenizationOperation',
    'VocabExtensionMethod',
    'EmbeddingInitMethod',
    'PruningMethod',

    # Factory functions
    'create_korean_adaptation_config',
    'create_chinese_adaptation_config',
    'create_multilingual_adaptation_config',
]
