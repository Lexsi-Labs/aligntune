"""
Configuration system for Tokenization Training.

Supports:
- Continued BPE training for vocabulary extension
- Naive extension (train separate tokenizer)
- Vocabulary pruning (leaf-based, frequency-based)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class VocabExtensionMethod(Enum):
    """Methods for vocabulary extension."""
    CONTINUED_BPE = "continued_bpe"      # Continue BPE merge training (recommended)
    NAIVE_EXTENSION = "naive_extension"  # Train separate tokenizer + diff


class PruningMethod(Enum):
    """Methods for vocabulary pruning."""
    LEAF_FREQUENCY = "leaf_frequency"    # Leaf-based + frequency (structure-aware, recommended)
    FREQUENCY = "frequency"              # Frequency-based only
    LAST_N = "last_n"                    # Remove last N tokens


@dataclass
class TokenizationModelConfig:
    """Model configuration for tokenization training."""

    # Base model to adapt
    base_model: str

    # Tokenizer configuration
    base_tokenizer: Optional[str] = None  # If None, use base_model tokenizer
    new_tokens_count: int = 20000  # Number of new tokens to add

    # Model loading
    precision: str = "bf16"  # bf16, fp16, fp32, auto
    device_map: str = "auto"
    trust_remote_code: bool = False
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model configuration."""
        if not self.base_model:
            raise ValueError("base_model is required")
        if self.new_tokens_count <= 0:
            raise ValueError("new_tokens_count must be positive")


@dataclass
class VocabExtensionConfig:
    """Configuration for vocabulary extension."""

    # Target languages for extension (e.g., ["hi", "zh", "ar"])
    target_languages: List[str] = field(default_factory=list)

    # Extension method
    method: VocabExtensionMethod = VocabExtensionMethod.CONTINUED_BPE

    def __post_init__(self):
        """Validate vocab extension configuration."""
        if not isinstance(self.method, VocabExtensionMethod):
            if isinstance(self.method, str):
                self.method = VocabExtensionMethod(self.method)


@dataclass
class PruningConfig:
    """Configuration for vocabulary pruning."""

    # Whether to enable pruning
    enabled: bool = False

    # Pruning parameters
    pruning_ratio: float = 0.1  # Prune 10% of vocabulary
    method: PruningMethod = PruningMethod.LEAF_FREQUENCY

    # Evaluation corpus for pruning (if different from extension corpus)
    eval_corpus_dataset: Optional[str] = None
    eval_corpus_split: str = "train"
    eval_corpus_samples: int = 100000

    def __post_init__(self):
        """Validate pruning configuration."""
        if not isinstance(self.method, PruningMethod):
            if isinstance(self.method, str):
                self.method = PruningMethod(self.method)

        if self.enabled and not (0.0 < self.pruning_ratio < 1.0):
            raise ValueError(f"pruning_ratio must be between 0 and 1, got {self.pruning_ratio}")


@dataclass
class TokenizationDatasetConfig:
    """Configuration for tokenization dataset."""

    # Dataset configuration
    name: str = "wikimedia/wikipedia"
    config_name: Optional[str] = None  # e.g., "20231101.hi" for Hindi Wikipedia
    split: str = "train"
    text_column: str = "text"

    # Sampling
    max_samples: Optional[int] = None
    streaming: bool = False

    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.name:
            raise ValueError("dataset name is required")


@dataclass
class TokenizationLoggingConfig:
    """Configuration for logging and output."""

    output_dir: str = "./tokenization_output"
    save_intermediate: bool = False
    log_level: str = "info"
    hub_model_id: Optional[str] = None  # Push to HF Hub if provided (e.g., "username/model-name")


@dataclass
class UnifiedTokenizationConfig:
    """
    Main configuration for tokenization training.

    Examples:
        >>> # Continued BPE for Hindi
        >>> config = UnifiedTokenizationConfig(
        ...     model=TokenizationModelConfig(
        ...         base_model="meta-llama/Llama-2-7b-hf",
        ...         new_tokens_count=20000
        ...     ),
        ...     vocab_extension=VocabExtensionConfig(
        ...         target_languages=["hi"],
        ...         method=VocabExtensionMethod.CONTINUED_BPE
        ...     ),
        ...     dataset=TokenizationDatasetConfig(
        ...         name="wikimedia/wikipedia",
        ...         config_name="20231101.hi",
        ...         max_samples=100000
        ...     ),
        ...     logging=TokenizationLoggingConfig(
        ...         output_dir="./hindi-llama"
        ...     )
        ... )

        >>> # With pruning enabled
        >>> config = UnifiedTokenizationConfig(
        ...     model=TokenizationModelConfig(base_model="gpt2", new_tokens_count=10000),
        ...     vocab_extension=VocabExtensionConfig(target_languages=["zh"]),
        ...     pruning=PruningConfig(enabled=True, pruning_ratio=0.1),
        ...     dataset=TokenizationDatasetConfig(name="wikimedia/wikipedia", config_name="20231101.zh"),
        ...     logging=TokenizationLoggingConfig(output_dir="./chinese-gpt2")
        ... )
    """

    model: TokenizationModelConfig
    vocab_extension: VocabExtensionConfig
    dataset: TokenizationDatasetConfig
    logging: TokenizationLoggingConfig
    pruning: PruningConfig = field(default_factory=PruningConfig)

    def __post_init__(self):
        """Validate complete configuration."""
        if not isinstance(self.model, TokenizationModelConfig):
            raise TypeError("model must be TokenizationModelConfig")
        if not isinstance(self.vocab_extension, VocabExtensionConfig):
            raise TypeError("vocab_extension must be VocabExtensionConfig")
        if not isinstance(self.dataset, TokenizationDatasetConfig):
            raise TypeError("dataset must be TokenizationDatasetConfig")
        if not isinstance(self.logging, TokenizationLoggingConfig):
            raise TypeError("logging must be TokenizationLoggingConfig")
        if not isinstance(self.pruning, PruningConfig):
            raise TypeError("pruning must be PruningConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": {
                "base_model": self.model.base_model,
                "base_tokenizer": self.model.base_tokenizer,
                "new_tokens_count": self.model.new_tokens_count,
                "precision": self.model.precision,
                "device_map": self.model.device_map,
                "trust_remote_code": self.model.trust_remote_code,
            },
            "vocab_extension": {
                "target_languages": self.vocab_extension.target_languages,
                "method": self.vocab_extension.method.value,
            },
            "pruning": {
                "enabled": self.pruning.enabled,
                "pruning_ratio": self.pruning.pruning_ratio,
                "method": self.pruning.method.value,
                "eval_corpus_dataset": self.pruning.eval_corpus_dataset,
                "eval_corpus_split": self.pruning.eval_corpus_split,
                "eval_corpus_samples": self.pruning.eval_corpus_samples,
            },
            "dataset": {
                "name": self.dataset.name,
                "config_name": self.dataset.config_name,
                "split": self.dataset.split,
                "text_column": self.dataset.text_column,
                "max_samples": self.dataset.max_samples,
                "streaming": self.dataset.streaming,
            },
            "logging": {
                "output_dir": self.logging.output_dir,
                "save_intermediate": self.logging.save_intermediate,
                "log_level": self.logging.log_level,
            },
        }


# Alias for backward compatibility
TokenizationConfig = UnifiedTokenizationConfig
