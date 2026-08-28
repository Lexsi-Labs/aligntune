"""
Evolution Strategies (ES) Configuration.

Only contains parameters actually used by ES trainer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class ESPeftConfig:
    """PEFT/LoRA configuration for ES trainer."""

    rank: int = 8
    """LoRA rank."""

    alpha: int = 16
    """LoRA alpha parameter."""

    dropout: float = 0.05
    """LoRA dropout rate."""

    target_modules: Optional[List[str]] = None
    """Target modules for LoRA."""

    peft_type: str = "lora"
    """PEFT type."""

    variant: str = "standard"
    """LoRA variant: standard, moa, etc."""

    use_dora: bool = False
    """Enable DoRA (not available in this build; falls back to standard LoRA)."""

    use_rslora: bool = False
    """Enable rsLoRA."""

    init_lora_weights: Union[str, bool] = True
    """Initialize LoRA weights."""

    bias: str = "none"
    """Bias type."""


@dataclass
class ESConfig:
    """ES training hyperparameters - only what trainer uses."""

    # ============================================
    # POPULATION PARAMETERS
    # ============================================
    population_size: int = 64
    """Number of perturbed variants per iteration."""

    sigma: float = 0.5
    """Mutation standard deviation for parameter perturbation."""

    learning_rate: float = 0.01
    """ES learning rate for updating base parameters."""

    # ============================================
    # GENERATION PARAMETERS
    # ============================================
    prompt_batch_size: int = 4
    """Number of prompts to evaluate per iteration."""

    max_new_tokens: int = 256
    """Maximum tokens to generate per completion."""

    temperature: float = 0.7
    """Sampling temperature for generation."""

    top_p: float = 0.95
    """Nucleus sampling top-p parameter."""

    top_k: int = -1
    """Top-k sampling parameter. -1 = disabled."""

    num_return_sequences: int = 1
    """Number of sequences to generate per prompt."""

    # ============================================
    # TRAINING LOOP PARAMETERS
    # ============================================
    num_iterations: int = 1000
    """Number of ES iterations to run."""

    save_freq: int = 100
    """Save checkpoint every N iterations."""

    log_freq: int = 10
    """Log metrics every N iterations."""

    # ============================================
    # HARDWARE & PERFORMANCE
    # ============================================
    tensor_parallel_size: int = 1
    """Tensor parallelism for vLLM rollout."""

    dtype: str = "auto"
    """Model dtype for vLLM: auto, bf16, fp16, fp32."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ============================================
    # REWARD CONFIGURATION
    # ============================================
    reward_type: str = "math_correctness"
    """Default reward function type. Overridden by config.rewards if provided."""

    # ============================================
    # ADVANCED OPTIONS
    # ============================================
    use_unsloth: bool = False
    """Enable Unsloth acceleration."""

    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint to resume from."""

    verbose: bool = True
    """Enable verbose logging."""


@dataclass
class ESModelConfig:
    """Model configuration for ES trainer."""

    name_or_path: str
    """Model name from HuggingFace or local path."""

    dtype: str = "auto"
    """Model dtype: auto, bf16, fp16, fp32."""

    max_seq_length: int = 2048
    """Maximum sequence length."""

    peft: ESPeftConfig = field(default_factory=ESPeftConfig)
    """PEFT/LoRA configuration."""

    quantization: Dict[str, Any] = field(default_factory=dict)
    """Quantization config."""

    use_peft: bool = True
    """Whether to use PEFT/LoRA."""

    device_map: str = "auto"
    """Device map for model loading."""

    # tokenizer_name_or_path: Optional[str] = None
    # """Tokenizer path if different from model path. TODO: Add in next sprint"""

    # embedding_init_method: str = "random"
    # """Embedding initialization method for extended tokenizers. TODO: Add in next sprint"""


@dataclass
class ESDatasetConfig:
    """Dataset configuration for ES trainer."""

    name: str
    """Dataset name from HuggingFace or local path."""

    split: Optional[str] = None
    """Dataset split to use."""

    subset: Optional[str] = None
    """Dataset subset/config to use (e.g., 'main', 'socratic' for GSM8K)."""

    column_mapping: Dict[str, str] = field(default_factory=dict)
    """Column name mappings."""

    system_prompt: Optional[str] = None
    """System prompt for chat templates."""

    enable_thinking: bool = False
    """Whether to request thinking mode from supported chat templates."""

    task_type: Optional[str] = None
    """Optional override for the trainer's DataManager task type."""

    format_type: Optional[str] = None
    """Expected CuratorKIT source format."""

    keep_columns: Optional[bool] = None
    """Optional override for the trainer's source-column preservation default."""

    max_samples: Optional[int] = None
    """Maximum examples to keep from each source split."""

    processing_fn: Optional[Any] = None
    """Custom processing function."""

    processing_batched: bool = False
    """Whether the custom processing function expects batches."""

    val_split_ratio: Optional[float] = None
    test_split_ratio: Optional[float] = None
    split_seed: int = 42

    curator_schema_gate: bool = True
    curator_clean: bool = False
    curator_dedup: str = "none"
    curator_use_tiktoken: bool = False
    curator_max_tokens: int = 1_000_000

    loader_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for dataset loading."""


@dataclass
class UnifiedESConfig:
    """Unified configuration for ES training."""

    es: ESConfig
    """ES-specific hyperparameters."""

    model: ESModelConfig
    """Model configuration."""

    dataset: ESDatasetConfig
    """Dataset configuration (singular)."""

    rewards: List[Dict[str, Any]] = field(default_factory=list)
    """List of reward function configurations."""

    output_dir: str = "./es_output"
    """Output directory for checkpoints and logs."""

    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.es, ESConfig):
            raise ValueError("es must be an ESConfig instance")
        if not isinstance(self.model, ESModelConfig):
            raise ValueError("model must be an ESModelConfig instance")
        if not isinstance(self.dataset, ESDatasetConfig):
            raise ValueError("dataset must be an ESDatasetConfig instance")
