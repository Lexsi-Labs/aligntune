"""
Unified configuration system for RLHF training.

This module defines the complete configuration schema for the unified training system,
with no placeholder defaults - all critical parameters must be explicitly provided.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class AlgorithmType(Enum):
    """Supported RLHF algorithms."""
    PPO = "ppo"
    DPO = "dpo"
    GRPO = "grpo"
    GSPO = "gspo"
    COUNTERFACT_GRPO = "counterfact_grpo"
    GBMPO = "gbmpo"
    DRGRPO = "drgrpo"
    DAPO = "dapo"
    PACE = "pace"  # Baseline-Optimized Learning Technique
    NMGRPO="nmgrpo"
    METAES="metaes"



# At the top, add AUTO to PrecisionType enum
class PrecisionType(Enum):
    """Model precision types."""
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    AUTO = "auto"  # ADD THIS


class BackendType(Enum):
    """Distributed training backends."""
    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"


@dataclass
class ModelConfig:
    """Model configuration with no default model names."""
    name_or_path: str
    backend: str = "trl"  # Backend: "trl" or "unsloth"
    sft_path: Optional[str] = None
    reward_path: Optional[str] = None
    reward_model_name: Optional[str] = None  # NEW: Separate reward model
    reward_model_source: Optional['RewardModelSourceConfig'] = None  # NEW: Reward model source configuration
    precision: PrecisionType = PrecisionType.AUTO
    quantization: Dict[str, Any] = field(default_factory=dict)
    attn_implementation: str = "auto"
    gradient_checkpointing: bool = False
    max_memory: Optional[Dict[str, str]] = None
    device_map: Optional[Union[str, Dict[str, int]]] = None  # Added device_map
    use_unsloth: bool = False  # Enable Unsloth acceleration
    max_seq_length: int = 2048  # For Unsloth
    reward_value_model: str = None  # Llama model for reward/value (loaded pre-Unsloth to avoid patches)
    reward_value_loading_type: Optional[str] = None  # 'unsloth' or 'standard'
    reward_model_quantization: Dict[str, Any] = field(default_factory=dict)
    value_model_quantization: Dict[str, Any] = field(default_factory=dict)
    use_peft: bool = True  # Enable PEFT (LoRA) by default
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    trust_remote_code: bool = True  # Trust remote code for model loading


    # New parameters from model initialization
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    ref_model_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    force_use_ref_model: bool = False
    disable_dropout: bool = True
    use_logits_to_keep: bool = False
    reward_device: str = "auto"


    
    
    def __post_init__(self):
        """Validate model configuration."""
        # Validate backend
        if self.backend not in ["trl", "unsloth"]:
            raise ValueError(f"backend must be 'trl' or 'unsloth', got '{self.backend}'")
        
        # Validate precision
        if not isinstance(self.precision, PrecisionType):
            if isinstance(self.precision, str):
                self.precision = PrecisionType(self.precision)
            else:
                raise ValueError(f"Invalid precision type: {self.precision}")
        
        # Set default device_map if not specified
        if self.device_map is None:
            self.device_map = "auto"

        # Normalize reward_device
        if not self.reward_device:
            self.reward_device = "auto"
        else:
            normalized = self.reward_device.lower()
            if normalized not in {"auto", "cpu", "cuda"}:
                raise ValueError(
                    f"reward_device must be 'auto', 'cpu', or 'cuda', got '{self.reward_device}'"
                )
            self.reward_device = normalized

@dataclass
class DatasetConfig:
    """Dataset configuration with flexible field mapping support."""
    name: str
    split: str = "train"
    percent: Optional[float] = None
    max_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    # Enhanced field mapping system
    field_mappings: Dict[str, str] = field(default_factory=dict)
    format_type: Optional[str] = None  # "alpaca", "dolly", "ultrachat", "custom"
    auto_detect_fields: bool = True
    # Legacy support
    column_mapping: Dict[str, str] = field(default_factory=dict)
    task_type: str = "conversation"
    weight: float = 1.0
    chat_template: Optional[str] = None
    
    system_prompt: Optional[str] = None  


    # New parameters from dataset processing
    dataset_num_proc: Optional[int] = None
    pad_token: Optional[str] = None
    label_pad_token_id: int = -100
    truncation_mode: str = 'keep_end'
    padding_free: bool = False
    precompute_ref_log_probs: bool = False
    precompute_ref_batch_size: Optional[int] = None
    tools: Optional[Any] = None  



    # ADD THESE NEW PARAMETERS
    preserve_columns: Optional[List[str]] = None  # NEW: Columns to preserve
    processing_fn: Optional[Any] = None  # NEW: Custom processing function
    processing_batched: bool = False  # NEW: Whether processing is batched
    processing_fn_kwargs: Dict[str, Any] = field(default_factory=dict)  # NEW: Args for processing_fn
    
    config_name: Optional[str] = None 
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.name:
            raise ValueError("Dataset name is required and cannot be empty")
        
        if self.percent is not None and (self.percent <= 0 or self.percent > 100):
            raise ValueError("Dataset percent must be between 0 and 100")
        
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        
        if self.weight <= 0:
            raise ValueError("Dataset weight must be positive")


@dataclass
class RewardConfig:
    """Reward function configuration."""
    type: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    shield: bool = False
    clip: Optional[float] = None
    normalize: bool = False


@dataclass
class RewardModelTrainingConfig:
    """Configuration for reward model training - ALL fields validated."""
    base_model_name: str  # NO default - must be explicit
    training_texts: List[str]  # NO default - must provide data
    reward_functions: List[str]  # NO default - must specify
    output_dir: str  # NO default - must specify where to save
    
    # Optional but validated if provided
    reference_texts: Optional[List[str]] = None
    reward_weights: Optional[List[float]] = None
    
    # Training params with JUSTIFIED defaults
    num_epochs: int = 3  # Standard for reward model training
    learning_rate: float = 1e-5  # Standard LM fine-tuning rate
    batch_size: int = 8  # Balanced for memory/speed
    gradient_accumulation_steps: int = 4  # For effective batch size 32
    max_length: int = 512  # Standard transformer length
    
    
    def __post_init__(self):
        """Validate ALL fields with strict validation."""
        # Validate required fields
        if not self.base_model_name:
            raise ValueError("base_model_name cannot be empty")
        if not self.training_texts:
            raise ValueError("training_texts cannot be empty")
        if not isinstance(self.training_texts, list):
            raise TypeError(f"training_texts must be list, got {type(self.training_texts)}")
        if len(self.training_texts) < 10:
            raise ValueError(f"Need at least 10 training texts, got {len(self.training_texts)}")
        if not self.reward_functions:
            raise ValueError("reward_functions cannot be empty")
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")
        
        # Validate optional fields if provided
        if self.reference_texts and len(self.reference_texts) != len(self.training_texts):
            raise ValueError("reference_texts length must match training_texts")
        if self.reward_weights:
            if len(self.reward_weights) != len(self.reward_functions):
                raise ValueError("reward_weights length must match reward_functions")
            if not all(w > 0 for w in self.reward_weights):
                raise ValueError("All reward_weights must be positive")
        
        # Validate training params
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


@dataclass
class RewardModelSourceConfig:
    """Reward model source - EXACTLY ONE must be specified."""
    source_type: str  # Must be set explicitly
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    training_config: Optional[RewardModelTrainingConfig] = None
    fine_tune_with_rewards: bool = False  # NEW: Enable hybrid mode to fine-tune pretrained with reward functions
    
    def __post_init__(self):
        """Validate source configuration with strict mutual exclusivity."""
        # Validate source_type
        valid_types = ["pretrained_hf", "pretrained_local", "custom_trained"]
        if self.source_type not in valid_types:
            raise ValueError(
                f"source_type must be one of {valid_types}, got '{self.source_type}'"
            )
        
        # Validate exactly one source
        sources = [self.model_name, self.model_path, self.training_config]
        source_count = sum(x is not None for x in sources)
        
        if source_count != 1:
            raise ValueError(
                f"Exactly ONE source must be specified. "
                f"Found {source_count}: "
                f"model_name={self.model_name}, "
                f"model_path={self.model_path}, "
                f"training_config={self.training_config is not None}"
            )
        
        # Validate source matches type
        if self.source_type == "pretrained_hf" and not self.model_name:
            raise ValueError("source_type='pretrained_hf' requires model_name")
        if self.source_type == "pretrained_local" and not self.model_path:
            raise ValueError("source_type='pretrained_local' requires model_path")
        if self.source_type == "custom_trained" and not self.training_config:
            raise ValueError("source_type='custom_trained' requires training_config")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result
    



@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training parameters
    per_device_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    epochs: Optional[int] = None
    eval_interval: int = 100
    save_interval: int = 500
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01 
    use_cache: bool = False,
    
    # Optimizer and scheduler
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0  # If 0, calculated from warmup_ratio or defaults to 5% of max_steps
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    
    # PPO/RL-specific parameters
    rollout_batch_size: int = 1
    kl_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    num_ppo_epochs: Optional[int] = None
    temperature: float = 0.6
    whiten_rewards: bool = False
    kl_estimator: str = 'k1'
    vf_coef: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    
    # Generation parameters
    response_length: int = 128
    stop_token: str = 'eos'
    missing_eos_penalty: float = 1.0
    ds3_gather_for_generation: bool = True
    generation_kwargs: dict = None
    
    # Length parameters
    max_length: int = 1024
    max_prompt_length: int = 512
    max_target_length: Optional[int] = None
    max_completion_length: int = 256

    # Generation sampling parameters
    top_p: float = 0.95
    
    # Processing parameters
    padding_free: bool = False
    truncation_mode: str = 'keep_end'
    
    # DPO-specific parameters
    beta: float = 0.1
    loss_type: str = None
    loss_weights: Optional[dict] = None
    f_divergence_type: str = 'reverse_kl'
    f_alpha_divergence_coef: float = 1.0
    reference_free: bool = False
    label_smoothing: float = 0.0
    use_weighting: bool = False
    
    # Algorithm-specific parameters
    rpo_alpha: Optional[float] = None
    ld_alpha: Optional[float] = None
    discopop_tau: float = 0.05
    
    # Reference model parameters
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.6
    ref_model_sync_steps: int = 512
    
    # GRPO-specific parameters
    grpo_alpha: float = 0.1
    grpo_beta: float = 0.1
    
    # GSPO-specific parameters
    gspo_gamma: float = 0.1
    gspo_delta: float = 0.1
    
    
    # Evaluation parameters
    eval_steps: Optional[int] = 100
    eval_strategy: str = "no"
    # DPO-specific evaluation options
    dpo_eval_enabled: bool = False
    dpo_eval_max_samples: Optional[int] = None
    dpo_zero_shot_max_samples: int = 50
    dpo_few_shot_max_samples: int = 30
    dpo_few_shot_examples_text: Optional[str] = None
    
    # Checkpointing parameters
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: Optional[int] = None
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: bool = False
    
    # Logging parameters
    logging_steps: int = 10
    logging_strategy: str = "steps"
    
    # Generation parameters (GRPO-specific)
    num_generations: Optional[int] = None
    mask_truncated_completions: bool = True
    scale_rewards: str = "group"
    reward_weights: Optional[List[float]] = None
    enable_thinking: bool = False  # Qwen3 thinking mode
    fast_inference: bool = False  # Unsloth vLLM fast inference (2-3x faster generation)
    vllm_gpu_memory_utilization: float = 0.7  # vLLM GPU memory (0.95 for max speed with spare VRAM)

    # Seed parameters
    seed: Optional[int] = 42
    data_seed: Optional[int] = 47  # Match training_script.py for fair comparison
    

    # Meta-ES specific
    meta_iterations: int = 15
    patience: int = 5
    min_delta: float = 0.001
    init_scale: float = 0.01
    N: int = 10  # Population size (must be even)
    T: int = 100  # Training steps per evaluation
    sigma: float = 0.01  # ES noise std
    sigma_decay: float = 0.99  # ES sigma decay per iteration
    alpha: float = 0.01  # ES learning rate
    mirror_coefficient: float = 0.0001
    debug_mode: bool = False
    eval_timeout: int = 5
    eval_max_tokens: int = 512
    eval_k: int = 1
    eval_temperature: float = 0.8
    num_workers: int = 1
    no_wandb: bool = False
    wandb_project: str = "neural-mirror-es"
    resume: Optional[str] = None
    use_rewards_directly: Optional[List[callable]] = None
    
    
    # ============================================================================
    # NEW: Counterfactual GRPO-specific parameters
    # ============================================================================
    boost_factor: float = 2.0
    """Importance weight boost factor for counterfactual tokens.
    Higher values give more weight to important tokens identified by the counterfactual method.
    Typical range: 1.5-3.0
    """
    
    min_weight: float = 0.5
    """Minimum importance weight to prevent underflow.
    Ensures no token weight drops below this value.
    Typical range: 0.1-0.5
    """
    
    max_spans: int = 10
    """Maximum number of counterfactual spans to detect per sequence.
    Limits the number of important token regions identified.
    Typical range: 5-20
    """

    answer_weight: float = 1.5
    """Weight multiplier for answer sections in reasoning tasks.
    Emphasizes the final answer/conclusion in math/reasoning problems.
    Typical range: 1.0-2.5
    """

    weighting_mode: Optional[str] = None
    """Unified weighting mode selector. Options: counterfactual, random, inverted, vanilla.
    If set, overrides random_importance and invert_importance flags.
    """

    method_name: str = "counterfactual"
    """Weighting method to use.
    Options:
    - "counterfactual": Standard counterfactual importance weighting
    - "uniform": Equal weights for all tokens (baseline)
    - "random": Random importance weights (ablation)
    - "inverse": Inverted importance weights (ablation)
    """
    
    random_importance: bool = False
    """Use random importance weights for ablation studies.
    When True, assigns random weights instead of computed counterfactual weights.
    Used to validate that counterfactual weighting improves over random.
    """
    
    invert_importance: bool = False
    """Invert importance weights for ablation studies.
    When True, flips the importance weights (important becomes unimportant).
    Used to validate that weighting direction matters.
    """
    
    enable_gradient_conservation: bool = True
    """Enable gradient conservation to maintain gradient flow.
    When True, normalizes weights to preserve total gradient magnitude.
    Recommended: True (prevents gradient vanishing/explosion)
    """
    
    weight_debug: bool = False
    """Enable detailed logging of importance weights for debugging.
    When True, logs weight statistics, span detection, and per-token weights.
    Use for development/debugging only (generates verbose logs).
    """
    # ============================================================================
    # NEW: GBMPO-specific parameters (Generalized Bregman Mirror Descent)
    # ============================================================================
    gbmpo_l2_coefficient: float = 0.0001
    """L2 regularization coefficient for GBMPO variants.
    Controls the strength of L2 regularization in log-space (L2/L2KL) or 
    probability-space (ProbL2/ProbL2KL) variants.
    Typical range: 0.00001-0.001
    - Math tasks (GSM8K): 0.0001
    - Code tasks (MBPP): 0.0001-0.0005
    """
    
    gbmpo_divergence_type: Optional[str] = None
    """Type of divergence for GBMPO algorithms.
    Options:
    - "l2": L2 norm regularization in log-space
    - "l2kl": Dual L2 + KL divergence
    - "prob_l2": L2 norm in probability space
    - "prob_l2kl": Dual probability-space L2 + KL
    Required when using GBMPO algorithms.
    """
    
    gbmpo_epsilon: float = 0.2
    """PPO-style clipping parameter for GBMPO.
    Used for advantage clipping in policy updates.
    Typical range: 0.1-0.3
    """
    
    
    # Performance optimization
    use_liger_kernel: bool = False
    use_liger_loss: Optional[bool] = None

    # BOLT-specific parameters (Baseline-Optimized Learning Technique)
    curriculum_enabled: bool = False  # Enable uncertainty-based sampling
    curriculum_epsilon: float = 0.05  # Floor for sampling weights
    curriculum_update_freq: int = 10  # Steps between weight updates
    baseline_enabled: bool = False  # Enable persistent baseline tracking
    baseline_rho_min: float = 0.875  # Min forgetting factor (fast adaptation)
    baseline_rho_max: float = 0.96  # Max forgetting factor (slow adaptation)
    baseline_D_half: float = 0.5  # KL half-life for adaptive forgetting
    baseline_warm_start: Optional[str] = None  # Path to JSON/PKL for warm-start
    use_baseline_advantages: bool = False  # Use A = r - v̂(x) vs group mean


    group_by_length: bool = True 

    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate training configuration."""
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        # if self.max_steps is not None and self.max_steps <= 0:
        #     raise ValueError("max_steps must be positive")
        
        if self.epochs is not None and self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.max_steps is None and self.epochs is None:
            raise ValueError("Either max_steps or epochs must be specified")
        
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        
        if self.rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")
        
        if self.cliprange <= 0:
            raise ValueError("cliprange must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")

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
class SampleLoggingConfig:
    """Configuration for qualitative sample logging."""
    enabled: bool = False
    prompts: Optional[List[str]] = None
    interval_steps: Optional[int] = None
    percent_of_max_steps: Optional[float] = None
    max_new_tokens: int = 80
    temperature: float = 0.6
    top_p: float = 0.9
    num_samples: int = 3
    
    def __post_init__(self):
        if self.interval_steps is not None and self.interval_steps <= 0:
            raise ValueError("sample_logging.interval_steps must be positive when set")
        if self.percent_of_max_steps is not None:
            if self.percent_of_max_steps <= 0 or self.percent_of_max_steps > 1:
                raise ValueError("sample_logging.percent_of_max_steps must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("sample_logging.max_new_tokens must be positive")
        if not (0 < self.temperature):
            raise ValueError("sample_logging.temperature must be positive")
        if not (0 < self.top_p <= 1):
            raise ValueError("sample_logging.top_p must be in (0, 1]")
        if self.num_samples <= 0:
            raise ValueError("sample_logging.num_samples must be positive")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    loggers: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    output_dir: str = "./output"
    log_level: str = "INFO"
    sample_logging: SampleLoggingConfig = field(default_factory=SampleLoggingConfig)
    report_to: str = "none"
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_loggers = {"tensorboard", "wandb"}
        for logger in self.loggers:
            if logger not in valid_loggers:
                raise ValueError(f"Invalid logger: {logger}. Must be one of {valid_loggers}")
        
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}. Must be one of {valid_levels}")


# @dataclass
# class RewardModelTrainingConfig:
#     """Configuration for reward model training."""
#     enabled: bool = False
#     base_model: Optional[str] = None  # If None, use main model
#     reward_functions: List[Union[dict, str]] = field(default_factory=list)
#     training_texts: Optional[List[str]] = None
#     dataset_name: Optional[str] = None
#     dataset_path: Optional[str] = None
#     dataset_split: str = "train"
#     text_column: str = "text"
#     output_dir: str = "./output/reward_model"
#     num_epochs: int = 3
#     batch_size: int = 4
#     learning_rate: float = 1e-5
#     save_steps: int = 100
#     logging_steps: int = 10
#     max_samples: Optional[int] = None


@dataclass
class UnifiedConfig:
    """Unified configuration for RLHF training with no placeholder defaults."""
    algo: AlgorithmType
    model: ModelConfig
    datasets: List[DatasetConfig]
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[RewardConfig] = field(default_factory=list)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    chat_template: Optional[str] = None
    caching: Dict[str, Any] = field(default_factory=dict)
    reward_training: Optional[RewardModelTrainingConfig] = None
    
    def __post_init__(self):
        """Validate unified configuration."""
        if not isinstance(self.algo, AlgorithmType):
            if isinstance(self.algo, str):
                self.algo = AlgorithmType(self.algo)
            else:
                raise ValueError(f"Invalid algorithm type: {self.algo}")
        
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def _serialize(value):
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "to_dict"):
                return value.to_dict()
            if isinstance(value, list):
                return [_serialize(item) for item in value]
            if isinstance(value, dict):
                return {k: _serialize(v) for k, v in value.items()}
            if hasattr(value, "__dict__"):
                return {k: _serialize(v) for k, v in value.__dict__.items()}
            return value
        
        return {field_name: _serialize(field_value) for field_name, field_value in self.__dict__.items()}
