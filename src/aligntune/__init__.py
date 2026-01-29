"""
AlignTune: A comprehensive fine-tuning library supporting both SFT and RL training methods.
"""

import logging
import os
import sys

# Import colored logging utilities
try:
    from .utils.colored_logging import (
        init_aligntune_logging,
        print_aligntune_banner,
        print_section_banner,
        print_subsection,
        aligntune_info,
        aligntune_warning,
        aligntune_error,
        aligntune_success,
        aligntune_step,
        setup_colored_logging,
    )
    COLORED_LOGGING_AVAILABLE = True
except ImportError:
    COLORED_LOGGING_AVAILABLE = False
    # Fallback to basic logging
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("aligntune")
    logger.setLevel(logging.WARNING)

# Initialize colored logging if available
if COLORED_LOGGING_AVAILABLE:
    logger = setup_colored_logging("aligntune", logging.WARNING)
else:
    # Configure basic logging as fallback
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("aligntune")
    logger.setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# VERSION & METADATA
# -----------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound
    __version__ = _pkg_version("aligntune")
except Exception:  # Fallback during editable installs before metadata exists
    __version__ = "0.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# -----------------------------------------------------------------------------
# CORE IMPORTS
# -----------------------------------------------------------------------------
# Import all components from the centralized import management system
from ._imports import (
    # Core availability flags
    UNSLOTH_AVAILABLE,
    UNSLOTH_ERROR_INFO,
    TRL_AVAILABLE,
    UNIFIED_RL_AVAILABLE,
    UNIFIED_SFT_AVAILABLE,
    EVAL_AVAILABLE,
    REWARDS_AVAILABLE,
    BACKEND_FACTORY_AVAILABLE,
    UNIFIED_AVAILABLE,
    CLI_AVAILABLE,
    
    # Unified system components
    UnifiedConfig,
    AlgorithmType,
    PrecisionType,
    BackendType,
    UnifiedModelConfig,
    UnifiedDatasetConfig,
    RewardConfig,
    UnifiedTrainingConfig,
    DistributedConfig,
    UnifiedLoggingConfig,
    ConfigLoader,
    TrainerBase,
    TrainingState,
    TrainerFactory,
    DatasetRegistry,
    RewardRegistry,
    TaskRegistry,
    UnifiedLogger,
    UnifiedEvaluator,
    PolicyModel,
    ReferenceModel,
    ValueModel,
    RolloutEngine,
    # Optimization components
    OptimizerRegistry,
    SchedulerRegistry,
    OptimizerType,
    SchedulerType,
    get_optimizer_for_config,
    get_scheduler_for_config,
    validate_optimizer_availability,
    validate_scheduler_availability,
    # Recipe system
    RecipeRegistry,
    RecipeTypeEnum,
    ModelFamily,
    RecipeMetadata,
    Recipe,
    load_recipe_from_yaml,
    load_builtin_recipes,
    # Validation and diagnostics
    ConfigValidator,
    validate_config,
    TrainingDiagnostics,
    TrainingMonitor,
    DiagnosticsCollector,
    generate_training_report,
    run_config_validation,
    run_comprehensive_diagnostics,
    # Error handling and UX
    AlignTuneError,
    ConfigurationError,
    TrainingError,
    EnvironmentError,
    ValidationError,
    handle_error,
    create_progress_display,
    HealthMonitor,
    config_error,
    training_error,
    env_error,
    validation_error,
    PPOTrainer,
    DPOTrainer,
    GRPOTrainer,
    GSPOTrainer,
    
    # SFT system components
    SFTConfig,
    SFTTaskType,
    SFTModelConfig,
    SFTDatasetConfig,
    SFTTrainingConfig,
    SFTLoggingConfig,
    SFTConfigLoader,
    SFTTrainerFactory,
    InstructionTrainer,
    ClassificationTrainer,
    ChatTrainer,
    
    # Evaluation system
    EvalType,
    TaskCategory,
    EvalConfig,
    EvalTask,
    EvalResult,
    EvalRunner,
    LMEvalConfig,
    LMEvalRunner,
    get_available_lm_eval_tasks,
    run_standard_benchmark,
    
    # Rewards system
    RewardType,
    RewardConfig as RewardsRewardConfig,
    RewardFunction,
    RewardFunctionFactory,
    CompositeReward,
    rewards_registry,
    
    # Backend factory
    BackendFactory,
    TrainingType,
    BackendFactoryType as BackendType,
    FactoryBackendType,
    RLAlgorithm,
    BackendConfig,
    create_sft_trainer,
    create_rl_trainer,
    list_backends,
    
    # CLI components
    cli_main,
    
    # Helper functions
    get_available_trainers,
    print_available_trainers,
    check_dependencies,
    get_missing_dependencies,
)

# Import fallback functions from dedicated module
from ._fallbacks import (
    train_dpo_from_yaml,
    train_ppo_from_yaml,
    train_grpo_from_yaml,
    train_grpo_from_config,
    create_sample_dpo_config,
    create_minimal_dpo_config,
    create_sample_ppo_config,
    create_minimal_ppo_config,
    create_specialized_ppo_configs,
    load_ppo_config_from_yaml,
    show_ppo_configuration_menu,
    select_ppo_dataset_size,
    create_grpo_configurations,
    select_grpo_config_interactively,
    evaluate_grpo_model,
)

# =============================================================================
# EXPORTS
# =============================================================================

# Core exports that are always available
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Core availability flags
    "UNSLOTH_AVAILABLE",
    "TRL_AVAILABLE",
    "UNIFIED_RL_AVAILABLE",
    "UNIFIED_SFT_AVAILABLE",
    "EVAL_AVAILABLE",
    "REWARDS_AVAILABLE",
    "BACKEND_FACTORY_AVAILABLE",
    "UNIFIED_AVAILABLE",
    "CLI_AVAILABLE",
    
    # Helper functions
    "get_available_trainers",
    "print_available_trainers",
    "check_dependencies",
    "get_missing_dependencies",
]

# Add unified system components if available
if UNIFIED_AVAILABLE:
    __all__.extend([
        "UnifiedConfig",
        "AlgorithmType",
        "PrecisionType",
        "BackendType",
        "UnifiedModelConfig",
        "UnifiedDatasetConfig",
        "RewardConfig",
        "UnifiedTrainingConfig",
        "DistributedConfig",
        "UnifiedLoggingConfig",
        "ConfigLoader",
        "TrainerBase",
        "TrainingState",
        "TrainerFactory",
        "DatasetRegistry",
        "RewardRegistry",
        "TaskRegistry",
        "UnifiedLogger",
        "UnifiedEvaluator",
        "PolicyModel",
        "ReferenceModel",
        "ValueModel",
        "RolloutEngine",
        # Optimization components
        "OptimizerRegistry",
        "SchedulerRegistry",
        "OptimizerType",
        "SchedulerType",
        "get_optimizer_for_config",
        "get_scheduler_for_config",
        "validate_optimizer_availability",
        "validate_scheduler_availability",
        # Recipe system
        "RecipeRegistry",
        "RecipeTypeEnum",
        "ModelFamily",
        "RecipeMetadata",
        "Recipe",
        "load_recipe_from_yaml",
        "load_builtin_recipes",
        # Validation and diagnostics
        "ConfigValidator",
        "validate_config",
        "TrainingDiagnostics",
        "TrainingMonitor",
        "DiagnosticsCollector",
        "generate_training_report",
        "run_config_validation",
        "run_comprehensive_diagnostics",
        # Error handling and UX
        "AlignTuneError",
        "ConfigurationError",
        "TrainingError",
        "EnvironmentError",
        "ValidationError",
        "handle_error",
        "create_progress_display",
        "HealthMonitor",
        "config_error",
        "training_error",
        "env_error",
        "validation_error",
        "PPOTrainer",
        "DPOTrainer",
        "GRPOTrainer",
        "GSPOTrainer",
    ])

# Add SFT system components if available
if UNIFIED_SFT_AVAILABLE:
    __all__.extend([
        "SFTConfig",
        "SFTTaskType",
        "SFTModelConfig",
        "SFTDatasetConfig",
        "SFTTrainingConfig",
        "SFTLoggingConfig",
        "SFTConfigLoader",
        "SFTTrainerFactory",
        "InstructionTrainer",
        "ClassificationTrainer",
        "ChatTrainer",
    ])

# Add evaluation components if available
if EVAL_AVAILABLE:
    __all__.extend([
        "EvalType",
        "TaskCategory",
        "EvalConfig",
        "EvalTask",
        "EvalResult",
        "EvalRunner",
        "LMEvalConfig",
        "LMEvalRunner",
        "get_available_lm_eval_tasks",
        "run_standard_benchmark",
    ])

# Add rewards components if available
if REWARDS_AVAILABLE:
    __all__.extend([
        "RewardType",
        "RewardsRewardConfig",
        "RewardFunction",
        "RewardFunctionFactory",
        "CompositeReward",
        "rewards_registry",
    ])

    # Add backend factory components if available
if BACKEND_FACTORY_AVAILABLE:
    __all__.extend([
        "BackendFactory",
        "TrainingType",
        "BackendType",
        "FactoryBackendType",
        "RLAlgorithm",
        "BackendConfig",
        "create_sft_trainer",
        "create_rl_trainer",
        "list_backends",
        
        # CLI components
        "cli_main",
    ])