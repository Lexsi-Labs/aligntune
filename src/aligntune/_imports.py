"""
Import management module for AlignTune.

This module consolidates all conditional imports and provides availability flags
for different components. It handles import errors gracefully and provides
clear error messages when dependencies are missing.
"""

import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# =============================================================================
# CORE DEPENDENCY CHECKS
# =============================================================================

# Conditional import for Unsloth - NEVER import if TRL-only mode is active
UNSLOTH_AVAILABLE = None  # Will be set lazily
UNSLOTH_ERROR_INFO = None  # Will store detailed error information

# Check if we should prevent Unsloth from loading at all
def _should_prevent_unsloth():
    """Check if Unsloth should be prevented from loading."""
    import os
    return (
        os.environ.get('TRL_ONLY_MODE', '0') == '1' or
        os.environ.get('DISABLE_UNSLOTH_FOR_TRL', '0') == '1' or
        os.environ.get('PURE_TRL_MODE', '0') == '1'
    )

# Set PURE_TRL_MODE at module level to prevent Unsloth import
import os
if os.environ.get('PURE_TRL_MODE', '0') == '1':
    # Force Unsloth to be unavailable
    UNSLOTH_AVAILABLE = False
    UNSLOTH_ERROR_INFO = {
        'error': 'Pure TRL mode - Unsloth completely disabled',
        'error_type': 'disabled_for_compatibility',
        'environment': {},
        'suggestion': ['Pure TRL mode active', 'Only HuggingFace Transformers and TRL will be used']
    }

def _categorize_unsloth_error(error):
    """Categorize unsloth import errors for better diagnostics."""
    error_str = str(error).lower()
    
    if 'undefined symbol' in error_str and 'cuda' in error_str:
        return 'cuda_symbol_error'
    elif 'flash_attn' in error_str or 'flash attention' in error_str:
        return 'flash_attention_error'
    elif 'version' in error_str and ('incompatible' in error_str or 'mismatch' in error_str):
        return 'version_mismatch'
    elif 'not found' in error_str or 'no module named' in error_str:
        return 'missing_dependency'
    elif 'not implemented' in error_str:
        return 'not_implemented'
    else:
        return 'unknown_error'

def _get_unsloth_fix_suggestion(error_type, env_info):
    """Provide actionable fix suggestions based on error type."""
    suggestions = {
        'cuda_symbol_error': [
            "CUDA symbol errors indicate version incompatibility between PyTorch and Unsloth",
            f"Current PyTorch: {env_info.get('pytorch_version', 'unknown')}",
            f"Current CUDA: {env_info.get('cuda_version', 'unknown')}",
            "Try: pip install --upgrade unsloth",
            "Or: Use TRL backends instead (--backend trl)"
        ],
        'flash_attention_error': [
            "Flash Attention compatibility issue detected",
            "Try: pip install --upgrade flash-attn",
            "Or: Unsloth will fallback to Xformers automatically"
        ],
        'version_mismatch': [
            "Version mismatch between dependencies",
            "Check PyTorch/CUDA compatibility matrix",
            "Try: pip install --upgrade torch unsloth"
        ],
        'missing_dependency': [
            "Unsloth or its dependencies are not installed",
            "Try: pip install unsloth",
            "Or: Use TRL backends instead (--backend trl)"
        ],
        'not_implemented': [
            "Unsloth feature not implemented for this configuration",
            "Try: Use TRL backends instead (--backend trl)"
        ],
        'unknown_error': [
            "Unknown error occurred during Unsloth import",
            "Check logs for more details",
            "Try: pip install --upgrade unsloth",
            "Or: Use TRL backends instead (--backend trl)"
        ]
    }
    return suggestions.get(error_type, suggestions['unknown_error'])

def _check_unsloth_available():
    """Lazy check for Unsloth availability with detailed diagnostics."""
    global UNSLOTH_AVAILABLE, UNSLOTH_ERROR_INFO
    if UNSLOTH_AVAILABLE is None:
        # NEVER import Unsloth if we should prevent it
        if _should_prevent_unsloth():
            UNSLOTH_AVAILABLE = False
            UNSLOTH_ERROR_INFO = {
                'error': 'Unsloth prevented from loading to avoid TRL interference',
                'error_type': 'disabled_for_compatibility',
                'environment': {},
                'suggestion': ['Unsloth prevented from loading', 'Will be loaded after TRL training']
            }
            logger.info("🚫 Unsloth prevented from loading to avoid TRL interference")
            return False
        
        # Check if transformers/trl/peft are already imported (bad import order)
        # Only warn once to avoid spam
        if not hasattr(_check_unsloth_available, '_import_warning_shown'):
            import sys
            already_imported = []
            if 'transformers' in sys.modules:
                already_imported.append('transformers')
            if 'trl' in sys.modules:
                already_imported.append('trl')
            if 'peft' in sys.modules:
                already_imported.append('peft')
            
            if already_imported:
                logger.warning(
                    f"WARNING: Unsloth should be imported before {', '.join(already_imported)} "
                    f"to ensure all optimizations are applied. Your code may run slower or encounter "
                    f"memory issues without these optimizations.\n"
                    f"Please restructure your imports with 'import unsloth' at the top of your file."
                )
                _check_unsloth_available._import_warning_shown = True
            
        env_info = {}
        try:
            import torch
            # Pre-check: Validate PyTorch and CUDA versions
            env_info = {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
            
            # Import Unsloth (should be before trl/transformers/peft, but we need torch for version info)
            import unsloth
            from unsloth import FastLanguageModel
            UNSLOTH_AVAILABLE = True
            UNSLOTH_ERROR_INFO = None
            logger.info(f"Unsloth successfully imported (PyTorch {env_info['pytorch_version']}, CUDA {env_info['cuda_version']})")
        except Exception as e:
            UNSLOTH_AVAILABLE = False
            # Categorize error type
            error_type = _categorize_unsloth_error(e)
            UNSLOTH_ERROR_INFO = {
                'error': str(e),
                'error_type': error_type,
                'environment': env_info,
                'suggestion': _get_unsloth_fix_suggestion(error_type, env_info)
            }
            logger.warning(f"Unsloth not available: {error_type} - {e}")
            # Set environment variables to disable Unsloth
            os.environ.setdefault('DISABLE_UNSLOTH', '1')
    return UNSLOTH_AVAILABLE

# Lazy import for TRL to avoid Unsloth interference
TRL_AVAILABLE = None  # Will be set lazily
TRL_ERROR_INFO = None  # Will store detailed error information

def _check_trl_available():
    """Lazy check for TRL availability."""
    global TRL_AVAILABLE, TRL_ERROR_INFO
    if TRL_AVAILABLE is None:
        try:
            import trl
            TRL_AVAILABLE = True
            TRL_ERROR_INFO = None
            logger.info("TRL successfully imported")
        except ImportError as e:
            TRL_AVAILABLE = False
            TRL_ERROR_INFO = {
                'error': str(e),
                'error_type': 'missing_dependency',
                'suggestion': ["TRL not installed", "Try: pip install trl"]
            }
            logger.warning(f"TRL not available: {e}")
    return TRL_AVAILABLE

def enable_unsloth_after_trl():
    """Re-enable Unsloth after TRL training is complete."""
    global UNSLOTH_AVAILABLE, UNSLOTH_ERROR_INFO
    
    # Clear environment variables that disable Unsloth
    import os
    os.environ.pop('DISABLE_UNSLOTH_FOR_TRL', None)
    os.environ.pop('TRL_ONLY_MODE', None)
    os.environ.pop('UNSLOTH_DISABLE_PATCHING', None)
    os.environ.pop('UNSLOTH_COMPILED_CACHE', None)
    
    # Reset Unsloth availability to force re-check
    UNSLOTH_AVAILABLE = None
    UNSLOTH_ERROR_INFO = None
    
    # Force re-check Unsloth availability
    logger.info("🔄 Re-enabling Unsloth after TRL training...")
    _check_unsloth_available()
    
    if UNSLOTH_AVAILABLE:
        logger.info("✅ Unsloth successfully re-enabled")
    else:
        logger.warning("⚠️ Unsloth could not be re-enabled")
    
    return UNSLOTH_AVAILABLE

# =============================================================================
# LEGACY BACKEND REMOVED
# =============================================================================
# All legacy backend functionality has been removed.
# Use TRL or Unsloth backends instead via create_sft_trainer() or create_rl_trainer()

# =============================================================================
# UNIFIED RLHF SYSTEM IMPORTS
# =============================================================================

try:
    from .core.rl import (
        UnifiedConfig,
        AlgorithmType,
        PrecisionType,
        BackendType,
        ModelConfig as UnifiedModelConfig,
        DatasetConfig as UnifiedDatasetConfig,
        RewardConfig,
        TrainingConfig as UnifiedTrainingConfig,
        DistributedConfig,
        LoggingConfig as UnifiedLoggingConfig,
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
    )
    # Optimization components
    from .core.optimization import (
        OptimizerRegistry,
        SchedulerRegistry,
        OptimizerType,
        SchedulerType,
        get_optimizer_for_config,
        get_scheduler_for_config,
        validate_optimizer_availability,
        validate_scheduler_availability,
    )
    # Recipe system
    from .recipes import (
        RecipeRegistry,
        RecipeType as RecipeTypeEnum,
        ModelFamily,
        RecipeMetadata,
        Recipe,
        load_recipe_from_yaml,
        load_builtin_recipes,
    )
    # Validation and diagnostics
    from .utils.validation import (
        ConfigValidator,
        validate_config,
    )
    from .utils.diagnostics import (
        TrainingDiagnostics,
        TrainingMonitor,
        DiagnosticsCollector,
        generate_training_report,
        run_config_validation,
        run_comprehensive_diagnostics,
    )
    # Error handling and UX
    from .utils.errors import (
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
    )
    UNIFIED_RL_AVAILABLE = True
    logger.info("Unified RLHF system imported successfully")
except ImportError as e:
    UNIFIED_RL_AVAILABLE = False
    logger.warning(f"Unified RLHF system not available: {e}")
    # Set all unified components to None
    UnifiedConfig = None
    AlgorithmType = None
    PrecisionType = None
    BackendType = None
    UnifiedModelConfig = None
    UnifiedDatasetConfig = None
    RewardConfig = None
    UnifiedTrainingConfig = None
    DistributedConfig = None
    UnifiedLoggingConfig = None
    ConfigLoader = None
    TrainerBase = None
    TrainingState = None
    TrainerFactory = None
    DatasetRegistry = None
    RewardRegistry = None
    TaskRegistry = None
    UnifiedLogger = None
    UnifiedEvaluator = None
    PolicyModel = None
    ReferenceModel = None
    ValueModel = None
    RolloutEngine = None
    # Optimization components
    OptimizerRegistry = None
    SchedulerRegistry = None
    OptimizerType = None
    SchedulerType = None
    get_optimizer_for_config = None
    get_scheduler_for_config = None
    validate_optimizer_availability = None
    validate_scheduler_availability = None
    # Recipe system
    RecipeRegistry = None
    RecipeTypeEnum = None
    ModelFamily = None
    RecipeMetadata = None
    Recipe = None
    load_recipe_from_yaml = None
    load_builtin_recipes = None
    # Validation and diagnostics
    ConfigValidator = None
    validate_config = None
    TrainingDiagnostics = None
    TrainingMonitor = None
    DiagnosticsCollector = None
    generate_training_report = None
    run_config_validation = None
    run_comprehensive_diagnostics = None
    # Error handling and UX
    AlignTuneError = None
    ConfigurationError = None
    TrainingError = None
    EnvironmentError = None
    ValidationError = None
    handle_error = None
    create_progress_display = None
    HealthMonitor = None
    config_error = None
    training_error = None
    env_error = None
    validation_error = None

# Unified RL trainers - These modules don't exist, so set to None
UNIFIED_RL_TRAINERS_AVAILABLE = False
logger.info("Unified RL trainers not available - modules don't exist at specified paths")
PPOTrainer = None
DPOTrainer = None
GRPOTrainer = None
GSPOTrainer = None

# =============================================================================
# UNIFIED SFT SYSTEM IMPORTS
# =============================================================================

# Unified SFT trainers - These modules don't exist, so set to None
UNIFIED_SFT_AVAILABLE = False
logger.info("Unified SFT system not available - modules don't exist at specified paths")
# Set all SFT components to None
SFTConfig = None
SFTTaskType = None
SFTModelConfig = None
SFTDatasetConfig = None
SFTTrainingConfig = None
SFTLoggingConfig = None
SFTConfigLoader = None
SFTTrainerFactory = None
InstructionTrainer = None
ClassificationTrainer = None
ChatTrainer = None

# =============================================================================
# EVALUATION SYSTEM IMPORTS
# =============================================================================

try:
    from .eval import (
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
    )
    EVAL_AVAILABLE = True
    logger.info("Evaluation system imported successfully")
except ImportError as e:
    EVAL_AVAILABLE = False
    logger.warning(f"Evaluation system not available: {e}")
    # Set all eval components to None
    EvalType = None
    TaskCategory = None
    EvalConfig = None
    EvalTask = None
    EvalResult = None
    EvalRunner = None
    LMEvalConfig = None
    LMEvalRunner = None
    get_available_lm_eval_tasks = None
    run_standard_benchmark = None

# =============================================================================
# REWARDS SYSTEM IMPORTS
# =============================================================================

try:
    from .rewards import (
        RewardType,
        RewardConfig,
        RewardFunction,
        RewardFunctionFactory,
        CompositeReward,
        RewardModelTrainer,
        RewardModelDataset,
        RewardModelValidator,
        RewardModelLoader,
        registry as rewards_registry,
    )
    REWARDS_AVAILABLE = True
    REWARD_TRAINING_AVAILABLE = True
    logger.info("Rewards system and training imported successfully")
except ImportError as e:
    REWARDS_AVAILABLE = False
    REWARD_TRAINING_AVAILABLE = False
    logger.warning(f"Rewards system not available: {e}")
    # Set all rewards components to None
    RewardType = None
    RewardConfig = None
    RewardFunction = None
    RewardFunctionFactory = None
    CompositeReward = None
    RewardModelTrainer = None
    RewardModelDataset = None
    RewardModelValidator = None
    RewardModelLoader = None
    rewards_registry = None

# =============================================================================
# BACKEND FACTORY IMPORTS (LAZY LOADING)
# =============================================================================

# Don't import backend factory here to avoid triggering imports before Unsloth
# The factory will be imported lazily when needed
BACKEND_FACTORY_AVAILABLE = True  # Assume available, will be checked when used

# Lazy import wrapper for backend factory
def _lazy_import_backend_factory():
    """Lazy import backend factory to ensure proper import order."""
    try:
        from .core.backend_factory import (
            BackendFactory,
            TrainingType,
            BackendType as FactoryBackendType,
            RLAlgorithm,
            BackendConfig,
            create_sft_trainer,
            create_rl_trainer,
            list_backends,
        )
        return {
            'BackendFactory': BackendFactory,
            'TrainingType': TrainingType,
            'FactoryBackendType': FactoryBackendType,
            'RLAlgorithm': RLAlgorithm,
            'BackendConfig': BackendConfig,
            'create_sft_trainer': create_sft_trainer,
            'create_rl_trainer': create_rl_trainer,
            'list_backends': list_backends,
        }
    except ImportError as e:
        logger.warning(f"Backend factory not available: {e}")
        return None

# Import backend factory components directly
try:
    from .core.backend_factory import (
        BackendFactory,
        TrainingType,
        BackendType as BackendFactoryType,
        RLAlgorithm,
        BackendConfig,
        create_sft_trainer,
        create_rl_trainer,
        list_backends,
    )
    # Use BackendFactoryType as the main BackendType for backend factory
    FactoryBackendType = BackendFactoryType
except ImportError as e:
    logger.warning(f"Backend factory not available: {e}")
    BackendFactory = None
    TrainingType = None
    BackendFactoryType = None
    FactoryBackendType = None
    RLAlgorithm = None
    BackendConfig = None
    create_sft_trainer = None
    create_rl_trainer = None
    list_backends = None

# =============================================================================
# CLI IMPORTS
# =============================================================================

try:
    from .cli import main as cli_main
    CLI_AVAILABLE = True
    logger.info("CLI imported successfully")
except ImportError as e:
    CLI_AVAILABLE = False
    logger.warning(f"CLI not available: {e}")
    cli_main = None

# =============================================================================
# COMPUTED AVAILABILITY FLAGS
# =============================================================================

UNIFIED_AVAILABLE = UNIFIED_RL_AVAILABLE or UNIFIED_SFT_AVAILABLE

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_available_trainers() -> dict:
    """Returns a dictionary of available trainer types."""
    return {
        "Unsloth": _check_unsloth_available(),
        "TRL": TRL_AVAILABLE,
        "Unified": UNIFIED_AVAILABLE,
        "BackendFactory": BACKEND_FACTORY_AVAILABLE,
    }

def print_available_trainers():
    """Print all available trainers."""
    trainers = get_available_trainers()
    print("\n" + "="*60)
    print("FINETUNEHUB - AVAILABLE TRAINERS")
    print("="*60)
    for trainer, available in trainers.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{trainer:15s}: {status}")
    print("="*60 + "\n")

def check_dependencies() -> dict:
    """Check all dependencies and return status."""
    return {
        "unsloth": _check_unsloth_available(),
        "trl": TRL_AVAILABLE,
        "unified_rl": UNIFIED_RL_AVAILABLE,
        "unified_sft": UNIFIED_SFT_AVAILABLE,
        "eval": EVAL_AVAILABLE,
        "rewards": REWARDS_AVAILABLE,
        "backend_factory": BACKEND_FACTORY_AVAILABLE,
    }

def get_missing_dependencies() -> list:
    """Get list of missing dependencies with installation instructions."""
    missing = []
    
    if not TRL_AVAILABLE:
        missing.append("TRL: pip install trl")
    
    if not UNSLOTH_AVAILABLE:
        missing.append("Unsloth: pip install unsloth")
    
    return missing

