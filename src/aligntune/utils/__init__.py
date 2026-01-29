"""
Utility modules for AlignTune
"""

from .auth import (
    setup_hf_auth, 
    check_hf_auth, 
    get_user_info,
    get_hf_token,
    logout_hf,
    test_hf_connection,
    interactive_hf_setup
)

from .model_loader import (
    ModelLoader,
    load_local_model,
    load_model_auto,
    get_model_info
)

from .checkpointing import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint
)

from .logging import (
    LoggingManager,
    create_logging_manager,
    create_wandb_config,
    create_tensorboard_config,
    create_full_logging_config
)

from .device import (
    DeviceManager,
    setup_device_config,
    get_optimal_batch_size,
    check_gpu_compatibility,
    get_device_manager
)

from .config_utils import (
    load_config,
    save_config,
    validate_config,
    merge_configs,
    create_config_template,
    update_config_paths,
    export_config_summary
)

from .colored_logging import (
    print_aligntune_banner,
    print_section_banner,
    print_subsection,
    aligntune_info,
    aligntune_warning,
    aligntune_error,
    aligntune_success,
    aligntune_step,
    setup_colored_logging,
    init_aligntune_logging,
    print_progress_bar,
    Fore,
    Back,
    Style,
)

__all__ = [
    # Auth utilities
    "setup_hf_auth",
    "check_hf_auth", 
    "get_user_info",
    "get_hf_token",
    "logout_hf",
    "test_hf_connection",
    "interactive_hf_setup",
    
    # Model loading utilities
    "ModelLoader",
    "load_local_model",
    "load_model_auto",
    "get_model_info",
    
    # Checkpointing utilities
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "get_latest_checkpoint",
    
    # Logging utilities
    "LoggingManager",
    "create_logging_manager",
    "create_wandb_config",
    "create_tensorboard_config",
    "create_full_logging_config",
    
    # Device utilities
    "DeviceManager",
    "setup_device_config",
    "get_optimal_batch_size",
    "check_gpu_compatibility",
    "get_device_manager",
    
    # Config utilities
    "load_config",
    "save_config",
    "validate_config",
    "merge_configs",
    "create_config_template",
    "update_config_paths",
    "export_config_summary",
    
    # Colored logging utilities
    "print_aligntune_banner",
    "print_section_banner",
    "print_subsection",
    "aligntune_info",
    "aligntune_warning",
    "aligntune_error",
    "aligntune_success",
    "aligntune_step",
    "setup_colored_logging",
    "init_aligntune_logging",
    "print_progress_bar",
    "Fore",
    "Back",
    "Style",
]