"""
CLI module for AlignTune.


This module provides command-line interface functionality for AlignTune,
including the unified CLI with backend factory integration.
"""

try:
    from .unified import app as unified_app
    UNIFIED_CLI_AVAILABLE = True
except ImportError:
    UNIFIED_CLI_AVAILABLE = False
    unified_app = None

try:
    from .arg_parser import (
        create_base_parser,
        parse_args,
        args_to_dict,
        add_base_args,
        add_model_args,
        add_training_args,
        add_dataset_args,
        add_backend_args,
        add_evaluation_args,
    )
    ARG_PARSER_AVAILABLE = True
except ImportError:
    ARG_PARSER_AVAILABLE = False
    create_base_parser = None
    parse_args = None
    args_to_dict = None
    add_base_args = None
    add_model_args = None
    add_training_args = None
    add_dataset_args = None
    add_backend_args = None
    add_evaluation_args = None

try:
    from .config_builders import (
        build_sft_config,
        build_dpo_config,
        build_ppo_config,
        build_grpo_config,
        build_gspo_config,
        create_trainer_from_config,
    )
    CONFIG_BUILDERS_AVAILABLE = True
except ImportError:
    CONFIG_BUILDERS_AVAILABLE = False
    build_sft_config = None
    build_dpo_config = None
    build_ppo_config = None
    build_grpo_config = None
    build_gspo_config = None
    create_trainer_from_config = None

__all__ = [
    # Unified CLI
    "unified_app",
    "UNIFIED_CLI_AVAILABLE",
    
    # Argument Parser
    "create_base_parser",
    "parse_args",
    "args_to_dict",
    "add_base_args",
    "add_model_args",
    "add_training_args",
    "add_dataset_args",
    "add_backend_args",
    "add_evaluation_args",
    "ARG_PARSER_AVAILABLE",
    
    # Configuration Builders
    "build_sft_config",
    "build_dpo_config",
    "build_ppo_config",
    "build_grpo_config",
    "build_gspo_config",
    "create_trainer_from_config",
    "CONFIG_BUILDERS_AVAILABLE",
]


def main():
    """Main CLI entry point."""
    if UNIFIED_CLI_AVAILABLE and unified_app:
        unified_app()
    else:
        print("CLI not available. Please check your installation.")
        return 1
    return 0