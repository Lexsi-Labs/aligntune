"""
<<<<<<< HEAD
=======
Fallback handler module for AlignTune.

This module provides fallback functions when dependencies are missing.
All functions provide clear error messages with installation instructions.
"""

def train_dpo_from_yaml(*args, **kwargs):
    """Fallback for DPO training when TRL is not available."""
    raise ImportError(
        "DPO training requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def train_ppo_from_yaml(*args, **kwargs):
    """Fallback for PPO training when TRL is not available."""
    raise ImportError(
        "PPO training requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def train_grpo_from_yaml(*args, **kwargs):
    """Fallback for GRPO training when TRL is not available."""
    raise ImportError(
        "GRPO training requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def train_grpo_from_config(*args, **kwargs):
    """Fallback for GRPO training from config when TRL is not available."""
    raise ImportError(
        "GRPO training requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def create_sample_dpo_config(*args, **kwargs):
    """Fallback for DPO config creation when TRL is not available."""
    raise ImportError(
        "DPO config creation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def create_minimal_dpo_config(*args, **kwargs):
    """Fallback for minimal DPO config when TRL is not available."""
    raise ImportError(
        "DPO config creation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def create_sample_ppo_config(*args, **kwargs):
    """Fallback for PPO config creation when TRL is not available."""
    raise ImportError(
        "PPO config creation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def create_minimal_ppo_config(*args, **kwargs):
    """Fallback for minimal PPO config when TRL is not available."""
    raise ImportError(
        "PPO config creation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def create_specialized_ppo_configs(*args, **kwargs):
    """Fallback for specialized PPO configs when TRL is not available."""
    raise ImportError(
        "PPO config creation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def load_ppo_config_from_yaml(*args, **kwargs):
    """Fallback for PPO config loading when TRL is not available."""
    raise ImportError(
        "PPO config loading requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def show_ppo_configuration_menu(*args, **kwargs):
    """Fallback for PPO config menu when TRL is not available."""
    raise ImportError(
        "PPO config menu requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def select_ppo_dataset_size(*args, **kwargs):
    """Fallback for PPO dataset selection when TRL is not available."""
    raise ImportError(
        "PPO dataset selection requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def create_grpo_configurations(*args, **kwargs):
    """Fallback for GRPO config creation when TRL is not available."""
    raise ImportError(
        "GRPO config creation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def select_grpo_config_interactively(*args, **kwargs):
    """Fallback for GRPO config selection when TRL is not available."""
    raise ImportError(
        "GRPO config selection requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

def evaluate_grpo_model(*args, **kwargs):
    """Fallback for GRPO evaluation when TRL is not available."""
    raise ImportError(
        "GRPO evaluation requires TRL. Install with: pip install trl\n"
        "For more information, visit: https://github.com/huggingface/trl"
    )

# Fallback classes for when trainers are not available
class FallbackSFTTrainer:
    """Fallback SFT trainer when trainer is not available."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "SFT trainer not available. Check your AlignTune installation.\n"
            "Install with: pip install -e .\n"
            "For more information, visit: https://github.com/yourusername/aligntune"
        )

class FallbackClassificationTrainer:
    """Fallback classification trainer when trainer is not available."""
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Classification trainer not available. Check your AlignTune installation.\n"
            "Install with: pip install -e .\n"
            "For more information, visit: https://github.com/yourusername/aligntune"
        )
