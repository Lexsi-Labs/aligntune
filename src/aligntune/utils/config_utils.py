"""
Configuration utilities for loading, saving, and validating configurations
"""

import yaml
import json
import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_config_to_unified(config_dict: Dict[str, Any], training_type: str = "rl") -> Dict[str, Any]:
    """
    Parse a config dictionary (from YAML/JSON) into the format expected by UnifiedConfig or SFTConfig.
    
    This handles nested structures like model, datasets, train, logging, etc.
    Returns a flattened dict suitable for create_rl_trainer/create_sft_trainer kwargs.
    """
    parsed = {}
    
    # Extract top-level fields
    if 'algo' in config_dict or 'algorithm' in config_dict:
        parsed['algorithm'] = config_dict.get('algo') or config_dict.get('algorithm')
    
    # Parse model section
    if 'model' in config_dict:
        model = config_dict['model']
        parsed['model_name'] = model.get('name_or_path')
        parsed['max_seq_length'] = model.get('max_seq_length', 512)
        parsed['precision'] = model.get('precision')
        parsed['quantization'] = model.get('quantization', {})
        parsed['use_gradient_checkpointing'] = model.get('gradient_checkpointing', False)
        parsed['use_peft'] = model.get('use_peft', True)
        parsed['lora_r'] = model.get('lora_r', 16)
        parsed['lora_alpha'] = model.get('lora_alpha', 32)
        parsed['lora_dropout'] = model.get('lora_dropout', 0.05)
        parsed['lora_target_modules'] = model.get('lora_target_modules')
        parsed['reward_value_model'] = model.get('reward_value_model')
        parsed['reward_model_name'] = model.get('reward_model_name')
        parsed['reward_model_path'] = model.get('reward_model_path')
        parsed['reward_value_loading_type'] = model.get('reward_value_loading_type')
        parsed['reward_model_quantization'] = model.get('reward_model_quantization', {})
        parsed['value_model_quantization'] = model.get('value_model_quantization', {})
        parsed['model_init_kwargs'] = model.get('model_init_kwargs', {})
        parsed['ref_model_init_kwargs'] = model.get('ref_model_init_kwargs', {})
        parsed['reward_device'] = model.get('reward_device', 'auto')
        parsed['device_map'] = model.get('device_map', 'auto')
        parsed['disable_dropout'] = model.get('disable_dropout', True)
    
    # Parse datasets section (handle both single dict and list)
    if 'datasets' in config_dict:
        datasets = config_dict['datasets']
        if isinstance(datasets, list) and len(datasets) > 0:
            dataset = datasets[0]  # Use first dataset
        else:
            dataset = datasets
            
        parsed['dataset_name'] = dataset.get('name')
        parsed['split'] = dataset.get('split', 'train')
        parsed['max_samples'] = dataset.get('max_samples')
        parsed['percent'] = dataset.get('percent')
        parsed['field_mappings'] = dataset.get('field_mappings', {})
        parsed['column_mapping'] = dataset.get('column_mapping', {})
        parsed['system_prompt'] = dataset.get('system_prompt')
        parsed['dataset_num_proc'] = dataset.get('dataset_num_proc')
        parsed['pad_token'] = dataset.get('pad_token')
        parsed['truncation_mode'] = dataset.get('truncation_mode', 'keep_end')
        parsed['padding_free'] = dataset.get('padding_free', False)
        parsed['tools'] = dataset.get('tools')
        parsed['config_name'] = dataset.get('config_name')
        parsed['subset'] = dataset.get('subset')
        parsed['config'] = dataset.get('config')
        
        # SFT-specific dataset fields
        if training_type == "sft":
            parsed['task_type'] = dataset.get('task_type', 'supervised_fine_tuning')
            parsed['text_column'] = dataset.get('text_column', 'text')
            parsed['label_column'] = dataset.get('label_column', 'label')
            parsed['dataset_text_field'] = dataset.get('dataset_text_field', 'text')
            parsed['chat_template'] = dataset.get('chat_template')
    
    # Parse dataset section (alternative format for SFT)
    elif 'dataset' in config_dict:
        dataset = config_dict['dataset']
        parsed['dataset_name'] = dataset.get('name')
        parsed['split'] = dataset.get('split', 'train')
        parsed['max_samples'] = dataset.get('max_samples')
        parsed['percent'] = dataset.get('percent')
        parsed['column_mapping'] = dataset.get('column_mapping', {})
        parsed['system_prompt'] = dataset.get('system_prompt')
        parsed['subset'] = dataset.get('subset')
        parsed['config'] = dataset.get('config')
        
        if training_type == "sft":
            parsed['task_type'] = dataset.get('task_type', 'supervised_fine_tuning')
            parsed['text_column'] = dataset.get('text_column', 'text')
            parsed['label_column'] = dataset.get('label_column', 'label')
    
    # Parse train/training section
    train_section = config_dict.get('train') or config_dict.get('training', {})
    if train_section:
        parsed['num_epochs'] = train_section.get('epochs', 3)
        parsed['max_steps'] = train_section.get('max_steps')
        parsed['batch_size'] = train_section.get('per_device_batch_size', 4)
        parsed['per_device_batch_size'] = train_section.get('per_device_batch_size', 4)
        parsed['eval_batch_size'] = train_section.get('per_device_eval_batch_size')
        parsed['gradient_accumulation_steps'] = train_section.get('gradient_accumulation_steps', 1)
        parsed['learning_rate'] = train_section.get('learning_rate', 2e-4)
        parsed['warmup_steps'] = train_section.get('warmup_steps', 0)
        parsed['warmup_ratio'] = train_section.get('warmup_ratio', 0.1)
        parsed['weight_decay'] = train_section.get('weight_decay', 0.01)
        parsed['seed'] = train_section.get('seed', 42)
        parsed['data_seed'] = train_section.get('data_seed', 47)
        
        # RL-specific training params
        if training_type == "rl":
            parsed['beta'] = train_section.get('beta', 0.1)
            parsed['kl_coef'] = train_section.get('kl_coef', 0.1)
            parsed['num_generations'] = train_section.get('num_generations')
            parsed['cliprange'] = train_section.get('cliprange', 0.2)
            parsed['max_prompt_length'] = train_section.get('max_prompt_length', 512)
            parsed['max_completion_length'] = train_section.get('max_completion_length', 256)
            parsed['temperature'] = train_section.get('temperature', 0.6)
            parsed['top_p'] = train_section.get('top_p', 0.95)
            parsed['max_grad_norm'] = train_section.get('max_grad_norm', 1.0)
            parsed['whiten_rewards'] = train_section.get('whiten_rewards', False)
            parsed['loss_type'] = train_section.get('loss_type', 'sigmoid')
            parsed['response_length'] = train_section.get('response_length', 128)
            parsed['generation_kwargs'] = train_section.get('generation_kwargs', {})
            parsed['scale_rewards'] = train_section.get('scale_rewards', 'group')
            parsed['enable_thinking'] = train_section.get('enable_thinking', False)
            
            # PPO-specific
            parsed['num_ppo_epochs'] = train_section.get('num_ppo_epochs')
            parsed['vf_coef'] = train_section.get('vf_coef', 0.1)
            parsed['cliprange_value'] = train_section.get('cliprange_value', 0.2)
            parsed['gamma'] = train_section.get('gamma', 1.0)
            parsed['lam'] = train_section.get('lam', 0.95)
            
            # Counterfactual GRPO params
            parsed['boost_factor'] = train_section.get('boost_factor', 2.0)
            parsed['min_weight'] = train_section.get('min_weight', 0.5)
            parsed['max_spans'] = train_section.get('max_spans', 10)
            parsed['answer_weight'] = train_section.get('answer_weight', 1.5)
            parsed['method_name'] = train_section.get('method_name', 'counterfactual')
            parsed['weighting_mode'] = train_section.get('weighting_mode')
            parsed['random_importance'] = train_section.get('random_importance', False)
            parsed['invert_importance'] = train_section.get('invert_importance', False)
            parsed['enable_gradient_conservation'] = train_section.get('enable_gradient_conservation', True)
            parsed['weight_debug'] = train_section.get('weight_debug', False)
            
            # GBMPO params
            parsed['gbmpo_divergence_type'] = train_section.get('gbmpo_divergence_type')
            parsed['gbmpo_l2_coefficient'] = train_section.get('gbmpo_l2_coefficient', 0.0001)
            parsed['gbmpo_epsilon'] = train_section.get('gbmpo_epsilon', 0.2)
            
            # BOLT params
            parsed['curriculum_enabled'] = train_section.get('curriculum_enabled', False)
            parsed['baseline_enabled'] = train_section.get('baseline_enabled', False)
        
        # SFT-specific training params
        if training_type == "sft":
            parsed['packing'] = train_section.get('packing', False)
            parsed['packing_strategy'] = train_section.get('packing_strategy', 'bfd')
            parsed['padding_free'] = train_section.get('padding_free', False)
            parsed['group_by_length'] = train_section.get('group_by_length', True)
            parsed['completion_only_loss'] = train_section.get('completion_only_loss')
            parsed['assistant_only_loss'] = train_section.get('assistant_only_loss', False)
        
        # Checkpointing
        parsed['save_steps'] = train_section.get('save_steps', 500)
        parsed['save_total_limit'] = train_section.get('save_total_limit')
        parsed['save_strategy'] = train_section.get('save_strategy', 'steps')
        parsed['save_interval'] = train_section.get('save_interval', 100)
        parsed['logging_steps'] = train_section.get('logging_steps', 10)
        parsed['eval_steps'] = train_section.get('eval_steps', 100)
        parsed['eval_interval'] = train_section.get('eval_interval', 100)
        parsed['eval_strategy'] = train_section.get('eval_strategy', 'no')
    
    # Parse logging section
    if 'logging' in config_dict:
        logging_cfg = config_dict['logging']
        parsed['output_dir'] = logging_cfg.get('output_dir', './output')
        parsed['run_name'] = logging_cfg.get('run_name')
        parsed['loggers'] = logging_cfg.get('loggers', ['tensorboard'])
        parsed['report_to'] = logging_cfg.get('report_to', 'none')
        parsed['wandb_project'] = logging_cfg.get('wandb_project')
        
        # Sample logging
        if 'sample_logging' in logging_cfg:
            parsed['sample_logging'] = logging_cfg['sample_logging']
    
    # Parse rewards section
    if 'rewards' in config_dict:
        parsed['rewards'] = config_dict['rewards']
    
    # Parse evaluation section (for SFT)
    if 'evaluation' in config_dict and training_type == "sft":
        eval_cfg = config_dict['evaluation']
        parsed['compute_perplexity'] = eval_cfg.get('compute_perplexity', True)
        parsed['compute_rouge'] = eval_cfg.get('compute_rouge', True)
        parsed['compute_bleu'] = eval_cfg.get('compute_bleu', True)
    
    # Top-level fields
    parsed['chat_template'] = config_dict.get('chat_template', 'auto')
    parsed['backend'] = config_dict.get('backend', 'auto')
    
    # Caching
    if 'caching' in config_dict:
        caching = config_dict['caching']
        parsed['cache_dir'] = caching.get('root', 'cache')
        parsed['caching_enabled'] = caching.get('enabled', True)
    
    # Reward training
    if 'reward_training' in config_dict:
        parsed['reward_training'] = config_dict['reward_training']
    
    # Remove None values to avoid overriding defaults
    return {k: v for k, v in parsed.items() if v is not None}



def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def validate_config(config: Dict[str, Any], config_type: str = "sft") -> bool:
    """
    Validate configuration structure
    
    Args:
        config: Configuration dictionary
        config_type: Type of configuration ("sft" or "rl")
        
    Returns:
        True if valid, raises exception if invalid
    """
    try:
        if config_type == "sft":
            # Required sections for SFT
            required_sections = ['model', 'dataset', 'training']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            # Validate model section
            model_config = config['model']
            if 'name' not in model_config:
                raise ValueError("Missing 'name' in model configuration")
            
            # Validate dataset section
            dataset_config = config['dataset']
            if 'name' not in dataset_config:
                raise ValueError("Missing 'name' in dataset configuration")
            
            # Validate training section
            training_config = config['training']
            if 'task_type' not in training_config:
                raise ValueError("Missing 'task_type' in training configuration")
                
        elif config_type == "rl":
            # Required fields for RL
            required_fields = ['model_name', 'dataset_name', 'trainer_type']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate trainer type
            valid_trainers = ['dpo', 'grpo', 'ppo']
            if config['trainer_type'] not in valid_trainers:
                raise ValueError(f"Invalid trainer_type: {config['trainer_type']}. Must be one of {valid_trainers}")
        
        else:
            raise ValueError(f"Unknown config_type: {config_type}")
        
        logger.info(f"Configuration validation passed for {config_type}")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return deep_merge(base_config, override_config)


def create_config_template(config_type: str = "sft", task_type: str = "instruction") -> Dict[str, Any]:
    """
    Create a configuration template
    
    Args:
        config_type: Type of configuration ("sft" or "rl")
        task_type: Specific task type
        
    Returns:
        Configuration template
    """
    if config_type == "sft":
        if task_type == "instruction":
            from ..sft.configs import create_instruction_following_config
            return create_instruction_following_config()
        elif task_type == "classification":
            from ..sft.configs import create_text_classification_config
            return create_text_classification_config()
        elif task_type == "chat":
            from ..sft.configs import create_chat_completion_config
            return create_chat_completion_config()
        else:
            from ..sft.configs import create_sft_config
            return create_sft_config()
            
    elif config_type == "rl":
        if task_type == "dpo":
            from ..rl.configs import create_dpo_config
            return create_dpo_config().__dict__
        elif task_type == "grpo":
            from ..rl.configs import create_grpo_config
            return create_grpo_config().__dict__
        elif task_type == "ppo":
            from ..rl.configs import create_ppo_config
            return create_ppo_config().__dict__
        else:
            from ..rl.configs import create_dpo_config
            return create_dpo_config().__dict__
    
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


def update_config_paths(config: Dict[str, Any], base_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Update relative paths in configuration to be relative to base_path
    
    Args:
        config: Configuration dictionary
        base_path: Base path for relative paths
        
    Returns:
        Updated configuration
    """
    base_path = Path(base_path)
    updated_config = config.copy()
    
    # Common path fields that might need updating
    path_fields = [
        'output_dir',
        'model.name',
        'dataset.name'
    ]
    
    def update_nested_path(config_dict: dict, field_path: str, base: Path):
        """Update nested path in configuration"""
        keys = field_path.split('.')
        current = config_dict
        
        # Navigate to the parent of the target field
        for key in keys[:-1]:
            if key in current and isinstance(current[key], dict):
                current = current[key]
            else:
                return  # Path doesn't exist
        
        # Update the target field if it exists
        target_key = keys[-1]
        if target_key in current:
            current_path = Path(current[target_key])
            if not current_path.is_absolute():
                current[target_key] = str(base / current_path)
    
    for field_path in path_fields:
        update_nested_path(updated_config, field_path, base_path)
    
    return updated_config


def export_config_summary(config: Dict[str, Any], output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Export a human-readable summary of the configuration
    
    Args:
        config: Configuration dictionary
        output_path: Optional path to save summary
        
    Returns:
        Configuration summary as string
    """
    summary_lines = ["AlignTune Configuration Summary", "=" * 50]
    
    def format_section(section_name: str, section_data: Any, indent: int = 0) -> List[str]:
        """Format a configuration section"""
        lines = []
        prefix = "  " * indent
        
        if isinstance(section_data, dict):
            lines.append(f"{prefix}{section_name}:")
            for key, value in section_data.items():
                if isinstance(value, dict):
                    lines.extend(format_section(key, value, indent + 1))
                else:
                    lines.append(f"{prefix}  {key}: {value}")
        else:
            lines.append(f"{prefix}{section_name}: {section_data}")
        
        return lines
    
    # Add configuration sections
    for key, value in config.items():
        summary_lines.extend(format_section(key, value))
        summary_lines.append("")  # Empty line between sections
    
    summary = "\n".join(summary_lines)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(summary)
        logger.info(f"Configuration summary saved to: {output_path}")
    
    return summary