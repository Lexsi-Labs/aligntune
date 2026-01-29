"""
Unified Config Parameter Extractor - Focus on config.train

This extractor:
1. Checks config.train for ALL parameters
2. Compares with backend_config to find MISSING params
3. Extracts from config.train.extra_params
4. Merges with kwargs
5. Returns only the missing params to add to backend_config

Usage:
    # Manual extraction
    grpo_config = GRPOConfig(
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        # ... manual params ...
    )
    
    # Get missing params
    missing_params = extract_extra_and_missing_params(
        backend_config=grpo_config,
        config=self.config,
        algorithm='grpo',
        **kwargs
    )
    
    # Add missing params
    for key, value in missing_params.items():
        setattr(grpo_config, key, value)
"""

import logging
from typing import Dict, Any, Set, Optional
from dataclasses import fields, is_dataclass
import inspect

logger = logging.getLogger(__name__)


class ConfigExtractor:
    """Extractor focused on config.train + extra_params."""
    
    # Algorithm to config class mapping
    ALGORITHM_CONFIG_MAP = {
        # GRPO and variants
        'grpo': 'GRPOConfig',
        'counterfact_grpo': 'GRPOConfig',
        'gbmpo': 'GRPOConfig',
        'drgrpo': 'GRPOConfig',
        'dapo': 'GRPOConfig',
        'nmgrpo': 'GRPOConfig',
        'bolt': 'GRPOConfig',
        
        # DPO
        'dpo': 'DPOConfig',
        
        # PPO
        'ppo': 'PPOConfig',
        
        # SFT
        'sft': 'SFTConfig',
    }
    
    @classmethod
    def get_backend_config_class(cls, algorithm: str):
        """Get the TRL/backend config class for an algorithm."""
        try:
            from trl import GRPOConfig, DPOConfig, PPOConfig
            from transformers import TrainingArguments  # For SFT
            
            config_class_name = cls.ALGORITHM_CONFIG_MAP.get(algorithm.lower(), 'GRPOConfig')
            
            config_class_map = {
                'GRPOConfig': GRPOConfig,
                'DPOConfig': DPOConfig,
                'PPOConfig': PPOConfig,
                'SFTConfig': TrainingArguments,
            }
            
            backend_class = config_class_map.get(config_class_name)
            if backend_class:
                logger.debug(f"Algorithm '{algorithm}' → {config_class_name}")
            return backend_class
            
        except ImportError as e:
            logger.error(f"Failed to import backend configs: {e}")
            return None
    
    @classmethod
    def get_valid_params(cls, config_class) -> Set[str]:
        """Extract all valid parameter names from a config class."""
        if config_class is None:
            return set()
        
        valid_params = set()
        
        # Method 1: Get from __init__ signature
        try:
            sig = inspect.signature(config_class.__init__)
            valid_params.update(sig.parameters.keys())
        except Exception as e:
            logger.debug(f"Could not extract params from __init__: {e}")
        
        # Method 2: Get from dataclass fields
        if is_dataclass(config_class):
            try:
                valid_params.update(f.name for f in fields(config_class))
            except Exception as e:
                logger.debug(f"Could not extract dataclass fields: {e}")
        
        # Method 3: Get from class annotations
        try:
            if hasattr(config_class, '__annotations__'):
                valid_params.update(config_class.__annotations__.keys())
        except Exception as e:
            logger.debug(f"Could not extract annotations: {e}")
        
        # Clean up
        valid_params.discard('self')
        valid_params.discard('args')
        valid_params.discard('kwargs')
        valid_params = {p for p in valid_params if not p.startswith('_')}
        
        return valid_params
    
    @classmethod
    def get_already_set_params(cls, backend_config) -> Dict[str, Any]:
        """
        Get params already set in backend_config with their values.
        
        Returns:
            Dict mapping param_name -> value for all non-None params
        """
        if backend_config is None:
            return {}
        
        already_set = {}
        
        # Get all attributes from backend_config
        if is_dataclass(backend_config):
            # For dataclass, get all fields with non-None values
            for field_info in fields(backend_config):
                field_name = field_info.name
                field_value = getattr(backend_config, field_name, None)
                
                # Include if not None and not empty
                if field_value is not None and field_value != {} and field_value != ():
                    already_set[field_name] = field_value
        else:
            # For regular class, get all non-None attributes
            for attr_name in dir(backend_config):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(backend_config, attr_name, None)
                        if attr_value is not None and not callable(attr_value):
                            already_set[attr_name] = attr_value
                    except:
                        pass
        
        return already_set
    
    @classmethod
    def extract_from_config_train(cls, config) -> Dict[str, Any]:
        """
        Extract ALL parameters from config.train (excluding extra_params).
        
        Returns:
            Dict of all params in config.train
        """
        params = {}
        
        if config is None or not hasattr(config, 'train'):
            return params
        
        train_config = config.train
        
        if is_dataclass(train_config):
            for field in fields(train_config):
                field_name = field.name
                field_value = getattr(train_config, field_name, None)
                
                # Skip None, empty values, and extra_params field itself
                if (field_value is None or 
                    field_value == {} or 
                    field_value == () or 
                    field_name == 'extra_params'):
                    continue
                
                params[field_name] = field_value
        
        logger.debug(f"Extracted {len(params)} params from config.train")
        return params
    
    @classmethod
    def extract_extra_params(cls, config) -> Dict[str, Any]:
        """
        Extract from config.train.extra_params.
        
        Returns:
            Dict of extra_params
        """
        extra_params = {}
        
        if config is None:
            return extra_params
        
        # Check config.train.extra_params
        if hasattr(config, 'train') and hasattr(config.train, 'extra_params'):
            if config.train.extra_params:
                extra_params.update(config.train.extra_params)
                logger.debug(f"Found {len(config.train.extra_params)} params in config.train.extra_params")
        
        return extra_params
    
    @classmethod
    def extract_extra_and_missing_params(
        cls,
        backend_config,
        config=None,
        algorithm: str = 'grpo',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract missing params from config.train + config.train.extra_params.
        
        Process:
        1. Get all params from config.train (EXCEPT extra_params field)
        2. Get params from config.train.extra_params
        3. Check which ones are MISSING from backend_config
        4. Merge with kwargs (kwargs override)
        5. Validate against backend config class
        6. Return ONLY the missing params
        
        Args:
            backend_config: Already created backend config (GRPOConfig, etc.)
            config: UnifiedConfig or SFTConfig
            algorithm: Algorithm name for validation
            **kwargs: Runtime kwargs (highest priority)
        
        Returns:
            Dict of params to add to backend_config
        """
        # Step 1: Get params already set in backend_config
        already_set = cls.get_already_set_params(backend_config)
        already_set_names = set(already_set.keys())
        
        logger.debug(f"✓ Already set in backend_config: {len(already_set_names)} params")
        if already_set_names:
            logger.debug(f"  Already set: {sorted(already_set_names)}")
        
        # Step 2: Extract from config.train (all params except extra_params)
        config_train_params = cls.extract_from_config_train(config)
        
        # Step 3: Extract from config.train.extra_params
        extra_params = cls.extract_extra_params(config)
        
        # Step 4: Combine config.train params + extra_params
        all_config_params = {**config_train_params, **extra_params}
        
        # Step 5: Find MISSING params (in config but NOT in backend_config)
        missing_params = {}
        for key, value in all_config_params.items():
            if key not in already_set_names:
                missing_params[key] = value
        
        if missing_params:
            logger.debug(f"✓ Found {len(missing_params)} missing params in config.train")
            logger.debug(f"  Missing: {sorted(missing_params.keys())}")
        
        # Step 6: Merge with kwargs (kwargs override everything)
        final_params = {**missing_params, **kwargs}
        
        if not final_params:
            logger.debug("No missing params found")
            return {}
        
        # Step 7: Validate against backend config class
        backend_config_class = cls.get_backend_config_class(algorithm)
        if backend_config_class is None:
            logger.warning(f"No backend config class found for '{algorithm}', passing all params")
            return final_params
        
        valid_params = cls.get_valid_params(backend_config_class)
        
        # Filter to only valid params
        validated_params = {}
        invalid_params = []
        
        for key, value in final_params.items():
            if key in valid_params:
                validated_params[key] = value
            else:
                invalid_params.append(key)
        
        if invalid_params:
            logger.debug(
                f"⚠️  Filtered out {len(invalid_params)} invalid params for {algorithm}: "
                f"{sorted(invalid_params)}"
            )
        
        if validated_params:
            logger.info(
                f"✓ Found {len(validated_params)} missing params to add to {backend_config_class.__name__}"
            )
            logger.info(f"  Params to add: {sorted(validated_params.keys())}")
        
        return validated_params


# Convenience function
def extract_extra_and_missing_params(
    backend_config,
    config=None,
    algorithm: str = 'grpo',
    **kwargs
) -> Dict[str, Any]:
    """
    Extract missing params from config.train + config.train.extra_params.
    
    Usage:
        # Manual extraction
        grpo_config = GRPOConfig(
            num_train_epochs=3,
            learning_rate=2e-4,
            # ... manual params ...
        )
        
        # Get missing params from config.train
        missing = extract_extra_and_missing_params(
            backend_config=grpo_config,
            config=self.config,
            algorithm='grpo',
            **self.kwargs
        )
        
        # Add missing params
        for key, value in missing.items():
            setattr(grpo_config, key, value)
    """
    return ConfigExtractor.extract_extra_and_missing_params(
        backend_config=backend_config,
        config=config,
        algorithm=algorithm,
        **kwargs
    )

