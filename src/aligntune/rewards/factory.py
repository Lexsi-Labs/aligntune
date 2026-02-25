"""
Factory for creating reward functions from flexible specifications.

This module provides utility functions to create reward functions from various formats,
making it easy to configure reward functions in different ways.
"""

from typing import List, Union, Optional, Dict, Any
import logging
from .core import RewardConfig, RewardFunction, RewardFunctionFactory
from .registry import RewardRegistry

logger = logging.getLogger(__name__)


def create_reward_functions(
    reward_specs: List[Union[RewardConfig, dict, str]],
    registry: Optional[RewardRegistry] = None
) -> List[RewardFunction]:
    """
    Create reward functions from flexible specifications.
    
    Args:
        reward_specs: List of reward specifications. Can be:
            - RewardConfig objects
            - Dict configs with 'type', 'weight', 'params' keys
            - String names of registered reward functions
        registry: Optional RewardRegistry instance. If None, uses class methods.
    
    Returns:
        List of RewardFunction instances
        
    Examples:
        # Using string names
        reward_functions = create_reward_functions(['length', 'sentiment'])
        
        # Using dict configs
        reward_functions = create_reward_functions([
            {'type': 'length', 'weight': 0.3, 'params': {'min_length': 10}},
            {'type': 'sentiment', 'weight': 0.7, 'params': {'positive_weight': 1.0}}
        ])
        
        # Using RewardConfig objects
        from aligntune.rewards import RewardConfig, RewardType
        reward_functions = create_reward_functions([
            RewardConfig(RewardType.LENGTH, weight=0.3),
            RewardConfig(RewardType.SENTIMENT, weight=0.7)
        ])
    """
    if registry is None:
        registry = RewardRegistry
    
    reward_functions = []
    
    for spec in reward_specs:
        if isinstance(spec, RewardConfig):
            # Direct RewardConfig object
            reward_func = RewardFunctionFactory.create_reward(spec)
            reward_functions.append(reward_func)
            
        elif isinstance(spec, dict):
            # Dict config - convert to RewardConfig
            reward_type_str = spec.get('type', '').upper()
            try:
                from .core import RewardType
                reward_type = RewardType[reward_type_str]
            except KeyError:
                raise ValueError(f"Unknown reward type: {reward_type_str}. Available: {[t.name for t in RewardType]}")
            
            config = RewardConfig(
                reward_type=reward_type,
                weight=spec.get('weight', 1.0),
                params=spec.get('params', {}),
                model_name=spec.get('model_name')
            )
            reward_func = RewardFunctionFactory.create_reward(config)
            reward_functions.append(reward_func)
            
        elif isinstance(spec, str):
            # String name - use registry
            reward_func = registry.get_reward_function(spec)
            reward_functions.append(reward_func)
            
        else:
            raise ValueError(f"Invalid reward specification: {spec}. Must be RewardConfig, dict, or string.")
    
    logger.info(f"Created {len(reward_functions)} reward functions")
    return reward_functions


def create_composite_reward_from_specs(
    reward_specs: List[Union[RewardConfig, dict, str]],
    weights: Optional[List[float]] = None,
    registry: Optional[RewardRegistry] = None
):
    """
    Create a composite reward function from flexible specifications.
    
    Args:
        reward_specs: List of reward specifications (same as create_reward_functions)
        weights: Optional list of weights for each reward function
        registry: Optional RewardRegistry instance
    
    Returns:
        CompositeReward instance
    """
    from .core import CompositeReward
    
    reward_functions = create_reward_functions(reward_specs, registry)
    return CompositeReward(reward_functions, weights)


def validate_reward_specs(reward_specs: List[Union[RewardConfig, dict, str]]) -> List[str]:
    """
    Validate reward specifications and return any errors.
    
    Args:
        reward_specs: List of reward specifications to validate
    
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    for i, spec in enumerate(reward_specs):
        if isinstance(spec, RewardConfig):
            # RewardConfig is always valid
            continue
            
        elif isinstance(spec, dict):
            # Validate dict config
            if 'type' not in spec:
                errors.append(f"Reward spec {i}: Missing 'type' field")
                continue
                
            reward_type_str = spec.get('type', '').upper()
            try:
                from .core import RewardType
                RewardType[reward_type_str]
            except KeyError:
                errors.append(f"Reward spec {i}: Unknown reward type '{reward_type_str}'")
                
            # Validate weight if present
            if 'weight' in spec and not isinstance(spec['weight'], (int, float)):
                errors.append(f"Reward spec {i}: Weight must be a number")
                
        elif isinstance(spec, str):
            # Validate string name
            if not spec.strip():
                errors.append(f"Reward spec {i}: Empty string name")
                continue
                
            # Check if it's registered
            try:
                RewardRegistry.get_reward_function(spec)
            except ValueError as e:
                errors.append(f"Reward spec {i}: {str(e)}")
                
        else:
            errors.append(f"Reward spec {i}: Invalid type {type(spec)}. Must be RewardConfig, dict, or string.")
    
    return errors
