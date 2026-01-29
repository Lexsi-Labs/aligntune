"""
Registry for reward functions.

This module provides a centralized registry for all reward functions,
making it easy to register, discover, and use reward functions in RL training.
"""

from typing import Dict, List, Optional, Callable, Union
import logging
from .core import RewardType, RewardConfig, RewardFunction, RewardFunctionFactory

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class RewardRegistry:
    """Central registry for reward functions."""
    
    _registered_rewards: Dict[str, RewardType] = {}
    _reward_configs: Dict[str, RewardConfig] = {}
    _custom_rewards: Dict[str, Callable] = {}
    
    @classmethod
    def register_reward_type(cls, name: str, reward_type: RewardType, default_config: Optional[RewardConfig] = None):
        """Register a reward type with optional default configuration."""
        cls._registered_rewards[name] = reward_type
        if default_config:
            cls._reward_configs[name] = default_config
        logger.info(f"Registered reward type: {name}")
    
    @classmethod
    def register_custom_reward(cls, name: str, reward_func: Callable, config: Optional[RewardConfig] = None):
        """Register a custom reward function."""
        cls._custom_rewards[name] = reward_func
        if config:
            cls._reward_configs[name] = config
        logger.info(f"Registered custom reward: {name}")
    
    @classmethod
    def _dict_to_config(cls, config_dict: dict) -> RewardConfig:
        """Convert dict config to RewardConfig."""
        reward_type_str = config_dict.get('type', '')
        if not reward_type_str:
            raise ValueError("Reward type is required in config dict")
        
        # Normalize the reward type string
        # Handle cases like "RewardType.MATH_REASONING" -> "MATH_REASONING"
        if '.' in reward_type_str:
            reward_type_str = reward_type_str.split('.')[-1]
        
        # Handle truncated strings like "RewardT" -> try to match "RewardType" enum members
        reward_type_str = reward_type_str.strip()
        original_str = reward_type_str
        reward_type_str = reward_type_str.upper()
        
        # Try multiple parsing strategies
        reward_type = None
        
        # Strategy 1: Try direct enum name match (e.g., "MATH_REASONING")
        try:
            reward_type = RewardType[reward_type_str]
        except KeyError:
            pass
        
        # Strategy 2: Try enum value match (e.g., "math_reasoning" -> MATH_REASONING)
        if reward_type is None:
            for rt in RewardType:
                if rt.value.upper() == reward_type_str:
                    reward_type = rt
                    break
        
        # Strategy 3: Try partial match for truncated strings (e.g., "RewardT" -> try to find matches)
        if reward_type is None and len(reward_type_str) < 8:  # Likely truncated
            matches = [rt for rt in RewardType if rt.name.startswith(reward_type_str)]
            if len(matches) == 1:
                reward_type = matches[0]
            elif len(matches) > 1:
                # Multiple matches, log warning but don't fail
                logger.warning(
                    f"Reward type '{original_str}' is ambiguous (matches {[m.name for m in matches]}). "
                    f"Using first match: {matches[0].name}"
                )
                reward_type = matches[0]
        
        # Strategy 4: Try case-insensitive partial match on enum values
        if reward_type is None:
            for rt in RewardType:
                if reward_type_str in rt.value.upper() or rt.value.upper().startswith(reward_type_str):
                    reward_type = rt
                    break
        
        if reward_type is None:
            available = [t.name for t in RewardType]
            raise ValueError(
                f"Unknown reward type: '{original_str}'. Available: {available}. "
                f"Note: If you see 'RewardT' or similar, the reward type string may be truncated."
            )
        
        return RewardConfig(
            reward_type=reward_type,
            weight=config_dict.get('weight', 1.0),
            params=config_dict.get('params', {}),
            model_name=config_dict.get('model_name')
        )
    
    @classmethod
    def get_reward_function(cls, name: str, config: Optional[Union[RewardConfig, dict]] = None) -> RewardFunction:
        """Get a reward function by name with flexible config support."""
        # Convert dict config to RewardConfig if needed
        if isinstance(config, dict):
            config = cls._dict_to_config(config)
        
        if name in cls._custom_rewards:
            # Custom reward function
            reward_func = cls._custom_rewards[name]
            if config is None:
                config = cls._reward_configs.get(name)
            if config is None:
                raise ValueError(f"No configuration provided for custom reward: {name}")
            return reward_func(config)
        
        elif name in cls._registered_rewards:
            # Standard reward type
            reward_type = cls._registered_rewards[name]
            if config is None:
                config = cls._reward_configs.get(name, RewardConfig(reward_type=reward_type))
            return RewardFunctionFactory.create_reward(config)
        
        else:
            raise ValueError(f"Unknown reward: {name}. Available: {list(cls._registered_rewards.keys()) + list(cls._custom_rewards.keys())}")
    
    @classmethod
    def list_rewards(cls) -> List[str]:
        """List all registered reward functions."""
        return list(cls._registered_rewards.keys()) + list(cls._custom_rewards.keys())
    
    @classmethod
    def get_reward_config(cls, name: str) -> Optional[RewardConfig]:
        """Get default configuration for a reward function."""
        return cls._reward_configs.get(name)
    
    @classmethod
    def create_composite_reward(cls, reward_names: List[str], weights: Optional[List[float]] = None):
        """Create a composite reward from multiple reward functions."""
        from .core import CompositeReward
        
        reward_functions = []
        for name in reward_names:
            reward_func = cls.get_reward_function(name)
            reward_functions.append(reward_func)
        
        return CompositeReward(reward_functions, weights)


# Initialize default reward types
def _initialize_default_rewards():
    """Initialize default reward types."""
    
    # ========================================================================
    # BASIC REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "length",
        RewardType.LENGTH,
        RewardConfig(RewardType.LENGTH, weight=0.1, params={"min_length": 10, "max_length": 500})
    )
    
    RewardRegistry.register_reward_type(
        "coherence",
        RewardType.COHERENCE,
        RewardConfig(RewardType.COHERENCE, weight=0.2)
    )
    
    # ========================================================================
    # QUALITY REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "sentiment",
        RewardType.SENTIMENT,
        RewardConfig(RewardType.SENTIMENT, weight=0.3, params={"target_sentiment": "positive"})
    )
    
    RewardRegistry.register_reward_type(
        "safety",
        RewardType.SAFETY,
        RewardConfig(RewardType.SAFETY, weight=0.4)
    )
    
    RewardRegistry.register_reward_type(
        "toxicity",
        RewardType.TOXICITY,
        RewardConfig(RewardType.TOXICITY, weight=0.3)
    )
    
    RewardRegistry.register_reward_type(
        "factuality",
        RewardType.FACTUALITY,
        RewardConfig(RewardType.FACTUALITY, weight=0.3)
    )
    
    RewardRegistry.register_reward_type(
        "bias",
        RewardType.BIAS,
        RewardConfig(RewardType.BIAS, weight=0.3)
    )
    
    # ========================================================================
    # GENERATION QUALITY REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "bleu",
        RewardType.BLEU,
        RewardConfig(RewardType.BLEU, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "rouge",
        RewardType.ROUGE,
        RewardConfig(RewardType.ROUGE, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "meteor",
        RewardType.METEOR,
        RewardConfig(RewardType.METEOR, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "bertscore",
        RewardType.BERTSCORE,
        RewardConfig(RewardType.BERTSCORE, weight=0.5)
    )
    
    # ========================================================================
    # MATH AND REASONING REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "math_correctness",
        RewardType.MATH_CORRECTNESS,
        RewardConfig(RewardType.MATH_CORRECTNESS, weight=0.6)
    )
    
    RewardRegistry.register_reward_type(
        "math_reasoning",
        RewardType.MATH_REASONING,
        RewardConfig(
            RewardType.MATH_REASONING, 
            weight=0.6,
            params={
                "check_correctness": True,
                "check_reasoning": True,
                "check_format": True
            }
        )
    )
    
    RewardRegistry.register_reward_type(
        "logical_consistency",
        RewardType.LOGICAL_CONSISTENCY,
        RewardConfig(RewardType.LOGICAL_CONSISTENCY, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "commonsense",
        RewardType.COMMONSENSE,
        RewardConfig(RewardType.COMMONSENSE, weight=0.4)
    )
    
    # ========================================================================
    # CODE REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "code_syntax",
        RewardType.CODE_SYNTAX,
        RewardConfig(RewardType.CODE_SYNTAX, weight=0.6)
    )
    
    RewardRegistry.register_reward_type(
        "code_quality",
        RewardType.CODE_QUALITY,
        RewardConfig(
            RewardType.CODE_QUALITY,
            weight=0.3,
            params={
                "check_syntax": True,
                "check_style": True,
                "check_comments": True,
                "check_efficiency": True,
                "language": "python"
            }
        )
    )
    
    RewardRegistry.register_reward_type(
        "code_execution",
        RewardType.CODE_EXECUTION,
        RewardConfig(RewardType.CODE_EXECUTION, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "code_completeness",
        RewardType.CODE_COMPLETENESS,
        RewardConfig(RewardType.CODE_COMPLETENESS, weight=0.4)
    )
    
    RewardRegistry.register_reward_type(
        "code_correctness",
        RewardType.CODE_CORRECTNESS,
        RewardConfig(
            RewardType.CODE_CORRECTNESS,
            weight=0.3,
            params={
                "test_cases": [],
                "timeout": 5.0
            }
        )
    )
    
    # ========================================================================
    # SPECIALIZED REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "hallucination",
        RewardType.HALLUCINATION,
        RewardConfig(RewardType.HALLUCINATION, weight=0.4)
    )
    
    RewardRegistry.register_reward_type(
        "politeness",
        RewardType.POLITENESS,
        RewardConfig(RewardType.POLITENESS, weight=0.3)
    )
    
    RewardRegistry.register_reward_type(
        "helpfulness",
        RewardType.HELPFULNESS,
        RewardConfig(RewardType.HELPFULNESS, weight=0.4)
    )
    
    RewardRegistry.register_reward_type(
        "honesty",
        RewardType.HONESTY,
        RewardConfig(RewardType.HONESTY, weight=0.3)
    )
    
    # ========================================================================
    # NEW: ENHANCED QUALITY REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "diversity",
        RewardType.DIVERSITY,
        RewardConfig(
            RewardType.DIVERSITY,
            weight=0.2,
            params={
                "check_vocabulary": True,
                "check_sentence_variety": True
            }
        )
    )
    
    RewardRegistry.register_reward_type(
        "fluency",
        RewardType.FLUENCY,
        RewardConfig(
            RewardType.FLUENCY,
            weight=0.2,
            params={
                "check_grammar": True,
                "check_coherence": True
            }
        )
    )
    
    RewardRegistry.register_reward_type(
        "relevance",
        RewardType.RELEVANCE,
        RewardConfig(
            RewardType.RELEVANCE,
            weight=0.2,
            params={"keywords": []}
        )
    )
    
    RewardRegistry.register_reward_type(
        "brevity",
        RewardType.BREVITY,
        RewardConfig(
            RewardType.BREVITY,
            weight=0.2,
            params={
                "ideal_length": 100,
                "max_length": 300
            }
        )
    )
    
    # ========================================================================
    # MULTI-MODAL REWARDS (Placeholders for future implementation)
    # ========================================================================
    RewardRegistry.register_reward_type(
        "image_relevance",
        RewardType.IMAGE_RELEVANCE,
        RewardConfig(RewardType.IMAGE_RELEVANCE, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "audio_quality",
        RewardType.AUDIO_QUALITY,
        RewardConfig(RewardType.AUDIO_QUALITY, weight=0.5)
    )
    
    # ========================================================================
    # NEW: INSTRUCTION FOLLOWING & ALIGNMENT
    # ========================================================================
    RewardRegistry.register_reward_type(
        "instruction_following",
        RewardType.INSTRUCTION_FOLLOWING,
        RewardConfig(RewardType.INSTRUCTION_FOLLOWING, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "harmlessness",
        RewardType.HARMLESSNESS,
        RewardConfig(RewardType.HARMLESSNESS, weight=0.4)
    )
    
    RewardRegistry.register_reward_type(
        "conciseness",
        RewardType.CONCISENESS,
        RewardConfig(RewardType.CONCISENESS, weight=0.3)
    )
    
    # ========================================================================
    # NEW: CONTEXT & TEMPORAL AWARENESS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "context_relevance",
        RewardType.CONTEXT_RELEVANCE,
        RewardConfig(RewardType.CONTEXT_RELEVANCE, weight=0.4)
    )
    
    RewardRegistry.register_reward_type(
        "temporal_consistency",
        RewardType.TEMPORAL_CONSISTENCY,
        RewardConfig(RewardType.TEMPORAL_CONSISTENCY, weight=0.3)
    )
    
    # ========================================================================
    # NEW: ADVANCED QUALITY METRICS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "semantic_similarity",
        RewardType.SEMANTIC_SIMILARITY,
        RewardConfig(RewardType.SEMANTIC_SIMILARITY, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "readability",
        RewardType.READABILITY,
        RewardConfig(RewardType.READABILITY, weight=0.3)
    )
    
    RewardRegistry.register_reward_type(
        "engagement",
        RewardType.ENGAGEMENT,
        RewardConfig(RewardType.ENGAGEMENT, weight=0.3)
    )
    
    # ========================================================================
    # NEW: DOMAIN-SPECIFIC REWARDS
    # ========================================================================
    RewardRegistry.register_reward_type(
        "medical_accuracy",
        RewardType.MEDICAL_ACCURACY,
        RewardConfig(RewardType.MEDICAL_ACCURACY, weight=0.6)
    )
    
    RewardRegistry.register_reward_type(
        "legal_compliance",
        RewardType.LEGAL_COMPLIANCE,
        RewardConfig(RewardType.LEGAL_COMPLIANCE, weight=0.6)
    )
    
    RewardRegistry.register_reward_type(
        "financial_accuracy",
        RewardType.FINANCIAL_ACCURACY,
        RewardConfig(RewardType.FINANCIAL_ACCURACY, weight=0.6)
    )
    
    # ========================================================================
    # NEW: ADVANCED REASONING
    # ========================================================================
    RewardRegistry.register_reward_type(
        "causal_reasoning",
        RewardType.CAUSAL_REASONING,
        RewardConfig(RewardType.CAUSAL_REASONING, weight=0.5)
    )
    
    RewardRegistry.register_reward_type(
        "counterfactual_reasoning",
        RewardType.COUNTERFACTUAL_REASONING,
        RewardConfig(RewardType.COUNTERFACTUAL_REASONING, weight=0.5)
    )
    # ========================================================================
    # NEW: COUNTERFACTUAL MATH REWARD (Process-Level)
    # ========================================================================
    RewardRegistry.register_reward_type(
        "counterfactual_math",
        RewardType.COUNTERFACTUAL_MATH,
        RewardConfig(
            RewardType.COUNTERFACTUAL_MATH,
            weight=1.0,
            params={}
        )
    )
    RewardRegistry.register_reward_type(
        "mbpp_reward",
        RewardType.MBPP_REWARD,
        RewardConfig(
            RewardType.MBPP_REWARD,
            weight=1.0,
            params={}
        )
    )


# Initialize on import
_initialize_default_rewards()