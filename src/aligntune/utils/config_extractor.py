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
from dataclasses import fields, is_dataclass, MISSING
import inspect

logger = logging.getLogger(__name__)


class ConfigExtractor:
    """Extractor focused on config.train + extra_params."""
    
    # Algorithm to config class mapping. Every RL/SFT algorithm the backend
    # factory registers for the TRL backend must have an entry here -- an
    # algorithm missing from this map silently falls back to 'GRPOConfig'
    # (see get_backend_config_class below), which means its own genuinely
    # valid params (e.g. KTO's desirable_weight) get rejected as "invalid"
    # during validation, on top of never being looked up correctly.
    ALGORITHM_CONFIG_MAP = {
        # GRPO and variants -- all share trl.GRPOConfig.
        'grpo': 'GRPOConfig',
        'gspo': 'GRPOConfig',
        'counterfact_grpo': 'GRPOConfig',
        'gbmpo': 'GRPOConfig',
        'drgrpo': 'GRPOConfig',
        'dapo': 'GRPOConfig',
        'pace': 'GRPOConfig',
        'nmgrpo': 'GRPOConfig',
        'bolt': 'GRPOConfig',

        # DPO and DPO-based algorithms.
        'dpo': 'DPOConfig',
        'spin': 'DPOConfig',

        # PPO-based algorithms.
        'ppo': 'PPOConfig',
        'cgpo': 'PPOConfig',

        # Everything else gets its own real TRL config class.
        'kto': 'KTOConfig',
        'rloo': 'RLOOConfig',
        'orpo': 'ORPOConfig',
        'simpo': 'CPOConfig',
        'online_dpo': 'OnlineDPOConfig',

        # SFT.
        'sft': 'SFTConfig',

        # Distillation trainers.
        'distillation': 'DistillationConfig',
        'gold': 'GOLDConfig',
        'sdft': 'SDFTConfig',
        'sdpo': 'SDPOConfig',
    }

    #: Cached {class_name: class_or_None}, built lazily so one missing/renamed
    #: TRL class (e.g. after a version bump) can't break every other lookup.
    _config_class_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _build_config_class_map(cls) -> Dict[str, Any]:
        import os
        os.environ.setdefault('TRL_EXPERIMENTAL_SILENCE', '1')

        importers = {
            'GRPOConfig':    lambda: __import__('trl', fromlist=['GRPOConfig']).GRPOConfig,
            'DPOConfig':     lambda: __import__('trl', fromlist=['DPOConfig']).DPOConfig,
            'KTOConfig':     lambda: __import__('trl', fromlist=['KTOConfig']).KTOConfig,
            'RLOOConfig':    lambda: __import__('trl', fromlist=['RLOOConfig']).RLOOConfig,
            'SFTConfig':     lambda: __import__('trl', fromlist=['SFTConfig']).SFTConfig,
            'PPOConfig':     lambda: __import__('trl.experimental.ppo', fromlist=['PPOConfig']).PPOConfig,
            'ORPOConfig':    lambda: __import__('trl.experimental.orpo', fromlist=['ORPOConfig']).ORPOConfig,
            'CPOConfig':     lambda: __import__('trl.experimental.cpo', fromlist=['CPOConfig']).CPOConfig,
            'OnlineDPOConfig': lambda: __import__(
                'trl.experimental.online_dpo', fromlist=['OnlineDPOConfig']
            ).OnlineDPOConfig,
            'DistillationConfig': lambda: __import__(
                'trl.experimental.distillation', fromlist=['DistillationConfig']
            ).DistillationConfig,
            'GOLDConfig': lambda: __import__(
                'trl.experimental.gold', fromlist=['GOLDConfig']
            ).GOLDConfig,
            'SDFTConfig': lambda: __import__(
                'trl.experimental.sdft', fromlist=['SDFTConfig']
            ).SDFTConfig,
            'SDPOConfig': lambda: __import__(
                'trl.experimental.sdpo', fromlist=['SDPOConfig']
            ).SDPOConfig,
        }

        config_class_map = {}
        for name, load in importers.items():
            try:
                config_class_map[name] = load()
            except Exception as e:
                logger.debug(f"Could not import {name}: {e}")
                config_class_map[name] = None
        return config_class_map

    @classmethod
    def get_backend_config_class(cls, algorithm: str):
        """Get the TRL/backend config class for an algorithm."""
        if cls._config_class_cache is None:
            cls._config_class_cache = cls._build_config_class_map()

        config_class_name = cls.ALGORITHM_CONFIG_MAP.get(algorithm.lower(), 'GRPOConfig')
        backend_class = cls._config_class_cache.get(config_class_name)
        if backend_class:
            logger.debug(f"Algorithm '{algorithm}' → {config_class_name}")
        else:
            logger.warning(
                f"Backend config class '{config_class_name}' for algorithm "
                f"'{algorithm}' could not be imported"
            )
        return backend_class
    
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
        Get params deliberately set on backend_config -- i.e. whose current
        value differs from the config class's OWN declared default.

        Why not "any non-None value", as before: TRL's dataclasses fill in
        their own defaults on construction (e.g. GRPOConfig.epsilon=0.2,
        PPOConfig.num_ppo_epochs=4), and essentially none of those are None,
        {} or (). Treating "not None" as "already set" made almost every
        field on every TRL config look pre-set, which permanently blocked
        this function from ever backfilling them -- regardless of what the
        caller had explicitly provided via config.train/extra_params. A field
        left exactly at its class's default is indistinguishable from "never
        touched", so it stays eligible for backfill; a field whose value
        differs from that default was necessarily set on purpose and must be
        protected from being overwritten.

        Returns:
            Dict mapping param_name -> value for genuinely-set params.
        """
        if backend_config is None:
            return {}

        already_set = {}

        # Get all attributes from backend_config
        if is_dataclass(backend_config):
            for field_info in fields(backend_config):
                field_name = field_info.name
                field_value = getattr(backend_config, field_name, None)

                if field_info.default is not MISSING:
                    class_default = field_info.default
                elif field_info.default_factory is not MISSING:  # type: ignore[misc]
                    try:
                        class_default = field_info.default_factory()
                    except Exception:
                        # Can't tell what "untouched" looks like -- be
                        # conservative and treat the current value as set.
                        already_set[field_name] = field_value
                        continue
                else:
                    # No declared default at all: a required field, so any
                    # value present was necessarily passed explicitly.
                    already_set[field_name] = field_value
                    continue

                try:
                    differs = field_value != class_default
                except Exception:
                    differs = field_value is not class_default

                if differs:
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
                # Factory kwargs may contain optional keys with value None.
                # Those mean "not provided" and must not overwrite native
                # backend defaults (for example DPO/SimPO loss_type).
                extra_params.update({
                    key: value
                    for key, value in config.train.extra_params.items()
                    if value is not None and value != {} and value != ()
                })
                logger.debug(f"Found {len(extra_params)} params in config.train.extra_params")
        
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
        3. Fill gaps from config.train using whatever's MISSING from backend_config
        4. Apply config.train.extra_params unconditionally (see note below)
        5. Merge with kwargs (kwargs override everything)
        6. Validate against backend config class
        7. Return the params to add/override on backend_config

        Note on precedence: config.train fields only fill gaps (backend_config
        wins if already set) because some of them are trainer-derived special
        cases, not raw user input. config.train.extra_params always wins
        regardless, because it holds only the leftover kwargs no *named*
        factory parameter consumed -- unambiguously explicit user input, by
        construction (extra_params=kwargs at the factory call site).

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

        # Step 4: Find MISSING params from config.train (gap-fill only -- these
        # can include values a trainer deliberately derived/special-cased from
        # a *named* factory parameter, e.g. num_train_epochs=1 when max_steps
        # is set, so they must not blindly override something already set).
        missing_params = {}
        for key, value in config_train_params.items():
            if key not in already_set_names:
                missing_params[key] = value

        if missing_params:
            logger.debug(f"✓ Found {len(missing_params)} missing params in config.train")
            logger.debug(f"  Missing: {sorted(missing_params.keys())}")

        # Step 5: extra_params ALWAYS win, gap or not. Unlike config_train_params,
        # extra_params is built as extra_params=kwargs at the factory call site --
        # it holds only leftover keyword arguments the caller passed that no
        # *named* factory parameter consumed, so a key present here is
        # unambiguous evidence the caller explicitly asked for it. This matters
        # for fields like GRPOConfig.generation_batch_size/steps_per_generation,
        # whose declared class default is None but whose __post_init__ always
        # computes a concrete value regardless of what was passed -- making
        # them look "already set" to the already_set_names check above even
        # when nothing in aligntune's trainer code ever touched them.
        final_params = {**missing_params, **extra_params, **kwargs}
        
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
