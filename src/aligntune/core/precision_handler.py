"""
Unified Precision Handler for AlignTune

This module provides consistent precision handling across all backends.
"""

import torch
import logging
from enum import Enum
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class PrecisionType(Enum):
    """Supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"
    INT8 = "int8"  # 8-bit quantization (bitsandbytes)
    FP8 = "fp8"    # 8-bit floating point (bitsandbytes paged)


class PrecisionHandler:
    """Handles precision configuration consistently across all backends."""
    
    @staticmethod
    def get_precision_from_config(
        config: Any, 
        default: str = "auto"
    ) -> str:
        """
        Extract precision from config with multiple fallback strategies.
        
        Args:
            config: Config object (can be dict or object)
            default: Default precision if not found
            
        Returns:
            Precision string: "fp32", "fp16", "bf16", or "auto"
        """
        # Try multiple attribute names for flexibility
        precision_attrs = ['precision', 'dtype', 'torch_dtype', 'model_precision']
        
        for attr in precision_attrs:
            # Try as dict
            if isinstance(config, dict):
                value = config.get(attr)
            # Try as object attribute
            elif hasattr(config, attr):
                value = getattr(config, attr)
            else:
                continue
            
            if value is not None:
                # Handle enum
                if hasattr(value, 'value'):
                    return value.value.lower()
                # Handle string
                elif isinstance(value, str):
                    return value.lower()
        
        # Try nested model config
        if hasattr(config, 'model'):
            return PrecisionHandler.get_precision_from_config(config.model, default)
        
        return default.lower()
    
    @staticmethod
    def get_torch_dtype(precision: str) -> torch.dtype:
        """
        Convert precision string to torch dtype.

        Args:
            precision: "fp32", "fp16", "bf16", "int8", "fp8", or "auto"

        Returns:
            torch.dtype
        """
        precision = precision.lower()

        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        elif precision == "fp32":
            return torch.float32
        elif precision in ["int8", "fp8"]:
            # 8-bit quantization is handled separately via bitsandbytes,
            # so we return float32 as the base dtype
            return torch.float32
        elif precision == "auto":
            # Auto-detect based on CUDA availability
            if torch.cuda.is_available():
                # Check if bf16 is supported (Ampere+ GPUs)
                if torch.cuda.is_bf16_supported():
                    logger.info("Auto-detected bf16 support, using bfloat16")
                    return torch.bfloat16
                else:
                    logger.info("bf16 not supported, using float16")
                    return torch.float16
            else:
                logger.info("CUDA not available, using float32")
                return torch.float32
        else:
            logger.warning(f"Unknown precision '{precision}', using fp32")
            return torch.float32
    
    @staticmethod
    def get_training_args_precision(precision: str) -> Dict[str, bool]:
        """
        Get training arguments for fp16/bf16 flags.
        
        Args:
            precision: "fp32", "fp16", "bf16", or "auto"
            
        Returns:
            Dict with 'fp16' and 'bf16' boolean flags
        """
        precision = precision.lower()
        
        if precision == "auto":
            # Auto-detect
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return {"fp16": False, "bf16": True}
                else:
                    return {"fp16": True, "bf16": False}
            else:
                return {"fp16": False, "bf16": False}
        
        return {
            "fp16": (precision == "fp16"),
            "bf16": (precision == "bf16")
        }
    
    @staticmethod
    def validate_precision(precision: str) -> str:
        """
        Validate precision string and provide helpful error.

        Args:
            precision: Precision string to validate

        Returns:
            Validated precision string

        Raises:
            ValueError: If precision is invalid
        """
        valid = ["fp32", "fp16", "bf16", "int8", "fp8", "auto"]
        precision = precision.lower()

        if precision not in valid:
            raise ValueError(
                f"Invalid precision '{precision}'. "
                f"Valid options: {valid}"
            )

        # Warn if bf16 requested but not supported
        if precision == "bf16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                logger.warning(
                    "⚠️  bf16 requested but not supported by GPU. "
                    "Falling back to fp16."
                )
                return "fp16"

        # Warn if int8/fp8 requested but bitsandbytes not available
        if precision in ["int8", "fp8"]:
            try:
                import bitsandbytes
                logger.info(f"8-bit quantization ({precision}) will use bitsandbytes paged AdamW optimizer")
            except ImportError:
                logger.warning(
                    f"⚠️  {precision} requested but bitsandbytes not installed. "
                    "Install with: pip install bitsandbytes"
                )

        return precision
    
    @staticmethod
    def log_precision_info(precision: str, backend: str = "unknown"):
        """Log precision configuration info."""
        dtype = PrecisionHandler.get_torch_dtype(precision)
        logger.info("=" * 60)
        logger.info(f"PRECISION CONFIGURATION - {backend.upper()} Backend")
        logger.info("=" * 60)
        logger.info(f"  Requested: {precision}")
        logger.info(f"  PyTorch dtype: {dtype}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
        logger.info("=" * 60)
    
    @staticmethod
    def get_model_load_kwargs(
        precision: str,
        device_map: Optional[Union[str, Dict]] = "auto",
        quantization_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get complete model loading kwargs with precision.

        Args:
            precision: Precision string
            device_map: Device map for model
            quantization_config: Optional quantization config

        Returns:
            Dict of kwargs for model loading
        """
        kwargs = {
            "torch_dtype": PrecisionHandler.get_torch_dtype(precision),
            "device_map": device_map or "auto",
        }

        if quantization_config:
            # If quantizing, dtype might be overridden by quantization
            if quantization_config.get("load_in_4bit") or quantization_config.get("load_in_8bit"):
                logger.info("Quantization enabled, precision will be handled by quantization config")

        return kwargs

    @staticmethod
    def get_8bit_optimizer_kwargs(precision: str) -> Dict[str, Any]:
        """
        Get optimizer kwargs for 8-bit quantization (bitsandbytes paged optimizer).

        Args:
            precision: Precision string ("int8" or "fp8")

        Returns:
            Dict of optimizer kwargs for TrainingArguments
        """
        if precision not in ["int8", "fp8"]:
            return {}

        # Configure 8-bit optimizer state with bitsandbytes
        # Using paged AdamW for better memory efficiency
        optimizer_kwargs = {
            "optim": "paged_adamw_8bit",  # Paged 8-bit AdamW from bitsandbytes
            "optim_target_modules": ["linear"],  # Apply 8-bit to linear layers
            "optim_args": "eps=1e-8",  # Optional: tuning parameters
        }

        logger.info(f"8-bit optimizer state configured for {precision} precision")
        return optimizer_kwargs

    @staticmethod
    def _apply_per_expert_quantization(
        model: Any,
        expert_bitwidth_config: Dict[str, Any]
    ) -> None:
        """
        Apply per-expert quantization to MoE model.

        This method applies different quantization bitwidths to different experts,
        allowing critical experts to use higher precision (8-bit) while other
        experts use lower precision (4-bit) to save memory.

        Router networks always remain in fp16 (not quantized) to maintain routing quality.

        Args:
            model: The MoE model to quantize
            expert_bitwidth_config: Configuration dict with format:
                {
                    'critical_experts': [list of expert indices],  # e.g., [0, 1]
                    'bitwidth': int,  # bitwidth for critical experts (e.g., 8)
                    'default': int,  # bitwidth for other experts (e.g., 4)
                }

        Example:
            >>> config = {
            ...     'critical_experts': [0, 1],  # First two experts at 8-bit
            ...     'bitwidth': 8,
            ...     'default': 4,
            ... }
            >>> PrecisionHandler._apply_per_expert_quantization(model, config)

        Note:
            This is a convenience method that applies quantization configuration.
            Actual quantization is performed by bitsandbytes or similar backends.
            This method primarily marks which experts should receive which bitwidth.
        """
        critical_experts = expert_bitwidth_config.get('critical_experts', [])
        critical_bitwidth = expert_bitwidth_config.get('bitwidth', 8)
        default_bitwidth = expert_bitwidth_config.get('default', 4)

        logger.info("=" * 70)
        logger.info("APPLYING PER-EXPERT QUANTIZATION")
        logger.info("=" * 70)

        # Find expert modules in the model
        expert_modules = []
        for name, module in model.named_modules():
            # Match common MoE expert naming patterns
            if any(pattern in name.lower() for pattern in ['expert', 'moe_layer']):
                expert_modules.append((name, module))

        if not expert_modules:
            logger.warning("No expert modules found in model. Skipping per-expert quantization.")
            return

        logger.info(f"Found {len(expert_modules)} expert modules")
        logger.info(f"Critical experts (using {critical_bitwidth}-bit): {critical_experts}")
        logger.info(f"Other experts (using {default_bitwidth}-bit): auto-assigned")

        # Mark each expert with its target bitwidth
        for idx, (name, module) in enumerate(expert_modules):
            if idx in critical_experts:
                bitwidth = critical_bitwidth
                marker = "CRITICAL"
            else:
                bitwidth = default_bitwidth
                marker = "STANDARD"

            # Store bitwidth as attribute for downstream quantization
            setattr(module, '_target_bitwidth', bitwidth)
            logger.info(f"  Expert {idx} ({name}): {bitwidth}-bit [{marker}]")

        # Ensure router networks are never quantized (keep fp16)
        for name, module in model.named_modules():
            if any(pattern in name.lower() for pattern in ['router', 'gate', 'selector']):
                setattr(module, '_target_bitwidth', 16)  # fp16
                logger.info(f"  Router ({name}): fp16 [UNQUANTIZED]")

        logger.info("=" * 70)
        logger.info("Per-expert quantization configuration applied successfully")
        logger.info("=" * 70)