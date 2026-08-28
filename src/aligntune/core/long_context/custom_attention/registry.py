"""
Attention implementation registry utilities for AlignTune.

This module provides centralized registration and management of custom
attention implementations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AttentionRegistry:
    """
    Centralized registry for managing custom attention implementations.

    Tracks which attention implementations have been registered to avoid
    duplicate registrations and provides status information.
    """

    _registered_implementations = set()

    @classmethod
    def is_registered(cls, implementation: str) -> bool:
        """
        Check if an attention implementation has been registered.

        Args:
            implementation: Name of the attention implementation (e.g., "swa", "s2")

        Returns:
            True if already registered, False otherwise
        """
        return implementation in cls._registered_implementations

    @classmethod
    def mark_registered(cls, implementation: str) -> None:
        """
        Mark an attention implementation as registered.

        Args:
            implementation: Name of the attention implementation
        """
        cls._registered_implementations.add(implementation)
        logger.debug(f"Marked attention implementation '{implementation}' as registered")

    @classmethod
    def get_registered(cls) -> set:
        """
        Get all registered attention implementations.

        Returns:
            Set of registered implementation names
        """
        return cls._registered_implementations.copy()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._registered_implementations.clear()


def register_attention_implementation(
    name: str,
    register_func: callable,
    force: bool = False
) -> bool:
    """
    Register a custom attention implementation with safety checks.

    Args:
        name: Name of the attention implementation (e.g., "swa")
        register_func: Function that performs the actual registration
        force: If True, register even if already registered

    Returns:
        True if registration was performed, False if skipped

    Example:
        >>> from aligntune.core.long_context.attention.sliding_window import register_sliding_window_attention
        >>> register_attention_implementation(
        ...     "swa",
        ...     register_sliding_window_attention
        ... )
    """
    if AttentionRegistry.is_registered(name) and not force:
        logger.debug(f"Attention implementation '{name}' already registered, skipping")
        return False

    try:
        register_func()
        AttentionRegistry.mark_registered(name)
        logger.info(f"Successfully registered attention implementation: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to register attention implementation '{name}': {e}")
        raise


__all__ = [
    "AttentionRegistry",
    "register_attention_implementation",
]
