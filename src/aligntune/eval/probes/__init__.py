"""
Probe set management for alignment auditing.

Provides utilities to load pre-built probe sets for:
- Refusal testing
- Sycophancy detection
- Verbosity analysis
- Reward hacking (reserved for future)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

PROBE_DIR = Path(__file__).parent


def load_refusal_probes() -> List[Dict[str, Any]]:
    """Load refusal test probes from JSONL file."""
    return _load_jsonl(PROBE_DIR / "refusal_probes.jsonl")


def load_sycophancy_probes() -> List[Dict[str, Any]]:
    """Load sycophancy detection probes from JSONL file."""
    return _load_jsonl(PROBE_DIR / "sycophancy_probes.jsonl")


def load_verbosity_probes() -> List[Dict[str, Any]]:
    """Load verbosity analysis probes from JSONL file."""
    return _load_jsonl(PROBE_DIR / "verbosity_probes.jsonl")


def load_reward_hacking_probes() -> List[Dict[str, Any]]:
    """
    Load reward hacking detection probes.

    Reserved for future integration with per-component reward tracking.
    """
    reward_hacking_path = PROBE_DIR / "reward_hacking_probes.jsonl"
    if not reward_hacking_path.exists():
        logger.debug("reward_hacking_probes.jsonl not found; reward hacking detection requires J integration")
        return []
    return _load_jsonl(reward_hacking_path)


def load_routing_safety_probes() -> List[Dict[str, Any]]:
    """
    Load routing safety probes for MoE alignment auditing.

    These probes are designed to test whether the router makes different
    decisions for semantically similar safe vs unsafe prompts, enabling
    detection of routing-based safety failures.
    """
    routing_safety_path = Path(__file__).parent.parent.parent / "recipes" / "probe_sets" / "routing_safety.jsonl"
    if not routing_safety_path.exists():
        logger.debug(f"routing_safety_probes not found at {routing_safety_path}")
        return []
    return _load_jsonl(routing_safety_path)


def load_all_probe_sets() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all available probe sets.

    Returns:
        Dict with keys: "refusal", "sycophancy", "verbosity", "reward_hacking", "routing_safety"
    """
    return {
        "refusal": load_refusal_probes(),
        "sycophancy": load_sycophancy_probes(),
        "verbosity": load_verbosity_probes(),
        "reward_hacking": load_reward_hacking_probes(),
        "routing_safety": load_routing_safety_probes(),
    }


def load_custom_probes(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load probes from a custom JSONL file.

    Args:
        filepath: Path to JSONL file with probe dicts

    Returns:
        List of probe dictionaries
    """
    return _load_jsonl(filepath)


def _load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dicts."""
    if not filepath.exists():
        logger.warning(f"Probe file not found: {filepath}")
        return []

    probes = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    probes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSONL line in {filepath}: {e}")
    except Exception as e:
        logger.error(f"Error loading probes from {filepath}: {e}")
        return []

    logger.info(f"Loaded {len(probes)} probes from {filepath}")
    return probes
