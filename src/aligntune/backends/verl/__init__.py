"""
veRL Backend for AlignTune - High-throughput RLHF via HybridFlow

veRL (https://github.com/volcengine/verl) provides 2-3x throughput improvements
over TRL for PPO/GRPO on large models through HybridFlow architecture
(co-locating actor+rollout on the same GPU).

This backend is optional - AlignTune works without veRL installed.
"""

__all__ = ["VerlBackendBase"]
