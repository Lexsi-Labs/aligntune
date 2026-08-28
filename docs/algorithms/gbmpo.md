# GBMPO (Group-Based Mirror Policy Optimization)

Group-Based Mirror Policy Optimization (GBMPO) replaces the standard PPO clipping mechanism with trust-region mirror-descent updates built on top of a GRPO-style grouped sampling strategy.

## Overview

GBMPO avoids the complexities of PPO value networks by generating groups of responses and scoring them relatively. It applies a forward KL-divergence constraint directly inside the optimization step to guarantee monotonic policy improvement, often outperforming both PPO and standard GRPO.

### When to use GBMPO?
- When standard GRPO fails due to over-optimization collapses.
- When you want guaranteed stable improvements similar to TRPO/PPO but without the memory footprint of a value network.

As of PR #34, GBMPO was refactored into a single `GBMPOConfig` (extends TRL's
`GRPOConfig`) covering all four divergence variants through one `divergence_type`
field, instead of four separate config classes:

| `divergence_type` | Description |
|---|---|
| `"l2"` (default) | L2 norm regularization in log-space |
| `"l2kl"` | Dual L2 + KL divergence (uses both `l2_coefficient` and `beta`) |
| `"prob_l2"` | L2 norm in probability space |
| `"prob_l2kl"` | Dual probability-space L2 + KL |

## Configuration

To use GBMPO, pass `algorithm="gbmpo"` to the factory method:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="gbmpo",
    backend="trl",  # Supported on Unsloth as well
    reward_functions=["length"],
    reward_function_weights=[1.0],
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-6,
    num_generations=4,
    max_completion_length=256,
    temperature=0.7,
    top_p=0.95,
    beta=0.1,
    epsilon=0.2,
    gbmpo_divergence_type="l2",  # "l2", "l2kl", "prob_l2", or "prob_l2kl"
    gbmpo_l2_coefficient=0.0001,
)

trainer.train()
```

`reward_functions` is required because GBMPO uses the same prompt-and-reward
rollout contract as GRPO. It accepts registry names such as
`"math_correctness"` and TRL-compatible Python callables.
`reward_function_weights` optionally supplies one weight per reward. See the
[reward functions guide](../user-guide/reward-functions.md) for registry and
custom-callable examples.

GBMPO expects prompt-oriented data with a `prompt` column. A ground-truth
column such as `reference`, `answer`, or `solution` is only needed by the
selected reward function and is forwarded to it by the data pipeline.

The trainer first computes the parent GRPO loss, then adds the configured L2
divergence penalty using the reference-policy log probabilities. The TRL and
Unsloth implementations both use this same loss-patching flow.

Note: at the `create_rl_trainer()` factory level the kwargs are prefixed
(`gbmpo_divergence_type`, `gbmpo_l2_coefficient`) to avoid clashing with other
algorithms' parameters. If you construct `GBMPOConfig` directly instead of
going through the factory, the unprefixed field names (`divergence_type`,
`l2_coefficient`) are what `GBMPOConfig` itself takes.

When `gbmpo_divergence_type` is omitted, the factory selects `"l2kl"`.
For `"l2"` and `"prob_l2"`, setting `beta=0.0` disables the separate KL term;
the trainer temporarily uses a tiny positive beta internally when required
by the parent TRL GRPO configuration.

## Configuration Parameters

GBMPO subclasses the full GRPO trainer, so it inherits every parameter on the
[GRPO page](grpo.md#configuration-parameters), plus:

--8<-- "docs/PARAMETERS.md:gbmpo"

## See Also

- [Algorithms Overview](overview.md)
- [GRPO Algorithm](grpo.md) - Full inherited parameter reference
