# Counterfactual GRPO (C-GRPO)

Counterfactual GRPO extends standard Group Relative Policy Optimization by incorporating explicit counterfactual analysis across the generative group.

## Overview

In traditional GRPO, rewards are normalized within an independently generated group of responses. Counterfactual GRPO introduces a baseline swapping penalty to determine what the reward *would have been* under different stylistic counterfactuals, effectively neutralizing superficial hacking (like length hacking) without needing an explicit length-penalty.

### When to use C-GRPO?
- When your standard GRPO training results in models that exploit the reward (e.g., generating incredibly long responses to gain higher scores).

## Configuration

To use Counterfactual GRPO, pass `algorithm="counterfact_grpo"` to the factory method:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="openai/gsm8k",
    algorithm="counterfact_grpo",
    backend="trl",
    num_epochs=1,
    batch_size=4,
    learning_rate=1e-6,
    num_generations=4,
    max_completion_length=256,
    temperature=0.7,
    top_p=0.95,
    beta=0.005,
    boost_factor=2.0,
    min_weight=0.5,
    max_spans=10,
    answer_weight=1.5,
    weighting_mode="counterfactual",
)

trainer.train()
```

Counterfactual GRPO currently uses its built-in math/code reward path rather
than the normal GRPO reward-function bridge. For GSM8K-style data it extracts
the numeric answer and scores generated completions against it; for MBPP-style
data it executes the generated code against tests. The configured
`reward_functions`/`rewards` entries are not used by the active TRL
counterfactual reward calculation, so do not rely on them to replace this
domain-specific scorer.

Use a supported math or code dataset with the answer/test fields needed by the
selected task. A generic conversational preference dataset is not an
equivalent input for this trainer.

The counterfactual weighting controls (`boost_factor`, `min_weight`,
`max_spans`, `answer_weight`, and `weighting_mode`) modify the policy-gradient
weights after the built-in reward is computed.

### Configuration Options

Subclasses the full GRPO trainer, so it inherits every parameter on the
[GRPO page](grpo.md#configuration-parameters), plus:

--8<-- "docs/PARAMETERS.md:counterfactual-grpo"

## See Also

- [Algorithms Overview](overview.md)
- [Counterfactual GRPO Parameters](../PARAMETERS.md#counterfactual-grpo) - Full parameter reference, including TRL vs. Unsloth backend differences
