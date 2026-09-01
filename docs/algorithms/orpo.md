# ORPO (Odds Ratio Preference Optimization)

Odds Ratio Preference Optimization (ORPO) integrates Supervised Fine-Tuning (SFT) and alignment into a single training objective.

## Overview

Unlike standard RLHF methods (PPO, DPO) which require a separate SFT phase and a frozen reference model, ORPO adds an odds ratio penalty directly to the negative log-likelihood loss during fine-tuning. This eliminates the need for an explicit reference model, drastically reducing memory requirements.

### When to use ORPO?
- When you want to combine instruction-tuning and alignment in one phase.
- When you are heavily memory-constrained and cannot host a reference model in memory.

### How ORPO works

1. The model scores both the chosen and rejected completion for each prompt.
2. A normal supervised language-model loss trains the chosen completion.
3. An odds-ratio term compares the model's relative preference for chosen over
   rejected tokens.
4. The language-model loss and the odds-ratio penalty are combined with
   `beta`, and one optimizer update is applied. There is no separate SFT phase,
   reward model, or reference-model forward pass.

## Configuration

To use ORPO, pass `algorithm="orpo"` to the `create_rl_trainer` factory method:

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="orpo",
    backend="unsloth" # ORPO is highly efficient on Unsloth
)

trainer.train()
```

### Data format

ORPO requires paired preference rows:

```json
{
  "prompt": "Explain compound interest.",
  "chosen": "A clear explanation...",
  "rejected": "An incorrect explanation..."
}
```

Use `column_mapping` if the source columns have different names. ORPO combines
the language-model loss and preference loss in one pass. It does not use
`reward_functions`, a reward model, or a reference model.

### Important parameters

- `beta`: weight of the odds-ratio preference term.
- `max_seq_length`: mapped to TRL's `max_length`, the total sequence limit.
- `batch_size`, `gradient_accumulation_steps`, `learning_rate`, and
  `num_epochs` control the SFT-style optimization loop.

The installed TRL `ORPOConfig` does not accept `max_prompt_length` or
`truncation_mode`; configure the overall limit with `max_seq_length`.

### Configuration Options

--8<-- "docs/PARAMETERS.md:orpo"

## See Also

- [Algorithms Overview](overview.md)
- [ORPO Parameters](../PARAMETERS.md#orpo-odds-ratio-preference-optimization) - Full parameter reference
