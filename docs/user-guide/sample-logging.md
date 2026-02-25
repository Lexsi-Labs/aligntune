# Qualitative Sample Logging

AlignTune can periodically generate qualitative samples during RL training to spot regressions long before metrics change. The feature is powered by `aligntune.core.rl.sample_logger` and governed by `SampleLoggingConfig`.

## When Samples Are Logged

- Available on TRL RL trainers (`ppo`, `grpo`, `gspo`) and Unsloth PPO via their calls to `generate_and_log_samples`.
- Samples run **after** key checkpoints (for PPO this happens at the end of training, and additional hooks log mid-training when `interval_steps` or `percent_of_max_steps` triggers fire).
- Responses and optional reward scores are streamed to both the configured logger and stdout so they appear in notebooks and terminal logs.

## Configuration Fields

`SampleLoggingConfig` lives under the `logging` section of `UnifiedConfig`:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `False` | Master switch for sample generation |
| `prompts` | list[str] | built-in prompt trio | Prompts to feed into `model.generate`; falls back to defaults when empty |
| `interval_steps` | int? | `None` | Log every N optimizer steps |
| `percent_of_max_steps` | float? | `None` | Log at percentages of `train.max_steps` (e.g., `0.25`) |
| `num_samples` | int | `3` | How many prompts to evaluate (capped at the number of prompts) |
| `max_new_tokens` | int | `80` | Max tokens per response |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling parameter |

Validation in `__post_init__` ensures all numeric values are positive and that percentages fall inside `(0, 1]`.

## YAML Example

```yaml
logging:
 output_dir: ./output/ppo_run
 loggers: [tensorboard]
 sample_logging:
 enabled: true
 prompts:
 - "Summarize the core idea of reinforcement learning in two sentences."
 - "Write a concise plan for debugging PPO training instabilities."
 interval_steps: 200
 num_samples: 2
 max_new_tokens: 96
 temperature: 0.6
 top_p: 0.95
```

## Python API Example

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 backend="unsloth",
 num_epochs=1,
 batch_size=1,
 sample_logging={
 "enabled": True,
 "percent_of_max_steps": 0.5,
 "num_samples": 2,
 "max_new_tokens": 72,
 "temperature": 0.8,
 },
)
trainer.train()
```

Behind the scenes, the backend factory merges the provided dictionary into a `SampleLoggingConfig` instance before passing it to the trainer.

## Reward-Aware Samples

When custom reward functions are registered on the trainer (e.g., PPO with multiple heuristic rewards), each generated response is scored and the per-reward outputs are printed alongside the prompt/response pair:

```
================================================================================
Qualitative samples (post-train) - prompts: 2
[Sample 1/2] Prompt: Summarize the core idea...
[Sample 1/2] Response: ...
[Sample 1/2] Rewards: {'reward_0': 0.81, 'reward_1': 0.42}
================================================================================
```

## Best Practices

- Keep prompts short and deterministic so diffs are easier to read in CI logs.
- Use `percent_of_max_steps` for long-running jobs where `max_steps` is known, otherwise rely on step intervals.
- On distributed training setups the samples run on the trainer's primary device; make sure enough GPU memory remains to run generation with the configured `max_new_tokens`.
- Disable qualitative samples on highly resource-constrained runs to avoid extra generation overhead.

For more context on logging knobs, revisit the [`logging.sample_logging` section of the README](#) or the API docs for `SampleLoggingConfig`.