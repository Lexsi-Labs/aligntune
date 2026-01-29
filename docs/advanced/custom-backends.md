# Custom Backends

Create custom training backends by implementing the trainer base classes.

## Overview

AlignTune uses a backend factory and base trainer classes so you can add new backends (e.g., other frameworks) while keeping the same configuration and API.

## Implementing a Custom SFT Backend

Subclass `SFTTrainerBase` and implement the required methods:

```python
from aligntune.core.sft.trainer_base import SFTTrainerBase

class CustomSFTTrainer(SFTTrainerBase):
    def train(self):
        # Your training loop
        ...

    def evaluate(self, eval_dataset=None, **kwargs):
        # Your evaluation logic
        ...
```

## Implementing a Custom RL Backend

Subclass the RL trainer base and implement the lifecycle:

```python
from aligntune.core.rl.trainer_base import TrainerBase

class CustomRLTrainer(TrainerBase):
    def train(self):
        ...

    def evaluate(self, eval_dataset=None, **kwargs):
        ...
```

## Registration

Register your backend with the factory so it can be selected by name (e.g., in config or `backend="custom"`). See the source for `BackendFactory` and backend registration patterns.

## Next Steps

- [Architecture](architecture.md) - System design overview
- [Distributed Training](distributed.md) - Multi-GPU and distributed setups
- [Performance](performance.md) - Optimization tips
