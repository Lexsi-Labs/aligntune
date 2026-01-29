# Core API Reference

Complete API reference for AlignTune core functions and utilities.

---

## Overview

This page covers core utility functions, error classes, and helper functions. For main API functions, see [Backend Factory](backend-factory.md).

---

## Configuration Loaders

### `SFTConfigLoader`

Load SFT configuration from YAML files.

::: core.sft.config_loader.SFTConfigLoader
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.sft.config_loader import SFTConfigLoader

config = SFTConfigLoader.load_from_yaml("config.yaml")
```

### `ConfigLoader` (RL)

Load RL configuration from YAML files.

::: core.rl.config_loader.ConfigLoader
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.rl.config_loader import ConfigLoader

config = ConfigLoader.load_from_yaml("ppo_config.yaml")
```

---

## Trainer Factories

### `SFTTrainerFactory`

Factory for creating SFT trainers.

::: core.sft.trainer_factory.SFTTrainerFactory
 options:
 show_source: true
 heading_level: 3

### `TrainerFactory` (RL)

Factory for creating RL trainers.

::: core.rl.trainer_factory.TrainerFactory
 options:
 show_source: true
 heading_level: 3

---

## Error Classes

### `AlignTuneError`

Base exception for AlignTune errors.

::: utils.errors.AlignTuneError
 options:
 show_source: true
 heading_level: 3

### `ConfigurationError`

Configuration-related errors.

::: utils.errors.ConfigurationError
 options:
 show_source: true
 heading_level: 3

### `TrainingError`

Training-related errors.

::: utils.errors.TrainingError
 options:
 show_source: true
 heading_level: 3

### `EnvironmentError`

Environment-related errors.

::: utils.errors.EnvironmentError
 options:
 show_source: true
 heading_level: 3

### `ValidationError`

Validation-related errors.

::: utils.errors.ValidationError
 options:
 show_source: true
 heading_level: 3

---

## Utility Functions

### `validate_backend_selection()`

Validate backend selection and provide helpful errors.

::: core.backend_factory.validate_backend_selection
 options:
 show_source: true
 heading_level: 3

**Example**:

```python
from aligntune.core.backend_factory import validate_backend_selection

backend_type = validate_backend_selection("trl", "SFT")
```

---

## See Also

- [Backend Factory](backend-factory.md) - Main API for creating trainers
- [Configuration Classes](configuration.md) - Configuration API
- [Trainers](trainers.md) - Trainer API