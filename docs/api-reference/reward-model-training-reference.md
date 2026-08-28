# Reward Model Training System

## Overview

The Reward Model Training System provides comprehensive capabilities for training neural reward models from rule-based reward functions and integrating them into PPO training pipelines. This system enables scalable reward model training with strict validation, no fallbacks, and production-ready implementations.

## Architecture

### Core Components

1. **RewardModelTrainer**: Main training class using TRL's RewardTrainer
2. **RewardModelDataset**: PyTorch Dataset for reward training data
3. **RewardModelValidator**: Strict validation with no fallbacks
4. **RewardModelLoader**: Loads HF Hub, local, or custom trained models
5. **RewardModelTrainingConfig**: Configuration for training parameters
6. **RewardModelSourceConfig**: Configuration for reward model sources

### Integration Points

- **PPO Integration**: Both Unsloth and TRL PPO backends support reward model loading
- **Reward Functions**: Uses existing CompositeReward system for training data generation
- **Configuration**: Integrates with existing dataclass-based config system
- **Validation**: Strict validation throughout with clear error messages

## Usage Modes

### 1. Standalone Reward Model Training

Train reward models independently without PPO:

```python
from aligntune.rewards.training import RewardModelTrainer
from aligntune.rewards.registry import RewardRegistry

# Get reward functions
registry = RewardRegistry()
length_func = registry.get_reward_function("length")
sentiment_func = registry.get_reward_function("sentiment")

# Create trainer
trainer = RewardModelTrainer(
 base_model_name="microsoft/DialoGPT-medium",
 reward_functions=[length_func, sentiment_func],
 composite_weights=[0.5, 0.5]
)

# Generate training data
training_data = trainer.generate_training_data(
 texts=["This is a good response.", "This is a bad response."],
 batch_size=32
)

# Train model
model_path = trainer.train_reward_model(
 training_data=training_data,
 output_dir="./reward_models/custom",
 num_epochs=3,
 learning_rate=1e-5,
 batch_size=8
)
```

### 2. PPO with Custom Reward Model Training

Train custom reward models as part of PPO training:

```python
from aligntune.core.backend_factory import create_rl_trainer

# Create PPO trainer with custom reward model training
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 
 # Custom reward model training
 train_custom_reward_model=True,
 reward_training_texts=load_training_texts(),
 reward_functions=["length", "sentiment", "safety", "coherence"],
 reward_function_weights=[0.2, 0.3, 0.3, 0.2],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom_ppo",
 
 # PPO configuration
 num_epochs=1,
 batch_size=1,
 max_samples=100
)
```

### 3. PPO with Pre-trained Reward Models

Use existing reward models from HuggingFace Hub or local paths:

```python
# HuggingFace Hub model
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 reward_model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
 # ... other config
)

# Local model
trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 reward_model_path="./reward_models/my_custom_model",
 # ... other config
)
```

## Configuration

### RewardModelTrainingConfig

Configuration for reward model training with strict validation:

```python
@dataclass
class RewardModelTrainingConfig:
 base_model_name: str # Required - no default
 training_texts: List[str] # Required - no default
 reward_functions: List[str] # Required - no default
 output_dir: str # Required - no default
 
 # Optional but validated if provided
 reference_texts: Optional[List[str]] = None
 reward_weights: Optional[List[float]] = None
 
 # Training parameters with justified defaults
 num_epochs: int = 3
 learning_rate: float = 1e-5
 batch_size: int = 8
 gradient_accumulation_steps: int = 4
 max_length: int = 512
```

### RewardModelSourceConfig

Configuration for reward model source with strict mutual exclusivity:

```python
@dataclass
class RewardModelSourceConfig:
 source_type: str # "pretrained_hf", "pretrained_local", "custom_trained"
 model_name: Optional[str] = None # For HF Hub
 model_path: Optional[str] = None # For local
 training_config: Optional[RewardModelTrainingConfig] = None # For custom
```

## Validation

### Strict Validation Rules

1. **Exactly One Source**: Must specify exactly one reward source (HF Hub, local, or custom)
2. **No Empty Fields**: All required fields must be non-empty
3. **Minimum Data**: At least 10 training texts required
4. **Valid Functions**: All reward functions must exist in registry
5. **Consistent Lengths**: Reference texts must match training texts length
6. **Positive Weights**: All reward weights must be positive

### Error Handling

- **No Fallbacks**: System fails fast with clear error messages
- **No Silent Failures**: All errors are logged and raised
- **Comprehensive Validation**: Every input is validated before processing
- **Clear Messages**: Error messages include actionable information

## Examples

### Complete Custom Training Example

```python
import logging
from aligntune.core.backend_factory import create_rl_trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_texts():
 """Load your training texts here."""
 return [
 "This is a helpful and informative response.",
 "I'm not sure about this, but here's what I think.",
 "That's a great question! Let me explain step by step.",
 # ... more texts
 ]

def main():
 # Load training data
 training_texts = load_training_texts()
 
 # Create PPO trainer with custom reward model
 trainer = create_rl_trainer(
 model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
 algorithm="ppo",
 backend="unsloth",
 dataset_name="HuggingFaceH4/ultrafeedback_binarized",
 
 # Custom reward model training
 train_custom_reward_model=True,
 reward_training_texts=training_texts,
 reward_functions=["length", "sentiment", "safety", "coherence"],
 reward_function_weights=[0.2, 0.3, 0.3, 0.2],
 reward_training_base_model="microsoft/DialoGPT-medium",
 reward_training_output_dir="./reward_models/custom",
 
 # Training parameters
 num_epochs=1,
 batch_size=1,
 max_samples=100,
 output_dir="./output/ppo_custom_reward"
 )
 
 # Setup and train
 trainer.setup_data()
 trainer.setup_trainer()
 results = trainer.train()
 
 logger.info(f"Training completed: {results}")

if __name__ == "__main__":
 main()
```

### Standalone Training Example

```python
from aligntune.rewards.training import RewardModelTrainer
from aligntune.rewards.registry import RewardRegistry

def standalone_training():
 # Get reward functions
 registry = RewardRegistry()
 reward_functions = [
 registry.get_reward_function("length"),
 registry.get_reward_function("sentiment"),
 registry.get_reward_function("safety")
 ]
 
 # Create trainer
 trainer = RewardModelTrainer(
 base_model_name="microsoft/DialoGPT-medium",
 reward_functions=reward_functions,
 composite_weights=[0.3, 0.4, 0.3]
 )
 
 # Generate training data
 texts = ["Your training texts here..."] * 20
 training_data = trainer.generate_training_data(texts)
 
 # Train model
 model_path = trainer.train_reward_model(
 training_data=training_data,
 output_dir="./reward_models/standalone",
 num_epochs=3,
 learning_rate=1e-5,
 batch_size=8
 )
 
 print(f"Model trained and saved to: {model_path}")

if __name__ == "__main__":
 standalone_training()
```

## Supported Reward Functions

The system supports all reward functions available in the registry:

- **Length**: Text length rewards
- **Sentiment**: Sentiment analysis rewards
- **Safety**: Safety and toxicity rewards
- **Coherence**: Text coherence rewards
- **BLEU**: BLEU score rewards
- **ROUGE**: ROUGE score rewards
- **Math Correctness**: Mathematical reasoning rewards
- **Code Syntax**: Code quality rewards
- **Toxicity**: Toxicity detection rewards

## Troubleshooting

### Common Issues

1. **"Need at least 10 training texts"**
 - Solution: Provide at least 10 training texts
 - Check: `len(training_texts) >= 10`

2. **"Missing reward functions"**
 - Solution: Use functions available in registry
 - Check: `RewardRegistry().list_reward_functions()`

3. **"Multiple reward sources specified"**
 - Solution: Specify exactly one source (HF Hub, local, or custom)
 - Check: Only one of `reward_model_name`, `reward_model_path`, `train_custom_reward_model` should be True

4. **"reward_model_path does not exist"**
 - Solution: Ensure local path exists and contains model files
 - Check: Path exists and contains `config.json` and model files

5. **"Model not accessible on HuggingFace Hub"**
 - Solution: Check model name and internet connection
 - Check: `huggingface_hub.model_info(model_name)`

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation Helpers

Use validation helpers to check configuration:

```python
from aligntune.rewards.training import RewardModelValidator

# Validate training config
RewardModelValidator.validate_training_config(config)

# Validate reward source
RewardModelValidator.validate_reward_source(source_config)

# Validate HF model access
RewardModelValidator.validate_hf_model_access("model-name")

# Validate local path
RewardModelValidator.validate_local_path("/path/to/model")
```

## Performance Considerations

### Memory Usage

- **Batch Processing**: Use appropriate batch sizes for your GPU memory
- **Gradient Accumulation**: Use gradient accumulation for effective larger batch sizes
- **Model Quantization**: Consider 4-bit quantization for memory efficiency

### Training Speed

- **Batch Size**: Larger batches generally train faster
- **Gradient Accumulation**: Use gradient accumulation to simulate larger batches
- **Model Size**: Smaller base models train faster but may have lower quality

### Scalability

- **Distributed Training**: Use multiple GPUs for large-scale training
- **Data Parallelism**: Process multiple samples in parallel
- **Model Parallelism**: Split large models across devices

## Best Practices

1. **Data Quality**: Use high-quality training texts for better reward models
2. **Function Selection**: Choose reward functions relevant to your task
3. **Weight Tuning**: Experiment with different reward function weights
4. **Validation**: Always validate your configuration before training
5. **Monitoring**: Monitor training progress and adjust parameters as needed
6. **Testing**: Test trained models before using in production

## API Reference

### RewardModelTrainer

```python
class RewardModelTrainer:
 def __init__(self, base_model_name: str, reward_functions: List[RewardFunction], composite_weights: List[float])
 def generate_training_data(self, texts: List[str], references: Optional[List[str]] = None, batch_size: int = 32) -> RewardModelDataset
 def train_reward_model(self, training_data: RewardModelDataset, output_dir: str, **kwargs) -> str
 def create_scalable_pipeline(self, training_texts: List[str], output_dir: str, **kwargs) -> str
```

### RewardModelDataset

```python
class RewardModelDataset(Dataset):
 def __init__(self, texts: List[str], reward_scores: List[float], tokenizer: AutoTokenizer, max_length: int = 512)
 def __len__(self) -> int
 def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
```

### RewardModelValidator

```python
class RewardModelValidator:
 @staticmethod
 def validate_reward_source(source_config: RewardModelSourceConfig) -> None
 @staticmethod
 def validate_training_config(config: RewardModelTrainingConfig) -> None
 @staticmethod
 def validate_hf_model_access(model_name: str) -> None
 @staticmethod
 def validate_local_path(path: str) -> None
```

### RewardModelLoader

```python
class RewardModelLoader:
 def load_from_huggingface(self, model_name: str, **kwargs) -> AutoModelForSequenceClassification
 def load_from_local(self, model_path: str, **kwargs) -> AutoModelForSequenceClassification
```

## Contributing

When contributing to the reward model training system:

1. **Follow Validation**: Always use strict validation with no fallbacks
2. **Add Tests**: Include comprehensive tests for new features
3. **Document Changes**: Update documentation for any API changes
4. **Error Handling**: Provide clear, actionable error messages
5. **Performance**: Consider performance implications of changes

## License

This reward model training system is part of the AlignTune project and follows the same license terms.