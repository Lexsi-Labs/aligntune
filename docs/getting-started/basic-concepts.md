# Basic Concepts

This guide introduces the core concepts you need to understand to effectively use AlignTune.

---

## Training Types

AlignTune supports two main training paradigms:

### Supervised Fine-Tuning (SFT)

**SFT** is the process of fine-tuning a pre-trained language model on a task-specific dataset. It's used for:

- Instruction following
- Text classification
- Chat completion
- Domain adaptation

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl"
)
trainer.train()
```

### Reinforcement Learning (RL)

**RL** training uses reinforcement learning algorithms to align models with human preferences or optimize for specific objectives. Supported algorithms include:

- **DPO** (Direct Preference Optimization) - No reward model needed
- **PPO** (Proximal Policy Optimization) - Requires reward model
- **GRPO** (Group Relative Policy Optimization) - Multi-criteria optimization
- **GSPO** (Group Sequential Policy Optimization) - Sequential group learning
- **DAPO** (Decouple Clip and Dynamic sAmpling Policy Optimization)
- **Dr. GRPO** (GRPO Done Right) - Unbiased GRPO variant

```python
from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="dpo",
 backend="trl"
)
trainer.train()
```

---

## Backends

AlignTune provides two backend implementations:

### TRL Backend

**TRL** (Transformers Reinforcement Learning) backend is the standard, battle-tested implementation:

- **Reliable**: Production-ready, extensively tested
- **Complete**: Supports all algorithms (SFT, DPO, PPO, GRPO, GSPO, etc.)
- **Compatible**: Works with all HuggingFace models
- **Slower**: Standard training speed

**Use TRL when:**
- You need maximum reliability
- You're using algorithms not supported by Unsloth (GSPO)
- You're on CPU or have compatibility issues with Unsloth

### Unsloth Backend

**Unsloth** backend provides optimized training with significant speed improvements:

- **Fast**: faster training
- **Memory Efficient**: Optimized memory usage with LoRA/QLoRA
- **GPU Optimized**: CUDA-specific optimizations
- **Limited**: Some algorithms not supported (GSPO)
- **GPU Required**: Needs CUDA-capable GPU

**Use Unsloth when:**
- You have a CUDA-capable GPU
- You want maximum training speed
- You're using supported algorithms (DPO, PPO, GRPO, etc.)

### Backend Selection

You can specify the backend explicitly or use auto-selection:

```python
# Explicit backend selection
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="trl" # or "unsloth"
)

# Auto-selection (recommended)
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 backend="auto" # Automatically selects best available backend
)
```

---

## Configuration

AlignTune supports three configuration methods:

### 1. Python API

Direct function calls with keyword arguments:

```python
trainer = create_sft_trainer(
 model_name="microsoft/DialoGPT-small",
 dataset_name="tatsu-lab/alpaca",
 num_epochs=3,
 batch_size=4,
 learning_rate=5e-5
)
```

### 2. YAML Configuration

Declarative configuration files:

```yaml
# config.yaml
model:
 name_or_path: "microsoft/DialoGPT-small"

datasets:
 - name: "tatsu-lab/alpaca"
 max_samples: 1000

train:
 num_epochs: 3
 per_device_batch_size: 4
 learning_rate: 5e-5
```

```python
from aligntune.core.sft.config_loader import SFTConfigLoader
from aligntune.core.backend_factory import create_sft_trainer


# Load configuration
config = SFTConfigLoader.load_from_yaml("config.yaml")

# Create trainer from config
trainer = create_sft_trainer(config=config)
```

### 3. CLI

Command-line interface:

```bash
aligntune train \
 --model microsoft/DialoGPT-small \
 --dataset tatsu-lab/alpaca \
 --type sft \
 --epochs 3 \
 --batch-size 4
```

---

## Reward Functions

Reward functions are used in RL training to evaluate and guide model behavior. AlignTune provides 27+ built-in reward functions:

### Quality Metrics
- `helpfulness` - Measures how helpful the response is
- `coherence` - Evaluates text coherence and flow
- `relevance` - Measures relevance to the prompt
- `fluency` - Evaluates language fluency

### Safety Metrics
- `toxicity` - Detects toxic content
- `bias` - Measures bias in responses
- `privacy` - Detects privacy violations
- `harmful_content` - Identifies harmful content

### Style Metrics
- `formality` - Measures formality level
- `tone` - Evaluates tone appropriateness
- `length` - Measures response length
- `structure` - Evaluates text structure

### Task-Specific Metrics
- `code_quality` - Evaluates code quality
- `math_accuracy` - Measures mathematical accuracy
- `factual_correctness` - Evaluates factual accuracy

### Using Reward Functions

```python
trainer = create_rl_trainer(
 model_name="microsoft/DialoGPT-medium",
 dataset_name="Anthropic/hh-rlhf",
 algorithm="ppo",
 reward_functions=["helpfulness", "safety", "coherence"],
 reward_function_weights=[0.4, 0.3, 0.3]
)
```

---

## Evaluation

AlignTune provides comprehensive evaluation capabilities:

### Basic Metrics
- **Loss**: Training loss
- **Perplexity**: Language modeling quality
- **Runtime**: Training time

### Quality Metrics
- **BLEU**: Bilingual Evaluation Understudy score
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **Task-Specific**: Code syntax, math accuracy, etc.

### Safety Metrics
- **Toxicity**: Toxicity detection
- **Bias**: Bias measurement
- **Factual Accuracy**: Factual correctness

```python
# Evaluate trained model
metrics = trainer.evaluate()
print(metrics)
```

---

## Model Management

### Saving Models

```python
# Save model to local directory
path = trainer.save_model("./output/my_model")

# Push to HuggingFace Hub
trainer.push_to_hub("username/my-model", token="hf_...")
```

### Loading Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/my_model")
tokenizer = AutoTokenizer.from_pretrained("./output/my_model")
```

---

## Next Steps

- [Configuration Guide](configuration.md) - Learn about all configuration options
- [Backend Selection](backend-selection.md) - Detailed backend comparison
- [SFT Guide](../user-guide/sft.md) - Deep dive into SFT training
- [RL Guide](../user-guide/rl.md) - Deep dive into RL training