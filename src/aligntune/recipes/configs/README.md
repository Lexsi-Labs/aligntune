# AlignTune Recipe Library

This directory contains production-ready training recipes for popular models and tasks.

## 📚 Available Recipes (20 total)

### Supervised Fine-Tuning (SFT) - 3 recipes
- **llama3-instruct-orca**: LLaMA 3 8B on SlimOrca for instruction following
- **qwen-instruct-ultrachat**: Qwen 2.5 3B on UltraChat for conversation
- **gemma-instruct-finetome**: Gemma 2 2B on FineTome for high-quality instruction following

### Direct Preference Optimization (DPO) - 4 recipes
- **llama3-ultrafeedback-dpo**: LLaMA 3 8B on UltraFeedback for alignment
- **qwen-preferences-dpo**: Qwen 2.5 3B on preference data for alignment
- **mistral-hhrlhf-dpo**: Mistral 7B on Anthropic HH-RLHF for safety
- **gemma-helpfulness-dpo**: Gemma 2 2B for improved helpfulness

### Proximal Policy Optimization (PPO) - 2 recipes
- **llama3-helpful-harmless-ppo**: Classic RLHF with LLaMA 3 for safety
- **qwen-safety-ppo**: Qwen 2.5 3B for improved safety alignment

### Group Relative Policy Optimization (GRPO) - 3 recipes
- **llama3-gsm8k-grpo**: LLaMA 3 8B on GSM8K for math reasoning
- **qwen-math-grpo**: Qwen 2.5 3B on GSM8K for math
- **deepseek-humaneval-grpo**: DeepSeek Coder on HumanEval for code generation

### BOLT (Best-of-N Learning) - 2 recipes
- **qwen-mbpp-bolt**: Qwen 3 1.7B on MBPP with baseline optimization
- **llama-code-bolt**: LLaMA 3.2 3B on code generation with curriculum

### Advanced Algorithms - 3 recipes
- **neural-mirror-grpo-math**: Learnable mirror maps for adaptive optimization
- **dr-grpo-robust-math**: Distribution robust GRPO for multi-domain math
- **gbmpo-l2-code**: Generalized Bregman mirror descent for code generation

---

## 🚀 Usage

### CLI Method

```bash
# List all recipes
aligntune recipes list

# Show recipe details
aligntune recipes show llama3-ultrafeedback-dpo

# Copy recipe to customize
aligntune recipes copy llama3-ultrafeedback-dpo --output my_config.yaml

# Run recipe directly
aligntune recipes run llama3-ultrafeedback-dpo
```

### Python API

```python
from aligntune.recipes import get_recipe, list_recipes

# Get specific recipe
recipe = get_recipe("llama3-ultrafeedback-dpo")
print(recipe.description)

# List all DPO recipes
dpo_recipes = list_recipes(algorithm="dpo")

# Search recipes
math_recipes = search_recipes("math")
```

---

## 📋 Recipe Selection Guide

### **For Beginners**
Start with **SFT recipes** - they're the simplest and most reliable:
- `qwen-instruct-ultrachat` (smallest, fastest)
- `gemma-instruct-finetome` (good quality)

### **For Alignment**
Use **DPO recipes** - most popular for RLHF:
- `llama3-ultrafeedback-dpo` (general alignment)
- `mistral-hhrlhf-dpo` (safety-focused)

### **For Math/Reasoning**
Use **GRPO recipes**:
- `qwen-math-grpo` (efficient, good for testing)
- `llama3-gsm8k-grpo` (higher quality)

### **For Code Generation**
Use **BOLT or GRPO recipes**:
- `qwen-mbpp-bolt` (with curriculum learning)
- `deepseek-humaneval-grpo` (code-specialized model)

### **For Research**
Try **advanced algorithms**:
- `neural-mirror-grpo-math` (adaptive optimization)
- `gbmpo-l2-code` (custom divergence measures)

---

## 🔧 Customizing Recipes

All recipes can be customized:

```bash
# Copy and edit
aligntune recipes copy llama3-gsm8k-grpo --output custom.yaml

# Edit the YAML file
# - Change model size
# - Adjust hyperparameters
# - Modify reward weights
# - Update dataset size

# Run your custom config
aligntune train --config custom.yaml
```

---

## 💾 Resource Requirements

| Category | Min VRAM | Recommended |
|----------|----------|-------------|
| **SFT** | 12-24GB | 16-32GB |
| **DPO** | 12-28GB | 16-32GB |
| **PPO** | 20-32GB | 24-40GB |
| **GRPO** | 12-28GB | 18-32GB |
| **BOLT** | 16-18GB | 20-24GB |
| **Advanced** | 24-32GB | 32-48GB |

*All recipes use LoRA by default for memory efficiency*

---

## 🏷️ Recipe Tags

Filter recipes by tags:

```bash
aligntune recipes list --tag llama      # All LLaMA recipes
aligntune recipes list --tag math       # Math reasoning recipes
aligntune recipes list --tag efficient  # Memory-efficient recipes
aligntune recipes list --tag advanced   # Research algorithms
```

Available tags:
- **Models**: `llama`, `qwen`, `gemma`, `mistral`, `deepseek`
- **Tasks**: `math`, `code`, `safety`, `conversation`, `reasoning`
- **Properties**: `efficient`, `popular`, `advanced`, `research`
- **Features**: `curriculum`, `baseline`, `adaptive`, `robust`

---

## 📖 Recipe Format

Each recipe YAML contains:

```yaml
recipe:
  name: "unique-recipe-name"
  description: "What this recipe does"
  model: "huggingface/model-name"
  dataset: "dataset-name"
  task: "task-type"
  algorithm: "algorithm-name"
  backend: "trl|unsloth"
  tags: ["tag1", "tag2"]
  requires_auth: true|false
  estimated_time: "X-Y hours"
  estimated_memory: "XGB VRAM"

config:
  # Full training configuration
  algo: algorithm_name
  model: {...}
  datasets: [...]
  rewards: [...]
  train: {...}
  logging: {...}
```

---

## 🤝 Contributing Recipes

To contribute a new recipe:

1. Create YAML file in appropriate subdirectory
2. Follow the format above
3. Test with `aligntune recipes run <name>`
4. Submit PR with recipe details

---

## 🔗 Related Documentation

- [AlignTune Main README](../../../../README.md)
- [Algorithm Documentation](../../../../docs/algorithms/)
- [Training Examples](../../../../examples/)
- [API Reference](../../../../docs/api/)

---

## 📝 Notes

- All recipes use **LoRA** by default for memory efficiency
- **bf16** precision is used where supported
- Recipes are configured for NVIDIA GPUs (A100, H100, RTX 4090)
- Training times are estimates for reference hardware
- Actual times vary based on GPU, batch size, and dataset size

---

## 🧪 Testing Status

**Configuration Validated**: All 17 recipes have validated YAML structure and parameter correctness.

**Pending Full Training Tests**: The following recipes have correct configurations but await end-to-end training validation:

### Priority for Testing (High Impact)
- [ ] **llama3-ultrafeedback-dpo** - Most popular alignment recipe
- [ ] **qwen-math-grpo** - Math reasoning (efficient model)
- [ ] **llama3-gsm8k-grpo** - Math reasoning (standard reference)
- [ ] **qwen-instruct-ultrachat** - Conversation fine-tuning
- [ ] **qwen-mbpp-bolt** - Code generation with baseline

### Standard Algorithms (Medium Priority)
- [ ] **llama3-instruct-orca** - SFT instruction-following
- [ ] **gemma-instruct-finetome** - SFT instruction-following
- [ ] **qwen-preferences-dpo** - Preference optimization
- [ ] **mistral-hhrlhf-dpo** - Safety alignment
- [ ] **gemma-helpfulness-dpo** - Helpfulness optimization
- [ ] **llama3-helpful-harmless-ppo** - Classic RLHF
- [ ] **qwen-safety-ppo** - Safety RLHF
- [ ] **deepseek-humaneval-grpo** - Code generation
- [ ] **llama-code-bolt** - Code generation

### Advanced Algorithms (Research/Low Priority)
- [ ] **neural-mirror-grpo-math** - Adaptive optimization (8-12 hours)
- [ ] **dr-grpo-robust-math** - Distribution robustness
- [ ] **gbmpo-l2-code** - Custom divergence measures

### Testing Notes
- Configurations based on established best practices and literature
- Hyperparameters aligned with successful training runs in similar setups
- Memory requirements estimated conservatively
- Some recipes may need minor adjustments for specific GPU architectures
- Please report results or issues via GitHub Issues

**Community Testing Welcome!** If you test any recipe, please share:
- GPU type and VRAM
- Actual training time
- Final metrics (loss, accuracy, pass@k, etc.)
- Any configuration adjustments needed

---

**Last Updated**: January 2026  
**AlignTune Version**: 0.2.0+  
**Testing Status**: Configuration validated, training tests pending
