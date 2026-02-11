
<p align="center">
  <img src="https://github.com/Lexsi-Labs/aligntune/blob/main/docs/assets/aligntune-banner.png" alt="AlignTune Banner" width="1000px"/>
</p>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg"/></a>
  <a href="https://github.com/Lexsi-Labs/aligntune/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://badge.fury.io/py/aligntune"><img src="https://badge.fury.io/py/aligntune.svg"/></a>
</div>

---

**AlignTune** is a production-ready fine-tuning library designed to simplify training and fine-tuning of Large Language Models (LLMs) with both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) methods. It provides a high-level, unified API that abstracts away the complexities of backend selection, algorithm configuration, and training loops, letting you focus on delivering results.

## Core Features

**Multi-Backend Architecture**: Choose between TRL (reliable, battle-tested) and Unsloth (faster) backends with intelligent auto-selection.

**Complete RLHF Coverage**: 12+ RL algorithms including DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO, GBMPO, Counterfactual GRPO, and PACE.

**Production-Ready**: No mock code, comprehensive error handling, extensive testing, and robust validation.

## Quick Start

### Supervised Fine-Tuning (SFT)

```python
from aligntune.core.backend_factory import create_sft_trainer

# Create and train SFT model
trainer = create_sft_trainer(
    model_name="microsoft/DialoGPT-small",
    dataset_name="tatsu-lab/alpaca",
    backend="trl",
    num_epochs=3,
    max_steps = -1,
    batch_size=4,
    learning_rate=5e-5
)

# Train the model
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(metrics)
```

### Reinforcement Learning (DPO)

```python
from aligntune.core.backend_factory import create_rl_trainer

# Create and train DPO model
trainer = create_rl_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="Anthropic/hh-rlhf",
    algorithm="dpo",
    backend="trl",
    num_epochs=1,
    batch_size=4,
    learning_rate=5e-5
)

# Train the model
trainer.train()
```

## Supported Algorithms

| Algorithm | TRL Backend | Unsloth Backend | Description |
|-----------|-------------|-----------------|-------------|
| **SFT** | Yes | Yes | Supervised Fine-Tuning |
| **DPO** | Yes | Yes | Direct Preference Optimization |
| **PPO** | Yes | Yes | Proximal Policy Optimization |
| **GRPO** | Yes | Yes | Group Relative Policy Optimization |
| **GSPO** | Yes | Yes | Group Sequential Policy Optimization  |
| **DAPO** | Yes | Yes | Decouple Clip and Dynamic sAmpling Policy Optimization |
| **Dr. GRPO** | Yes | Yes | GRPO Done Right (unbiased variant) |
| **GBMPO** | Yes | No | Group-Based Mirror Policy Optimization |
| **Counterfactual GRPO** | Yes | Yes | Counterfactual GRPO variant |
| **PACE** | Yes | Yes | Baseline-Optimized Learning Technique |

## Installation

```bash
# Or install from source
git clone https://github.com/Lexsi-Labs/aligntune.git
cd aligntune
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

## Demo Notebooks

Interactive Colab notebooks demonstrating various AlignTune workflows:
Here are the organized tables containing the Colab links, models, and datasets provided in your text.

### Supervised Fine-Tuning (SFT)

| Backend| Model | Dataset | Link |
| --- | --- | --- | --- |
| TRL | **Qwen/Qwen3-4B-Instruct-2507** | sohamb37lexsi/bitext-wealth-management-llm-chatbot-splits | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1okAnfMlkch-G5Dy2dmj_dfR07rtxmYoq?usp=sharing) |
| TRL | **Qwen3-4B-Instruct** | sohamb37lexsi/bitext-retail-banking-llm-chatbot-splits | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1JnFvEWZ6PDqrDAyWznCdOA9oNn8aqN2v?usp=sharing) |
| Unsloth| **Qwen/Qwen2.5-0.5B-Instruct** | bebechien/MobileGameNPC | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/15R3JOrzAUuMagCamDHsqe0wNyb44rw2j?usp=sharing) |
| TRL | **google/txgemma-2b-predict** | trialbench_adverse-event-rate-prediction | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1r91SS7Lb5LjkjzWXp7uzkpR3vlWZ6gwD?usp=sharing) |
| Unsloth| **Qwen/Qwen2.5-0.5B-Instruct** | bebechien/MobileGameNP |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/11DrRbG30MUCegZTDXwR9nxEdvKaLfWCb?usp=sharing) |

### Reinforcement Learning (RL)

| Backend| Algorithm | Model | Dataset | Link |
| --- | --- | --- | --- | --- |
| Unsloth| **DPO** | microsoft/phi-2 | argilla/distilabel-intel-orca-dpo-pairs | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1aKdQzT7KGs0PEr7pO9rOXFkwQK6LzyRZ#scrollTo=dOnIJIAMxP9J) |
| TRL | **DPO** | google/gemma-2-2b-it | Anthropic/hh-rlhf | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1WQhek61Z0v1zHFWT4mQwVX7n10ln-j3d?usp=sharing) |
| TRL | **DPO** | sohamb37lexsi/wealth_management_Qwen3-4B-Instruct-2507 | sohamb37lexsi/bitext_wealth_management_preference_data | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1LR3KWcjQAFjSHf0MR9271ShchjPUVSpS?usp=sharing) |
| Unsloth| **PPO** | Qwen/Qwen2.5-0.5B-Instruct | HuggingFaceH4/ultrachat_200k | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1G4fdtO4DhBNwjOYhuaGmvIrBU_lc159H?usp=sharing) |
| TRL | **PPO** | EleutherAI/pythia-1.4b | CarperAI/openai_summarize_tldr | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1wlbSvQAJv8ZFM2qGD4XlUOKPzZHXstQo?usp=sharing) |
| TRL | **GRPO** (Coding) | Qwen/Qwen3-4B | google-research-datasets/mbpp | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/13HYZ-EkLC3-6wxJG_1NyWXPEuN9bIeM4?usp=sharing) |
| Unsloth| **GRPO** (Math) | meta-llama/Llama-3.2-3B-Instruct | openai/gsm8k | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/11tj2odJa7v55VvQkpOlm7_WBclqFdaVi?usp=sharing) |
| TRL | **GRPO** | meta-llama/Llama-3.2-3B-Instruct | openai/gsm8k | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/14b2dt0_iSVL8Z_8mlOx--f-d_rDCqVOP?usp=sharing) |
| Unsloth | **DRGRPO** | Qwen/Qwen2.5-3B-Instruct | yahma/alpaca-cleaned | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1IzDtjONyL6CZ038faTTp8AvLfhzkqSpb?usp=sharing) |
| TRL | **DRGRPO** | Qwen/Qwen2-0.5B-Instruct | AI-MO/NuminaMath-TIR | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1KfrMzuGRPrZwyRuNTGep1ZL6C6h2aVjk?usp=sharing) |
| Unsloth | **GSPO** | Qwen/Qwen3-1.7B | CyberNative/Code_Vulnerability_Security_DPO | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/120FjMEAZRXUoOMRsWHCn5tmpGfLTAY4g?usp=sharing) |
| TRL | **GSPO** | meta-llama/Llama-3.2-3B-Instruct | HuggingFaceH4/ultrachat_200k | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1vDP7ukBHWwSiD7KVSgSekCvk-x_MkWdC?usp=sharing) |
| Unsloth| **DAPO** | microsoft/Phi-3.5-mini-instruct | HuggingFaceH4/ultrachat_200k | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1gAa6EPc5345XRfog1nzQIFTmp_OJqNvk?usp=sharing) |
|TRL | **DAPO** | meta-llama/Llama-3.2-3B-Instruct | google-research-datasets/mbpp | [![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1aF3LxEMmLl0fCyA5yVBAkU_Rgsy3dsh5?usp=sharing) |
## Documentation

- **[Getting Started](docs/getting-started/installation.md)**: Installation, setup, and basic usage
- **[User Guide](docs/user-guide/overview.md)**: In-depth tutorials for SFT and RL training
- **[API Reference](docs/api-reference/overview.md)**: Complete Python API and class/method details
- **[Examples](docs/examples/overview.md)**: End-to-end code examples
- **[Advanced Topics](docs/advanced/architecture.md)**: Architecture, custom backends, and performance optimization
- **[Notebooks](docs/notebooks/index.md)**: Interactive Colab notebooks and local Jupyter notebooks

## Key Capabilities

- **Multiple Training Paradigms**: Supports SFT, DPO, PPO, GRPO, and advanced RL algorithms
- **Backend Flexibility**: TRL and Unsloth backends with automatic fallback
- **Reward Model Training**: Train custom reward models from rule-based functions
- **Comprehensive Evaluation**: Multi-level evaluation with lm-eval integration
- **Production Ready**: Model serialization, reproducible training, and deployment-ready pipelines
- **Extensible Architecture**: Modular design for easy integration of custom algorithms and backends

## Architecture

AlignTune uses a flexible backend architecture:

```mermaid
flowchart TD
    Factory[Backend Factory] --> TRL[TRL Backend]
    Factory --> Unsloth[Unsloth Backend]
    TRL --> TRL_Algos[TRL Algorithms]
    Unsloth --> Unsloth_Algos[Unsloth Algorithms]
```

**TRL Backend:** SFT, DPO, PPO, GRPO, GSPO, DAPO, Dr. GRPO, GBMPO, Counterfactual GRPO, PACE

**Unsloth Backend:** SFT, DPO, PPO, GRPO, DAPO, Dr. GRPO, Counterfactual GRPO, PACE

See [Architecture](docs/advanced/architecture.md) for details.

## Contributing

We welcome contributions! See our [Contributing Guide](docs/contributing/guide.md) for details.

## License

This project is released under the MIT License. Please cite appropriately if used in academic or production projects.
See the [LICENSE](LICENSE) file for details.

**Key Points:**
- **Free for Research & Learning**: Use, modify, and study for personal, academic, or research purposes
- **Source Available**: Full access to source code
- **Commercial Use Restricted**: Requires separate commercial license
- **Contact**: For commercial licensing, partnership, or redistribution rights, contact [support@lexsi.ai](mailto:support@lexsi.ai)

This is **not** an open-source license as defined by OSI, but provides broad access for non-commercial use.

## Citation

If you use AlignTune in your research, please cite:

**BibTeX:**
```bibtex
@software{alignTune2025,
  title        = {{AlignTune}: Modular Toolkit for Post-Training Alignment of Large Language Models},
  author       = {Lyngkhoi, R E Zera Marveen and Chawla, Chirag and Seth, Pratinav and Avaiya, Utsav and Bhattacharjee, Soham and Khandoga, Mykola and Yuan, Rui and Sankarapu, Vinay Kumar},
  year         = {2025},
  note         = {Equal contribution: R E Zera Marveen Lyngkhoi, Chirag Chawla, Pratinav Seth},
  organization = {Lexsi Labs},
  url          = {https://github.com/Lexsi-Labs/aligntune},
  version      = {0.0.0}
}
```

**Plain Text:**
```
Lyngkhoi, R. E. Z. M., Chawla, C., Seth, P., Avaiya, U., Bhattacharjee, S., Khandoga, M.,
Yuan, R., & Sankarapu, V. K. (2025). AlignTune: Modular Toolkit for Post-Training Alignment
of Large Language Models. Lexsi Labs. https://github.com/Lexsi-Labs/aligntune

*Equal contribution: R E Zera Marveen Lyngkhoi, Chirag Chawla, Pratinav Seth
```

## Acknowledgments

AlignTune is built upon the excellent work of the following projects:

- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** - Model architectures and tokenizers
- **[TRL](https://github.com/huggingface/trl)** - Transformer Reinforcement Learning library
- **[Unsloth](https://github.com/unslothai/unsloth)** - Fast and memory-efficient training
- **[HuggingFace Datasets](https://github.com/huggingface/datasets)** - Dataset loading and processing

## Support

- **Documentation**: [aligntune.lexsi.ai/](https://aligntune.lexsi.ai/)
- **GitHub Issues**: [github.com/Lexsi-Labs/aligntune/issues](https://github.com/Lexsi-Labs/aligntune/issues)
- **Discussions**: [github.com/Lexsi-Labs/aligntune/discussions](https://github.com/Lexsi-Labs/aligntune/discussions)
- **Email**: [support@lexsi.ai](mailto:support@lexsi.ai)

---

**Get started with AlignTune and accelerate your LLM fine-tuning workflows today!**
