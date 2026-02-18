# AlignTune Notebooks

## Running the Notebooks

### Local Setup

1. **Clone the repository:**
 ```bash
 git clone https://github.com/Lexsi-Labs/aligntune.git
 cd AlignTune
 ```

2. **Install dependencies:**
 ```bash
 pip install -e .
 pip install jupyter notebook
 ```

3. **Launch Jupyter:**
 ```bash
 jupyter notebook docs/notebooks/
 ```

### Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

## Demo Notebooks

Interactive Colab notebooks demonstrating various AlignTune workflows:

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
