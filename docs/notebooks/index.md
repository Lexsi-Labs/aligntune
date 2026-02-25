# AlignTune Notebooks

Two notebooks demonstrate the core AlignTune workflows.

<div class="grid cards" markdown>

- __01 - Basic SFT Training__

 ---

 End-to-end Supervised Fine-Tuning on a small dataset for a fast demo.

 :octicons-arrow-right-24: [Open notebook](./01_basic_sft_training.ipynb)

- __02 - DPO Preference Training__

 ---

 Direct Preference Optimization training using a preference dataset.

 :octicons-arrow-right-24: [Open notebook](./02_dpo_preference_training.ipynb)

</div>

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

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

## Demo Notebooks

Interactive Colab notebooks demonstrating various AlignTune workflows:

### Supervised Fine-Tuning (SFT)

| Backend | Model | Dataset | Link |
|---------|-------|---------|------|
| TRL | Gemma-7b | philschmid/dolly-15k-oai-style (messages format) | [Open in Colab](https://colab.research.google.com/drive/1iKnvNS1vanAv_-ZmMbCrN1dbaS01clmE?usp=sharing) |
| TRL | GemmaTX | TrialBench adverse event prediction | [Open in Drive](https://drive.google.com/file/d/1rA80FTpiQHuQJHBs5KyN2UYgU-ie4-lO/view?usp=sharing) |
| Unsloth | Qwen-2.5-0.5-I | gaming | [Open in Colab](https://colab.research.google.com/drive/11DrRbG30MUCegZTDXwR9nxEdvKaLfWCb?usp=sharing) |

### Reinforcement Learning (RL)

| Backend | Algorithm | Dataset | Link |
|---------|-----------|---------|------|
| TRL | DPO | - | [Open in Colab](https://colab.research.google.com/drive/11G3W3otAuD7OqmQBhOqT0n5Hoj6AypfG?usp=sharing) |
| TRL | GRPO | GSM8K | [Open in Colab](https://colab.research.google.com/drive/1A_u9xlJN6oBMtYfUqK1AapsjOFC_qgNG?usp=sharing) |

## Contributing

Want to improve a notebook? See our [Contributing Guide](../contributing/guide.md).