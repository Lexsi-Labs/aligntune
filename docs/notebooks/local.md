# Local Notebooks

## Running the Notebooks

### Local Setup

1. **Clone the repository:**
 ```bash
 git clone https://github.com/Lexsi-Labs/aligntune.git
 cd aligntune
 ```

2. **Install dependencies:**
 ```bash
 pip install -e .
 pip install jupyter notebook
 ```

3. **Launch Jupyter:**
 ```bash
 jupyter notebook notebooks/
 ```

### Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for faster training)

## Local Notebooks (in this repo)

`notebooks/` holds one self-contained notebook per training method or capability.
Each ships with a config cell, a train cell, and optional "push to the Hub"
cells, and defaults to a tiny model with a handful of samples so the API call
can be smoke-tested quickly before scaling up. The first cell installs AlignTune
(`pip install aligntune`); on a machine where it is already installed you can
skip it. Launch with `jupyter notebook notebooks/` from the repo root.

### Core algorithms — `notebooks/01`–`29`

SFT, Sequence Packing, DPO, Online-DPO, ORPO, GRPO, GSPO, RLVR, DAPO,
Dr. GRPO, GBMPO, Counterfactual-GRPO, PACE, PPO, SPIN, Distillation, SDFT,
ES, RAFT, Alignment Audit, and end-to-end Composition. See
[Algorithms](../algorithms/overview.md) for what each one does.

### Long-context, PEFT, merging & tokenization — `notebooks/26`–`47`

One notebook per capability, for techniques that don't have their own dedicated
RL/SFT algorithm page:

| Range | Capability | Docs |
|---|---|---|
| `26`–`28` | RoPE scaling, S2-Attention, Sliding Window Attention | [Long-Context & Attention](../advanced/long-context.md) |
| `32`, `34` | MoA, Text2LoRA / Doc2LoRA | [Advanced Adapters](../advanced/adapters.md) |
| `35`, `42` | Model merging (linear / SLERP / TIES / DARE / task-arithmetic), direct PEFT merge | [Model Merging](../advanced/merging.md) |
| `43`–`46` | Tokenization: continued BPE, naive extension, vocab pruning, FVT embedding init | [Tokenization & Multilingual](../advanced/tokenization.md) |
| `47` | Data curation with CuratorKit | — |

Most core-algorithm notebooks also have an Unsloth-backend counterpart under
`notebooks/unsloth/`.

Standalone (doesn't import AlignTune):
[Tokenizer Embedding Adaptation Experiment](tokenizer_embedding_adaptation_experiment.ipynb) —
continually extends a Hugging Face tokenizer, adapts a model's embeddings without
shrinking padded vocabularies, and compares `mean` vs. `mean_of_constituents`
initialization.
