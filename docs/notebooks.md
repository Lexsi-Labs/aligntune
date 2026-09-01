# AlignTune Notebooks

This page contains a collection of interactive Colab and local Jupyter notebooks demonstrating various AlignTune workflows. These notebooks are designed to be a "living" introduction to the library's capabilities.

For local setup instructions see [Local Notebooks](notebooks/local.md); for the
full Colab demo table see [Demo Notebooks](notebooks/demo.md).

---

## Local Notebooks (in this repo)

`notebooks/` holds one self-contained notebook per training method or capability —
config cell, train cell, and optional "push to the Hub" cells — defaulting to a
tiny model and a handful of samples so the API call can be smoke-tested quickly.
Launch with `jupyter notebook notebooks/` from the repo root.

### Core algorithms, `notebooks/01`–`29`

SFT, Sequence Packing, DPO, Online-DPO, ORPO, GRPO, GSPO, RLVR, DAPO,
Dr. GRPO, GBMPO, Counterfactual-GRPO, PACE, PPO, SPIN, Distillation, SDFT, ES,
RAFT, Alignment Audit, and end-to-end Composition. See
[Algorithms](algorithms/overview.md) for what each one does.

### Long-context, PEFT, merging & tokenization, `notebooks/26`–`47`

One notebook per capability, for techniques that don't have their own dedicated RL/SFT algorithm page:

| Range | Capability | Docs |
|---|---|---|
| `26`–`28` | RoPE scaling, S2-Attention, Sliding Window Attention | [Long-Context & Attention](advanced/long-context.md) |
| `32`, `34` | MoA, Text2LoRA/Doc2LoRA | [Advanced Adapters](advanced/adapters.md) |
| `35`, `42` | Model merging (linear / SLERP / TIES / DARE / task-arithmetic), direct PEFT merge | [Model Merging](advanced/merging.md) |
| `43`–`46` | Tokenization: continued BPE, naive extension, vocab pruning, FVT embedding init | [Tokenization & Multilingual](advanced/tokenization.md) |
| `47` | Data curation with CuratorKit | — |

Most core-algorithm notebooks also have an Unsloth-backend counterpart under `notebooks/unsloth/`.

---

## Demo Notebooks (Colab)

Full table of Colab-hosted demo notebooks by backend/algorithm: see [Demo Notebooks](notebooks/demo.md) on the Notebooks page.
