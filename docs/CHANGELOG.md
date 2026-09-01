# Changelog

All notable changes to AlignTune are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.10] — Initial public release

### Highlights

- **Distillation framework**: Standard (external frozen teacher) and SDFT (self-distillation, no external teacher required) methods under a single unified config.
- **Shared reward-handling for RL trainers**: single or multiple reward functions, averaged per completion, with both synchronous and concurrent-async computation paths, shared consistently across the GRPO-family trainers.
- **ES backend rollout abstraction**: generation for the gradient-free ES backend is abstracted behind a common interface, with a standard `transformers`-based implementation and a vLLM-backed implementation (continuous batching, PagedAttention, LoRA-adapter support) for substantially faster generation at scale.
- **GBMPO**: four divergence variants are available through a single configurable trainer, selected via one `divergence_type` option.
- **SFT trainer & PEFT**: SFT trainers are split by task type (text generation, classification, token classification) behind a common entry point. A dedicated PEFT module supports multiple adapter variants (standard LoRA with rsLoRA/LoftQ init, plus experimental variants) behind one factory interface. Model loading (HF, Unsloth, quantized, VLM) is centralized in a single loader.
- **VLM (vision-language model) fine-tuning support** via a dedicated model/dataset configuration — use `VLMModelConfig` or set `is_vlm=True` on `ModelConfig`.
- **SPIN** (self-play fine-tuning): trains against its own prior-round outputs rather than a fixed opponent checkpoint.
- **Online DPO**: built on TRL's online DPO trainer, supporting reward functions passed as callables, string reward types, or weighted configs.
- **GRPO-family reward setup**: reward configuration is shared consistently across the whole family (GRPO, GSPO, DAPO, DR-GRPO), and the data pipeline preserves all original dataset columns so reward functions can access columns that earlier preprocessing would otherwise have dropped.

### Note

If you construct a TRL trainer manually alongside AlignTune, TRL's own `SFTTrainer` takes `processing_class=` rather than the older `tokenizer=` argument.

### Known issues

- Several GRPO-family algorithms on the Unsloth backend are incompatible with `trl>=1.0` because of a version mismatch in Unsloth's own generated trainer code. AlignTune currently pins an older, compatible `trl` version; bumping it repo-wide is tracked as future work and would additionally require updating the PPO backend, which depends on APIs removed in `trl>=1.0`.
