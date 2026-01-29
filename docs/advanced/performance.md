# Performance Optimization

Tips for faster training and lower memory use with AlignTune.

## Backend Choice

- **Unsloth**: Generally faster and more memory-efficient for supported models (SFT, DPO, PPO, GRPO, etc.). Prefer when you have a compatible GPU and model.
- **TRL**: Use when you need GSPO or maximum compatibility; still supports mixed precision, LoRA, and gradient checkpointing.

## Memory

- Enable **gradient checkpointing** to trade compute for memory.
- Use **LoRA/QLoRA** (e.g., `use_peft=True`, 4-bit via Unsloth) to reduce VRAM.
- Reduce **batch size** and increase **gradient_accumulation_steps** to keep effective batch size.
- Reduce **max sequence length** when possible.

## Speed

- Use **mixed precision** (e.g., bf16/fp16) when your hardware supports it.
- Prefer **Unsloth** for supported setups.
- Increase **batch size** (within memory limits) and tune **gradient_accumulation_steps**.

## Next Steps

- [Backend Support Matrix](../compatibility/backend-matrix.md) - Algorithm and backend support
- [Troubleshooting](../user-guide/troubleshooting.md) - Common issues
- [Architecture](architecture.md) - System design
