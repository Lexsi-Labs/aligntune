#!/bin/bash
# Examples of using the AlignTune advisor (cost estimator and algorithm recommendations)

# List available GPUs
echo "=== List Available GPUs ==="
aligntune advise list-gpus
echo ""

# Estimate resources for DPO training
echo "=== Estimate DPO Training Resources ==="
aligntune advise estimate \
    --model "Qwen/Qwen2.5-7B" \
    --dataset-size 10000 \
    --algorithm dpo \
    --gpu a100-40gb
echo ""

# Estimate resources for LoRA fine-tuning (memory efficient)
echo "=== Estimate LoRA Fine-tuning (Memory Efficient) ==="
aligntune advise estimate \
    --model "meta-llama/Llama-2-70b-hf" \
    --dataset-size 50000 \
    --algorithm lora \
    --gpu a100-40gb \
    --batch-size 8
echo ""

# Estimate with QLoRA (ultra memory efficient)
echo "=== Estimate QLoRA Training (Ultra Efficient) ==="
aligntune advise estimate \
    --model "meta-llama/Llama-2-70b-hf" \
    --dataset-size 10000 \
    --algorithm qlora \
    --gpu l4 \
    --batch-size 4
echo ""

# Recommend algorithms for alignment task
echo "=== Recommend Algorithms for Alignment Task ==="
aligntune advise recommend \
    --task "alignment optimization" \
    --dataset-size 10000
echo ""

# Recommend algorithms with budget constraint
echo "=== Recommend Algorithms with Budget Constraint ==="
aligntune advise recommend \
    --task "general fine-tuning" \
    --dataset-size 50000 \
    --budget 15.0
echo ""

# Recommend algorithms for fast training
echo "=== Recommend Algorithms for Speed ==="
aligntune advise recommend \
    --task "fast training" \
    --dataset-size 5000 \
    --model-size 7b
echo ""

# Get optimization suggestions for 70B model
echo "=== Optimization Suggestions for 70B Model ==="
aligntune advise optimize \
    --model-size 70b \
    --precision fp32 \
    --gpu a100-40gb \
    --dataset-size 100000
echo ""

# Get optimization suggestions with tight VRAM
echo "=== Optimization Suggestions (Tight VRAM) ==="
aligntune advise optimize \
    --model-size 13b \
    --precision fp32 \
    --gpu l4 \
    --dataset-size 50000 \
    --vram-tight
echo ""

# Get optimization suggestions for QLoRA setup
echo "=== Optimization Suggestions for QLoRA ==="
aligntune advise optimize \
    --model-size 70b \
    --precision int4 \
    --gpu t4 \
    --dataset-size 10000
echo ""

echo "All examples completed!"
