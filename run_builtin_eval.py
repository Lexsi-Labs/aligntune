#!/usr/bin/env python3
"""Run built-in aligntune eval on GSM8K with vLLM."""
import gc
import torch

from aligntune.eval.runner import EvalConfig, run_eval


def main():
    # --- Eval base model ---
    print("=" * 60)
    print("Evaluating BASE model with built-in eval (vLLM)")
    print("=" * 60)

    config_base = EvalConfig(
        model_path="Qwen/Qwen3-1.7B",
        output_dir="./eval_results/builtin_gsm8k_base_vllm",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        split="test",
        task_type="math",
        batch_size=128,
        use_vllm=True,
        precision="bf16",
        max_length=2048,
        max_new_tokens=2048,
        do_sample=False,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
    )

    results_base = run_eval(config_base)
    base_acc = results_base.get('math_accuracy', 'N/A')
    print(f"\nBase model math_accuracy: {base_acc}")

    # Free memory
    gc.collect()
    torch.cuda.empty_cache()

    # --- Eval trained model (counterfact GRPO with length norm) ---
    print("\n" + "=" * 60)
    print("Evaluating TRAINED model with built-in eval (vLLM)")
    print("=" * 60)

    config_trained = EvalConfig(
        model_path="./output/qwen3_counterfact_grpo_200_lenorm/checkpoint-200",
        base_model="Qwen/Qwen3-1.7B",
        output_dir="./eval_results/builtin_gsm8k_lenorm_vllm",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        split="test",
        task_type="math",
        batch_size=128,
        use_vllm=True,
        use_lora=True,
        precision="bf16",
        max_length=2048,
        max_new_tokens=2048,
        do_sample=False,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
    )

    results_trained = run_eval(config_trained)
    trained_acc = results_trained.get('math_accuracy', 'N/A')
    print(f"\nTrained model math_accuracy: {trained_acc}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY (Built-in eval, vLLM, NO chat template)")
    print("=" * 60)
    print(f"Base Qwen3-1.7B:          {base_acc}")
    print(f"Counterfact GRPO+LenNorm: {trained_acc}")


if __name__ == '__main__':
    main()
