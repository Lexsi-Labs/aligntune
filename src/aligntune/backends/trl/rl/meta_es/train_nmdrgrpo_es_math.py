# train_nmdrgrpo_es_math.py
# Neural Mirror Dr GRPO training on GSM8K with ES support
# Dr GRPO improvements: loss_type="dr_grpo" (eliminates length bias), scale_rewards=False (eliminates difficulty bias)
# - 126-neuron parametrized mirror map for learnable divergence
# - Exact-match reward on GSM8K numeric answer
# - ES MODE: Load pre-computed mirror parameters from ES coordinator
# - Supports both full-parameter and LoRA training

import re
import math
import argparse
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from neural_mirror_grpo import NeuralMirrorGRPOConfig, NeuralMirrorGRPOTrainer
from peft import LoraConfig, get_peft_model
from logging_config import get_adaptive_logging_config, print_logging_config

# -----------------------
# Helpers: answer parsing
# -----------------------
ANS_RE = re.compile(r"####\s*(.+)$", flags=re.MULTILINE)


def parse_gsm8k_gold(ans: str) -> str:
    """Extract the numeric target from GSM8K 'answer' field (after '####')."""
    m = ANS_RE.search(ans)
    if not m:
        # Fallback: last number in the string
        nums = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        return nums[-1] if nums else ""
    s = m.group(1).strip()
    s = s.replace(",", "")
    # Handle simple fractions like "3/4"
    if re.fullmatch(r"-?\d+/\d+", s):
        num, den = s.split("/")
        try:
            return str(float(num) / float(den))
        except Exception:
            return s
    return s


def parse_pred_number(text: str) -> str:
    """Extract the last number from the output."""
    # Look for the last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    candidate = nums[-1] if nums else ""
    # Normalize simple fractions
    if re.fullmatch(r"-?\d+/\d+", candidate):
        num, den = candidate.split("/")
        try:
            return str(float(num) / float(den))
        except Exception:
            return candidate
    return candidate


def numeric_equal(a: str, b: str, rtol=1e-4, atol=1e-8) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    except Exception:
        # Pure string fallback
        return a.strip() == b.strip()


# -----------------------
# Reward functions
# -----------------------
def exact_answer_reward(
        completions: List[List[dict]], answers: List[str], **kwargs) -> List[float]:
    """
    0/1 reward based on exact numeric match.
    `completions` is a list of completion strings.
    `answers` is the GSM8K gold answer (one per prompt). We expand it to completions.
    """
    # Expand answers to match completions length if needed
    if len(answers) == 0:
        return [0.0] * len(completions)
    factor = max(1, len(completions) // len(answers))
    expanded = [a for a in answers for _ in range(factor)]
    expanded = expanded[: len(completions)]

    rewards = []
    for comp, gold in zip(completions, expanded):
        # Handle both string and dict formats
        if isinstance(comp, str):
            text = comp
        elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
            text = comp[0]["content"]
        else:
            print(f"Warning: Unexpected completion format: {type(comp)}")
            text = str(comp)

        pred = parse_pred_number(text)
        rewards.append(1.0 if numeric_equal(pred, gold) else 0.0)
    return rewards


# -----------------------
# Build GSM8K prompts
# -----------------------
SYSTEM = "You are a careful math tutor. Solve step by step and provide the final numeric answer."


def to_prompt(q: str, tokenizer=None) -> str:
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        # Use proper chat template for Qwen
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Problem:\n{q}"}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            # Switches between thinking and non-thinking modes. Default is
            # True.
        )
    else:
        # Fallback to plain text format
        return f"{SYSTEM}\n\n" f"Problem:\n{q}\n\n" "Answer:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        type=str,
        help="Model to use for Neural Mirror Dr GRPO")
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument(
        "--max_steps",
        default=1000,
        type=int,
        help="Training steps for Neural Mirror Dr GRPO")
    parser.add_argument("--per_device_train_batch_size", default=2, type=int)
    parser.add_argument(
        "--num_generations",
        default=4,
        type=int,
        help="Number of generations per prompt")
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--max_prompt_length", default=512, type=int)
    parser.add_argument("--max_completion_length", default=256, type=int)
    parser.add_argument(
        "--output_dir",
        default="./Qwen3-4B-NeuralMirrorDrGRPO-GSM8K",
        type=str)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for training")
    parser.add_argument("--lora_r", default=64, type=int, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha",
        default=64,
        type=int,
        help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout",
        default=0.1,
        type=float,
        help="LoRA dropout")
    parser.add_argument(
        "--wandb_project",
        default="nmdrgrpo-math-training",
        type=str,
        help="Wandb project name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name (auto-generated if not provided)")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging")
    # Neural Mirror Dr GRPO specific parameters
    parser.add_argument(
        "--mirror_coefficient",
        default=0.0001,
        type=float,
        help="Bregman divergence regularization coefficient")
    parser.add_argument(
        "--mirror_init_scale",
        default=0.01,
        type=float,
        help="Initialization scale for mirror map parameters")
    parser.add_argument(
        "--mirror_seed",
        default=42,
        type=int,
        help="Random seed for mirror map initialization")
    # ES mode parameters
    parser.add_argument(
        "--es_mode",
        action="store_true",
        help="Enable ES mode (uses 80% train split for inner training)")
    parser.add_argument(
        "--mirror_params_path",
        type=str,
        help="Path to pre-computed mirror map parameters (for ES)")
    args = parser.parse_args()

    # Configure run naming (used by both TRL and wandb)
    run_name = (
        args.wandb_run_name
        or f"{args.model.split('/')[-1]}-nmdrgrpo-gsm8k-mc{args.mirror_coefficient}-seed{args.mirror_seed}-lr{args.lr}-steps{args.max_steps}"
    )

    # Configure wandb
    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                "algorithm": "NeuralMirrorDrGRPO",
                "divergence_type": "neural_mirror",
                "loss_type": "dr_grpo",
                "scale_rewards": False,
                "mirror_map_neurons": 126,
                "dataset": "gsm8k",
            },
        )

    print("=" * 80)
    print("Neural Mirror Dr GRPO Math Training on GSM8K")
    print("=" * 80)
    print(f"Algorithm: Neural Mirror Dr GRPO (Learnable Bregman divergence + GRPO Done Right)")
    print(f"Dr GRPO optimizations:")
    print(f"  - loss_type='dr_grpo': eliminates response-level length bias")
    print(f"  - scale_rewards=False: eliminates question-level difficulty bias")
    print(f"Model: {args.model}")
    print(f"Dataset: GSM8K ({args.split} split)")
    print(f"Mirror coefficient: {args.mirror_coefficient}")
    print(f"Mirror init scale: {args.mirror_init_scale}")
    print(f"Mirror seed: {args.mirror_seed}")
    print(f"Mirror map: 126 neurons (6 activation types)")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    print(
        f"Batch size: {
            args.per_device_train_batch_size} x {
            args.grad_accum} = {
                args.per_device_train_batch_size *
            args.grad_accum}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"LoRA: {'Yes' if args.use_lora else 'No'}")
    print("=" * 80)

    # 1) Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # safer for generation training

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if (
            args.bf16 and torch.cuda.is_available()) else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Critical: Enable gradient checkpointing and disable cache for RL training
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Apply LoRA if requested
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(
            f"✅ LoRA applied with rank={
                args.lora_r}, alpha={
                args.lora_alpha}")
        model.print_trainable_parameters()

        # Debug: Check LoRA parameter devices and gradients
        print("\nLoRA Parameter Debug:")
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                print(
                    f"  {name}: device={
                        param.device}, shape={
                        param.shape}, requires_grad={
                        param.requires_grad}")
                break  # Just show first LoRA param

    # 2) Dataset: GSM8K train split -> {"prompt", "answer_clean"}
    # q: 'question', a: 'answer'
    ds = load_dataset("openai/gsm8k", "main", split=args.split)

    # ES mode: Use only 80% of train split for inner training (leave 20% for
    # validation)
    if args.es_mode:
        print(f"🔬 ES MODE: Using 80% of GSM8K train split for inner training")
        total_size = len(ds)
        train_size = int(0.8 * total_size)
        ds = ds.select(range(train_size))
        print(
            f"   Original: {total_size} examples → Inner train: {
                len(ds)} examples")

    # Dataset loaded successfully
    print(f"✅ Dataset loaded: {len(ds)} examples")

    def _map(ex):
        gold = parse_gsm8k_gold(ex["answer"])
        prompt = to_prompt(ex["question"], tokenizer)
        return {"prompt": prompt, "answer_clean": gold}

    ds = ds.map(
        _map, remove_columns=[
            c for c in ds.column_names if c not in (
                "prompt", "answer_clean")])

    # Dataset processed successfully
    print(f"✅ Dataset processed: prompts and answers extracted")

    # 3) Rewards: exact answer only
    def exact_answer_reward_wrapped(completions, prompts=None, **kwargs):
        # NMDRGRPO should pass prompts to help us match answers correctly
        if prompts is not None:
            # Extract answers from the prompts by matching against our dataset
            prompt_to_answer = {
                item["prompt"]: item["answer_clean"] for item in ds}
            matched_answers = []
            for prompt in prompts:
                if prompt in prompt_to_answer:
                    matched_answers.append(prompt_to_answer[prompt])
                else:
                    # Fallback: search for similar prompt
                    print(
                        f"WARNING: Could not find exact prompt match. Prompt starts with: {prompt[:100]}...")
                    matched_answers.append("0")  # Default to wrong answer

            rewards = exact_answer_reward(completions, answers=matched_answers)
        else:
            # Fallback to old method (likely buggy)
            print("WARNING: No prompts provided to reward function - using dataset order")
            rewards = exact_answer_reward(
                completions, answers=list(
                    ds["answer_clean"]))

        # Debug: Print first few examples every step
        if hasattr(exact_answer_reward_wrapped, "step_count"):
            exact_answer_reward_wrapped.step_count += 1
        else:
            exact_answer_reward_wrapped.step_count = 1

        # Print basic progress every step
        print(
            f"\n🔄 NMDRGRPO STEP {
                exact_answer_reward_wrapped.step_count}: Reward computed - Mean: {
                sum(rewards) / len(rewards):.4f}, Correct: {
                sum(
                    1 for r in rewards if r > 0)}/{
                        len(rewards)}")

        # Enhanced debug output that combines math accuracy with neural mirror
        # regularization info
        if True:  # Print detailed analysis every step
            print("\n" + "=" * 120)
            print(
                f"NEURAL MIRROR DR GRPO MATH DEBUG STEP {
                    exact_answer_reward_wrapped.step_count}: COMPLETE GENERATION ANALYSIS")
            print("=" * 120)

            # Show first prompt fully
            if prompts is not None and len(prompts) > 0:
                print("FIRST PROMPT (FULL):")
                print("-" * 60)
                print(prompts[0])
                print("-" * 60)
                print()

            # Group completions by prompt (NMDRGRPO generates multiple per
            # prompt)
            num_gens = args.num_generations if hasattr(
                args, "num_generations") else 4
            num_prompts = len(prompts) if prompts else len(
                completions) // num_gens

            print(
                f"ANALYSIS: {
                    len(completions)} completions, {num_prompts} prompts, {num_gens} generations per prompt")
            print()

            # Show all generations for first prompt
            for gen_idx in range(min(num_gens, len(completions))):
                comp = completions[gen_idx]
                reward = rewards[gen_idx]

                # Handle both string and dict formats for display
                if isinstance(comp, str):
                    text = comp
                elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
                    text = comp[0]["content"]
                else:
                    text = str(comp)

                predicted = parse_pred_number(text)

                if prompts is not None and gen_idx < len(matched_answers):
                    gold_answer = matched_answers[gen_idx]
                else:
                    gold_answer = "UNKNOWN"

                print(f"GENERATION {gen_idx + 1}:")
                print(f"Full Response: {text}")
                print(f"Extracted Answer: '{predicted}'")
                print(f"Gold Answer: '{gold_answer}'")
                print(f"Reward: {reward} ({'✓' if reward > 0 else '✗'})")
                print("-" * 80)

            # Summary statistics
            print(f"\nSUMMARY:")
            print(f"Total Rewards - Mean: {sum(rewards) / len(rewards):.4f}")
            print(
                f"Correct Answers: {
                    sum(
                        1 for r in rewards if r > 0)}/{
                    len(rewards)} ({
                    100 * sum(
                        1 for r in rewards if r > 0) / len(rewards):.1f}%)")
            print(
                f"Neural Mirror Dr GRPO: Watch for [NEURAL MIRROR] initialization and [MIRROR DEBUG] messages showing Bregman divergence"
            )
            print("=" * 120 + "\n")

            # Force output flush
            import sys

            sys.stdout.flush()

        return rewards

    reward_fcns = [
        exact_answer_reward_wrapped,
    ]
    reward_weights = [1.0]

    # 4) Neural Mirror Dr GRPO config — learnable Bregman divergence with Dr
    # GRPO optimizations

    # Adaptive logging configuration based on training length
    logging_steps, save_steps, save_total_limit = get_adaptive_logging_config(
        args.max_steps)
    print_logging_config(
        args.max_steps,
        logging_steps,
        save_steps,
        save_total_limit)

    gen_batch_size = args.per_device_train_batch_size * \
        args.grad_accum  # must be divisible by num_generations
    training_args = NeuralMirrorGRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        # Neural Mirror Dr GRPO specific settings
        divergence_type="neural_mirror",
        mirror_coefficient=args.mirror_coefficient,
        mirror_init_scale=args.mirror_init_scale,
        mirror_seed=args.mirror_seed,
        beta=0.0,  # Disable KL regularization
        # Dr GRPO specific settings
        loss_type="dr_grpo",  # Use Dr GRPO loss (eliminates length bias)
        scale_rewards=False,
        # Do not scale rewards (eliminates difficulty bias)
        # Learning settings
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16 and torch.cuda.is_available(),
        max_steps=args.max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=["wandb"] if not args.no_wandb else [],
        remove_unused_columns=False,  # keep our extra 'answer_clean' column around
        # Learning rate scheduler
        lr_scheduler_type="cosine",
        warmup_steps=int(0.1 * args.max_steps),  # 10% warmup
        # Generation params
        num_generations=args.num_generations,
        generation_batch_size=gen_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=0.6,
        top_p=0.95,
        mask_truncated_completions=True,
        # Neural Mirror Dr GRPO specifics
        epsilon=0.2,  # PPO-style clip
        reward_weights=reward_weights,
        gradient_checkpointing=True,
        tf32=torch.cuda.is_available() and not (
            args.bf16 and torch.cuda.is_available()),
        # Critical for numerical stability
        max_grad_norm=1.0,  # Gradient clipping for numerical stability
    )

    print(f"\n🚀 Neural Mirror Dr GRPO Training Configuration:")
    print(f"   Divergence type: {training_args.divergence_type}")
    print(f"   Mirror coefficient: {training_args.mirror_coefficient}")
    print(f"   Mirror init scale: {training_args.mirror_init_scale}")
    print(f"   Mirror seed: {training_args.mirror_seed}")
    print(f"   Mirror map neurons: 126 (21 per activation type)")
    print(f"   Beta (KL): {training_args.beta} (disabled)")
    print(
        f"   Loss type: {
            training_args.loss_type} (Dr GRPO - eliminates length bias)")
    print(
        f"   Scale rewards: {
            training_args.scale_rewards} (eliminates difficulty bias)")
    print(f"   Max grad norm: {training_args.max_grad_norm}")
    print(
        f"   Expected debug: [NEURAL MIRROR] + [MIRROR DEBUG] messages during training")
    print()

    # 5) Neural Mirror Dr GRPO Trainer
    trainer = NeuralMirrorGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fcns,
        args=training_args,
        train_dataset=ds,
    )

    # ES mode: Load pre-computed mirror map parameters
    if args.mirror_params_path:
        print(
            f"\n🔬 ES MODE: Loading pre-computed mirror map parameters from {args.mirror_params_path}")
        mirror_params = torch.load(args.mirror_params_path, map_location="cpu")

        # Load parameters into trainer's mirror module
        trainer.mirror_module.v.copy_(mirror_params["v"])
        trainer.mirror_module.w.copy_(mirror_params["w"])
        trainer.mirror_module.b.copy_(mirror_params["b"])
        trainer.mirror_module.a.copy_(mirror_params["a"])
        trainer.mirror_module.c.copy_(mirror_params["c"])

        print(f"✅ ES MODE: Mirror map parameters loaded successfully")
        print(f"   Parameter ranges:")
        print(
            f"   v: [{
                mirror_params['v'].min():.6f}, {
                mirror_params['v'].max():.6f}]")
        print(
            f"   w: [{
                mirror_params['w'].min():.6f}, {
                mirror_params['w'].max():.6f}]")
        print(
            f"   b: [{
                mirror_params['b'].min():.6f}, {
                mirror_params['b'].max():.6f}]")
        print(f"   a: {mirror_params['a'].item():.6f}")
        print(f"   c: {mirror_params['c'].item():.6f}")

    print("\n🚀 Starting Neural Mirror Dr GRPO training on GSM8K...")
    print(
        "Expected output: Math accuracy + [NEURAL MIRROR] initialization + [MIRROR DEBUG] Bregman divergence analysis"
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save mirror map parameters for analysis/ES
    mirror_params_path = f"{args.output_dir}/mirror_map_params.pt"
    torch.save(
        {
            "v": trainer.mirror_module.v,
            "w": trainer.mirror_module.w,
            "b": trainer.mirror_module.b,
            "a": trainer.mirror_module.a,
            "c": trainer.mirror_module.c,
            "seed": args.mirror_seed,
            "init_scale": args.mirror_init_scale,
        },
        mirror_params_path,
    )

    print(
        "✅ Done. Neural Mirror Dr GRPO model and mirror map parameters saved to:",
        args.output_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("NEURAL MIRROR Dr GRPO MATH TRAINING COMPLETED")
    print("=" * 80)
    print(f"Algorithm: Neural Mirror Dr GRPO with 126-neuron mirror map")
    print(f"Dr GRPO optimizations:")
    print(f"  - loss_type='dr_grpo': constant normalization (eliminates length bias)")
    print(f"  - scale_rewards=False: no std scaling (eliminates difficulty bias)")
    print(f"Mirror coefficient: {args.mirror_coefficient}")
    print(f"Training completed: {args.max_steps} steps")
    print(f"Model saved to: {args.output_dir}")
    print(f"Mirror map parameters saved to: {mirror_params_path}")
    print(f"Check logs above for:")
    print(f"  - Math accuracy progress")
    print(f"  - [NEURAL MIRROR] initialization messages")
    print(f"  - [MIRROR DEBUG] Bregman divergence analysis")
    print("=" * 80)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
