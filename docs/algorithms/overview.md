# RL Algorithms Overview

AlignTune supports a comprehensive set of Reinforcement Learning algorithms for aligning Large Language Models with human preferences and optimizing for specific objectives. Algorithms below are grouped by family, pick the family that matches your data/objective, then the specific algorithm within it.

## Algorithm Directory

<div class="algo-family-label">Preference-pair methods</div>
<div class="grid cards" markdown>

-   __[DPO](dpo.md)__

    ---

    Chosen/rejected pairs, no reward model needed.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[Online-DPO](online-dpo.md)__

    ---

    DPO run iteratively inside an active-learning loop against a reward model.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[ORPO](orpo.md)__

    ---

    Combines SFT and preference alignment in one pass, no reference model.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

</div>

<div class="algo-family-label">Group-relative RL (GRPO family)</div>
<div class="grid cards" markdown>

-   __[GRPO](grpo.md)__

    ---

    Group-scored responses, no separate value/critic model.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[GSPO](gspo.md)__

    ---

    Thin GRPO subclass: sequence-level (not per-token) importance sampling.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[DAPO](dapo.md)__

    ---

    Thin GRPO subclass: decoupled clip / dynamic sampling loss variant.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[Dr. GRPO](dr-grpo.md)__

    ---

    Thin GRPO subclass: debiased loss variant ("GRPO Done Right").

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[GBMPO](gbmpo.md)__

    ---

    Mirror-descent trust-region updates on top of GRPO-style grouped sampling.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[Counterfactual GRPO](counterfactual-grpo.md)__

    ---

    Baseline-swapping penalty that neutralizes reward hacking (e.g. length hacking).

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[PACE](pace.md)__

    ---

    Per-prompt learned baseline tracking.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

</div>

<div class="algo-family-label">Classic policy-gradient / online RL</div>
<div class="grid cards" markdown>

-   __[PPO](ppo.md)__

    ---

    Reward model + value network, fine-grained control over reward signals.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

-   __[SPIN](spin.md)__

    ---

    Self-play against prior checkpoints, no external preference labels.

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

</div>

<div class="algo-family-label">Specialized trainers</div>
<div class="grid cards" markdown>

-   __[RAFT](raft.md)__

    ---

    Document-grounded SFT with golden + distractor context (separate factory function).

    <span class="chip-row"><span class="chip chip-trl">TRL</span><span class="chip chip-unsloth">Unsloth</span></span>

</div>

---

## Algorithm Comparison

| Algorithm | TRL Backend | Unsloth Backend | Reward Model | Description |
|-----------|-------------|-----------------|--------------|-------------|
| **DPO** | Yes | Yes | No | Direct Preference Optimization |
| **Online-DPO**| Yes | Yes | Optional | Online Iterative DPO |
| **PPO** | Yes | Yes | Yes | Proximal Policy Optimization |
| **GRPO** | Yes | Yes | No | Group Relative Policy Optimization |
| **GSPO** | Yes | Yes | No | Group Sequence Policy Optimization |
| **DAPO** | Yes | Yes | No | Decouple Clip and Dynamic sAmpling PO |
| **Dr. GRPO** | Yes | Yes | No | GRPO Done Right - Unbiased GRPO variant |
| **GBMPO** | Yes | Yes | No | Group-Based Mirror Policy Optimization |
| **C-GRPO** | Yes | Yes | No | Counterfactual GRPO |
| **PACE** | Yes | Yes | No | Baseline-Optimized Learning Technique |
| **ORPO** | Yes | Yes | No | Odds Ratio Preference Optimization |
| **SPIN** | Yes | Yes | No | Self-Play Fine-Tuning |
| **RAFT** | Yes | Yes | No | Retrieval Augmented Fine-Tuning (document-grounded SFT) |

---

## Algorithm Selection Guide

### When to Use DPO
**Use DPO when:**
- You have preference data (chosen vs rejected pairs)
- You want to avoid training a separate reward model
- You need fast training with minimal setup

### When to Use PPO
**Use PPO when:**
- You have a reward model (or want to train one)
- You need fine-grained control over reward signals
- You're doing online learning with environment interaction
- You want to optimize for complex, multi-objective rewards

### When to Use GRPO (and Variants like Dr. GRPO, GSPO, DAPO, GBMPO, PACE, C-GRPO)
**Use GRPO-based algorithms when:**
- You want multi-criteria optimization without a separate critic model
- You're working with group-based preferences/relative scoring
- You need memory-efficient RL (no value model required)
- *Note:* Use **GSPO** for sequential tasks, **DAPO** for dynamic sampling, **Dr. GRPO** to avoid biases, **GBMPO** for mirror-descent updates, or **PACE** for baseline optimization.

### When to Use ORPO
**Use ORPO when:**
- You want to combine SFT and preference alignment in a single training pass
- You have paired preference data (chosen/rejected) but want to skip a separate reference model

### When to Use SPIN
**Use this when:**
- **SPIN**: You want the model to improve itself through self-play without external human preference data.

---

## Next Steps

- **[Backend Selection](../getting-started/backend-selection.md)** - Choose the right backend
- **[Reward System](../user-guide/reward-functions.md)** - Create or train reward functions
