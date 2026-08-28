# AlignTune 2026: Utility-First Post-Training Roadmap

This document is a research roadmap for `aligntune.core`: proposed post-training
techniques focused specifically on model utility, not general-purpose alignment.

AlignTune targets four capabilities (reasoning, math, coding, and factuality)
and automates the multi-stage pipelines used to train frontier reasoning models
like DeepSeek R1 and Llama 3.1, rather than wrapping an existing RLHF library.

The sections below describe eight proposed techniques for `aligntune.core`.
None are implemented yet.

---

## 🏭 1. The Autonomous "R1" Pipeline: Reasoning Utility Scaler

Standard SFT datasets are fixed in advance. Frontier reasoning models instead
generate their own training data by exploring many solution paths and keeping
the ones that work.

**The Concept:**
An automated **Rejection Sampling Post-Training Pipeline** (`aligntune.core.composition.rejection_sampler`).

**The Pipeline Flow in AlignTune:**
1. **Exploration:** generate $N$ reasoning trajectories per problem for complex logic/math tasks.
2. **Filtering:** a step-level reward model or reward ensemble keeps the trajectories that reach the correct answer through sound reasoning, and discards the rest.
3. **Recursive training:** the kept trajectories are fed back into the SFT/DPO pipeline as synthetic preferred examples.
4. **Convergence check:** training stops once benchmark scores (GSM8K, HumanEval) plateau, instead of running a fixed number of cycles.

> [!TIP]
> Automates the R1-Zero self-improvement loop end to end, with no manual data labeling required to turn a base model into a reasoning model.

---

## ⚖️ 2. RLVR (Reinforcement Learning from Verifiable Rewards)

Neural reward models can be gamed by verbose or well-formatted output that
reads as correct without being correct. Verifiable rewards score against a
deterministic check instead of a learned preference model.

**The Concept:**
A natively integrated **Verifiable Reward Sandbox** (`aligntune.rewards.verifiable`) for code and logic.

**The Pipeline Flow in AlignTune:**
1. **Execution:** during GRPO/PPO, the model generates executable Python/SQL/JSON.
2. **Validation:** AlignTune runs it in a secured sandbox against unit tests or schema validators.
3. **Deterministic reward:** a `+1.0` reward is granted only if the code compiles and passes every functional test.
4. **Anchoring:** rewards actual test-passing behavior instead of output a neural judge happens to prefer.

> [!IMPORTANT]
> Replaces subjective preference scoring with pass/fail verification, the same approach behind current coding and math alignment results.

---

## 🧠 3. Capability-Specific MoA (Mixture of Adapters) RLHF

Training a specific capability with standard RLHF usually degrades performance
elsewhere. Routing-only RL avoids that by leaving the underlying experts
untouched.

**The Concept:**
Applying RL exclusively to the **routing matrix** of a multi-expert adapter setup.

**The Pipeline Flow in AlignTune:**
1. **Expert initialization:** load specialized LoRA experts (e.g. "SQL", "Creative Writing", "Reasoning").
2. **Routing:** an MLP router decides which expert combination to activate per prompt.
3. **Training:** RL updates only the router (learning which experts to activate for which prompts) without touching expert weights.

> [!NOTE]
> Avoids the regression where a model improves at one skill (say, coding) at the cost of another (say, math), since the experts themselves never receive RL updates.

---

## ✂️ 4. PKE (Precision Knowledge Editing) for Factuality Utility

Standard fine-tuning is a blunt tool for fixing a single factual error. It
touches the whole model instead of the specific layers holding the wrong
fact.

**The Concept:**
A **Knowledge Repair Trainer** (`aligntune.core.pke.KnowledgeRepairTrainer`) for factual updates.

**The Pipeline Flow in AlignTune:**
1. **Localization:** identify the exact MLP layers responsible for a factual error using causal tracing.
2. **Targeted update:** apply a pinpoint gradient update to correct the fact, without a whole-model training run.
3. **Consistency check:** verify the fix holds across different phrasings of the same question.

---

## ⚡ 5. Curriculum-Aged RLHF (Compute-Utility Optimization)

Not every training sample needs the same compute. Easy samples burn rollout
budget without teaching the model anything new.

**The Concept:**
A **Complexity-Aware RL Controller** that dynamically manages the compute budget per sample.

**The Pipeline Flow in AlignTune:**
1. **Difficulty probing:** the Advisor scores training samples for logic complexity.
2. **Dynamic depth:** the trainer skips layers or reduces rollout depth for easy samples, and increases compute for hard ones (e.g. advanced calculus).
3. **Result:** more compute goes to hard samples, on the same hardware, instead of being spent evenly across easy and hard cases alike.

---

## 📊 6. Contrastive Utility Infusion (CUI)

Standard alignment trains on clearly right vs. clearly wrong answers. This
targets a harder case: reasoning that looks correct but fails at one specific
step.

**The Concept:**
A contrastive training loop that targets **logically plausible but incorrect** traces.

**The Pipeline Flow in AlignTune:**
1. **Error mining:** the model generates reasoning chains that look correct but fail at a specific, verifiable step.
2. **Contrastive loss:** the model is trained to distinguish these near-miss failures from correct traces.
3. **Result:** improved reliability on edge cases, by training the model to catch its own near-miss reasoning.

---

## ⚡ 7. Multi-Token Prediction (MTP) Post-Training

Standard training predicts one token at a time. DeepSeek-V3 and similar
models get better reasoning and faster inference by predicting several
tokens ahead during training.

**The Concept:**
A **Multi-Token Auxiliary Trainer** that adds a secondary loss for $k$ future tokens.

**The Pipeline Flow in AlignTune:**
1. **Auxiliary head:** attach $K$ lightweight prediction heads to the transformer's hidden states during SFT/RLHF.
2. **Joint optimization:** train to minimize next-token loss and the distance to the future $K$ token embeddings together.
3. **Effect:** pushes the model toward representations that plan ahead, which shows up as better coherence on long-context tasks.

> [!TIP]
> Moves multi-token prediction from pre-training into the post-training stage, so existing models can gain planning behavior without retraining from scratch.

---

## 🎨 8. Discrete Diffusion Refinement (Iterative Utility Polishing)

Standard autoregressive LLMs generate left to right with no way to revise: a
mistake at token 5 propagates through the rest of the sequence.
Diffusion-based post-training lets the model revise earlier tokens based on
what it generates later.

**The Concept:**
A **Diffusion-based Post-Training Refiner** (`aligntune.core.composition.diffusion_refiner`).

**The Pipeline Flow in AlignTune:**
1. **Initial draft:** the model generates a fast, low-quality draft (potentially via MTP).
2. **Denoising training:** during post-training, a discrete diffusion objective trains the model to iteratively correct this draft.
3. **Global revision:** unlike autoregressive models, which commit to each token immediately, the refiner can revise earlier reasoning steps based on what comes later. That's useful for multi-step math and code.

**Technical Implementation: AR-to-Diffusion Transition:**
AlignTune does not train diffusion LLMs from scratch. Instead, it repurposes pre-trained autoregressive (AR) weights through **Continual Diffusion Fine-Tuning**:
*   **Objective swap:** replace the next-token-prediction (NTP) loss with a denoising contextual loss, while keeping the pretrained transformer backbone frozen.
*   **Block-wise attention:** add bidirectional attention within local blocks so segments can be denoised in parallel, while keeping the model's pretrained causal structure intact outside each block.
*   **Curriculum noise:** a scheduled noise-injection curriculum during SFT shifts the model gradually from sequential generation to whole-draft revision.

> [!IMPORTANT]
> Adds a revision step on top of standard transformer weights through a diffusion-refinement stage, instead of requiring a diffusion model trained from scratch.

---

## 🗺️ Execution Strategy

Priorities for this roadmap:
1. Keep documentation and benchmarks scoped to MMLU, HumanEval, and GSM8K, not general safety or agentic behavior.
2. Build RLVR first: it gives the most immediate value to developers training coding/math models.
3. Build the Composition API so the R1-Zero pipeline runs from a single YAML config.
