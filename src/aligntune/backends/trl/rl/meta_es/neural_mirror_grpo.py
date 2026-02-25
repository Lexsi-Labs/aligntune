import torch
import torch.nn as nn
from trl import GRPOTrainer, GRPOConfig


class NeuralMirrorGRPOConfig(GRPOConfig):
    """
    Configuration class for Neural Mirror GRPO trainer that uses a neural network
    parametrized w-potential mirror map for Bregman divergence regularization.

    The mirror map is defined by:
    - ϕ⁻¹(y) = Σ_{j=1}^{126} v_j * activation_j(w_j*y + b_j) + a*y + c*log(y)
    - h(y) = integral of ϕ⁻¹(y)
    - Bregman divergence: D(y || y0) = h(y) - h(y0) - h'(y0)*(y - y0)
    """

    def __init__(self, *args, **kwargs):
        # Extract neural mirror-specific parameters before calling parent
        self.divergence_type = kwargs.pop("divergence_type", "neural_mirror")
        self.mirror_coefficient = kwargs.pop("mirror_coefficient", 0.0001)
        self.mirror_init_scale = kwargs.pop("mirror_init_scale", 0.01)
        self.mirror_seed = kwargs.pop("mirror_seed", 42)

        # Call parent init with remaining kwargs
        super().__init__(*args, **kwargs)


class NeuralMirrorModule(nn.Module):
    """
    Neural network module for computing mirror map and Bregman divergence.

    Implements a 126-neuron network with 6 different activation types:
    - Units 1-21: x³
    - Units 22-42: x²
    - Units 43-63: x^(1/2)
    - Units 64-84: x^(1/3)
    - Units 85-105: log(x⁺ + 10⁻³)
    - Units 106-126: exp(x)

    All parameters {v_j, w_j, b_j, a, c} are non-negative and randomly initialized.
    They are NOT trainable (will be meta-learned via evolutionary strategies).
    """

    def __init__(self, init_scale=0.01, seed=42):
        super().__init__()

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Initialize 126 neurons with random non-negative parameters
        # Use register_buffer to make them part of state_dict but non-trainable

        # v_j: output weights (non-negative)
        v = torch.rand(126) * init_scale
        self.register_buffer("v", v)

        # w_j: input weights (non-negative)
        w = torch.rand(126) * init_scale
        self.register_buffer("w", w)

        # b_j: biases (non-negative)
        b = torch.rand(126) * init_scale
        self.register_buffer("b", b)

        # a, c: additional scalar parameters (non-negative)
        a = torch.rand(1) * init_scale
        self.register_buffer("a", a)

        c = torch.rand(1) * init_scale
        self.register_buffer("c", c)

        # Epsilon for numerical stability in log activations
        self.eps = 1e-3
        self.eps_prob = 1e-10  # for probability clamping

        print(
            f"[NEURAL MIRROR] Initialized with seed={seed}, init_scale={init_scale}")
        print(
            f"[NEURAL MIRROR] v range: [{
                v.min().item():.6f}, {
                v.max().item():.6f}]")
        print(
            f"[NEURAL MIRROR] w range: [{
                w.min().item():.6f}, {
                w.max().item():.6f}]")
        print(
            f"[NEURAL MIRROR] b range: [{
                b.min().item():.6f}, {
                b.max().item():.6f}]")
        print(f"[NEURAL MIRROR] a={a.item():.6f}, c={c.item():.6f}")

    def _get_affine_transform(self, y, j):
        """Compute u_j = w_j * y + b_j for neuron j."""
        return self.w[j] * y + self.b[j]

    def activation(self, y, j):
        """
        Apply activation function for neuron j.

        Args:
            y: input tensor (probabilities)
            j: neuron index (0-125)

        Returns:
            activation_j(w_j * y + b_j)
        """
        u = self._get_affine_transform(y, j)

        if j < 21:  # Units 0-20: x³
            return u**3
        elif j < 42:  # Units 21-41: x²
            return u**2
        elif j < 63:  # Units 42-62: x^(1/2)
            return torch.pow(u.clamp(min=0), 0.5)
        elif j < 84:  # Units 63-83: x^(1/3)
            return torch.pow(u.clamp(min=0), 1.0 / 3.0)
        elif j < 105:  # Units 84-104: log(x⁺ + eps)
            return torch.log(u.clamp(min=0) + self.eps)
        else:  # Units 105-125: exp(x)
            return torch.exp(u)

    def primitive(self, y, j):
        """
        Compute the antiderivative (primitive) H_j(y) for neuron j.

        Args:
            y: input tensor (probabilities)
            j: neuron index (0-125)

        Returns:
            H_j(y) such that H_j'(y) = activation_j(w_j * y + b_j)
        """
        w = self.w[j]
        u = self._get_affine_transform(y, j)

        # Handle w_j = 0 special case: activation is constant
        if w.abs() < 1e-12:
            return self.activation(y, j) * y

        if j < 21:  # x³ → x⁴/4
            return (u**4) / (4 * w)
        elif j < 42:  # x² → x³/3
            return (u**3) / (3 * w)
        elif j < 63:  # x^(1/2) → (2/3)*x^(3/2)
            return (2.0 / 3.0) * torch.pow(u.clamp(min=0), 1.5) / w
        elif j < 84:  # x^(1/3) → (3/4)*x^(4/3)
            return (3.0 / 4.0) * torch.pow(u.clamp(min=0), 4.0 / 3.0) / w
        elif j < 105:  # log → u*log(u) - u
            u_safe = u.clamp(min=0) + self.eps
            return (u_safe * torch.log(u_safe) - u_safe) / w
        else:  # exp → exp
            return torch.exp(u) / w

    def h(self, y):
        """
        Compute the mirror potential h(y).

        h(y) = Σ_{j=1}^{126} v_j * H_j(y) + (a/2)*y² + c*(y*log(y) - y)

        Args:
            y: input tensor (probabilities), shape [..., seq_len]

        Returns:
            h(y), same shape as input
        """
        y_safe = y.clamp(min=self.eps_prob)

        # Neural network terms
        result = torch.zeros_like(y)
        for j in range(126):
            result = result + self.v[j] * self.primitive(y, j)

        # Quadratic term: (a/2) * y²
        result = result + (self.a / 2) * (y**2)

        # Entropic term: c * (y*log(y) - y)
        result = result + self.c * (y * torch.log(y_safe) - y)

        return result

    def phi_inv(self, y):
        """
        Compute the inverse potential ϕ⁻¹(y) = h'(y).

        ϕ⁻¹(y) = Σ_{j=1}^{126} v_j * activation_j(w_j*y + b_j) + a*y + c*log(y)

        Args:
            y: input tensor (probabilities), shape [..., seq_len]

        Returns:
            ϕ⁻¹(y), same shape as input
        """
        y_safe = y.clamp(min=self.eps_prob)

        # Neural network terms
        result = torch.zeros_like(y)
        for j in range(126):
            result = result + self.v[j] * self.activation(y, j)

        # Linear term: a * y
        result = result + self.a * y

        # Log term: c * log(y)
        result = result + self.c * torch.log(y_safe)

        return result

    def bregman_divergence(self, y, y0):
        """
        Compute the Bregman divergence D_h(y || y0) induced by mirror map h.

        D(y || y0) = h(y) - h(y0) - h'(y0) * (y - y0)
                   = Σ v_j * [H_j(y) - H_j(y0) - activation_j(y0)*(y - y0)]
                     + (a/2)*(y - y0)²
                     + c*[y*log(y/y0) - (y - y0)]

        Args:
            y: current policy probabilities, shape [..., seq_len]
            y0: reference policy probabilities, shape [..., seq_len]

        Returns:
            Bregman divergence per token, same shape as input
        """
        y_safe = y.clamp(min=self.eps_prob)
        y0_safe = y0.clamp(min=self.eps_prob)

        divergence = torch.zeros_like(y)

        # Neural network terms: Σ v_j * [H_j(y) - H_j(y0) - activation_j(y0)*(y
        # - y0)]
        for j in range(126):
            H_y = self.primitive(y, j)
            H_y0 = self.primitive(y0, j)
            act_y0 = self.activation(y0, j)
            divergence = divergence + \
                self.v[j] * (H_y - H_y0 - act_y0 * (y - y0))

        # Quadratic term: (a/2) * (y - y0)²
        divergence = divergence + (self.a / 2) * ((y - y0) ** 2)

        # Entropic term: c * [y*log(y/y0) - (y - y0)]
        divergence = divergence + self.c * \
            (y * torch.log(y_safe / y0_safe) - (y - y0))

        return divergence


class NeuralMirrorGRPOTrainer(GRPOTrainer):
    """
    Neural Mirror GRPO trainer that uses a neural network parametrized mirror map
    for Bregman divergence regularization.

    The mirror map parameters are NOT trainable - they will be meta-learned via
    evolutionary strategies in future experiments.
    """

    def __init__(self, **kwargs):
        print("[MEMORY DEBUG] Starting NeuralMirrorGRPOTrainer init...")

        super().__init__(**kwargs)
        print("[MEMORY DEBUG] Parent init complete")

        # Extract neural mirror-specific configuration from args
        config = kwargs.get("args") or kwargs.get("config")
        self.divergence_type = getattr(
            config, "divergence_type", "neural_mirror")

        # Only extract mirror coefficient when using neural_mirror divergence
        if self.divergence_type == "neural_mirror":
            self.mirror_coefficient = getattr(
                config, "mirror_coefficient", 0.0001)
            mirror_init_scale = getattr(config, "mirror_init_scale", 0.01)
            mirror_seed = getattr(config, "mirror_seed", 42)

            # Create mirror map module (non-trainable)
            print(
                f"[MIRROR DEBUG] Creating neural mirror module with init_scale={mirror_init_scale}, seed={mirror_seed}"
            )
            self.mirror_module = NeuralMirrorModule(
                init_scale=mirror_init_scale, seed=mirror_seed).to(
                self.accelerator.device)

            # Ensure all parameters are non-trainable
            for param in self.mirror_module.parameters():
                param.requires_grad = False

            print(
                f"[MIRROR DEBUG] Mirror module created with {
                    sum(
                        p.numel() for p in self.mirror_module.buffers())} parameters")

        # Handle reference model for neural mirror GRPO (following TRL's
        # pattern)
        from accelerate.utils import is_peft_model
        from transformers import AutoConfig
        import transformers

        if not self.needs_ref_model:
            # If no regularization needed, no reference model required
            self.ref_model = None
        elif is_peft_model(self.model):
            # If PEFT is used, no separate reference model needed (use adapter
            # disabling)
            self.ref_model = None
        else:
            # For non-PEFT models, create a reference model from scratch
            model_id = kwargs.get("model")
            if model_id and isinstance(model_id, str):
                print(
                    f"[REF DEBUG] Creating reference model from {model_id}...")
                config_ref = AutoConfig.from_pretrained(model_id)
                architecture = getattr(
                    transformers, config_ref.architectures[0])
                self.ref_model = architecture.from_pretrained(
                    model_id, torch_dtype=self.model.dtype, device_map={
                        "": self.accelerator.device})
                self.ref_model.eval()
                print("[REF DEBUG] Reference model created successfully")
            else:
                print(
                    "[REF DEBUG] WARNING: Could not create reference model - model_id not available")
                self.ref_model = None

        if self.ref_model is not None:
            print(
                f"[MEMORY DEBUG] Reference model created: {
                    type(
                        self.ref_model).__name__}")
        else:
            print("[MEMORY DEBUG] No reference model created")

    @property
    def needs_ref_model(self):
        """
        Determine if reference model log probabilities are needed for regularization.

        Returns:
            bool: True if reference model logprobs should be computed
        """
        # Original GRPO KL divergence case
        if self.beta != 0.0:
            return True

        # Neural mirror GRPO case
        if (
            hasattr(self, "mirror_coefficient")
            and self.mirror_coefficient != 0.0
            and self.divergence_type == "neural_mirror"
        ):
            return True

        return False

    def _generate_and_score_completions(self, inputs):
        """
        Override to use needs_ref_model property and handle reference model computation.

        This enables neural mirror GRPO to compute reference model log probabilities
        when needed for Bregman divergence computation.
        """
        print(
            f"[MEMORY DEBUG] Models in memory: policy={
                type(
                    self.model).__name__}, ref={
                type(
                    self.ref_model).__name__ if self.ref_model else 'None'}")
        # Check if we need reference model computation
        if not self.needs_ref_model:
            # No regularization needed, call parent with beta=0
            return super()._generate_and_score_completions(inputs)

        # Use the original GRPO generation logic but with our custom ref model
        # handling
        original_beta = self.beta
        try:
            # Temporarily set beta to trigger ref model computation in parent
            # method
            if self.beta == 0.0:
                self.beta = 1e-10

            # Workaround for adapter compatibility issues in full-parameter
            # training
            model = self.accelerator.unwrap_model(self.model)
            added_disable_adapter = False

            # Check if we have adapters loaded
            has_adapters = hasattr(
                model, "peft_config") and getattr(
                model, "peft_config", None) is not None

            if not has_adapters:
                # For full-parameter training, create a no-op disable_adapter
                # method
                def no_op_disable_adapter():
                    from contextlib import nullcontext

                    return nullcontext()

                if not hasattr(model, "disable_adapter"):
                    model.disable_adapter = no_op_disable_adapter
                    added_disable_adapter = True
            elif not hasattr(model, "disable_adapter") and hasattr(model, "disable_adapters"):
                # For adapter training, alias the plural method
                model.disable_adapter = model.disable_adapters
                added_disable_adapter = True

            # Use the standard GRPO logic - reference model is already properly
            # set up in __init__
            result = super()._generate_and_score_completions(inputs)

            # Clean up temporary method if we added it
            if added_disable_adapter and hasattr(model, "disable_adapter"):
                delattr(model, "disable_adapter")

            return result

        finally:
            # Always restore original beta value
            self.beta = original_beta

    def _compute_loss(self, model, inputs):
        """
        Override the loss computation to use neural network Bregman divergence regularization.

        Computes: GRPO_base_loss + λ * D_Bregman(π_current || π_ref)
        """
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # we only need to compute the logits for the completion tokens
        logits_to_keep = completion_ids.size(1)

        # Safety check for NaN in inputs
        if torch.isnan(
                input_ids.float()).any() or torch.isnan(
                attention_mask.float()).any():
            print("[MIRROR DEBUG] ERROR: NaN detected in input_ids or attention_mask!")
            return torch.tensor(
                0.0,
                device=input_ids.device,
                requires_grad=True)

        # Compute the per_token_logps and the entropy at each position in the
        # completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute regularization: Bregman divergence in probability space
        if self.mirror_coefficient != 0.0 and self.divergence_type == "neural_mirror":
            if "ref_per_token_logps" in inputs:
                ref_per_token_logps = inputs["ref_per_token_logps"]

                # Check for NaN in the individual tensors
                if torch.isnan(per_token_logps).any():
                    print("[MIRROR DEBUG] ERROR: NaN found in per_token_logps!")
                if torch.isnan(ref_per_token_logps).any():
                    print("[MIRROR DEBUG] ERROR: NaN found in ref_per_token_logps!")

                # Convert log probabilities to probabilities (clamp to avoid exact zeros from underflow)
                # Use torch.maximum instead of clamp (more reliable for
                # preventing exact zeros)
                eps_tensor = torch.tensor(
                    self.mirror_module.eps_prob,
                    device=per_token_logps.device,
                    dtype=per_token_logps.dtype)

                per_token_probs_raw = torch.exp(per_token_logps)
                per_token_probs = torch.maximum(
                    per_token_probs_raw, eps_tensor)

                ref_per_token_probs_raw = torch.exp(ref_per_token_logps)
                print(
                    f"[CLAMP DEBUG] BEFORE maximum: ref has {
                        (
                            ref_per_token_probs_raw == 0.0).sum().item()} exact zeros, min={
                        ref_per_token_probs_raw.min().item():.12e}")
                ref_per_token_probs = torch.maximum(
                    ref_per_token_probs_raw, eps_tensor)
                print(
                    f"[CLAMP DEBUG] AFTER maximum: ref has {
                        (
                            ref_per_token_probs == 0.0).sum().item()} exact zeros, min={
                        ref_per_token_probs.min().item():.12e}, eps={
                        self.mirror_module.eps_prob}")

                # Compute Bregman divergence using neural mirror map
                per_token_bregman = self.mirror_module.bregman_divergence(
                    per_token_probs, ref_per_token_probs)

                # Debug logging
                prob_diff = per_token_probs - ref_per_token_probs
                prob_diff_mean = prob_diff.abs().mean().item()
                bregman_mean = per_token_bregman.mean().item()

                # Count exact zeros (should be 0 after clamping)
                num_zeros = (per_token_probs == 0.0).sum().item()
                num_ref_zeros = (ref_per_token_probs == 0.0).sum().item()

                print(
                    f"[MIRROR DEBUG] per_token_probs range: [{
                        per_token_probs.min().item():.12e}, {
                        per_token_probs.max().item():.6f}] (zeros: {num_zeros})")
                print(
                    f"[MIRROR DEBUG] ref_per_token_probs range: [{
                        ref_per_token_probs.min().item():.12e}, {
                        ref_per_token_probs.max().item():.6f}] (zeros: {num_ref_zeros})")
                print(
                    f"[MIRROR DEBUG] Probability diff abs mean: {
                        prob_diff_mean:.6f}")
                print(
                    f"[MIRROR DEBUG] Bregman divergence: mean={
                        bregman_mean:.6f}, min={
                        per_token_bregman.min().item():.6f}, max={
                        per_token_bregman.max().item():.6f}")
                print(
                    f"[MIRROR DEBUG] Bregman contribution to loss: {
                        self.mirror_coefficient *
                        bregman_mean:.6f}")

                if prob_diff_mean < 1e-6:
                    print(
                        "[MIRROR DEBUG] WARNING: Probability difference is near zero - policy and reference might be identical!"
                    )
            else:
                # Fallback: no regularization if ref logps not available
                per_token_bregman = torch.zeros_like(per_token_logps)
                print(
                    "[MIRROR DEBUG] WARNING: No ref_per_token_logps found - Bregman regularization disabled!")
        elif self.beta != 0.0 and self.divergence_type == "kl":
            # Fall back to original KL divergence if specified
            if "ref_per_token_logps" in inputs:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = (torch.exp(ref_per_token_logps -
                                          per_token_logps) -
                                (ref_per_token_logps -
                                 per_token_logps) -
                                1)
            else:
                per_token_kl = torch.zeros_like(per_token_logps)

        # Compute the loss (same as original GRPO)
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach(
        ) if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {
                    self.importance_sampling_level}. Possible values are 'token' " "and 'sequence'.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon_low,
            1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * \
                inputs["importance_sampling_ratio"]

        # Add regularization term
        if self.mirror_coefficient != 0.0 and self.divergence_type == "neural_mirror":
            per_token_loss = per_token_loss + self.mirror_coefficient * per_token_bregman
        elif self.beta != 0.0 and self.divergence_type == "kl":
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) /
                    completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / \
                completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / \
                (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / \
                self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Log Bregman divergence regularization
        if self.mirror_coefficient != 0.0 and self.divergence_type == "neural_mirror":
            mean_bregman = masked_batch_mean(per_token_bregman)
            self._metrics[mode]["bregman_divergence"] = self._metrics[mode].get(
                "bregman_divergence", [])
            self._metrics[mode]["bregman_divergence"].append(
                self.accelerator.gather(mean_bregman).nanmean().item())
        elif self.beta != 0.0 and self.divergence_type == "kl":
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"] = self._metrics[mode].get("kl", [])
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"] = self._metrics[mode].get("entropy", [])
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios (same as original)
        is_low_clipped = (
            coef_1 < 1 -
            self.epsilon_low) & (
            advantages.unsqueeze(1) < 0)
        is_high_clipped = (
            coef_1 > 1 +
            self.epsilon_high) & (
            advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"] = self._metrics[mode].get(
            "clip_ratio/low_mean", [])
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"] = self._metrics[mode].get(
            "clip_ratio/low_min", [])
        self._metrics[mode]["clip_ratio/low_min"].append(
            self.nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"] = self._metrics[mode].get(
            "clip_ratio/high_mean", [])
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"] = self._metrics[mode].get(
            "clip_ratio/high_max", [])
        self._metrics[mode]["clip_ratio/high_max"].append(
            self.nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"] = self._metrics[mode].get(
            "clip_ratio/region_mean", [])
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item())

        # Safety check: if loss is NaN or zero, handle gracefully
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                "[MIRROR DEBUG] WARNING: Loss is NaN or Inf! Returning zero loss to skip gradient update.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        if loss.item() == 0.0:
            print(
                "[MIRROR DEBUG] INFO: Loss is exactly zero. This is normal at initialization.")

        return loss

    def nanmin(self, tensor):
        """Helper function to compute nanmin (compatible with older PyTorch versions)"""
        if hasattr(torch, "nanmin"):
            return torch.nanmin(tensor)
        else:
            mask = ~torch.isnan(tensor)
            if torch.any(mask):
                return torch.min(tensor[mask])
            else:
                return torch.tensor(
                    float("inf"),
                    device=tensor.device,
                    dtype=tensor.dtype)

    def nanmax(self, tensor):
        """Helper function to compute nanmax (compatible with older PyTorch versions)"""
        if hasattr(torch, "nanmax"):
            return torch.nanmax(tensor)
        else:
            mask = ~torch.isnan(tensor)
            if torch.any(mask):
                return torch.max(tensor[mask])
            else:
                return torch.tensor(
                    float("-inf"),
                    device=tensor.device,
                    dtype=tensor.dtype)
