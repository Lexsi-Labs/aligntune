import torch
from trl import GRPOTrainer, GRPOConfig


class L2GRPOConfig(GRPOConfig):
    """
    Configuration class for L2-GRPO trainer that replaces KL divergence with L2 norm.
    """

    def __init__(self, *args, **kwargs):
        # Extract L2-specific parameters before calling parent
        self.divergence_type = kwargs.pop("divergence_type", "l2")
        self.l2_coefficient = kwargs.pop("l2_coefficient", 0.1)

        # Call parent init with remaining kwargs
        super().__init__(*args, **kwargs)


class L2GRPOTrainer(GRPOTrainer):
    """
    L2-GRPO trainer that replaces KL divergence regularization with L2 norm regularization.

    This is a step toward implementing general w-potential mirror maps with Evolutionary Strategies.
    """

    def __init__(self, **kwargs):
        print("[MEMORY DEBUG] Starting L2GRPOTrainer init...")
        super().__init__(**kwargs)
        print("[MEMORY DEBUG] Parent init complete")
        # Extract L2-specific configuration from args
        config = kwargs.get("args") or kwargs.get("config")
        self.divergence_type = getattr(config, "divergence_type", "l2")

        # Only extract L2 coefficient when using L2 divergence
        if self.divergence_type == "l2":
            self.l2_coefficient = getattr(config, "l2_coefficient", 0.1)
        # Note: KL divergence uses inherited self.beta from parent GRPOTrainer

        # Handle reference model for L2-GRPO (following TRL's pattern)
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

        This property enables extensibility for future GBMPO implementations with
        different Bregman divergences and mirror maps.

        Returns:
            bool: True if reference model logprobs should be computed
        """
        # Original GRPO KL divergence case
        if self.beta != 0.0:
            return True

        # L2-GRPO case
        if hasattr(
                self,
                "l2_coefficient") and self.l2_coefficient != 0.0 and self.divergence_type == "l2":
            return True

        # Future GBMPO extension point:
        # if hasattr(self, 'bregman_coefficient') and self.bregman_coefficient != 0.0:
        #     return True

        return False

    def _generate_and_score_completions(self, inputs):
        """
        Override to use needs_ref_model property and handle reference model computation.

        This enables L2-GRPO and future GBMPO variants to compute reference model log probabilities
        when needed for their respective divergence computations.
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

        # We need reference logprobs but need to handle the adapter case properly
        # device = self.accelerator.device
        # mode = "train" if self.model.training else "eval"

        # prompts = [x["prompt"] for x in inputs]

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
        Override the loss computation to replace KL divergence with L2 norm.
        Based on TRL's GRPOTrainer._compute_loss method.
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
            print("[L2 DEBUG] ERROR: NaN detected in input_ids or attention_mask!")
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
            # image_split_sizes=inputs.get("image_split_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute regularization: L2 norm instead of KL divergence
        if self.l2_coefficient != 0.0 and self.divergence_type == "l2":
            if "ref_per_token_logps" in inputs:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                # L2 regularization: squared difference between current and
                # reference log probabilities
                per_token_l2 = (per_token_logps - ref_per_token_logps) ** 2

                # Debug logging to verify L2 regularization is working
                l2_diff = per_token_logps - ref_per_token_logps

                # Check for NaN in the individual tensors
                if torch.isnan(per_token_logps).any():
                    print("[L2 DEBUG] ERROR: NaN found in per_token_logps!")
                if torch.isnan(ref_per_token_logps).any():
                    print("[L2 DEBUG] ERROR: NaN found in ref_per_token_logps!")

                l2_diff_mean = l2_diff.abs().mean().item()
                l2_mean = per_token_l2.mean().item()

                # Additional debugging
                print(
                    f"[L2 DEBUG] per_token_logps range: [{
                        per_token_logps.min().item():.3f}, {
                        per_token_logps.max().item():.3f}]")
                print(
                    f"[L2 DEBUG] ref_per_token_logps range: [{
                        ref_per_token_logps.min().item():.3f}, {
                        ref_per_token_logps.max().item():.3f}]")
                print(
                    f"[L2 DEBUG] L2 diff abs mean: {
                        l2_diff_mean:.6f}, L2 term mean: {
                        l2_mean:.6f}")

                if l2_diff_mean < 1e-6:
                    print(
                        "[L2 DEBUG] WARNING: L2 difference is near zero - policy and reference might be identical!")
            else:
                # Fallback: no regularization if ref logps not available
                per_token_l2 = torch.zeros_like(per_token_logps)
                print(
                    "[L2 DEBUG] WARNING: No ref_per_token_logps found - L2 regularization disabled!")
        elif self.beta != 0.0 and self.divergence_type == "kl":
            if "ref_per_token_logps" in inputs:
                # Fall back to original KL divergence if specified
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = (torch.exp(ref_per_token_logps -
                                          per_token_logps) -
                                (ref_per_token_logps -
                                 per_token_logps) -
                                1)
            else:
                # Fallback: no regularization if ref logps not available
                per_token_kl = torch.zeros_like(per_token_logps)

        # Compute the loss (same as original GRPO)
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
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
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level:
        # (B, 1)

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
        if self.l2_coefficient != 0.0 and self.divergence_type == "l2":
            per_token_loss = per_token_loss + self.l2_coefficient * per_token_l2
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

        # Log the metrics (similar to original, but for L2 instead of KL)
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Log L2 regularization instead of KL
        if self.l2_coefficient != 0.0 and self.divergence_type == "l2":
            mean_l2 = masked_batch_mean(per_token_l2)
            self._metrics[mode]["l2_regularization"] = self._metrics[mode].get(
                "l2_regularization", [])
            self._metrics[mode]["l2_regularization"].append(
                self.accelerator.gather(mean_l2).nanmean().item())
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
                "[L2 DEBUG] WARNING: Loss is NaN or Inf! Returning zero loss to skip gradient update.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        if loss.item() == 0.0:
            print(
                "[L2 DEBUG] INFO: Loss is exactly zero (likely untrained model or all rewards zero). This is normal at initialization."
            )

        return loss

    def nanmin(self, tensor):
        """Helper function to compute nanmin (compatible with older PyTorch versions)"""
        if hasattr(torch, "nanmin"):
            return torch.nanmin(tensor)
        else:
            # Fallback for older PyTorch versions
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
            # Fallback for older PyTorch versions
            mask = ~torch.isnan(tensor)
            if torch.any(mask):
                return torch.max(tensor[mask])
            else:
                return torch.tensor(
                    float("-inf"),
                    device=tensor.device,
                    dtype=tensor.dtype)


class L2KLGRPOConfig(GRPOConfig):
    """
    Configuration class for L2KL-GRPO trainer with dual regularization (both L2 and KL).

    This enables simultaneous L2 norm and KL divergence regularization:
    Loss = GRPO_base + β * KL_divergence + λ * L2_norm
    """

    def __init__(self, *args, **kwargs):
        # Extract L2KL-specific parameters before calling parent
        self.divergence_type = kwargs.pop("divergence_type", "l2kl")
        self.l2_coefficient = kwargs.pop("l2_coefficient", 0.0001)

        # Note: beta (KL coefficient) comes from parent GRPOConfig
        # Call parent init with remaining kwargs
        super().__init__(*args, **kwargs)


class L2KLGRPOTrainer(GRPOTrainer):
    """
    L2KL-GRPO trainer with dual regularization: both L2 norm and KL divergence.

    Supports flexible regularization combinations:
    - Pure KL: beta > 0, l2_coefficient = 0
    - Pure L2: beta = 0, l2_coefficient > 0
    - Dual: beta > 0, l2_coefficient > 0
    - None: beta = 0, l2_coefficient = 0
    """

    def __init__(self, **kwargs):
        print("[MEMORY DEBUG] Starting L2KLGRPOTrainer init...")
        super().__init__(**kwargs)
        print("[MEMORY DEBUG] Parent init complete")

        # Extract L2KL-specific configuration from args
        config = kwargs.get("args") or kwargs.get("config")
        self.divergence_type = getattr(config, "divergence_type", "l2kl")

        # Always extract L2 coefficient for dual mode
        if self.divergence_type == "l2kl":
            self.l2_coefficient = getattr(config, "l2_coefficient", 0.0001)
            # self.beta is inherited from parent GRPOTrainer

        # Handle reference model setup (similar to L2GRPOTrainer)
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
        Determine if reference model log probabilities are needed for dual regularization.

        Returns True if either KL or L2 regularization is active.
        """
        # L2KL dual regularization case - need ref model if either coefficient
        # is non-zero
        if self.divergence_type == "l2kl" and (
                self.beta != 0.0 or self.l2_coefficient != 0.0):
            return True

        # Future GBMPO extension point:
        # if hasattr(self, 'bregman_coefficient') and self.bregman_coefficient != 0.0:
        #     return True

        return False

    def _generate_and_score_completions(self, inputs):
        """
        Selective override for dual regularization reference model computation.

        Smart routing:
        - Use parent when beta > 0 (parent can handle ref model computation)
        - Override with beta manipulation when beta = 0 but L2 needed
        """
        print(
            f"[DUAL DEBUG] Models in memory: policy={
                type(
                    self.model).__name__}, ref={
                type(
                    self.ref_model).__name__ if self.ref_model else 'None'}")
        print(
            f"[DUAL DEBUG] Coefficients: beta={
                self.beta}, l2_coefficient={
                self.l2_coefficient}")

        # Case 1: No regularization needed
        if not self.needs_ref_model:
            print("[DUAL DEBUG] No regularization needed, using parent method")
            return super()._generate_and_score_completions(inputs)

        # Case 2: Parent can handle it (beta > 0, KL regularization active)
        if self.beta != 0.0:
            print(
                "[DUAL DEBUG] KL regularization active (beta > 0), using parent method")
            return super()._generate_and_score_completions(inputs)

        # Case 3: L2-only mode (beta = 0, l2_coefficient > 0)
        # Need beta manipulation trick like L2GRPOTrainer
        print("[DUAL DEBUG] L2-only mode detected, using beta manipulation")
        return self._generate_with_beta_manipulation(inputs)

    def _generate_with_beta_manipulation(self, inputs):
        """
        Beta manipulation logic for L2-only regularization mode.

        Temporarily sets beta to trigger parent's reference model computation,
        then restores original beta value.
        """
        original_beta = self.beta
        try:
            # Temporarily set beta to trigger ref model computation in parent
            # method
            self.beta = 1e-10
            print(
                f"[DUAL DEBUG] Temporarily set beta to {
                    self.beta} for ref model computation")

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
            print(f"[DUAL DEBUG] Restoring beta to {original_beta}")
            self.beta = original_beta

    def _compute_loss(self, model, inputs):
        """
        Override loss computation to support dual L2 + KL regularization.

        Computes: GRPO_base_loss + β * KL_divergence + λ * L2_norm
        """
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # we only need to compute the logits for the completion tokens
        logits_to_keep = completion_ids.size(1)

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
            # image_split_sizes=inputs.get("image_split_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Initialize regularization terms
        per_token_kl = torch.zeros_like(per_token_logps)
        per_token_l2 = torch.zeros_like(per_token_logps)

        # Compute dual regularization: both L2 and KL if coefficients are
        # non-zero
        if (
            self.divergence_type == "l2kl"
            and (self.beta != 0.0 or self.l2_coefficient != 0.0)
            and "ref_per_token_logps" in inputs
        ):

            ref_per_token_logps = inputs["ref_per_token_logps"]

            # Check for NaN in the individual tensors (critical for stability)
            if torch.isnan(per_token_logps).any():
                print("[DUAL DEBUG] ERROR: NaN found in per_token_logps!")
            if torch.isnan(ref_per_token_logps).any():
                print("[DUAL DEBUG] ERROR: NaN found in ref_per_token_logps!")

            # Compute KL divergence regularization if beta > 0
            if self.beta != 0.0:
                per_token_kl = (torch.exp(ref_per_token_logps -
                                          per_token_logps) -
                                (ref_per_token_logps -
                                 per_token_logps) -
                                1)

            # Compute L2 regularization if l2_coefficient > 0
            if self.l2_coefficient != 0.0:
                per_token_l2 = (per_token_logps - ref_per_token_logps) ** 2

            # Comprehensive dual regularization debugging
            if self.beta != 0.0 or self.l2_coefficient != 0.0:
                logps_diff = per_token_logps - ref_per_token_logps

                # Statistical analysis
                kl_mean = per_token_kl.mean().item() if self.beta != 0.0 else 0.0
                l2_mean = per_token_l2.mean().item() if self.l2_coefficient != 0.0 else 0.0
                logps_diff_mean = logps_diff.abs().mean().item()

                # Tensor ranges
                print(
                    f"[DUAL DEBUG] per_token_logps range: [{
                        per_token_logps.min().item():.3f}, {
                        per_token_logps.max().item():.3f}]")
                print(
                    f"[DUAL DEBUG] ref_per_token_logps range: [{
                        ref_per_token_logps.min().item():.3f}, {
                        ref_per_token_logps.max().item():.3f}]")

                # Regularization analysis
                print("[DUAL DEBUG] === Regularization Analysis ===")
                if self.beta != 0.0:
                    kl_contribution = self.beta * kl_mean
                    print(
                        f"[DUAL DEBUG] KL: mean={
                            kl_mean:.6f}, contribution={
                            kl_contribution:.6f}")
                if self.l2_coefficient != 0.0:
                    l2_contribution = self.l2_coefficient * l2_mean
                    print(
                        f"[DUAL DEBUG] L2: mean={
                            l2_mean:.6f}, contribution={
                            l2_contribution:.6f}")

                total_regularization = (self.beta * kl_mean if self.beta != 0.0 else 0.0) + (
                    self.l2_coefficient * l2_mean if self.l2_coefficient != 0.0 else 0.0)
                print(
                    f"[DUAL DEBUG] Total regularization contribution: {
                        total_regularization:.6f}")
                print(
                    f"[DUAL DEBUG] Policy-Reference difference (abs mean): {logps_diff_mean:.6f}")

                # Warning for identical models
                if logps_diff_mean < 1e-6:
                    print(
                        "[DUAL DEBUG] WARNING: Policy-Reference difference near zero - models might be identical!")

        elif self.divergence_type == "l2kl" and (self.beta != 0.0 or self.l2_coefficient != 0.0):
            # Fallback: regularization needed but no ref logps available
            print(
                "[DUAL DEBUG] WARNING: No ref_per_token_logps found - dual regularization disabled!")

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

        # Add dual regularization terms
        if self.beta != 0.0 and self.divergence_type == "l2kl":
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if self.l2_coefficient != 0.0 and self.divergence_type == "l2kl":
            per_token_loss = per_token_loss + self.l2_coefficient * per_token_l2

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

        # Log dual regularization metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Log both regularization metrics
        if self.beta != 0.0 and self.divergence_type == "l2kl":
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"] = self._metrics[mode].get("kl", [])
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item())

        if self.l2_coefficient != 0.0 and self.divergence_type == "l2kl":
            mean_l2 = masked_batch_mean(per_token_l2)
            self._metrics[mode]["l2_regularization"] = self._metrics[mode].get(
                "l2_regularization", [])
            self._metrics[mode]["l2_regularization"].append(
                self.accelerator.gather(mean_l2).nanmean().item())

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
                "[L2 DEBUG] WARNING: Loss is NaN or Inf! Returning zero loss to skip gradient update.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        if loss.item() == 0.0:
            print(
                "[L2 DEBUG] INFO: Loss is exactly zero (likely untrained model or all rewards zero). This is normal at initialization."
            )

        return loss

    def nanmin(self, tensor):
        """Helper function to compute nanmin (compatible with older PyTorch versions)"""
        if hasattr(torch, "nanmin"):
            return torch.nanmin(tensor)
        else:
            # Fallback for older PyTorch versions
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
            # Fallback for older PyTorch versions
            mask = ~torch.isnan(tensor)
            if torch.any(mask):
                return torch.max(tensor[mask])
            else:
                return torch.tensor(
                    float("-inf"),
                    device=tensor.device,
                    dtype=tensor.dtype)


class ProbL2GRPOConfig(GRPOConfig):
    """
    Configuration class for Probability-L2-GRPO trainer that replaces KL divergence
    with L2 norm in probability space: (p_current - p_ref)²

    This corrects the original L2GRPOConfig which computed (log_p_current - log_p_ref)²
    """

    def __init__(self, *args, **kwargs):
        # Extract L2-specific parameters before calling parent
        self.divergence_type = kwargs.pop("divergence_type", "prob_l2")
        self.l2_coefficient = kwargs.pop("l2_coefficient", 0.1)

        # Call parent init with remaining kwargs
        super().__init__(*args, **kwargs)


class ProbL2GRPOTrainer(GRPOTrainer):
    """
    Probability-L2-GRPO trainer that replaces KL divergence regularization with
    L2 norm regularization in probability space: (p_current - p_ref)²

    This corrects the L2GRPOTrainer which computed (log_p_current - log_p_ref)²
    instead of the true probability-space L2 norm.
    """

    def __init__(self, **kwargs):
        print("[MEMORY DEBUG] Starting ProbL2GRPOTrainer init...")
        super().__init__(**kwargs)
        print("[MEMORY DEBUG] Parent init complete")
        # Extract L2-specific configuration from args
        config = kwargs.get("args") or kwargs.get("config")
        self.divergence_type = getattr(config, "divergence_type", "prob_l2")

        # Only extract L2 coefficient when using prob_l2 divergence
        if self.divergence_type == "prob_l2":
            self.l2_coefficient = getattr(config, "l2_coefficient", 0.1)

        # Handle reference model for prob_l2-GRPO (following TRL's pattern)
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
        """
        # Original GRPO KL divergence case
        if self.beta != 0.0:
            return True

        # Prob-L2-GRPO case
        if hasattr(
                self,
                "l2_coefficient") and self.l2_coefficient != 0.0 and self.divergence_type == "prob_l2":
            return True

        return False

    def _generate_and_score_completions(self, inputs):
        """
        Override to use needs_ref_model property and handle reference model computation.
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
        Override the loss computation to use L2 norm in probability space: (p_current - p_ref)²
        instead of (log_p_current - log_p_ref)²
        """
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Safety check for NaN in inputs
        if torch.isnan(
                input_ids.float()).any() or torch.isnan(
                attention_mask.float()).any():
            print("[PROB_L2 DEBUG] ERROR: NaN detected in input_ids or attention_mask!")
            return torch.tensor(
                0.0,
                device=input_ids.device,
                requires_grad=True)

        # Compute the per_token_logps and the entropy
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

        # Compute regularization: L2 norm in probability space (p_current -
        # p_ref)²
        if self.l2_coefficient != 0.0 and self.divergence_type == "prob_l2":
            if "ref_per_token_logps" in inputs:
                ref_per_token_logps = inputs["ref_per_token_logps"]

                # Check for NaN in the individual tensors
                if torch.isnan(per_token_logps).any():
                    print("[PROB_L2 DEBUG] ERROR: NaN found in per_token_logps!")
                if torch.isnan(ref_per_token_logps).any():
                    print("[PROB_L2 DEBUG] ERROR: NaN found in ref_per_token_logps!")

                # Convert log probabilities to probabilities
                per_token_probs = torch.exp(per_token_logps)
                ref_per_token_probs = torch.exp(ref_per_token_logps)

                # L2 regularization in probability space: (p_current - p_ref)²
                per_token_l2 = (per_token_probs - ref_per_token_probs) ** 2

                # Debug logging
                prob_diff = per_token_probs - ref_per_token_probs
                prob_diff_mean = prob_diff.abs().mean().item()
                l2_mean = per_token_l2.mean().item()

                print(
                    f"[PROB_L2 DEBUG] per_token_probs range: [{
                        per_token_probs.min().item():.6f}, {
                        per_token_probs.max().item():.6f}]")
                print(
                    f"[PROB_L2 DEBUG] ref_per_token_probs range: [{
                        ref_per_token_probs.min().item():.6f}, {
                        ref_per_token_probs.max().item():.6f}]")
                print(
                    f"[PROB_L2 DEBUG] Probability diff abs mean: {
                        prob_diff_mean:.6f}, L2 term mean: {
                        l2_mean:.6f}")

                if prob_diff_mean < 1e-6:
                    print(
                        "[PROB_L2 DEBUG] WARNING: Probability difference is near zero - policy and reference might be identical!"
                    )
            else:
                # Fallback: no regularization if ref logps not available
                per_token_l2 = torch.zeros_like(per_token_logps)
                print(
                    "[PROB_L2 DEBUG] WARNING: No ref_per_token_logps found - L2 regularization disabled!")
        elif self.beta != 0.0 and self.divergence_type == "kl":
            if "ref_per_token_logps" in inputs:
                # Fall back to original KL divergence if specified
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
                    self.importance_sampling_level}.")

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
        if self.l2_coefficient != 0.0 and self.divergence_type == "prob_l2":
            per_token_loss = per_token_loss + self.l2_coefficient * per_token_l2
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

        # Log L2 regularization
        if self.l2_coefficient != 0.0 and self.divergence_type == "prob_l2":
            mean_l2 = masked_batch_mean(per_token_l2)
            self._metrics[mode]["prob_l2_regularization"] = self._metrics[mode].get(
                "prob_l2_regularization", [])
            self._metrics[mode]["prob_l2_regularization"].append(
                self.accelerator.gather(mean_l2).nanmean().item())
        elif self.beta != 0.0 and self.divergence_type == "kl":
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"] = self._metrics[mode].get("kl", [])
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"] = self._metrics[mode].get("entropy", [])
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
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
                "[PROB_L2 DEBUG] WARNING: Loss is NaN or Inf! Returning zero loss to skip gradient update.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        if loss.item() == 0.0:
            print(
                "[PROB_L2 DEBUG] INFO: Loss is exactly zero. This is normal at initialization.")

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


class ProbL2KLGRPOConfig(GRPOConfig):
    """
    Configuration class for Probability-L2KL-GRPO trainer with dual regularization.

    Uses L2 norm in probability space: (p_current - p_ref)² instead of (log_p_current - log_p_ref)²
    Loss = GRPO_base + β * KL_divergence + λ * (p_current - p_ref)²
    """

    def __init__(self, *args, **kwargs):
        # Extract L2KL-specific parameters before calling parent
        self.divergence_type = kwargs.pop("divergence_type", "prob_l2kl")
        self.l2_coefficient = kwargs.pop("l2_coefficient", 0.0001)

        # Note: beta (KL coefficient) comes from parent GRPOConfig
        # Call parent init with remaining kwargs
        super().__init__(*args, **kwargs)


class ProbL2KLGRPOTrainer(GRPOTrainer):
    """
    Probability-L2KL-GRPO trainer with dual regularization using probability-space L2 norm.

    Uses L2 norm in probability space: (p_current - p_ref)² instead of (log_p_current - log_p_ref)²
    Combined with KL divergence:
    Loss = GRPO_base + β * KL_divergence + λ * (p_current - p_ref)²

    Supports flexible regularization combinations:
    - Pure KL: beta > 0, l2_coefficient = 0
    - Pure prob_l2: beta = 0, l2_coefficient > 0
    - Dual: beta > 0, l2_coefficient > 0
    - None: beta = 0, l2_coefficient = 0
    """

    def __init__(self, **kwargs):
        print("[MEMORY DEBUG] Starting ProbL2KLGRPOTrainer init...")
        super().__init__(**kwargs)
        print("[MEMORY DEBUG] Parent init complete")

        # Extract L2KL-specific configuration from args
        config = kwargs.get("args") or kwargs.get("config")
        self.divergence_type = getattr(config, "divergence_type", "prob_l2kl")

        # Always extract L2 coefficient for dual mode
        if self.divergence_type == "prob_l2kl":
            self.l2_coefficient = getattr(config, "l2_coefficient", 0.0001)
            # self.beta is inherited from parent GRPOTrainer

        # Handle reference model setup
        from accelerate.utils import is_peft_model
        from transformers import AutoConfig
        import transformers

        if not self.needs_ref_model:
            self.ref_model = None
        elif is_peft_model(self.model):
            self.ref_model = None
        else:
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
        Determine if reference model log probabilities are needed for dual regularization.

        Returns True if either KL or prob_l2 regularization is active.
        """
        # Prob_L2KL dual regularization case
        if self.divergence_type == "prob_l2kl" and (
                self.beta != 0.0 or self.l2_coefficient != 0.0):
            return True
        return False

    def _generate_and_score_completions(self, inputs):
        """
        Selective override for dual regularization reference model computation.
        """
        print(
            f"[PROB_DUAL DEBUG] Models in memory: policy={
                type(
                    self.model).__name__}, ref={
                type(
                    self.ref_model).__name__ if self.ref_model else 'None'}")
        print(
            f"[PROB_DUAL DEBUG] Coefficients: beta={
                self.beta}, l2_coefficient={
                self.l2_coefficient}")

        # Case 1: No regularization needed
        if not self.needs_ref_model:
            print("[PROB_DUAL DEBUG] No regularization needed, using parent method")
            return super()._generate_and_score_completions(inputs)

        # Case 2: Parent can handle it (beta > 0, KL regularization active)
        if self.beta != 0.0:
            print(
                "[PROB_DUAL DEBUG] KL regularization active (beta > 0), using parent method")
            return super()._generate_and_score_completions(inputs)

        # Case 3: Prob_L2-only mode (beta = 0, l2_coefficient > 0)
        print("[PROB_DUAL DEBUG] Prob_L2-only mode detected, using beta manipulation")
        return self._generate_with_beta_manipulation(inputs)

    def _generate_with_beta_manipulation(self, inputs):
        """
        Beta manipulation logic for prob_l2-only regularization mode.
        """
        original_beta = self.beta
        try:
            self.beta = 1e-10
            print(
                f"[PROB_DUAL DEBUG] Temporarily set beta to {
                    self.beta} for ref model computation")

            model = self.accelerator.unwrap_model(self.model)
            added_disable_adapter = False

            has_adapters = hasattr(
                model, "peft_config") and getattr(
                model, "peft_config", None) is not None

            if not has_adapters:

                def no_op_disable_adapter():
                    from contextlib import nullcontext

                    return nullcontext()

                if not hasattr(model, "disable_adapter"):
                    model.disable_adapter = no_op_disable_adapter
                    added_disable_adapter = True
            elif not hasattr(model, "disable_adapter") and hasattr(model, "disable_adapters"):
                model.disable_adapter = model.disable_adapters
                added_disable_adapter = True

            result = super()._generate_and_score_completions(inputs)

            if added_disable_adapter and hasattr(model, "disable_adapter"):
                delattr(model, "disable_adapter")

            return result

        finally:
            print(f"[PROB_DUAL DEBUG] Restoring beta to {original_beta}")
            self.beta = original_beta

    def _compute_loss(self, model, inputs):
        """
        Override loss computation to support dual prob_l2 + KL regularization.

        Computes: GRPO_base_loss + β * KL_divergence + λ * (p_current - p_ref)²
        """
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute the per_token_logps and the entropy
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

        # Initialize regularization terms
        per_token_kl = torch.zeros_like(per_token_logps)
        per_token_l2 = torch.zeros_like(per_token_logps)

        # Compute dual regularization: both prob_l2 and KL if coefficients are
        # non-zero
        if (
            self.divergence_type == "prob_l2kl"
            and (self.beta != 0.0 or self.l2_coefficient != 0.0)
            and "ref_per_token_logps" in inputs
        ):

            ref_per_token_logps = inputs["ref_per_token_logps"]

            # Check for NaN
            if torch.isnan(per_token_logps).any():
                print("[PROB_DUAL DEBUG] ERROR: NaN found in per_token_logps!")
            if torch.isnan(ref_per_token_logps).any():
                print("[PROB_DUAL DEBUG] ERROR: NaN found in ref_per_token_logps!")

            # Compute KL divergence regularization if beta > 0
            if self.beta != 0.0:
                per_token_kl = (torch.exp(ref_per_token_logps -
                                          per_token_logps) -
                                (ref_per_token_logps -
                                 per_token_logps) -
                                1)

            # Compute prob_l2 regularization if l2_coefficient > 0
            if self.l2_coefficient != 0.0:
                # Convert log probabilities to probabilities
                per_token_probs = torch.exp(per_token_logps)
                ref_per_token_probs = torch.exp(ref_per_token_logps)

                # L2 regularization in probability space: (p_current - p_ref)²
                per_token_l2 = (per_token_probs - ref_per_token_probs) ** 2

            # Comprehensive dual regularization debugging
            if self.beta != 0.0 or self.l2_coefficient != 0.0:
                # Statistical analysis
                kl_mean = per_token_kl.mean().item() if self.beta != 0.0 else 0.0
                l2_mean = per_token_l2.mean().item() if self.l2_coefficient != 0.0 else 0.0

                # Tensor ranges
                print(
                    f"[PROB_DUAL DEBUG] per_token_logps range: [{
                        per_token_logps.min().item():.3f}, {
                        per_token_logps.max().item():.3f}]")
                print(
                    f"[PROB_DUAL DEBUG] ref_per_token_logps range: [{
                        ref_per_token_logps.min().item():.3f}, {
                        ref_per_token_logps.max().item():.3f}]")

                # Regularization analysis
                print("[PROB_DUAL DEBUG] === Regularization Analysis ===")
                if self.beta != 0.0:
                    kl_contribution = self.beta * kl_mean
                    print(
                        f"[PROB_DUAL DEBUG] KL: mean={
                            kl_mean:.6f}, contribution={
                            kl_contribution:.6f}")
                if self.l2_coefficient != 0.0:
                    l2_contribution = self.l2_coefficient * l2_mean
                    print(
                        f"[PROB_DUAL DEBUG] Prob_L2: mean={
                            l2_mean:.6f}, contribution={
                            l2_contribution:.6f}")

                    # Additional probability-space debugging
                    if self.l2_coefficient != 0.0:
                        per_token_probs = torch.exp(per_token_logps)
                        ref_per_token_probs = torch.exp(ref_per_token_logps)
                        prob_diff_mean = (
                            per_token_probs - ref_per_token_probs).abs().mean().item()
                        print(
                            f"[PROB_DUAL DEBUG] Probability diff abs mean: {
                                prob_diff_mean:.6f}")

                total_regularization = (self.beta * kl_mean if self.beta != 0.0 else 0.0) + (
                    self.l2_coefficient * l2_mean if self.l2_coefficient != 0.0 else 0.0)
                print(
                    f"[PROB_DUAL DEBUG] Total regularization contribution: {
                        total_regularization:.6f}")

        elif self.divergence_type == "prob_l2kl" and (self.beta != 0.0 or self.l2_coefficient != 0.0):
            print(
                "[PROB_DUAL DEBUG] WARNING: No ref_per_token_logps found - dual regularization disabled!")

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
                    self.importance_sampling_level}.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.epsilon_low,
            1 + self.epsilon_high)

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

        # Add dual regularization terms
        if self.beta != 0.0 and self.divergence_type == "prob_l2kl":
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if self.l2_coefficient != 0.0 and self.divergence_type == "prob_l2kl":
            per_token_loss = per_token_loss + self.l2_coefficient * per_token_l2

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

        # Log dual regularization metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Log both regularization metrics
        if self.beta != 0.0 and self.divergence_type == "prob_l2kl":
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"] = self._metrics[mode].get("kl", [])
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item())

        if self.l2_coefficient != 0.0 and self.divergence_type == "prob_l2kl":
            mean_l2 = masked_batch_mean(per_token_l2)
            self._metrics[mode]["prob_l2_regularization"] = self._metrics[mode].get(
                "prob_l2_regularization", [])
            self._metrics[mode]["prob_l2_regularization"].append(
                self.accelerator.gather(mean_l2).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"] = self._metrics[mode].get("entropy", [])
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
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

        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print("[PROB_DUAL DEBUG] WARNING: Loss is NaN or Inf! Returning zero loss.")
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        if loss.item() == 0.0:
            print(
                "[PROB_DUAL DEBUG] INFO: Loss is exactly zero. Normal at initialization.")

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
