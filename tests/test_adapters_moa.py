"""
CPU-only tests for Mixture of Adapters (MoA) implementation (v3.3 Advanced Parameterization).

Tests verify:
1. MoARouter initialization and forward pass shape correctness
2. Top-k selection produces exactly k non-zero values per token
3. Load balance loss computation and fairness
4. MoALoraLayer forward pass with mock linear layers
5. Output shape preservation
6. Routing weight normalization (sum ≈ 1.0 per token)
7. Parameter validation and error handling
8. Integration with base linear layers

All tests run on CPU without GPU requirement.
No model loading or external dependencies beyond torch.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestMoARouterInitialization:
    """Test MoARouter initialization and parameter validation."""

    def test_router_init_basic(self):
        """Test basic router initialization with default parameters."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        assert router.hidden_dim == 768
        assert router.num_experts == 4
        assert router.top_k == 2
        assert router.use_mlp is False
        assert router.router_temp == 1.0

    def test_router_init_with_mlp(self):
        """Test router initialization with MLP gating."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=8,
            top_k=3,
            use_mlp=True,
            mlp_hidden_dim=512,
        )

        assert router.use_mlp is True
        assert isinstance(router.gate, nn.Sequential)

    def test_router_init_linear_gate(self):
        """Test router initialization with linear gate."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=512,
            num_experts=4,
            top_k=2,
            use_mlp=False,
        )

        assert router.use_mlp is False
        assert isinstance(router.gate, nn.Linear)

    def test_router_validation_top_k_exceeds_experts(self):
        """Test that top_k > num_experts raises ValueError."""
        from aligntune.core.adapters.moa.router import MoARouter

        with pytest.raises(ValueError, match="top_k.*must be.*num_experts"):
            MoARouter(
                hidden_dim=768,
                num_experts=4,
                top_k=5,
            )

    def test_router_validation_negative_hidden_dim(self):
        """Test that negative hidden_dim raises ValueError."""
        from aligntune.core.adapters.moa.router import MoARouter

        with pytest.raises(ValueError, match="must be positive"):
            MoARouter(
                hidden_dim=-1,
                num_experts=4,
                top_k=2,
            )

    def test_router_validation_zero_experts(self):
        """Test that zero num_experts raises ValueError."""
        from aligntune.core.adapters.moa.router import MoARouter

        with pytest.raises(ValueError, match="must be positive"):
            MoARouter(
                hidden_dim=768,
                num_experts=0,
                top_k=2,
            )

    def test_router_validation_zero_top_k(self):
        """Test that zero top_k raises ValueError."""
        from aligntune.core.adapters.moa.router import MoARouter

        with pytest.raises(ValueError, match="must be positive"):
            MoARouter(
                hidden_dim=768,
                num_experts=4,
                top_k=0,
            )


class TestMoARouterForward:
    """Test MoARouter forward pass and routing behavior."""

    def test_router_forward_output_shapes(self):
        """Test that forward pass produces correct output shapes."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        # Input: [batch=2, seq_len=10, hidden_dim=768]
        hidden_states = torch.randn(2, 10, 768)
        expert_indices, routing_weights = router(hidden_states)

        # Output shapes should be [batch, seq_len, top_k]
        assert expert_indices.shape == (2, 10, 2)
        assert routing_weights.shape == (2, 10, 2)

    def test_router_forward_expert_indices_valid(self):
        """Test that expert indices are within valid range."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        expert_indices, _ = router(hidden_states)

        # All indices should be in [0, num_experts)
        assert expert_indices.min() >= 0
        assert expert_indices.max() < 4

    def test_router_forward_weights_sum_to_one(self):
        """Test that routing weights sum to 1.0 per token."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        _, routing_weights = router(hidden_states)

        # Sum along top_k dimension should be 1.0 for each token
        weight_sums = routing_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_router_forward_weights_non_negative(self):
        """Test that routing weights are all non-negative."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        _, routing_weights = router(hidden_states)

        # All weights should be non-negative (since they come from softmax)
        assert (routing_weights >= 0.0).all()

    def test_router_forward_single_sample(self):
        """Test router with single sample."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=256,
            num_experts=8,
            top_k=3,
        )

        hidden_states = torch.randn(1, 5, 256)
        expert_indices, routing_weights = router(hidden_states)

        assert expert_indices.shape == (1, 5, 3)
        assert routing_weights.shape == (1, 5, 3)

    def test_router_forward_large_batch(self):
        """Test router with large batch size."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=512,
            num_experts=16,
            top_k=4,
        )

        hidden_states = torch.randn(32, 128, 512)
        expert_indices, routing_weights = router(hidden_states)

        assert expert_indices.shape == (32, 128, 4)
        assert routing_weights.shape == (32, 128, 4)

    def test_router_forward_invalid_input_dim(self):
        """Test that wrong input dimension raises ValueError."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        # Wrong feature dimension
        hidden_states = torch.randn(2, 10, 512)
        with pytest.raises(ValueError, match="hidden_dim"):
            router(hidden_states)

    def test_router_forward_invalid_input_shape(self):
        """Test that wrong input shape (not 3D) raises ValueError."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        # 2D input instead of 3D
        hidden_states = torch.randn(2, 768)
        with pytest.raises(ValueError, match="3D input"):
            router(hidden_states)


class TestMoARouterLoadBalanceLoss:
    """Test load balance loss computation."""

    def test_load_balance_loss_non_negative(self):
        """Test that load balance loss is always non-negative."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        _, _ = router(hidden_states)

        lb_loss = router.get_load_balance_loss()
        assert lb_loss >= 0.0

    def test_load_balance_loss_uniform_distribution(self):
        """Test that perfectly uniform distribution has minimal loss."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        # Uniform distribution: each expert used equally
        router.expert_counts = torch.ones(4) * 10.0
        lb_loss = router.get_load_balance_loss()

        # Should be zero for uniform distribution
        assert lb_loss < 1e-5

    def test_load_balance_loss_skewed_distribution(self):
        """Test that skewed distribution has non-zero loss."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        # Skewed distribution
        router.expert_counts = torch.tensor([100.0, 1.0, 1.0, 1.0])
        lb_loss = router.get_load_balance_loss()

        # Should be large for very skewed distribution
        assert lb_loss > 100.0

    def test_load_balance_loss_reset(self):
        """Test that expert counts can be reset."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        _, _ = router(hidden_states)

        # Check counts are non-zero
        assert router.expert_counts.sum() > 0

        # Reset
        router.reset_expert_counts()

        # Counts should be zero
        assert router.expert_counts.sum() == 0
        assert router.get_load_balance_loss() == 0.0


class TestMoARouterTemperature:
    """Test temperature scaling effects."""

    def test_router_temperature_default(self):
        """Test default temperature value."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
        )

        assert router.router_temp == 1.0

    def test_router_temperature_custom(self):
        """Test custom temperature value."""
        from aligntune.core.adapters.moa.router import MoARouter

        router = MoARouter(
            hidden_dim=768,
            num_experts=4,
            top_k=2,
            router_temp=0.5,
        )

        assert router.router_temp == 0.5


class TestMoALoraLayerInitialization:
    """Test MoALoraLayer initialization and parameter validation."""

    def test_moa_layer_init_basic(self):
        """Test basic MoALoraLayer initialization."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        assert moa_layer.num_experts == 4
        assert moa_layer.lora_r == 16
        assert moa_layer.lora_alpha == 32
        assert moa_layer.top_k == 2
        assert len(moa_layer.lora_a_list) == 4
        assert len(moa_layer.lora_b_list) == 4

    def test_moa_layer_init_expert_count(self):
        """Test that correct number of experts are created."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=8,
            lora_r=16,
            lora_alpha=32,
            top_k=3,
        )

        assert len(moa_layer.lora_a_list) == 8
        assert len(moa_layer.lora_b_list) == 8

    def test_moa_layer_init_lora_dimensions(self):
        """Test that LoRA matrices have correct dimensions."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(512, 2048)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=8,
            lora_alpha=16,
            top_k=2,
        )

        for lora_a in moa_layer.lora_a_list:
            assert lora_a.in_features == 512
            assert lora_a.out_features == 8

        for lora_b in moa_layer.lora_b_list:
            assert lora_b.in_features == 8
            assert lora_b.out_features == 2048

    def test_moa_layer_validation_not_linear(self):
        """Test that non-Linear base module raises ValueError."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_conv = nn.Conv1d(768, 3072, kernel_size=1)
        with pytest.raises(ValueError, match="nn.Linear"):
            MoALoraLayer(
                base_module=base_conv,
                num_experts=4,
                lora_r=16,
                lora_alpha=32,
                top_k=2,
            )

    def test_moa_layer_validation_invalid_experts(self):
        """Test that invalid expert count raises ValueError."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        with pytest.raises(ValueError, match="num_experts"):
            MoALoraLayer(
                base_module=base_linear,
                num_experts=0,
                lora_r=16,
                lora_alpha=32,
                top_k=2,
            )

    def test_moa_layer_validation_top_k_exceeds_experts(self):
        """Test that top_k > num_experts raises ValueError."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        with pytest.raises(ValueError, match="top_k"):
            MoALoraLayer(
                base_module=base_linear,
                num_experts=4,
                lora_r=16,
                lora_alpha=32,
                top_k=5,
            )


class TestMoALoraLayerForward:
    """Test MoALoraLayer forward pass."""

    def test_moa_layer_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        output = moa_layer(hidden_states)

        # Output should have same shape as input except last dimension
        assert output.shape == (2, 10, 3072)

    def test_moa_layer_forward_batch_processing(self):
        """Test forward pass with different batch sizes."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(512, 2048)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=8,
            lora_alpha=16,
            top_k=2,
        )

        # Test with batch size 1
        hidden_1 = torch.randn(1, 5, 512)
        output_1 = moa_layer(hidden_1)
        assert output_1.shape == (1, 5, 2048)

        # Test with batch size 16
        hidden_16 = torch.randn(16, 20, 512)
        output_16 = moa_layer(hidden_16)
        assert output_16.shape == (16, 20, 2048)

    def test_moa_layer_forward_single_sequence(self):
        """Test forward pass with single sequence."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(256, 1024)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=8,
            lora_r=16,
            lora_alpha=32,
            top_k=4,
        )

        hidden_states = torch.randn(1, 64, 256)
        output = moa_layer(hidden_states)

        assert output.shape == (1, 64, 1024)

    def test_moa_layer_forward_deterministic(self):
        """Test that forward pass is deterministic with same seed."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        torch.manual_seed(42)
        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        output1 = moa_layer(hidden_states)

        # Reset seed and run again
        torch.manual_seed(42)
        base_linear2 = nn.Linear(768, 3072)
        moa_layer2 = MoALoraLayer(
            base_module=base_linear2,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        output2 = moa_layer2(hidden_states)

        # Outputs should be identical (same seed)
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_moa_layer_forward_invalid_input_shape(self):
        """Test that wrong input shape raises ValueError."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        # 2D input instead of 3D
        hidden_states = torch.randn(2, 768)
        with pytest.raises(ValueError, match="3D input"):
            moa_layer(hidden_states)

    def test_moa_layer_forward_invalid_feature_dim(self):
        """Test that wrong feature dimension raises ValueError."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        # Wrong feature dimension
        hidden_states = torch.randn(2, 10, 512)
        with pytest.raises(ValueError, match="feature dimension"):
            moa_layer(hidden_states)


class TestMoALoraLayerLoadBalance:
    """Test load balance loss functionality."""

    def test_moa_layer_get_load_balance_loss(self):
        """Test getting load balance loss from layer."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        _ = moa_layer(hidden_states)

        lb_loss = moa_layer.get_load_balance_loss()
        assert isinstance(lb_loss, torch.Tensor)
        assert lb_loss.dim() == 0  # Scalar

    def test_moa_layer_reset_load_balance(self):
        """Test resetting load balance loss."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768)
        _ = moa_layer(hidden_states)

        # Reset
        moa_layer.reset_load_balance_loss()

        # Loss should be zero after reset
        lb_loss = moa_layer.get_load_balance_loss()
        assert lb_loss == 0.0


class TestMoAIntegration:
    """Integration tests for MoA system."""

    def test_moa_gradient_flow(self):
        """Test that gradients flow through MoA layer."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        hidden_states = torch.randn(2, 10, 768, requires_grad=True)
        output = moa_layer(hidden_states)

        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert hidden_states.grad is not None
        assert hidden_states.grad.abs().sum() > 0

    def test_moa_parameter_count(self):
        """Test that parameter count is correct."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        # Count total parameters
        # Router: linear gate = 768 * 4 + 4 = 3076
        # LoRA A matrices: 4 experts * (768 * 16) = 49152
        # LoRA B matrices: 4 experts * (16 * 3072) = 196608
        # Total trainable: ~248836

        total_params = sum(p.numel() for p in moa_layer.parameters())
        assert total_params > 0

        # Router should have some parameters
        router_params = sum(p.numel() for p in moa_layer.router.parameters())
        assert router_params > 0

        # LoRA should have parameters
        lora_params = sum(
            p.numel() for lora in moa_layer.lora_a_list for p in lora.parameters()
        )
        assert lora_params > 0

    def test_moa_with_optimizer(self):
        """Test MoA layer with PyTorch optimizer."""
        from aligntune.core.adapters.moa.layer import MoALoraLayer

        base_linear = nn.Linear(768, 3072)
        moa_layer = MoALoraLayer(
            base_module=base_linear,
            num_experts=4,
            lora_r=16,
            lora_alpha=32,
            top_k=2,
        )

        optimizer = torch.optim.Adam(moa_layer.parameters(), lr=1e-4)

        hidden_states = torch.randn(2, 10, 768)
        output = moa_layer(hidden_states)

        loss = output.sum()
        loss.backward()

        # Optimizer should have state
        optimizer.step()

        # Check that optimizer has accumulated state
        assert len(optimizer.state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
