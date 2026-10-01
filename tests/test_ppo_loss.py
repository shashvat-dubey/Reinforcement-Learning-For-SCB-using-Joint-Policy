import pytest
import torch

from src.training.ppo_loss import (
    PPOLoss,
    PPOLossOutput,
)


# ============================================================
# HELPERS
# ============================================================

def make_inputs(batch_size=8):

    torch.manual_seed(42)

    new_log_probs = torch.randn(batch_size)
    old_log_probs = torch.randn(batch_size)

    advantages = torch.randn(batch_size)

    values = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    entropy = torch.rand(batch_size)

    return (
        new_log_probs,
        old_log_probs,
        advantages,
        values,
        returns,
        entropy,
    )


# ============================================================
# 1. BASIC OUTPUT
# ============================================================

def test_ppo_loss_returns_expected_structure():

    loss_fn = PPOLoss()

    inputs = make_inputs()

    output = loss_fn(*inputs)

    assert isinstance(
        output,
        PPOLossOutput,
    )

    assert output.total_loss.ndim == 0
    assert output.policy_loss.ndim == 0
    assert output.value_loss.ndim == 0
    assert output.entropy_loss.ndim == 0


# ============================================================
# 2. FINITE LOSSES
# ============================================================

def test_losses_are_finite():

    loss_fn = PPOLoss()

    output = loss_fn(
        *make_inputs()
    )

    assert torch.isfinite(
        output.total_loss
    )

    assert torch.isfinite(
        output.policy_loss
    )

    assert torch.isfinite(
        output.value_loss
    )

    assert torch.isfinite(
        output.entropy_loss
    )


# ============================================================
# 3. PPO RATIO
# ============================================================

def test_equal_log_probs_give_ratio_one():

    loss_fn = PPOLoss()

    old_log_probs = torch.tensor(
        [-1.0, -2.0, -3.0]
    )

    new_log_probs = old_log_probs.clone()

    advantages = torch.ones(3)

    values = torch.zeros(3)
    returns = torch.zeros(3)

    entropy = torch.ones(3)

    output = loss_fn(
        new_log_probs,
        old_log_probs,
        advantages,
        values,
        returns,
        entropy,
    )

    assert output.clip_fraction.item() == 0.0


# ============================================================
# 4. CLIPPING DETECTS LARGE POLICY CHANGE
# ============================================================

def test_clip_fraction_detects_large_policy_change():

    loss_fn = PPOLoss(
        clip_epsilon=0.2
    )

    old_log_probs = torch.tensor(
        [-2.0, -2.0, -2.0, -2.0]
    )

    new_log_probs = torch.tensor(
        [-1.0, -1.0, -1.0, -1.0]
    )

    advantages = torch.ones(4)

    values = torch.zeros(4)
    returns = torch.zeros(4)

    entropy = torch.ones(4)

    output = loss_fn(
        new_log_probs,
        old_log_probs,
        advantages,
        values,
        returns,
        entropy,
    )

    assert output.clip_fraction.item() == 1.0


# ============================================================
# 5. VALUE LOSS
# ============================================================

def test_value_loss_matches_mse():

    loss_fn = PPOLoss(
        value_coef=1.0,
        entropy_coef=0.0,
    )

    new_log_probs = torch.zeros(2)
    old_log_probs = torch.zeros(2)

    advantages = torch.zeros(2)

    values = torch.tensor(
        [1.0, 3.0]
    )

    returns = torch.tensor(
        [2.0, 1.0]
    )

    entropy = torch.zeros(2)

    output = loss_fn(
        new_log_probs,
        old_log_probs,
        advantages,
        values,
        returns,
        entropy,
    )

    expected = torch.tensor(
        (1.0 + 4.0) / 2
    )

    assert torch.allclose(
        output.value_loss,
        expected,
    )


# ============================================================
# 6. ENTROPY CONTRIBUTION
# ============================================================

def test_entropy_bonus_reduces_total_loss():

    inputs = make_inputs()

    loss_without_entropy = PPOLoss(
        entropy_coef=0.0
    )(*inputs)

    loss_with_entropy = PPOLoss(
        entropy_coef=0.1
    )(*inputs)

    assert (
        loss_with_entropy.total_loss
        < loss_without_entropy.total_loss
    )


# ============================================================
# 7. TOTAL LOSS FORMULA
# ============================================================

def test_total_loss_formula():

    loss_fn = PPOLoss(
        value_coef=0.5,
        entropy_coef=0.01,
    )

    output = loss_fn(
        *make_inputs()
    )

    expected = (
        output.policy_loss
        + 0.5 * output.value_loss
        + 0.01 * output.entropy_loss
    )

    assert torch.allclose(
        output.total_loss,
        expected,
    )


# ============================================================
# 8. GRADIENT FLOW
# ============================================================

def test_gradient_flows_through_loss():

    new_log_probs = torch.randn(
        8,
        requires_grad=True,
    )

    old_log_probs = torch.randn(8)

    advantages = torch.randn(8)

    values = torch.randn(
        8,
        requires_grad=True,
    )

    returns = torch.randn(8)

    entropy = torch.randn(
        8,
        requires_grad=True,
    )

    loss_fn = PPOLoss()

    output = loss_fn(
        new_log_probs,
        old_log_probs,
        advantages,
        values,
        returns,
        entropy,
    )

    output.total_loss.backward()

    assert new_log_probs.grad is not None
    assert values.grad is not None
    assert entropy.grad is not None


# ============================================================
# 9. INVALID SHAPES
# ============================================================

def test_invalid_shapes():

    loss_fn = PPOLoss()

    inputs = list(
        make_inputs()
    )

    inputs[2] = torch.randn(4)

    with pytest.raises(ValueError):

        loss_fn(*inputs)


# ============================================================
# 10. INVALID RANK
# ============================================================

def test_invalid_rank():

    loss_fn = PPOLoss()

    inputs = list(
        make_inputs()
    )

    inputs[0] = torch.randn(
        2,
        4,
    )

    with pytest.raises(ValueError):

        loss_fn(*inputs)


# ============================================================
# 11. INVALID CLIP EPSILON
# ============================================================

def test_invalid_clip_epsilon():

    with pytest.raises(ValueError):

        PPOLoss(
            clip_epsilon=0.0
        )


# ============================================================
# 12. INVALID VALUE COEFFICIENT
# ============================================================

def test_invalid_value_coef():

    with pytest.raises(ValueError):

        PPOLoss(
            value_coef=-1.0
        )


# ============================================================
# 13. INVALID ENTROPY COEFFICIENT
# ============================================================

def test_invalid_entropy_coef():

    with pytest.raises(ValueError):

        PPOLoss(
            entropy_coef=-1.0
        )


# ============================================================
# 14. DIAGNOSTIC VALUES
# ============================================================

def test_diagnostics_are_finite():

    output = PPOLoss()(
        *make_inputs()
    )

    assert torch.isfinite(
        output.approx_kl
    )

    assert torch.isfinite(
        output.clip_fraction
    )

    assert (
        0.0
        <= output.clip_fraction.item()
        <= 1.0
    )