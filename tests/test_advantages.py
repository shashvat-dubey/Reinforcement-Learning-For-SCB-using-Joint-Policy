import torch
import pytest

from src.training.advantages import compute_gae


# ============================================================
# 1. ZERO REWARD / ZERO VALUE
# ============================================================

def test_zero_rewards_and_values():

    rewards = torch.zeros(4)
    values = torch.zeros(4)
    next_values = torch.zeros(4)
    dones = torch.zeros(4)

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
    )

    assert torch.allclose(
        result.advantages,
        torch.zeros(4),
    )

    assert torch.allclose(
        result.returns,
        torch.zeros(4),
    )


# ============================================================
# 2. SINGLE TERMINAL REWARD
# ============================================================

def test_single_terminal_reward():

    rewards = torch.tensor([1.0])
    values = torch.tensor([0.0])
    next_values = torch.tensor([100.0])

    # Terminal means we MUST NOT bootstrap from V(s_next).
    dones = torch.tensor([1.0])

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert torch.allclose(
        result.advantages,
        torch.tensor([1.0]),
    )

    assert torch.allclose(
        result.returns,
        torch.tensor([1.0]),
    )


# ============================================================
# 3. BOOTSTRAP FROM NEXT VALUE
# ============================================================

def test_bootstrap_from_next_value():

    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([2.0])
    dones = torch.tensor([0.0])

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_delta = (
        1.0
        + 0.99 * 2.0
        - 0.5
    )

    assert torch.allclose(
        result.advantages,
        torch.tensor([expected_delta]),
    )

    assert torch.allclose(
        result.returns,
        torch.tensor(
            [expected_delta + 0.5]
        ),
    )


# ============================================================
# 4. MULTI-STEP GAE
# ============================================================

def test_multistep_gae():

    rewards = torch.tensor([
        1.0,
        1.0,
        1.0,
    ])

    values = torch.tensor([
        0.5,
        0.5,
        0.5,
    ])

    next_values = torch.tensor([
        0.5,
        0.5,
        0.5,
    ])

    dones = torch.tensor([
        0.0,
        0.0,
        1.0,
    ])

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Each non-terminal delta:
    #
    # 1 + 0.99 * 0.5 - 0.5
    # = 0.995
    #
    # Final timestep:
    #
    # 1 - 0.5
    # = 0.5

    delta = 0.995

    expected_last = 0.5

    expected_second = (
        delta
        + 0.99 * 0.95 * expected_last
    )

    expected_first = (
        delta
        + 0.99 * 0.95 * expected_second
    )

    expected = torch.tensor([
        expected_first,
        expected_second,
        expected_last,
    ])

    assert torch.allclose(
        result.advantages,
        expected,
        atol=1e-6,
    )


# ============================================================
# 5. TERMINAL STATE STOPS GAE PROPAGATION
# ============================================================

def test_terminal_stops_gae_propagation():

    rewards = torch.tensor([
        0.0,
        10.0,
    ])

    values = torch.zeros(2)
    next_values = torch.zeros(2)

    dones = torch.tensor([
        1.0,
        1.0,
    ])

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
    )

    # Because timestep 0 is terminal,
    # the reward from timestep 1 must NOT
    # propagate backward into timestep 0.

    assert torch.allclose(
        result.advantages,
        torch.tensor([
            0.0,
            10.0,
        ]),
    )


# ============================================================
# 6. RETURNS = ADVANTAGES + VALUES
# ============================================================

def test_returns_are_advantages_plus_values():

    rewards = torch.tensor([
        1.0,
        2.0,
        3.0,
    ])

    values = torch.tensor([
        0.5,
        1.0,
        1.5,
    ])

    next_values = torch.tensor([
        1.0,
        1.5,
        0.0,
    ])

    dones = torch.tensor([
        0.0,
        0.0,
        1.0,
    ])

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
    )

    assert torch.allclose(
        result.returns,
        result.advantages + values,
    )


# ============================================================
# 7. ADVANTAGE NORMALIZATION
# ============================================================

def test_advantage_normalization():

    rewards = torch.tensor([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    values = torch.zeros(4)
    next_values = torch.zeros(4)

    dones = torch.ones(4)

    result = compute_gae(
        rewards,
        values,
        next_values,
        dones,
        normalize=True,
    )

    assert torch.allclose(
        result.advantages.mean(),
        torch.tensor(0.0),
        atol=1e-6,
    )

    assert torch.allclose(
        result.advantages.std(unbiased=False),
        torch.tensor(1.0),
        atol=1e-6,
    )


# ============================================================
# 8. INVALID SHAPES
# ============================================================

def test_invalid_shapes():

    with pytest.raises(ValueError):

        compute_gae(
            torch.zeros(3),
            torch.zeros(2),
            torch.zeros(3),
            torch.zeros(3),
        )


# ============================================================
# 9. INVALID GAMMA
# ============================================================

def test_invalid_gamma():

    with pytest.raises(ValueError):

        compute_gae(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            gamma=1.5,
        )


# ============================================================
# 10. INVALID LAMBDA
# ============================================================

def test_invalid_lambda():

    with pytest.raises(ValueError):

        compute_gae(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            gae_lambda=-0.1,
        )