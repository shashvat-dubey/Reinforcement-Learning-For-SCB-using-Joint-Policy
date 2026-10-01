import torch
import pytest

from src.models.value_head import SCBValueHead


# ============================================================
# 1. SINGLE STATE OUTPUT
# ============================================================

def test_single_state_output_shape():

    critic = SCBValueHead(
        hidden_dim=128,
        value_hidden_dim=128,
    )

    embedding = torch.randn(128)

    value = critic(embedding)

    assert value.shape == torch.Size([1])


# ============================================================
# 2. BATCH OUTPUT
# ============================================================

def test_batch_output_shape():

    critic = SCBValueHead(
        hidden_dim=128,
        value_hidden_dim=128,
    )

    embeddings = torch.randn(8, 128)

    values = critic(embeddings)

    assert values.shape == torch.Size([8])


# ============================================================
# 3. DIFFERENT STATES CAN PRODUCE DIFFERENT VALUES
# ============================================================

def test_different_embeddings_can_produce_different_values():

    torch.manual_seed(42)

    critic = SCBValueHead(
        hidden_dim=128,
        value_hidden_dim=128,
    )

    embedding_a = torch.randn(128)
    embedding_b = torch.randn(128)

    value_a = critic(embedding_a)
    value_b = critic(embedding_b)

    assert not torch.allclose(
        value_a,
        value_b,
    )


# ============================================================
# 4. GRADIENT FLOWS THROUGH CRITIC
# ============================================================

def test_gradient_flow():

    critic = SCBValueHead(
        hidden_dim=128,
        value_hidden_dim=128,
    )

    embedding = torch.randn(
        128,
        requires_grad=True,
    )

    value = critic(embedding)

    loss = value.sum()

    loss.backward()

    assert embedding.grad is not None

    assert torch.isfinite(
        embedding.grad
    ).all()

    parameter_has_gradient = any(
        parameter.grad is not None
        for parameter in critic.parameters()
    )

    assert parameter_has_gradient


# ============================================================
# 5. BATCH GRADIENT FLOW
# ============================================================

def test_batch_gradient_flow():

    critic = SCBValueHead(
        hidden_dim=128,
    )

    embeddings = torch.randn(
        4,
        128,
        requires_grad=True,
    )

    values = critic(embeddings)

    loss = values.mean()

    loss.backward()

    assert embeddings.grad is not None

    assert embeddings.grad.shape == (
        4,
        128,
    )


# ============================================================
# 6. WRONG EMBEDDING DIMENSION
# ============================================================

def test_wrong_embedding_dimension():

    critic = SCBValueHead(
        hidden_dim=128,
    )

    embedding = torch.randn(64)

    with pytest.raises(ValueError):

        critic(embedding)


# ============================================================
# 7. WRONG BATCH EMBEDDING DIMENSION
# ============================================================

def test_wrong_batch_embedding_dimension():

    critic = SCBValueHead(
        hidden_dim=128,
    )

    embeddings = torch.randn(
        4,
        64,
    )

    with pytest.raises(ValueError):

        critic(embeddings)


# ============================================================
# 8. WRONG RANK
# ============================================================

def test_wrong_embedding_rank():

    critic = SCBValueHead(
        hidden_dim=128,
    )

    embeddings = torch.randn(
        2,
        4,
        128,
    )

    with pytest.raises(ValueError):

        critic(embeddings)


# ============================================================
# 9. CUSTOM DIMENSION
# ============================================================

def test_custom_hidden_dimension():

    critic = SCBValueHead(
        hidden_dim=64,
        value_hidden_dim=32,
    )

    embedding = torch.randn(64)

    value = critic(embedding)

    assert value.shape == torch.Size([1])


# ============================================================
# 10. CRITIC HAS TRAINABLE PARAMETERS
# ============================================================

def test_critic_has_trainable_parameters():

    critic = SCBValueHead(
        hidden_dim=128,
    )

    trainable_parameters = [
        parameter
        for parameter in critic.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


# ============================================================
# 11. OUTPUT IS FINITE
# ============================================================

def test_output_is_finite():

    critic = SCBValueHead(
        hidden_dim=128,
    )

    embedding = torch.randn(128)

    value = critic(embedding)

    assert torch.isfinite(value).all()