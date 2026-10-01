from dataclasses import dataclass

import torch


@dataclass
class AdvantageBatch:
    """
    Computed learning targets for one rollout.
    """

    advantages: torch.Tensor
    returns: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = False,
) -> AdvantageBatch:
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards:
        Reward at each timestep.
        Shape: [T]

    values:
        V(s_t) for each timestep.
        Shape: [T]

    next_values:
        V(s_{t+1}) for each timestep.
        Shape: [T]

    dones:
        1.0 where the transition terminates the episode,
        otherwise 0.0.
        Shape: [T]

    gamma:
        Discount factor.

    gae_lambda:
        GAE smoothing parameter.

    normalize:
        Whether to normalize advantages across the rollout.

    Returns
    -------
    AdvantageBatch
        advantages:
            GAE advantage estimate A_t

        returns:
            critic targets R_t = A_t + V(s_t)
    """

    if rewards.ndim != 1:
        raise ValueError("rewards must have shape [T].")

    if values.ndim != 1:
        raise ValueError("values must have shape [T].")

    if next_values.ndim != 1:
        raise ValueError("next_values must have shape [T].")

    if dones.ndim != 1:
        raise ValueError("dones must have shape [T].")

    if not (
        len(rewards)
        == len(values)
        == len(next_values)
        == len(dones)
    ):
        raise ValueError(
            "rewards, values, next_values, and dones "
            "must have the same length."
        )

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1].")

    T = len(rewards)

    advantages = torch.zeros_like(rewards)

    gae = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for t in reversed(range(T)):

        non_terminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_values[t] * non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * gae
        )

        advantages[t] = gae

    returns = advantages + values

    if normalize and T > 1:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)

        if std > 1e-8:
            advantages = (
                advantages - mean
            ) / (std + 1e-8)

    return AdvantageBatch(
        advantages=advantages,
        returns=returns,
    )