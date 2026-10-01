"""PPO loss computation for the SCB joint RL system."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PPOLossOutput:
    """Individual PPO loss components and final optimization loss."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor

    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class PPOLoss:
    """
    Computes the clipped PPO objective.

    The module itself contains no optimizer and does not update
    model parameters. It only computes losses/statistics.

    Policy objective:

        min(
            ratio * advantage,
            clipped_ratio * advantage
        )

    Value objective:

        MSE(value, return)

    Entropy:

        H(policy)

    Total loss:

        -policy_objective
        + value_coef * value_loss
        - entropy_coef * entropy
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        if clip_epsilon <= 0:
            raise ValueError(
                "clip_epsilon must be positive."
            )

        if value_coef < 0:
            raise ValueError(
                "value_coef must be non-negative."
            )

        if entropy_coef < 0:
            raise ValueError(
                "entropy_coef must be non-negative."
            )

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def __call__(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> PPOLossOutput:

        self._validate_shapes(
            new_log_probs,
            old_log_probs,
            advantages,
            values,
            returns,
            entropy,
        )

        # ---------------------------------------------------------
        # PPO probability ratio
        # ---------------------------------------------------------

        log_ratio = (
            new_log_probs - old_log_probs
        )

        ratio = torch.exp(log_ratio)

        # ---------------------------------------------------------
        # Clipped surrogate objective
        # ---------------------------------------------------------

        unclipped_objective = (
            ratio * advantages
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )

        clipped_objective = (
            clipped_ratio * advantages
        )

        policy_objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        policy_loss = -policy_objective.mean()

        # ---------------------------------------------------------
        # Value loss
        # ---------------------------------------------------------

        value_loss = F.mse_loss(
            values,
            returns,
        )

        # ---------------------------------------------------------
        # Entropy
        # ---------------------------------------------------------

        entropy_mean = entropy.mean()

        entropy_loss = -entropy_mean

        # ---------------------------------------------------------
        # Total PPO loss
        # ---------------------------------------------------------

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        approx_kl = (
            old_log_probs - new_log_probs
        ).mean()

        clip_fraction = (
            (
                torch.abs(ratio - 1.0)
                > self.clip_epsilon
            )
            .float()
            .mean()
        )

        return PPOLossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    @staticmethod
    def _validate_shapes(
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:

        tensors = {
            "new_log_probs": new_log_probs,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "values": values,
            "returns": returns,
            "entropy": entropy,
        }

        for name, tensor in tensors.items():

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

            if tensor.ndim != 1:
                raise ValueError(
                    f"{name} must be a 1D tensor, "
                    f"got shape {tuple(tensor.shape)}."
                )

        expected_shape = new_log_probs.shape

        for name, tensor in tensors.items():

            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} "
                    f"does not match "
                    f"{tuple(expected_shape)}."
                )
                
"""PPO loss computation for the SCB joint RL system."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PPOLossOutput:
    """Individual PPO loss components and final optimization loss."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor

    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class PPOLoss:
    """
    Computes the clipped PPO objective.

    The module itself contains no optimizer and does not update
    model parameters. It only computes losses/statistics.

    Policy objective:

        min(
            ratio * advantage,
            clipped_ratio * advantage
        )

    Value objective:

        MSE(value, return)

    Entropy:

        H(policy)

    Total loss:

        -policy_objective
        + value_coef * value_loss
        - entropy_coef * entropy
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        if clip_epsilon <= 0:
            raise ValueError(
                "clip_epsilon must be positive."
            )

        if value_coef < 0:
            raise ValueError(
                "value_coef must be non-negative."
            )

        if entropy_coef < 0:
            raise ValueError(
                "entropy_coef must be non-negative."
            )

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def __call__(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> PPOLossOutput:

        self._validate_shapes(
            new_log_probs,
            old_log_probs,
            advantages,
            values,
            returns,
            entropy,
        )

        # ---------------------------------------------------------
        # PPO probability ratio
        # ---------------------------------------------------------

        log_ratio = (
            new_log_probs - old_log_probs
        )

        ratio = torch.exp(log_ratio)

        # ---------------------------------------------------------
        # Clipped surrogate objective
        # ---------------------------------------------------------

        unclipped_objective = (
            ratio * advantages
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )

        clipped_objective = (
            clipped_ratio * advantages
        )

        policy_objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        policy_loss = -policy_objective.mean()

        # ---------------------------------------------------------
        # Value loss
        # ---------------------------------------------------------

        value_loss = F.mse_loss(
            values,
            returns,
        )

        # ---------------------------------------------------------
        # Entropy
        # ---------------------------------------------------------

        entropy_mean = entropy.mean()

        entropy_loss = -entropy_mean

        # ---------------------------------------------------------
        # Total PPO loss
        # ---------------------------------------------------------

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        approx_kl = (
            old_log_probs - new_log_probs
        ).mean()

        clip_fraction = (
            (
                torch.abs(ratio - 1.0)
                > self.clip_epsilon
            )
            .float()
            .mean()
        )

        return PPOLossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    @staticmethod
    def _validate_shapes(
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:

        tensors = {
            "new_log_probs": new_log_probs,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "values": values,
            "returns": returns,
            "entropy": entropy,
        }

        for name, tensor in tensors.items():

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

            if tensor.ndim != 1:
                raise ValueError(
                    f"{name} must be a 1D tensor, "
                    f"got shape {tuple(tensor.shape)}."
                )

        expected_shape = new_log_probs.shape

        for name, tensor in tensors.items():

            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} "
                    f"does not match "
                    f"{tuple(expected_shape)}."
                )