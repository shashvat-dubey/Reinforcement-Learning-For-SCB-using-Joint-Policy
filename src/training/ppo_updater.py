"""PPO optimization for the joint SCB RL system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor, nn


@dataclass
class PPOUpdateStats:
    """Statistics returned after one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float

    total_loss: float

    approx_kl: float
    clip_fraction: float

    grad_norm: float

    num_epochs: int
    num_samples: int


class PPOUpdater:
    """
    PPO optimizer for the joint SCB policy/value network.

    The updater is intentionally independent of:

        Environment
        Controller
        Policy phase logic
        GA
        Reward construction
        Rollout generation

    It operates only on the collected rollout and the trainable model.

    Expected rollout transition fields:

        log_prob
        entropy
        reward
        done

    GAE is expected to have already produced:

        advantages
        returns

    The updater therefore does not recompute GAE.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: Optional[int] = None,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if clip_epsilon <= 0:
            raise ValueError(
                "clip_epsilon must be positive."
            )

        if value_coef < 0:
            raise ValueError(
                "value_coef cannot be negative."
            )

        if entropy_coef < 0:
            raise ValueError(
                "entropy_coef cannot be negative."
            )

        if max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive."
            )

        if ppo_epochs <= 0:
            raise ValueError(
                "ppo_epochs must be positive."
            )

        if minibatch_size is not None and minibatch_size <= 0:
            raise ValueError(
                "minibatch_size must be positive."
            )

        self.model = model

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    # =========================================================
    # UTILITY
    # =========================================================

    @staticmethod
    def _to_tensor(
        values: Iterable,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:

        return torch.as_tensor(
            list(values),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _extract(
        rollout,
        field: str,
    ):
        """
        Extract one field from a rollout.

        Supports both:

            RolloutBuffer
            list[RolloutTransition]
        """

        if hasattr(rollout, "transitions"):
            transitions = rollout.transitions
        else:
            transitions = rollout

        return [
            getattr(transition, field)
            for transition in transitions
        ]

    # =========================================================
    # BATCH PREPARATION
    # =========================================================

    def _prepare_batch(
        self,
        rollout,
        advantages: Tensor,
        returns: Tensor,
    ):
        """
        Prepare old PPO quantities from the rollout.

        States/actions remain attached to the transition objects
        because the actual policy/value evaluation is model-specific.

        This method only validates the PPO bookkeeping tensors.
        """

        if hasattr(rollout, "__len__"):
            num_samples = len(rollout)
        elif hasattr(rollout, "transitions"):
            num_samples = len(rollout.transitions)
        else:
            raise TypeError(
                "rollout must be iterable or expose transitions."
            )

        if num_samples == 0:
            raise ValueError(
                "Cannot perform PPO update on an empty rollout."
            )

        if advantages.ndim != 1:
            raise ValueError(
                "advantages must be a 1-D tensor."
            )

        if returns.ndim != 1:
            raise ValueError(
                "returns must be a 1-D tensor."
            )

        if len(advantages) != num_samples:
            raise ValueError(
                "advantages length does not match rollout."
            )

        if len(returns) != num_samples:
            raise ValueError(
                "returns length does not match rollout."
            )

        old_log_probs = self._extract(
            rollout,
            "log_prob",
        )

        old_log_probs = self._to_tensor(
            [
                value.item()
                if isinstance(value, Tensor)
                else value
                for value in old_log_probs
            ],
            device=advantages.device,
        )

        return old_log_probs

    # =========================================================
    # PPO LOSS
    # =========================================================

    def compute_policy_loss(
        self,
        new_log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate PPO clipped policy loss.

        Returns:

            policy_loss
            clip_fraction
        """

        log_ratio = (
            new_log_probs - old_log_probs
        )

        ratio = torch.exp(log_ratio)

        unclipped = (
            ratio * advantages
        )

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )

        clipped = (
            clipped_ratio * advantages
        )

        policy_loss = -torch.min(
            unclipped,
            clipped,
        ).mean()

        clip_fraction = (
            (torch.abs(ratio - 1.0) > self.clip_epsilon)
            .float()
            .mean()
        )

        return (
            policy_loss,
            clip_fraction,
        )

    def compute_value_loss(
        self,
        values: Tensor,
        returns: Tensor,
    ) -> Tensor:
        """Calculate value-function MSE loss."""

        return 0.5 * (
            values - returns
        ).pow(2).mean()

    # =========================================================
    # MINIBATCH INDICES
    # =========================================================

    def _make_minibatches(
        self,
        num_samples: int,
        device: torch.device,
    ) -> list[Tensor]:

        indices = torch.randperm(
            num_samples,
            device=device,
        )

        if self.minibatch_size is None:
            return [indices]

        return list(
            indices.split(
                self.minibatch_size
            )
        )

    # =========================================================
    # MODEL EVALUATION
    # =========================================================

    def _evaluate_transition(
        self,
        transition,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Re-evaluate one transition using the current model.

        The model is expected to expose:

            evaluate_action(...)

        returning:

            new_log_prob
            entropy
            value

        This adapter intentionally lives here so that the PPO
        algorithm remains independent from the environment.
        """

        if not hasattr(
            self.model,
            "evaluate_action",
        ):
            raise AttributeError(
                "PPO model must implement "
                "evaluate_action(transition)."
            )

        result = self.model.evaluate_action(
            transition
        )

        if len(result) != 3:
            raise ValueError(
                "evaluate_action() must return "
                "(log_prob, entropy, value)."
            )

        new_log_prob, entropy, value = result

        return (
            new_log_prob,
            entropy,
            value,
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        rollout,
        advantages: Tensor,
        returns: Tensor,
    ) -> PPOUpdateStats:
        """
        Perform a complete PPO update.

        GAE must already have been computed before calling this.

        Returns:
            PPOUpdateStats
        """

        if not isinstance(
            advantages,
            Tensor,
        ):
            advantages = torch.as_tensor(
                advantages,
                dtype=torch.float32,
            )

        if not isinstance(
            returns,
            Tensor,
        ):
            returns = torch.as_tensor(
                returns,
                dtype=torch.float32,
            )

        device = next(
            self.model.parameters()
        ).device

        advantages = advantages.to(device)
        returns = returns.to(device)

        old_log_probs = self._prepare_batch(
            rollout,
            advantages,
            returns,
        )

        old_log_probs = old_log_probs.to(device)

        # -----------------------------------------------------
        # Normalize advantages.
        # -----------------------------------------------------

        if len(advantages) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False)
                + 1e-8
            )

        num_samples = len(
            advantages
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        total_grad_norm = 0.0

        update_count = 0

        # -----------------------------------------------------
        # PPO epochs.
        # -----------------------------------------------------

        for _ in range(self.ppo_epochs):

            minibatches = self._make_minibatches(
                num_samples,
                device,
            )

            for indices in minibatches:

                batch_transitions = [
                    rollout[int(index)]
                    for index in indices
                ]

                new_log_probs = []
                entropies = []
                values = []

                for transition in batch_transitions:

                    (
                        new_log_prob,
                        entropy,
                        value,
                    ) = self._evaluate_transition(
                        transition
                    )

                    new_log_probs.append(
                        new_log_prob
                    )

                    entropies.append(
                        entropy
                    )

                    values.append(
                        value
                    )

                new_log_probs = torch.stack(
                    new_log_probs
                )

                entropies = torch.stack(
                    entropies
                )

                values = torch.stack(
                    values
                ).reshape(-1)

                batch_old_log_probs = (
                    old_log_probs[indices]
                )

                batch_advantages = (
                    advantages[indices]
                )

                batch_returns = (
                    returns[indices]
                )

                # -------------------------------------------------
                # Policy loss
                # -------------------------------------------------

                policy_loss, clip_fraction = (
                    self.compute_policy_loss(
                        new_log_probs,
                        batch_old_log_probs,
                        batch_advantages,
                    )
                )

                # -------------------------------------------------
                # Value loss
                # -------------------------------------------------

                value_loss = (
                    self.compute_value_loss(
                        values,
                        batch_returns,
                    )
                )

                # -------------------------------------------------
                # Entropy
                # -------------------------------------------------

                entropy = entropies.mean()

                # -------------------------------------------------
                # Total PPO objective
                # -------------------------------------------------

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # -------------------------------------------------
                # Diagnostics
                # -------------------------------------------------

                with torch.no_grad():

                    log_ratio = (
                        new_log_probs
                        - batch_old_log_probs
                    )

                    approx_kl = (
                        0.5
                        * log_ratio.pow(2)
                    ).mean()

                total_policy_loss += (
                    policy_loss.detach().item()
                )

                total_value_loss += (
                    value_loss.detach().item()
                )

                total_entropy += (
                    entropy.detach().item()
                )

                total_loss += (
                    loss.detach().item()
                )

                total_kl += (
                    approx_kl.detach().item()
                )

                total_clip_fraction += (
                    clip_fraction.detach().item()
                )

                total_grad_norm += (
                    float(grad_norm)
                )

                update_count += 1

        if update_count == 0:
            raise RuntimeError(
                "PPO update produced zero optimization steps."
            )

        return PPOUpdateStats(
            policy_loss=(
                total_policy_loss
                / update_count
            ),
            value_loss=(
                total_value_loss
                / update_count
            ),
            entropy=(
                total_entropy
                / update_count
            ),
            total_loss=(
                total_loss
                / update_count
            ),
            approx_kl=(
                total_kl
                / update_count
            ),
            clip_fraction=(
                total_clip_fraction
                / update_count
            ),
            grad_norm=(
                total_grad_norm
                / update_count
            ),
            num_epochs=self.ppo_epochs,
            num_samples=num_samples,
        )