from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.environment.actions import Action, ActionType
from src.environment.state import SCBState
from src.models.action_scorer import ActionLogits, ActionScorer
from src.models.scb_encoder import SCBEncoder, SCBEncoderOutput
from src.scb.problem import SCBProblem


class PolicyType(Enum):
    VALIDITY = "validity"
    MINIMIZATION = "minimization"
    GLOBAL_EXPLORATION = "global_exploration"


@dataclass
class PolicyOutput:
    """
    Output of a policy decision.

    action:
        Concrete environment action.

    action_index:
        Flat index used internally by the categorical distribution.

    log_prob:
        Log probability of the selected action.

    entropy:
        Entropy of the action distribution.

    logits:
        Masked logits over the policy's complete action space.
    """

    action: Action
    action_index: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    logits: torch.Tensor


class BaseSCBPolicy(nn.Module):
    """
    Base class for the three SCB policies.

    The encoder is supplied externally and shared conceptually across
    the three policies, while each policy owns its own ActionScorer.

    Action ordering:

        0 ... E-1
            ADD_EDGE

        E ... 2E-1
            REMOVE_EDGE

        2E ... 2E+S-1
            ADD_SESSION

        2E+S ... 2E+2S-1
            REMOVE_SESSION

        2E+2S
            STOP
    """

    def __init__(
        self,
        policy_type: PolicyType,
        hidden_dim: int = 128,
        action_embedding_dim: int = 32,
        scorer_hidden_dim: int = 128,
    ):
        super().__init__()

        self.policy_type = policy_type

        self.scorer = ActionScorer(
            hidden_dim=hidden_dim,
            action_embedding_dim=action_embedding_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )

    # ---------------------------------------------------------
    # ACTION SPACE
    # ---------------------------------------------------------

    @staticmethod
    def action_space_size(
        num_edges: int,
        num_sessions: int,
    ) -> int:
        return (
            2 * num_edges
            + 2 * num_sessions
            + 1
        )

    def _build_flat_logits(
        self,
        action_logits: ActionLogits,
        num_edges: int,
        num_sessions: int,
    ) -> torch.Tensor:

        return torch.cat(
            [
                action_logits.add_edge,
                action_logits.remove_edge,
                action_logits.add_session,
                action_logits.remove_session,
                action_logits.stop.reshape(1),
            ],
            dim=0,
        )

    # ---------------------------------------------------------
    # ACTION MASK
    # ---------------------------------------------------------

    def _build_action_mask(
        self,
        problem: SCBProblem,
        state: SCBState,
    ) -> torch.Tensor:
        """
        Construct a syntactic/executable action mask.

        True  = action is executable
        False = action must be masked

        IMPORTANT:
            This does NOT determine SCB validity.

            It only prevents nonsensical operations such as:
                ADD_EDGE on an already-cut edge
                REMOVE_EDGE on an uncut edge
                ADD_SESSION on an already-selected session
                REMOVE_SESSION on an unselected session
        """

        num_edges = problem.num_edges
        num_sessions = problem.num_sessions

        mask = torch.zeros(
            self.action_space_size(
                num_edges,
                num_sessions,
            ),
            dtype=torch.bool,
        )

        candidate = state.candidate

        # -----------------------------------------------------
        # EDGE ACTIONS
        # -----------------------------------------------------

        for edge_index, edge in problem.idx_to_edge.items():

            add_index = edge_index

            remove_index = (
                num_edges + edge_index
            )

            if edge not in candidate.cut_edges:
                mask[add_index] = True

            if edge in candidate.cut_edges:
                mask[remove_index] = True

        # -----------------------------------------------------
        # SESSION ACTIONS
        # -----------------------------------------------------

        session_offset = 2 * num_edges

        selected_sessions = candidate.selected_sessions

        for session_index, session in enumerate(
            problem.sessions
        ):
            add_index = (
                session_offset
                + session_index
            )

            remove_index = (
                session_offset
                + num_sessions
                + session_index
            )

            if session not in selected_sessions:
                mask[add_index] = True

            if session in selected_sessions:
                mask[remove_index] = True

        # -----------------------------------------------------
        # STOP
        # -----------------------------------------------------

        stop_index = (
            2 * num_edges
            + 2 * num_sessions
        )

        mask[stop_index] = True

        # -----------------------------------------------------
        # VALIDITY POLICY RESTRICTION
        # -----------------------------------------------------

        if self.policy_type == PolicyType.VALIDITY:

            # Validity policy can only modify edges.
            mask[
                2 * num_edges:
                2 * num_edges + 2 * num_sessions + 1
            ] = False

        return mask

    # ---------------------------------------------------------
    # MASK LOGITS
    # ---------------------------------------------------------

    @staticmethod
    def _apply_mask(
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        if logits.shape != mask.shape:
            raise ValueError(
                "Logit/mask shape mismatch: "
                f"{logits.shape} vs {mask.shape}"
            )

        if not mask.any():
            raise RuntimeError(
                "Action mask contains no executable actions."
            )

        return logits.masked_fill(
            ~mask.to(logits.device),
            float("-inf"),
        )

    # ---------------------------------------------------------
    # DECODE
    # ---------------------------------------------------------

    @staticmethod
    @staticmethod
    def _decode_action(
        action_index: int,
        problem: SCBProblem,
    ) -> Action:

        num_edges = problem.num_edges
        num_sessions = problem.num_sessions

        # ADD_EDGE
        if action_index < num_edges:
            return Action(
                action_type=ActionType.ADD_EDGE,
                target=problem.idx_to_edge[action_index],
            )

        # REMOVE_EDGE
        action_index -= num_edges

        if action_index < num_edges:
            return Action(
                action_type=ActionType.REMOVE_EDGE,
                target=problem.idx_to_edge[action_index],
            )

        # ADD_SESSION
        action_index -= num_edges

        if action_index < num_sessions:
            return Action(
                action_type=ActionType.ADD_SESSION,
                target=problem.sessions[action_index],
            )

        # REMOVE_SESSION
        action_index -= num_sessions

        if action_index < num_sessions:
            return Action(
                action_type=ActionType.REMOVE_SESSION,
                target=problem.sessions[action_index],
            )

        # STOP
        action_index -= num_sessions

        if action_index == 0:
            return Action(
                action_type=ActionType.STOP,
                target=None,
            )

        raise ValueError(
            f"Invalid action index: {action_index}"
        )

    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------

    def forward(
        self,
        encoder_output: SCBEncoderOutput,
        problem: SCBProblem,
        state: SCBState,
        deterministic: bool = False,
    ) -> PolicyOutput:

        action_logits = self.scorer(
            encoder_output
        )

        flat_logits = self._build_flat_logits(
            action_logits,
            problem.num_edges,
            problem.num_sessions,
        )

        mask = self._build_action_mask(
            problem,
            state,
        )

        masked_logits = self._apply_mask(
            flat_logits,
            mask,
        )

        distribution = Categorical(
            logits=masked_logits
        )

        if deterministic:
            action_index_tensor = torch.argmax(
                masked_logits
            )
        else:
            action_index_tensor = distribution.sample()

        action_index = int(
            action_index_tensor.item()
        )

        action = self._decode_action(
            action_index,
            problem,
        )

        log_prob = distribution.log_prob(
            action_index_tensor
        )

        entropy = distribution.entropy()

        return PolicyOutput(
            action=action,
            action_index=action_index,
            log_prob=log_prob,
            entropy=entropy,
            logits=masked_logits,
        )


# =============================================================
# THREE POLICIES
# =============================================================


class ValidityPolicy(BaseSCBPolicy):
    """
    Learns to make the candidate valid.

    Available actions:
        ADD_EDGE
        REMOVE_EDGE
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        action_embedding_dim: int = 32,
        scorer_hidden_dim: int = 128,
    ):
        super().__init__(
            policy_type=PolicyType.VALIDITY,
            hidden_dim=hidden_dim,
            action_embedding_dim=action_embedding_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )


class MinimizationPolicy(BaseSCBPolicy):
    """
    Learns to reduce SCB after/while maintaining useful candidates.

    Available actions:
        ADD_EDGE
        REMOVE_EDGE
        ADD_SESSION
        REMOVE_SESSION
        STOP
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        action_embedding_dim: int = 32,
        scorer_hidden_dim: int = 128,
    ):
        super().__init__(
            policy_type=PolicyType.MINIMIZATION,
            hidden_dim=hidden_dim,
            action_embedding_dim=action_embedding_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )


class GlobalExplorationPolicy(BaseSCBPolicy):
    """
    Searches for alternative joint (I', E') structures.

    Available actions:
        ADD_EDGE
        REMOVE_EDGE
        ADD_SESSION
        REMOVE_SESSION
        STOP
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        action_embedding_dim: int = 32,
        scorer_hidden_dim: int = 128,
    ):
        super().__init__(
            policy_type=PolicyType.GLOBAL_EXPLORATION,
            hidden_dim=hidden_dim,
            action_embedding_dim=action_embedding_dim,
            scorer_hidden_dim=scorer_hidden_dim,
        )


__all__ = [
    "PolicyType",
    "PolicyOutput",
    "BaseSCBPolicy",
    "ValidityPolicy",
    "MinimizationPolicy",
    "GlobalExplorationPolicy",
]