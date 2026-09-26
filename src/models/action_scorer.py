from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.environment.actions import ActionType
from src.models.scb_encoder import SCBEncoderOutput


@dataclass
class ActionLogits:
    """
    Raw logits for every possible action.

    Shapes:
        add_edge:       [num_edges]
        remove_edge:    [num_edges]
        add_session:    [num_sessions]
        remove_session: [num_sessions]
        stop:            scalar
    """

    add_edge: torch.Tensor
    remove_edge: torch.Tensor
    add_session: torch.Tensor
    remove_session: torch.Tensor
    stop: torch.Tensor

    def for_action_type(self, action_type: ActionType) -> torch.Tensor:
        if action_type == ActionType.ADD_EDGE:
            return self.add_edge

        if action_type == ActionType.REMOVE_EDGE:
            return self.remove_edge

        if action_type == ActionType.ADD_SESSION:
            return self.add_session

        if action_type == ActionType.REMOVE_SESSION:
            return self.remove_session

        if action_type == ActionType.STOP:
            return self.stop.reshape(1)

        raise ValueError(f"Unsupported action type: {action_type}")


class ActionScorer(nn.Module):
    """
    Scores all candidate actions from the shared SCB encoder output.

    Important:
        This module does NOT perform action masking.

    It answers:
        "How attractive does the network think each action is?"

    The policy layer will later answer:
        "Which of those actions are currently executable?"
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        action_embedding_dim: int = 32,
        scorer_hidden_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.action_type_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=action_embedding_dim,
        )

        edge_input_dim = (
            hidden_dim
            + hidden_dim
            + action_embedding_dim
        )

        session_input_dim = (
            hidden_dim
            + hidden_dim
            + action_embedding_dim
        )

        stop_input_dim = hidden_dim + action_embedding_dim

        self.edge_scorer = nn.Sequential(
            nn.Linear(edge_input_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

        self.session_scorer = nn.Sequential(
            nn.Linear(session_input_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

        self.stop_scorer = nn.Sequential(
            nn.Linear(stop_input_dim, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, 1),
        )

    @staticmethod
    def _action_id(action_type: ActionType) -> int:
        mapping = {
            ActionType.ADD_EDGE: 0,
            ActionType.REMOVE_EDGE: 1,
            ActionType.ADD_SESSION: 2,
            ActionType.REMOVE_SESSION: 3,
            ActionType.STOP: 4,
        }

        return mapping[action_type]

    def _action_embedding(
        self,
        action_type: ActionType,
        device: torch.device,
    ) -> torch.Tensor:
        action_id = torch.tensor(
            self._action_id(action_type),
            dtype=torch.long,
            device=device,
        )

        return self.action_type_embedding(action_id)

    def forward(
        self,
        encoder_output: SCBEncoderOutput,
    ) -> ActionLogits:

        global_embedding = encoder_output.global_embedding
        edge_embeddings = encoder_output.edge_embeddings
        session_embeddings = encoder_output.session_embeddings

        device = global_embedding.device

        # ---------------------------------------------------------
        # EDGE ACTIONS
        # ---------------------------------------------------------

        add_edge_embedding = self._action_embedding(
            ActionType.ADD_EDGE,
            device,
        )

        remove_edge_embedding = self._action_embedding(
            ActionType.REMOVE_EDGE,
            device,
        )

        # [num_edges, hidden_dim]
        global_for_edges = global_embedding.unsqueeze(0).expand(
            edge_embeddings.shape[0],
            -1,
        )

        # [num_edges, action_embedding_dim]
        add_edge_action = add_edge_embedding.unsqueeze(0).expand(
            edge_embeddings.shape[0],
            -1,
        )

        remove_edge_action = remove_edge_embedding.unsqueeze(0).expand(
            edge_embeddings.shape[0],
            -1,
        )

        add_edge_input = torch.cat(
            [
                global_for_edges,
                edge_embeddings,
                add_edge_action,
            ],
            dim=-1,
        )

        remove_edge_input = torch.cat(
            [
                global_for_edges,
                edge_embeddings,
                remove_edge_action,
            ],
            dim=-1,
        )

        add_edge_logits = self.edge_scorer(
            add_edge_input
        ).squeeze(-1)

        remove_edge_logits = self.edge_scorer(
            remove_edge_input
        ).squeeze(-1)

        # ---------------------------------------------------------
        # SESSION ACTIONS
        # ---------------------------------------------------------

        add_session_embedding = self._action_embedding(
            ActionType.ADD_SESSION,
            device,
        )

        remove_session_embedding = self._action_embedding(
            ActionType.REMOVE_SESSION,
            device,
        )

        # [num_sessions, hidden_dim]
        global_for_sessions = global_embedding.unsqueeze(0).expand(
            session_embeddings.shape[0],
            -1,
        )

        add_session_action = add_session_embedding.unsqueeze(0).expand(
            session_embeddings.shape[0],
            -1,
        )

        remove_session_action = remove_session_embedding.unsqueeze(0).expand(
            session_embeddings.shape[0],
            -1,
        )

        add_session_input = torch.cat(
            [
                global_for_sessions,
                session_embeddings,
                add_session_action,
            ],
            dim=-1,
        )

        remove_session_input = torch.cat(
            [
                global_for_sessions,
                session_embeddings,
                remove_session_action,
            ],
            dim=-1,
        )

        add_session_logits = self.session_scorer(
            add_session_input
        ).squeeze(-1)

        remove_session_logits = self.session_scorer(
            remove_session_input
        ).squeeze(-1)

        # ---------------------------------------------------------
        # STOP
        # ---------------------------------------------------------

        stop_embedding = self._action_embedding(
            ActionType.STOP,
            device,
        )

        stop_input = torch.cat(
            [
                global_embedding,
                stop_embedding,
            ],
            dim=-1,
        )

        stop_logit = self.stop_scorer(
            stop_input
        )

        return ActionLogits(
            add_edge=add_edge_logits,
            remove_edge=remove_edge_logits,
            add_session=add_session_logits,
            remove_session=remove_session_logits,
            stop=stop_logit,
        )