from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.environment.state import SCBState
from src.scb.evaluator import evaluate_candidate
from src.scb.problem import SCBProblem


@dataclass
class SCBEncoderOutput:
    """
    Structured representation produced by the shared SCB encoder.

    These embeddings will later be consumed by the three policy heads.
    """

    node_embeddings: Tensor
    edge_embeddings: Tensor
    session_embeddings: Tensor
    global_embedding: Tensor

    # Session-specific representation of every edge.
    # Shape: [num_sessions, num_edges, hidden_dim]
    session_edge_embeddings: Tensor


class EdgeAwareBlock(nn.Module):
    """
    Residual edge-aware graph message-passing block.

    Nodes and edges are updated together.

    For an edge u -- e -- v:

        message(u -> v) = f(u, v, e)
        message(v -> u) = f(v, u, e)

    This allows edge state, including current cut status,
    to influence graph propagation.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_h: Tensor,
        edge_h: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:

        src = edge_index[0]
        dst = edge_index[1]

        src_h = node_h[src]
        dst_h = node_h[dst]

        # ------------------------------------------------------------
        # Update edges
        # ------------------------------------------------------------

        edge_input = torch.cat(
            [src_h, dst_h, edge_h],
            dim=-1,
        )

        new_edge = edge_h + self.edge_update(edge_input)
        new_edge = self.edge_norm(new_edge)

        # ------------------------------------------------------------
        # Send messages in both directions
        # ------------------------------------------------------------

        forward_msg = self.message(
            torch.cat(
                [src_h, dst_h, new_edge],
                dim=-1,
            )
        )

        backward_msg = self.message(
            torch.cat(
                [dst_h, src_h, new_edge],
                dim=-1,
            )
        )

        # ------------------------------------------------------------
        # Aggregate messages at nodes
        # ------------------------------------------------------------

        aggregated = torch.zeros_like(node_h)

        aggregated.index_add_(
            0,
            src,
            backward_msg,
        )

        aggregated.index_add_(
            0,
            dst,
            forward_msg,
        )

        # Normalize by node degree so high-degree nodes don't
        # automatically dominate simply because they receive
        # more messages.
        degree = torch.zeros(
            node_h.size(0),
            device=node_h.device,
            dtype=node_h.dtype,
        )

        ones = torch.ones(
            src.size(0),
            device=node_h.device,
            dtype=node_h.dtype,
        )

        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)

        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)

        # ------------------------------------------------------------
        # Residual node update
        # ------------------------------------------------------------

        new_node = node_h + self.node_update(
            torch.cat(
                [node_h, aggregated],
                dim=-1,
            )
        )

        new_node = self.node_norm(new_node)

        return new_node, new_edge


class SessionConditionedBlock(nn.Module):
    """
    Session-conditioned graph propagation.

    Every source-sink pair gets its own graph-level propagation state.

    Therefore the same physical edge can eventually have different
    representations for different sessions.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.session_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.session_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_h: Tensor,
        edge_h: Tensor,
        session_h: Tensor,
        session_nodes: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:

        # session_nodes:
        # [num_sessions, num_nodes, hidden_dim]

        num_sessions, num_nodes, hidden_dim = session_nodes.shape

        src = edge_index[0]
        dst = edge_index[1]

        num_edges = edge_index.size(1)

        src_h = session_nodes[:, src, :]
        dst_h = session_nodes[:, dst, :]

        edge_context = edge_h.unsqueeze(0).expand(
            num_sessions,
            num_edges,
            hidden_dim,
        )

        session_context = session_h.unsqueeze(1).expand(
            num_sessions,
            num_edges,
            hidden_dim,
        )

        # ------------------------------------------------------------
        # Session-conditioned edge messages
        # ------------------------------------------------------------

        messages = self.message(
            torch.cat(
                [
                    src_h,
                    dst_h,
                    edge_context,
                    session_context,
                ],
                dim=-1,
            )
        )

        # ------------------------------------------------------------
        # Aggregate into session-specific node states
        # ------------------------------------------------------------

        aggregated = torch.zeros_like(session_nodes)

        aggregated.index_add_(
            1,
            src,
            messages,
        )

        aggregated.index_add_(
            1,
            dst,
            messages,
        )

        degree = torch.zeros(
            num_nodes,
            device=node_h.device,
            dtype=node_h.dtype,
        )

        ones = torch.ones(
            num_edges,
            device=node_h.device,
            dtype=node_h.dtype,
        )

        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)

        aggregated = aggregated / degree.clamp_min(1.0).view(
            1,
            num_nodes,
            1,
        )

        # ------------------------------------------------------------
        # Update session-specific nodes
        # ------------------------------------------------------------

        updated_nodes = session_nodes + self.node_update(
            torch.cat(
                [
                    session_nodes,
                    aggregated,
                ],
                dim=-1,
            )
        )

        updated_nodes = self.node_norm(updated_nodes)

        # ------------------------------------------------------------
        # Update session representations
        # ------------------------------------------------------------

        pooled_nodes = updated_nodes.mean(dim=1)

        updated_sessions = session_h + self.session_update(
            torch.cat(
                [
                    session_h,
                    pooled_nodes,
                    session_h,
                ],
                dim=-1,
            )
        )

        updated_sessions = self.session_norm(updated_sessions)

        return updated_sessions, updated_nodes


class SCBEncoder(nn.Module):
    """
    Shared structural encoder for the joint SCB RL system.

    It does NOT:
        - choose actions
        - use GA results
        - calculate rewards
        - perform PPO
        - control curriculum

    Its job is to transform the current:

        Graph + Candidate (I', E')

    into learned node, edge, session and global representations.
    """

    NODE_INPUT_DIM = 4
    EDGE_INPUT_DIM = 3
    SESSION_FLAG_DIM = 2
    CANDIDATE_INPUT_DIM = 5

    def __init__(
        self,
        hidden_dim: int = 128,
        topology_layers: int = 3,
        session_layers: int = 2,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if topology_layers < 1:
            raise ValueError(
                "topology_layers must be at least 1"
            )

        if session_layers < 1:
            raise ValueError(
                "session_layers must be at least 1"
            )

        self.hidden_dim = hidden_dim

        # ------------------------------------------------------------
        # Initial projections
        # ------------------------------------------------------------

        self.node_projection = nn.Sequential(
            nn.Linear(
                self.NODE_INPUT_DIM,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_projection = nn.Sequential(
            nn.Linear(
                self.EDGE_INPUT_DIM,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.session_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 + self.SESSION_FLAG_DIM,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Learned markers identifying the source and sink of
        # each session during session-conditioned propagation.
        self.source_marker = nn.Parameter(
            torch.randn(hidden_dim) * 0.02
        )

        self.sink_marker = nn.Parameter(
            torch.randn(hidden_dim) * 0.02
        )

        # ------------------------------------------------------------
        # Graph topology reasoning
        # ------------------------------------------------------------

        self.topology_blocks = nn.ModuleList(
            EdgeAwareBlock(hidden_dim)
            for _ in range(topology_layers)
        )

        # ------------------------------------------------------------
        # Session-conditioned reasoning
        # ------------------------------------------------------------

        self.session_blocks = nn.ModuleList(
            SessionConditionedBlock(hidden_dim)
            for _ in range(session_layers)
        )

        # ------------------------------------------------------------
        # Session ↔ edge attention
        # ------------------------------------------------------------

        self.session_edge_attention = nn.Sequential(
            nn.Linear(
                hidden_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ------------------------------------------------------------
        # Global representation
        # ------------------------------------------------------------

        self.global_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 4 + self.CANDIDATE_INPUT_DIM,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        problem: SCBProblem,
        state: SCBState,
    ) -> SCBEncoderOutput:

        if state.candidate is None:
            raise ValueError(
                "SCBState must contain a Candidate."
            )

        device = next(self.parameters()).device

        (
            node_features,
            edge_features,
            edge_index,
        ) = self._build_graph_features(
            problem,
            state,
            device,
        )

        # ============================================================
        # INITIAL NODE / EDGE EMBEDDINGS
        # ============================================================

        node_h = self.node_projection(
            node_features
        )

        edge_h = self.edge_projection(
            edge_features
        )

        # ============================================================
        # TOPOLOGY REASONING
        # ============================================================

        for block in self.topology_blocks:
            node_h, edge_h = block(
                node_h,
                edge_h,
                edge_index,
            )

        # ============================================================
        # SESSION REPRESENTATIONS
        # ============================================================

        (
            session_features,
            source_idx,
            sink_idx,
        ) = self._build_session_features(
            problem,
            state,
            node_h,
            device,
        )

        session_h = self.session_projection(
            session_features
        )

        # ============================================================
        # INITIAL SESSION-CONDITIONED NODE STATES
        # ============================================================

        num_sessions = problem.num_sessions

        session_nodes = (
            node_h
            .unsqueeze(0)
            .expand(
                num_sessions,
                -1,
                -1,
            )
            .clone()
        )

        source_mask = self._node_session_mask(
            problem.num_nodes,
            source_idx,
            device,
        )

        sink_mask = self._node_session_mask(
            problem.num_nodes,
            sink_idx,
            device,
        )

        session_nodes = (
            session_nodes
            + self.source_marker * source_mask
            + self.sink_marker * sink_mask
        )

        # ============================================================
        # SESSION-CONDITIONED PROPAGATION
        # ============================================================

        for block in self.session_blocks:
            session_h, session_nodes = block(
                node_h,
                edge_h,
                session_h,
                session_nodes,
                edge_index,
            )

        # ============================================================
        # SESSION-SPECIFIC EDGE REPRESENTATIONS
        # ============================================================

        src = edge_index[0]
        dst = edge_index[1]

        session_src = session_nodes[:, src, :]
        session_dst = session_nodes[:, dst, :]

        edge_context = edge_h.unsqueeze(0).expand(
            num_sessions,
            -1,
            -1,
        )

        session_edge_embeddings = (
            session_src
            + session_dst
            + edge_context
        )

        # ============================================================
        # ATTENTION ACROSS SESSIONS
        # ============================================================

        attention_input = torch.cat(
            [
                session_edge_embeddings,
                session_h.unsqueeze(1).expand_as(
                    session_edge_embeddings
                ),
                edge_context,
            ],
            dim=-1,
        )

        attention_logits = (
            self.session_edge_attention(
                attention_input
            )
            .squeeze(-1)
        )

        attention = torch.softmax(
            attention_logits,
            dim=0,
        )

        # Joint edge representation.
        joint_edge_h = (
            attention.unsqueeze(-1)
            * session_edge_embeddings
        ).sum(dim=0)

        # ============================================================
        # GLOBAL REPRESENTATION
        # ============================================================

        global_node = node_h.mean(dim=0)
        global_edge = joint_edge_h.mean(dim=0)
        global_session = session_h.mean(dim=0)
        global_session_edge = (
            session_edge_embeddings.mean(dim=(0, 1))
        )

        candidate_features = self._candidate_features(
            problem,
            state,
            device,
        )

        global_input = torch.cat(
            [
                global_node,
                global_edge,
                global_session,
                global_session_edge,
                candidate_features,
            ],
            dim=-1,
        )

        global_h = self.global_projection(
            global_input
        )

        return SCBEncoderOutput(
            node_embeddings=node_h,
            edge_embeddings=joint_edge_h,
            session_embeddings=session_h,
            global_embedding=global_h,
            session_edge_embeddings=session_edge_embeddings,
        )

    # =================================================================
    # FEATURE CONSTRUCTION
    # =================================================================

    def _build_graph_features(
        self,
        problem: SCBProblem,
        state: SCBState,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:

        num_nodes = problem.num_nodes
        num_edges = problem.num_edges

        # ------------------------------------------------------------
        # Node degrees
        # ------------------------------------------------------------

        degrees = torch.zeros(
            num_nodes,
            dtype=torch.float32,
            device=device,
        )

        edge_pairs = []

        for edge in problem.edges:

            u = problem.node_to_idx[edge[0]]
            v = problem.node_to_idx[edge[1]]

            edge_pairs.append((u, v))

            degrees[u] += 1.0
            degrees[v] += 1.0

        max_degree = degrees.max().clamp_min(1.0)

        degree_norm = degrees / max_degree

        # ------------------------------------------------------------
        # Session participation
        # ------------------------------------------------------------

        selected = state.candidate.selected_sessions

        num_sessions = max(
            problem.num_sessions,
            1,
        )

        source_counts = torch.zeros(
            num_nodes,
            dtype=torch.float32,
            device=device,
        )

        sink_counts = torch.zeros(
            num_nodes,
            dtype=torch.float32,
            device=device,
        )

        touch_counts = torch.zeros(
            num_nodes,
            dtype=torch.float32,
            device=device,
        )

        for session in selected:

            source, target = session

            source_idx = problem.node_to_idx[source]
            target_idx = problem.node_to_idx[target]

            source_counts[source_idx] += 1.0
            sink_counts[target_idx] += 1.0

            touch_counts[source_idx] += 1.0
            touch_counts[target_idx] += 1.0

        node_features = torch.stack(
            [
                degree_norm,
                source_counts / num_sessions,
                sink_counts / num_sessions,
                touch_counts / (2.0 * num_sessions),
            ],
            dim=-1,
        )

        # ------------------------------------------------------------
        # Edge features
        # ------------------------------------------------------------

        cut_edges = state.candidate.cut_edges

        edge_features = []

        for u, v in edge_pairs:

            edge = problem.idx_to_edge[u]

            reverse = (
                edge[1],
                edge[0],
            )

            is_cut = float(
                edge in cut_edges
                or reverse in cut_edges
            )

            edge_features.append(
                [
                    is_cut,
                    float(degree_norm[u]),
                    float(degree_norm[v]),
                ]
            )

        if num_edges:

            edge_features_tensor = torch.tensor(
                edge_features,
                dtype=torch.float32,
                device=device,
            )

            edge_index = torch.tensor(
                edge_pairs,
                dtype=torch.long,
                device=device,
            ).t().contiguous()

        else:

            edge_features_tensor = torch.empty(
                (
                    0,
                    self.EDGE_INPUT_DIM,
                ),
                dtype=torch.float32,
                device=device,
            )

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=device,
            )

        return (
            node_features,
            edge_features_tensor,
            edge_index,
        )

    def _build_session_features(
        self,
        problem: SCBProblem,
        state: SCBState,
        node_h: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:

        evaluation = evaluate_candidate(
            problem,
            selected_sessions=set(
                state.candidate.selected_sessions
            ),
            cut_edges=set(
                state.candidate.cut_edges
            ),
        )

        separated = set(
            evaluation.separated_sessions
        )

        source_idx = torch.tensor(
            [
                problem.node_to_idx[source]
                for source, _ in problem.sessions
            ],
            dtype=torch.long,
            device=device,
        )

        sink_idx = torch.tensor(
            [
                problem.node_to_idx[target]
                for _, target in problem.sessions
            ],
            dtype=torch.long,
            device=device,
        )

        selected = state.candidate.selected_sessions

        session_flags = []

        for session in problem.sessions:

            session_flags.append(
                [
                    float(session in selected),
                    float(session in separated),
                ]
            )

        session_flags = torch.tensor(
            session_flags,
            dtype=torch.float32,
            device=device,
        )

        session_features = torch.cat(
            [
                node_h[source_idx],
                node_h[sink_idx],
                session_flags,
            ],
            dim=-1,
        )

        return (
            session_features,
            source_idx,
            sink_idx,
        )

    @staticmethod
    def _node_session_mask(
        num_nodes: int,
        indices: Tensor,
        device: torch.device,
    ) -> Tensor:

        mask = torch.zeros(
            indices.numel(),
            num_nodes,
            1,
            dtype=torch.float32,
            device=device,
        )

        mask.scatter_(
            1,
            indices.view(-1, 1, 1).expand(
                -1,
                -1,
                1,
            ),
            1.0,
        )

        return mask

    def _candidate_features(
        self,
        problem: SCBProblem,
        state: SCBState,
        device: torch.device,
    ) -> Tensor:

        candidate = state.candidate

        num_sessions = max(
            problem.num_sessions,
            1,
        )

        num_edges = max(
            problem.num_edges,
            1,
        )

        scb = (
            state.scb
            if state.scb is not None
            else 0.0
        )

        # Scale SCB so that it is not dramatically larger than
        # the other candidate-level features.
        scb_scale = float(
            max(problem.num_edges, 1)
        )

        return torch.tensor(
            [
                candidate.num_selected_sessions
                / num_sessions,

                candidate.num_cut_edges
                / num_edges,

                state.separated_sessions
                / num_sessions,

                float(state.valid),

                float(scb) / scb_scale,
            ],
            dtype=torch.float32,
            device=device,
        )