"""Canonical mathematical representation of one SCB graph."""

from dataclasses import dataclass, field
from typing import Any

from src.data.graph_schema import GraphInstance


@dataclass
class SCBProblem:
    nodes: list[Any]
    edges: list[tuple[Any, Any]]
    sessions: list[tuple[Any, Any]]

    node_to_idx: dict[Any, int] = field(init=False)
    idx_to_node: dict[int, Any] = field(init=False)
    edge_to_idx: dict[tuple[Any, Any], int] = field(init=False)
    idx_to_edge: dict[int, tuple[Any, Any]] = field(init=False)
    sessions_idx: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edges)}
        self.idx_to_edge = {i: edge for edge, i in self.edge_to_idx.items()}

        for edge in self.edges:
            if edge[0] not in self.node_to_idx or edge[1] not in self.node_to_idx:
                raise ValueError(f"Edge {edge} contains an unknown node")

        self.sessions_idx = []
        for source, target in self.sessions:
            if source not in self.node_to_idx or target not in self.node_to_idx:
                raise ValueError(f"Session {(source, target)} contains an unknown node")
            self.sessions_idx.append((self.node_to_idx[source], self.node_to_idx[target]))

    @classmethod
    def from_graph(cls, graph: GraphInstance) -> "SCBProblem":
        return cls(list(graph.nodes), list(graph.edges), list(graph.sessions))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)
