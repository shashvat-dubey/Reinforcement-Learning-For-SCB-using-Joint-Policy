"""Canonical data structures for the SCB dataset."""

from dataclasses import dataclass
from typing import Any

Edge = tuple[Any, Any]
Session = tuple[Any, Any]


@dataclass(frozen=True)
class GraphInstance:
    """The graph information visible to the new RL system."""

    graph_id: int
    nodes: tuple[Any, ...]
    edges: tuple[Edge, ...]
    sessions: tuple[Session, ...]

    @classmethod
    def from_record(cls, record: dict) -> "GraphInstance":
        return cls(
            graph_id=record["graph_id"],
            nodes=tuple(record["nodes"]),
            edges=tuple(tuple(edge) for edge in record["edges"]),
            sessions=tuple(tuple(session) for session in record["sessions"]),
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)


@dataclass(frozen=True)
class GABenchmark:
    """GA result stored with a graph, used only for later benchmarking."""

    graph_id: int
    objective: str
    fitness1: float
    fitness2: float
    cut: int
    sep: int
    cut_edges: tuple[Edge, ...]
    components: tuple[int, ...]
    sessions_sep: tuple[Session, ...]
    scb: float

    @classmethod
    def from_record(cls, record: dict) -> "GABenchmark":
        return cls(
            graph_id=record["graph_id"],
            objective=record["ga_objective"],
            fitness1=record["ga_fitness1"],
            fitness2=record["ga_fitness2"],
            cut=record["ga_cut"],
            sep=record["ga_sep"],
            cut_edges=tuple(tuple(edge) for edge in record["ga_cut_edges"]),
            components=tuple(record["ga_components"]),
            sessions_sep=tuple(tuple(s) for s in record["ga_sessions_sep"]),
            scb=record["ga_scb"],
        )
