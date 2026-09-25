"""Candidate representation for the joint SCB search space."""

from dataclasses import dataclass, field
from typing import Any, Iterable

Edge = tuple[Any, Any]
Session = tuple[Any, Any]


@dataclass
class Candidate:
    """A candidate solution C = (I', E').

    The candidate stores only the selected sessions and cut edges. Mathematical
    validity and SCB are evaluated by the SCB layer, not here.
    """

    selected_sessions: set[Session] = field(default_factory=set)
    cut_edges: set[Edge] = field(default_factory=set)

    @classmethod
    def from_sets(
        cls,
        selected_sessions: Iterable[Session] = (),
        cut_edges: Iterable[Edge] = (),
    ) -> "Candidate":
        return cls(set(selected_sessions), set(cut_edges))

    def copy(self) -> "Candidate":
        return Candidate(
            selected_sessions=set(self.selected_sessions),
            cut_edges=set(self.cut_edges),
        )

    def add_session(self, session: Session) -> None:
        self.selected_sessions.add(session)

    def remove_session(self, session: Session) -> None:
        self.selected_sessions.discard(session)

    def add_edge(self, edge: Edge) -> None:
        self.cut_edges.add(edge)

    def remove_edge(self, edge: Edge) -> None:
        self.cut_edges.discard(edge)

    @property
    def num_selected_sessions(self) -> int:
        return len(self.selected_sessions)

    @property
    def num_cut_edges(self) -> int:
        return len(self.cut_edges)

    @property
    def is_empty(self) -> bool:
        return not self.selected_sessions
