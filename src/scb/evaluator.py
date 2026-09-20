"""Mathematical evaluation of a joint candidate (I', E')."""

from dataclasses import dataclass

from .connectivity import connected_components, remaining_adjacency
from .problem import SCBProblem
from .scb import calculate_scb


@dataclass(frozen=True)
class CandidateEvaluation:
    cut_size: int
    selected_session_count: int
    separated_sessions: tuple[tuple, ...]
    valid: bool
    components: tuple[int, ...]
    scb: float | None


def evaluate_candidate(
    problem: SCBProblem,
    selected_sessions: set[tuple],
    cut_edges: set[tuple],
) -> CandidateEvaluation:
    """Evaluate whether E' separates every session in I' and compute SCB if valid."""
    selected_sessions = set(selected_sessions)
    cut_edges = set(cut_edges)

    unknown_edges = cut_edges.difference(problem.edge_to_idx)
    if unknown_edges:
        raise ValueError(f"Candidate contains edges not in graph: {unknown_edges}")

    known_sessions = set(problem.sessions)
    unknown_sessions = selected_sessions.difference(known_sessions)
    if unknown_sessions:
        raise ValueError(f"Candidate contains sessions not in graph: {unknown_sessions}")

    adjacency = remaining_adjacency(problem, cut_edges)
    components = connected_components(adjacency)

    separated = []
    for session in selected_sessions:
        source, target = session
        if components[problem.node_to_idx[source]] != components[problem.node_to_idx[target]]:
            separated.append(session)

    valid = bool(selected_sessions) and len(separated) == len(selected_sessions)
    scb = calculate_scb(len(cut_edges), len(selected_sessions)) if valid else None

    return CandidateEvaluation(
        cut_size=len(cut_edges),
        selected_session_count=len(selected_sessions),
        separated_sessions=tuple(sorted(separated)),
        valid=valid,
        components=tuple(components),
        scb=scb,
    )
