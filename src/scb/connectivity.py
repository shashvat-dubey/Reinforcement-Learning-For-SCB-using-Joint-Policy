"""Graph connectivity operations used by SCB evaluation."""

from collections import deque

from .problem import SCBProblem


def remaining_adjacency(problem: SCBProblem, cut_edges: set[tuple]) -> list[list[int]]:
    """Build adjacency after removing the candidate cut edges."""
    cut_edges = set(cut_edges)
    adjacency = [[] for _ in range(problem.num_nodes)]

    for edge in problem.edges:
        if edge in cut_edges:
            continue
        u, v = problem.node_to_idx[edge[0]], problem.node_to_idx[edge[1]]
        adjacency[u].append(v)
        adjacency[v].append(u)

    return adjacency


def connected_components(adjacency: list[list[int]]) -> list[int]:
    """Return a component ID for every node using BFS."""
    component = [-1] * len(adjacency)
    component_id = 0

    for start in range(len(adjacency)):
        if component[start] != -1:
            continue

        queue = deque([start])
        component[start] = component_id

        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if component[neighbor] == -1:
                    component[neighbor] = component_id
                    queue.append(neighbor)

        component_id += 1

    return component
