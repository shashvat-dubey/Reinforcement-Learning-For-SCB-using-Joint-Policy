import torch

from src.data.graph_schema import GraphInstance
from src.environment.candidate import Candidate
from src.environment.state import SCBState
from src.models.scb_encoder import SCBEncoder
from src.scb.evaluator import evaluate_candidate
from src.scb.problem import SCBProblem


def make_problem():
    graph = GraphInstance(
        graph_id=1,
        nodes=(0, 1, 2, 3, 4),
        edges=(
            (0, 1),
            (1, 2),
            (2, 4),
            (1, 3),
            (3, 4),
        ),
        sessions=(
            (0, 4),
            (1, 4),
            (0, 3),
        ),
    )

    return SCBProblem.from_graph(graph)


def make_state(problem):
    selected = {
        problem.sessions[0],
        problem.sessions[1],
    }

    cut = {
        problem.edges[1],
    }

    result = evaluate_candidate(
        problem,
        selected,
        cut,
    )

    return SCBState(
        candidate=Candidate(
            selected_sessions=selected,
            cut_edges=cut,
        ),
        valid=result.valid,
        scb=result.scb,
        cut_size=result.cut_size,
        selected_session_count=result.selected_session_count,
        separated_sessions=len(
            result.separated_sessions
        ),
    )


def test_encoder_shapes():

    problem = make_problem()
    state = make_state(problem)

    model = SCBEncoder(
        hidden_dim=32,
    )

    output = model(
        problem,
        state,
    )

    assert output.node_embeddings.shape == (
        problem.num_nodes,
        32,
    )

    assert output.edge_embeddings.shape == (
        problem.num_edges,
        32,
    )

    assert output.session_embeddings.shape == (
        problem.num_sessions,
        32,
    )

    assert output.session_edge_embeddings.shape == (
        problem.num_sessions,
        problem.num_edges,
        32,
    )

    assert output.global_embedding.shape == (
        32,
    )


def test_encoder_outputs_are_finite():

    problem = make_problem()
    state = make_state(problem)

    model = SCBEncoder(
        hidden_dim=32,
    )

    output = model(
        problem,
        state,
    )

    tensors = (
        output.node_embeddings,
        output.edge_embeddings,
        output.session_embeddings,
        output.session_edge_embeddings,
        output.global_embedding,
    )

    for tensor in tensors:
        assert torch.isfinite(tensor).all()


def test_encoder_backpropagates():

    problem = make_problem()
    state = make_state(problem)

    model = SCBEncoder(
        hidden_dim=32,
    )

    output = model(
        problem,
        state,
    )

    loss = output.global_embedding.square().mean()

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients

    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_cut_status_changes_representation():

    problem = make_problem()

    state_a = make_state(problem)

    state_b = make_state(problem)

    state_b.candidate.cut_edges = set()

    result = evaluate_candidate(
        problem,
        state_b.candidate.selected_sessions,
        set(),
    )

    state_b.valid = result.valid
    state_b.scb = result.scb
    state_b.cut_size = result.cut_size
    state_b.separated_sessions = len(
        result.separated_sessions
    )

    model = SCBEncoder(
        hidden_dim=32,
    )

    output_a = model(
        problem,
        state_a,
    ).global_embedding

    output_b = model(
        problem,
        state_b,
    ).global_embedding

    assert not torch.allclose(
        output_a,
        output_b,
    )