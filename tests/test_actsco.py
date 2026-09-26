import pickle
from pathlib import Path

import torch

from src.models.action_scorer import ActionScorer
from src.models.scb_encoder import SCBEncoder

from src.environment.candidate import Candidate
from src.environment.state import SCBState

from src.scb.problem import SCBProblem
from src.scb.evaluator import evaluate_candidate
from src.data.graph_schema import GraphInstance

DATASET_PATH = Path("data/raw/labelled_dataset.pkl")


def _make_problem_and_candidate():
    with DATASET_PATH.open("rb") as f:
        records = pickle.load(f)

    record = records[0]

    graph = GraphInstance.from_record(record)
    problem = SCBProblem.from_graph(graph)

    candidate = Candidate(
        selected_sessions=set(problem.sessions),
        cut_edges=set(),
    )

    return problem, candidate

def _make_encoder_output():
    problem, candidate = _make_problem_and_candidate()

    # Evaluate the initial candidate exactly as the environment does.
    from src.scb.evaluator import evaluate_candidate

    result = evaluate_candidate(
        problem,
        selected_sessions=candidate.selected_sessions,
        cut_edges=candidate.cut_edges,
    )

    state = SCBState(
        candidate=candidate,
        valid=result.valid,
        scb=result.scb,
        cut_size=result.cut_size,
        selected_session_count=result.selected_session_count,
        separated_sessions=len(result.separated_sessions),
    )

    encoder = SCBEncoder(
        hidden_dim=32,
        topology_layers=2,
        session_layers=1,
    )

    encoder_output = encoder(
        problem,
        state,
    )

    return problem, candidate, encoder_output

def test_action_scorer_shapes():
    problem, candidate, encoder_output = _make_encoder_output()

    scorer = ActionScorer(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    logits = scorer(encoder_output)

    assert logits.add_edge.shape == (
        problem.num_edges,
    )

    assert logits.remove_edge.shape == (
        problem.num_edges,
    )

    assert logits.add_session.shape == (
        problem.num_sessions,
    )

    assert logits.remove_session.shape == (
        problem.num_sessions,
    )

    assert logits.stop.shape == (1,)


def test_action_scorer_outputs_are_finite():
    _, _, encoder_output = _make_encoder_output()

    scorer = ActionScorer(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    logits = scorer(encoder_output)

    for tensor in (
        logits.add_edge,
        logits.remove_edge,
        logits.add_session,
        logits.remove_session,
        logits.stop,
    ):
        assert torch.isfinite(tensor).all()


def test_action_scorer_backpropagates():
    _, _, encoder_output = _make_encoder_output()

    scorer = ActionScorer(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    logits = scorer(encoder_output)

    loss = (
        logits.add_edge.sum()
        + logits.remove_edge.sum()
        + logits.add_session.sum()
        + logits.remove_session.sum()
        + logits.stop.sum()
    )

    loss.backward()

    scorer_grads = [
        parameter.grad
        for parameter in scorer.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient in scorer_grads
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in scorer_grads
        if gradient is not None
    )


def test_action_type_changes_scores():
    _, _, encoder_output = _make_encoder_output()

    scorer = ActionScorer(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    logits = scorer(encoder_output)

    assert not torch.allclose(
        logits.add_edge,
        logits.remove_edge,
    )