import pickle
from pathlib import Path

import torch

from src.data.graph_schema import GraphInstance

from src.scb.problem import SCBProblem
from src.scb.evaluator import evaluate_candidate

from src.environment.candidate import Candidate
from src.environment.state import SCBState
from src.environment.actions import ActionType

from src.models.scb_encoder import SCBEncoder
from src.models.policies import (
    ValidityPolicy,
    MinimizationPolicy,
    GlobalExplorationPolicy,
)


DATASET_PATH = Path(
    "data/raw/labelled_dataset.pkl"
)


def make_problem_and_state():
    with DATASET_PATH.open("rb") as f:
        records = pickle.load(f)

    graph = GraphInstance.from_record(
        records[0]
    )

    problem = SCBProblem.from_graph(
        graph
    )

    candidate = Candidate(
        selected_sessions=set(
            problem.sessions
        ),
        cut_edges=set(),
    )

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
        separated_sessions=len(
            result.separated_sessions
        ),
    )

    return problem, state


def make_encoder_output(problem, state):
    encoder = SCBEncoder(
        hidden_dim=32,
        topology_layers=2,
        session_layers=1,
    )

    return encoder(
        problem,
        state,
    )


def test_validity_policy_action_space():
    problem, state = make_problem_and_state()

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = ValidityPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    expected_size = (
        2 * problem.num_edges
        + 2 * problem.num_sessions
        + 1
    )

    assert output.logits.shape == (
        expected_size,
    )

    assert output.action.action_type in (
        ActionType.ADD_EDGE,
        ActionType.REMOVE_EDGE,
    )


def test_minimization_policy_can_select_full_action_space():
    problem, state = make_problem_and_state()

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = MinimizationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    expected_size = (
        2 * problem.num_edges
        + 2 * problem.num_sessions
        + 1
    )

    assert output.logits.shape == (
        expected_size,
    )


def test_global_policy_can_select_full_action_space():
    problem, state = make_problem_and_state()

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = GlobalExplorationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    expected_size = (
        2 * problem.num_edges
        + 2 * problem.num_sessions
        + 1
    )

    assert output.logits.shape == (
        expected_size,
    )


def test_validity_policy_masks_non_edge_actions():
    problem, state = make_problem_and_state()

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = ValidityPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    session_start = (
        2 * problem.num_edges
    )

    session_end = (
        2 * problem.num_edges
        + 2 * problem.num_sessions
        + 1
    )

    assert torch.isneginf(
        output.logits[
            session_start:session_end
        ]
    ).all()


def test_add_remove_edge_masking():
    problem, state = make_problem_and_state()

    # Start with one cut edge.
    edge = problem.edges[0]

    state.candidate.cut_edges.add(edge)

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = ValidityPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    edge_index = problem.edge_to_idx[edge]

    # ADD_EDGE must be masked because it is already cut.
    assert torch.isneginf(
        output.logits[edge_index]
    )

    # REMOVE_EDGE must remain executable.
    remove_index = (
        problem.num_edges
        + edge_index
    )

    assert not torch.isneginf(
        output.logits[remove_index]
    )


def test_selected_session_masks_add():
    problem, state = make_problem_and_state()

    selected_session = problem.sessions[0]

    state.candidate.selected_sessions.add(
        selected_session
    )

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = MinimizationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    session_index = problem.sessions.index(
        selected_session
    )

    add_index = (
        2 * problem.num_edges
        + session_index
    )

    assert torch.isneginf(
        output.logits[add_index]
    )


def test_unselected_session_masks_remove():
    problem, state = make_problem_and_state()

    unselected_session = problem.sessions[0]

    state.candidate.selected_sessions.discard(
        unselected_session
    )

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = MinimizationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    session_index = problem.sessions.index(
        unselected_session
    )

    remove_index = (
        2 * problem.num_edges
        + problem.num_sessions
        + session_index
    )

    assert torch.isneginf(
        output.logits[remove_index]
    )


def test_policy_returns_valid_action():
    problem, state = make_problem_and_state()

    encoder_output = make_encoder_output(
        problem,
        state,
    )

    policy = MinimizationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    output = policy(
        encoder_output,
        problem,
        state,
        deterministic=False,
    )

    assert output.action is not None
    assert torch.isfinite(
        output.log_prob
    )
    assert torch.isfinite(
        output.entropy
    )


def test_policy_backpropagates():
    problem, state = make_problem_and_state()

    encoder = SCBEncoder(
        hidden_dim=32,
        topology_layers=2,
        session_layers=1,
    )

    policy = MinimizationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    encoder_output = encoder(
        problem,
        state,
    )

    output = policy(
        encoder_output,
        problem,
        state,
    )

    loss = (
        -output.log_prob
        - 0.01 * output.entropy
    )

    loss.backward()

    policy_gradients = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient in policy_gradients
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in policy_gradients
        if gradient is not None
    )