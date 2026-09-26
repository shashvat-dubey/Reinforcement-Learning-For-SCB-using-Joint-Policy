import torch
import pytest

from src.controller.controller import ControllerPhase
from src.controller.policy_orchestrator import PolicyOrchestrator
from src.environment.candidate import Candidate
from src.environment.state import SCBState
from src.models.policies import (
    ValidityPolicy,
    MinimizationPolicy,
    GlobalExplorationPolicy,
)
from src.models.scb_encoder import SCBEncoder
from src.scb.evaluator import evaluate_candidate
from src.scb.problem import SCBProblem
from src.data.graph_schema import GraphInstance
from src.data.graph_loader import GraphDataset


DATASET_PATH = "data/raw/labelled_dataset.pkl"


def make_problem_and_state():
    dataset = GraphDataset(DATASET_PATH)

    graph = GraphInstance.from_record(dataset[0])
    problem = SCBProblem.from_graph(graph)

    candidate = Candidate(
        selected_sessions=set(problem.sessions),
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
        separated_sessions=len(result.separated_sessions),
    )

    return problem, state


def make_orchestrator():
    return PolicyOrchestrator(
        validity_policy=ValidityPolicy(
            hidden_dim=32,
            action_embedding_dim=16,
            scorer_hidden_dim=32,
        ),
        minimization_policy=MinimizationPolicy(
            hidden_dim=32,
            action_embedding_dim=16,
            scorer_hidden_dim=32,
        ),
        global_exploration_policy=GlobalExplorationPolicy(
            hidden_dim=32,
            action_embedding_dim=16,
            scorer_hidden_dim=32,
        ),
    )


@pytest.fixture
def problem_state_encoder():
    problem, state = make_problem_and_state()

    encoder = SCBEncoder(
        hidden_dim=32,
        topology_layers=2,
        session_layers=1,
    )

    encoder_output = encoder(problem, state)

    return problem, state, encoder_output


def test_phase_policy_mapping():
    orchestrator = make_orchestrator()

    assert (
        orchestrator.get_policy(ControllerPhase.VALIDITY)
        is orchestrator.validity_policy
    )

    assert (
        orchestrator.get_policy(ControllerPhase.MINIMIZATION)
        is orchestrator.minimization_policy
    )

    assert (
        orchestrator.get_policy(ControllerPhase.LOCAL_RETRY)
        is orchestrator.minimization_policy
    )

    assert (
        orchestrator.get_policy(ControllerPhase.GLOBAL_EXPLORATION)
        is orchestrator.global_exploration_policy
    )


def test_done_has_no_policy():
    orchestrator = make_orchestrator()

    with pytest.raises(RuntimeError):
        orchestrator.get_policy(ControllerPhase.DONE)


def test_validity_phase_routes_to_validity_policy(
    problem_state_encoder,
):
    problem, state, encoder_output = problem_state_encoder

    orchestrator = make_orchestrator()

    output = orchestrator.act(
        ControllerPhase.VALIDITY,
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    assert output.phase == ControllerPhase.VALIDITY
    assert output.policy is orchestrator.validity_policy
    assert output.action.action_type.name in {
        "ADD_EDGE",
        "REMOVE_EDGE",
    }


def test_minimization_phase_routes_correctly(
    problem_state_encoder,
):
    problem, state, encoder_output = problem_state_encoder

    orchestrator = make_orchestrator()

    output = orchestrator.act(
        ControllerPhase.MINIMIZATION,
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    assert output.phase == ControllerPhase.MINIMIZATION
    assert output.policy is orchestrator.minimization_policy


def test_local_retry_uses_minimization_policy(
    problem_state_encoder,
):
    problem, state, encoder_output = problem_state_encoder

    orchestrator = make_orchestrator()

    output = orchestrator.act(
        ControllerPhase.LOCAL_RETRY,
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    assert output.phase == ControllerPhase.LOCAL_RETRY
    assert output.policy is orchestrator.minimization_policy


def test_global_exploration_routes_correctly(
    problem_state_encoder,
):
    problem, state, encoder_output = problem_state_encoder

    orchestrator = make_orchestrator()

    output = orchestrator.act(
        ControllerPhase.GLOBAL_EXPLORATION,
        encoder_output,
        problem,
        state,
        deterministic=True,
    )

    assert output.phase == ControllerPhase.GLOBAL_EXPLORATION
    assert output.policy is orchestrator.global_exploration_policy


def test_orchestrator_preserves_policy_gradients(
    problem_state_encoder,
):
    problem, state, encoder_output = problem_state_encoder

    orchestrator = make_orchestrator()

    output = orchestrator.act(
        ControllerPhase.MINIMIZATION,
        encoder_output,
        problem,
        state,
        deterministic=False,
    )

    loss = -output.log_prob + 0.01 * output.entropy
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in orchestrator.minimization_policy.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        for gradient in gradients
    )