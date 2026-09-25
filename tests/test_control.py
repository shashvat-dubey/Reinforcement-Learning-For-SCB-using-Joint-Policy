from pathlib import Path

import pytest

from src.environment.candidate import Candidate
from src.data.graph_loader import GraphDataset
from src.scb.problem import SCBProblem
from src.environment.environment import SCBEnvironment
from src.environment.state import SCBState
from src.controller.controller import (
    ControllerPhase,
    SCBController,
)


DATASET_PATH = Path("data/raw/labelled_dataset.pkl")


@pytest.fixture
def environment():
    dataset = GraphDataset(DATASET_PATH)
    graph = dataset.graph(0)
    problem = SCBProblem.from_graph(graph)

    return SCBEnvironment(problem)


def make_valid_state(scb: float) -> SCBState:
    """
    Minimal synthetic valid state for controller testing.
    """

    candidate = Candidate(
        selected_sessions=set(),
        cut_edges=set(),
    )

    return SCBState(
        candidate=candidate,
        valid=True,
        scb=scb,
        cut_size=1,
        selected_session_count=2,
        separated_sessions=2,
    )


def make_invalid_state() -> SCBState:
    """
    Minimal synthetic invalid state for controller testing.
    """

    candidate = Candidate(
        selected_sessions=set(),
        cut_edges=set(),
    )

    return SCBState(
        candidate=candidate,
        valid=False,
        scb=None,
        cut_size=0,
        selected_session_count=2,
        separated_sessions=0,
    )


def test_controller_starts_in_validity(environment):
    state = environment.reset()

    controller = SCBController()
    controller.start(state)

    assert controller.current_policy() == ControllerPhase.VALIDITY


def test_controller_moves_to_minimization_when_valid():
    controller = SCBController()

    initial = make_invalid_state()
    valid = make_valid_state(0.50)

    controller.start(initial)

    assert controller.phase == ControllerPhase.VALIDITY

    controller.update(valid)

    assert controller.phase == ControllerPhase.MINIMIZATION
    assert controller.best_scb == 0.50
    assert controller.global_best_scb == 0.50


def test_minimization_improvement_resets_stall():
    controller = SCBController()

    initial = make_valid_state(0.50)

    controller.start(initial)

    # No improvement.
    controller.update(make_valid_state(0.50))

    assert controller.stall_count == 1

    # Improvement.
    controller.update(make_valid_state(0.40))

    assert controller.stall_count == 0
    assert controller.best_scb == 0.40
    assert controller.phase == ControllerPhase.MINIMIZATION


def test_minimization_reaches_local_retry():
    controller = SCBController()

    initial = make_valid_state(0.50)
    controller.start(initial)

    for _ in range(controller.N_STALL):
        controller.update(make_valid_state(0.50))

    assert controller.phase == ControllerPhase.LOCAL_RETRY
    assert controller.stall_count == 0
    assert controller.local_retry_start_scb == 0.50


def test_promising_local_retry_returns_to_minimization():
    controller = SCBController()

    initial = make_valid_state(0.50)
    controller.start(initial)

    for _ in range(controller.N_STALL):
        controller.update(make_valid_state(0.50))

    assert controller.phase == ControllerPhase.LOCAL_RETRY

    controller.update(make_valid_state(0.35))

    assert controller.phase == ControllerPhase.MINIMIZATION
    assert controller.best_scb == 0.35
    assert controller.local_retry_count == 0


def test_unsuccessful_local_retries_trigger_global_exploration():
    controller = SCBController()

    initial = make_valid_state(0.50)
    controller.start(initial)

    for _ in range(controller.N_STALL):
        controller.update(make_valid_state(0.50))

    assert controller.phase == ControllerPhase.LOCAL_RETRY

    # Retry 1: no improvement.
    controller.update(make_valid_state(0.50))

    assert controller.phase == ControllerPhase.LOCAL_RETRY
    assert controller.local_retry_count == 1

    # Retry 2: no improvement.
    controller.update(make_valid_state(0.50))

    assert controller.phase == ControllerPhase.GLOBAL_EXPLORATION
    assert controller.local_retry_count == 0


def test_global_improvement_returns_to_minimization():
    controller = SCBController()

    initial = make_valid_state(0.50)
    controller.start(initial)

    controller.phase = ControllerPhase.GLOBAL_EXPLORATION

    controller.update(make_valid_state(0.30))

    assert controller.phase == ControllerPhase.MINIMIZATION
    assert controller.global_best_scb == 0.30
    assert controller.best_scb == 0.30


def test_global_exploration_invalid_returns_to_validity():
    controller = SCBController()

    initial = make_valid_state(0.50)
    controller.start(initial)

    controller.phase = ControllerPhase.GLOBAL_EXPLORATION

    controller.update(make_invalid_state())

    assert controller.phase == ControllerPhase.VALIDITY


def test_minimization_invalid_returns_to_validity():
    controller = SCBController()

    initial = make_valid_state(0.50)
    controller.start(initial)

    controller.update(make_invalid_state())

    assert controller.phase == ControllerPhase.VALIDITY
    assert controller.stall_count == 0