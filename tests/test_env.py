import pytest
from pathlib import Path

from src.data.graph_loader import GraphDataset
from src.scb.problem import SCBProblem

from src.environment.actions import Action, ActionType
from src.environment.environment import SCBEnvironment


DATASET_PATH = Path("data/raw/labelled_dataset.pkl")


@pytest.fixture
def environment():
    dataset = GraphDataset(DATASET_PATH)

    graph = dataset.graph(0)

    problem = SCBProblem.from_graph(graph)

    return SCBEnvironment(problem)


def test_reset(environment):
    state = environment.reset()

    assert state.candidate is not None
    assert len(state.candidate.selected_sessions) > 0
    assert len(state.candidate.cut_edges) == 0


def test_add_edge_changes_candidate(environment):
    environment.reset()

    edge = next(iter(environment.problem.edges))

    action = Action(
        action_type=ActionType.ADD_EDGE,
        target=edge,
    )

    next_state, reward, done, info = environment.step(action)

    assert edge in next_state.candidate.cut_edges
    assert done is False


def test_add_session_changes_candidate(environment):
    state = environment.reset()

    all_sessions = set(environment.problem.sessions)

    available = all_sessions - state.candidate.selected_sessions

    if not available:
        pytest.skip("All sessions are already selected.")

    session = next(iter(available))

    action = Action(
        action_type=ActionType.ADD_SESSION,
        target=session,
    )

    next_state, reward, done, info = environment.step(action)

    assert session in next_state.candidate.selected_sessions

def test_stop_ends_episode(environment):
    environment.reset()

    action = Action(
        action_type=ActionType.STOP,
        target=None,
    )

    state, reward, done, info = environment.step(action)

    assert done is True
    assert reward == 0.0


def test_validity_reward(environment):
    environment.reset()

    environment.set_mode(SCBEnvironment.VALIDITY_MODE)

    assert environment.state.valid is False

    edge = next(iter(environment.problem.edges))

    action = Action(
        action_type=ActionType.ADD_EDGE,
        target=edge,
    )

    _, reward, _, _ = environment.step(action)

    assert reward in (0.0, 1.0)