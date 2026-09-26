import torch

from src.controller.controller import SCBController
from src.controller.episode_manager import SCBEpisodeManager
from src.controller.policy_orchestrator import PolicyOrchestrator

from src.data.graph_loader import GraphDataset

from src.environment.environment import SCBEnvironment

from src.models.policies import (
    ValidityPolicy,
    MinimizationPolicy,
    GlobalExplorationPolicy,
)

from src.models.scb_encoder import SCBEncoder

from src.scb.problem import SCBProblem


DATASET_PATH = "data/raw/labelled_dataset.pkl"

class EpisodeTestEnvironment(SCBEnvironment):

    @staticmethod
    def _normalize_state(state):
        state.separated_sessions = len(
            state.separated_sessions
        )
        return state

    def reset(self, selected_sessions=None):
        state = super().reset(selected_sessions)

        state = self._normalize_state(state)

        self.state = state

        return state.copy()

    def step(self, action):
        state, reward, done, info = super().step(action)

        state = self._normalize_state(state)

        self.state = state

        return state, reward, done, info

def build_episode_manager():
    # --------------------------------------------------------------
    # Load one real graph from the existing dataset
    # --------------------------------------------------------------

    dataset = GraphDataset(DATASET_PATH)

    graph_index = 0

    graph = dataset.graph(graph_index)

    print()
    print("=" * 60)
    print("BUILDING SCB RL EPISODE")
    print("=" * 60)
    print(f"Graph index : {graph_index}")
    print(f"Nodes       : {graph.num_nodes}")
    print(f"Edges       : {graph.num_edges}")
    print(f"Sessions    : {graph.num_sessions}")
    print("=" * 60)

    problem = SCBProblem.from_graph(graph)

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    environment = EpisodeTestEnvironment(
    problem=problem,
    mode=SCBEnvironment.VALIDITY_MODE,
    )

    # --------------------------------------------------------------
    # Controller
    # --------------------------------------------------------------

    controller = SCBController()

    # --------------------------------------------------------------
    # Encoder
    # --------------------------------------------------------------

    encoder = SCBEncoder(
        hidden_dim=32,
        topology_layers=2,
        session_layers=1,
    )

    # --------------------------------------------------------------
    # Policies
    # --------------------------------------------------------------

    validity_policy = ValidityPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    minimization_policy = MinimizationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    global_exploration_policy = GlobalExplorationPolicy(
        hidden_dim=32,
        action_embedding_dim=16,
        scorer_hidden_dim=32,
    )

    # --------------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------------

    orchestrator = PolicyOrchestrator(
        validity_policy=validity_policy,
        minimization_policy=minimization_policy,
        global_exploration_policy=global_exploration_policy,
    )

    # --------------------------------------------------------------
    # Episode Manager
    # --------------------------------------------------------------

    episode_manager = SCBEpisodeManager(
        environment=environment,
        controller=controller,
        encoder=encoder,
        orchestrator=orchestrator,
        max_steps=50,
    )

    return episode_manager


def test_episode_manager_runs_real_episode():
    """
    End-to-end smoke test.

    This intentionally runs one real episode through:

        Dataset
        → SCBProblem
        → Environment
        → Controller
        → Encoder
        → Policy
        → Orchestrator
        → Environment
        → Controller
    """

    torch.manual_seed(42)

    episode_manager = build_episode_manager()

    result = episode_manager.run(
        verbose=True,
        deterministic=True,
    )

    # --------------------------------------------------------------
    # Basic episode checks
    # --------------------------------------------------------------

    assert result is not None

    assert result.initial_state is not None
    assert result.final_state is not None

    assert len(result.steps) > 0

    assert len(result.steps) <= 50

    assert result.final_phase is not None

    # --------------------------------------------------------------
    # Every recorded step should contain valid information
    # --------------------------------------------------------------

    for step in result.steps:
        assert step.step_number >= 1

        assert step.phase is not None

        assert step.policy_name in {
            "ValidityPolicy",
            "MinimizationPolicy",
            "GlobalExplorationPolicy",
        }

        assert step.action is not None

        assert isinstance(step.reward, float)

        assert step.state is not None

    # --------------------------------------------------------------
    # The environment must have actually advanced
    # --------------------------------------------------------------

    assert (
        result.final_state.candidate
        is not None
    )


def test_episode_manager_quiet_mode():
    """
    Verify that verbose=False still executes the full episode
    without producing debug output as part of the API contract.
    """

    torch.manual_seed(42)

    episode_manager = build_episode_manager()

    result = episode_manager.run(
        verbose=False,
        deterministic=True,
    )

    assert result is not None
    assert len(result.steps) > 0