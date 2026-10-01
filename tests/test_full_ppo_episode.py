"""
Full PPO episode integration test on a tiny deterministic SCB graph.

The complete console output is also saved to:

    tests/output/full_ppo_episode.txt
"""

import io
import os
import sys
import torch

from contextlib import redirect_stdout

from src.controller.controller import SCBController
from src.controller.episode_manager import SCBEpisodeManager
from src.controller.policy_orchestrator import PolicyOrchestrator

from src.environment.environment import SCBEnvironment
from src.environment.state import SCBState

from src.models.policies import (
    ValidityPolicy,
    MinimizationPolicy,
    GlobalExplorationPolicy,
)

from src.models.scb_encoder import SCBEncoder

from src.scb.problem import SCBProblem


# ======================================================================
# OUTPUT FILE
# ======================================================================

OUTPUT_DIR = "tests/output"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "full_ppo_episode.txt",
)


# ======================================================================
# DUMMY GRAPH
# ======================================================================

class DummyGraph:
    """
    Tiny deterministic graph.

        v0 ---- v1
         |      |
         |      |
        v2 ---- v3

    Sessions:

        (v0, v3)
        (v1, v2)
    """

    def __init__(self):

        self.num_nodes = 4

        self.edges = [
            ("v0", "v1"),
            ("v1", "v3"),
            ("v3", "v2"),
            ("v2", "v0"),
        ]

        self.num_edges = len(self.edges)

        self.sessions = [
            ("v0", "v3"),
            ("v1", "v2"),
        ]

        self.num_sessions = len(self.sessions)

        self.nodes = [
            "v0",
            "v1",
            "v2",
            "v3",
        ]


# ======================================================================
# TEST ENVIRONMENT
# ======================================================================

class DummyPPOEnvironment(SCBEnvironment):

    @staticmethod
    def _normalize_state(state: SCBState):

        if hasattr(state.separated_sessions, "__len__"):
            state.separated_sessions = len(
                state.separated_sessions
            )

        return state

    def reset(self, selected_sessions=None):

        state = super().reset(
            selected_sessions=selected_sessions
        )

        state = self._normalize_state(state)

        self.state = state

        return state.copy()

    def step(self, action):

        state, reward, done, info = super().step(
            action
        )

        state = self._normalize_state(state)

        self.state = state

        return state, reward, done, info


# ======================================================================
# BUILD COMPONENTS
# ======================================================================

def build_components():

    graph = DummyGraph()

    print()
    print("=" * 70)
    print("BUILDING TINY PPO TEST GRAPH")
    print("=" * 70)

    print(f"Nodes    : {graph.num_nodes}")
    print(f"Edges    : {graph.num_edges}")
    print(f"Sessions : {graph.num_sessions}")

    for edge in graph.edges:
        print(f"  edge    : {edge}")

    for session in graph.sessions:
        print(f"  session : {session}")

    print("=" * 70)

    problem = SCBProblem.from_graph(
        graph
    )

    environment = DummyPPOEnvironment(
        problem=problem,
        mode=SCBEnvironment.VALIDITY_MODE,
    )

    controller = SCBController()

    encoder = SCBEncoder(
        hidden_dim=32,
        topology_layers=2,
        session_layers=1,
    )

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

    orchestrator = PolicyOrchestrator(
        validity_policy=validity_policy,
        minimization_policy=minimization_policy,
        global_exploration_policy=global_exploration_policy,
    )

    episode_manager = SCBEpisodeManager(
        environment=environment,
        controller=controller,
        encoder=encoder,
        orchestrator=orchestrator,
        max_steps=100,
    )

    return (
        graph,
        problem,
        environment,
        controller,
        encoder,
        orchestrator,
        episode_manager,
    )


# ======================================================================
# FULL EPISODE TEST
# ======================================================================

def _run_test():

    torch.manual_seed(42)

    print()
    print("=" * 70)
    print("FULL PPO TINY-GRAPH INTEGRATION TEST")
    print("=" * 70)

    (
        graph,
        problem,
        environment,
        controller,
        encoder,
        orchestrator,
        episode_manager,
    ) = build_components()

    # ==================================================================
    # ROLLOUT
    # ==================================================================

    print()
    print("PHASE 1: ROLLOUT")
    print("-" * 70)

    result = episode_manager.run(
        verbose=True,
        deterministic=False,
    )

    assert result is not None

    assert result.initial_state is not None
    assert result.final_state is not None

    # ==================================================================
    # BASIC STATE VALIDATION
    # ==================================================================

    initial_state = result.initial_state
    final_state = result.final_state

    print()
    print("-" * 70)
    print("EPISODE SUMMARY")
    print("-" * 70)

    print(
        f"Initial valid : {initial_state.valid}"
    )

    print(
        f"Final valid   : {final_state.valid}"
    )

    print(
        f"Initial SCB   : {initial_state.scb}"
    )

    print(
        f"Final SCB     : {final_state.scb}"
    )

    print(
        f"Cut size      : {final_state.cut_size}"
    )

    print(
        f"Selected sessions : "
        f"{final_state.selected_session_count}"
    )

    print(
        f"Steps         : {len(result.transitions)}"
    )

    # ==================================================================
    # TRANSITION CHECKS
    # ==================================================================

    transitions = result.transitions

    assert len(transitions) > 0

    for transition in transitions:

        assert transition.state is not None
        assert transition.next_state is not None

        assert transition.action is not None

        assert isinstance(
            transition.action_index,
            int,
        )

        assert isinstance(
            transition.reward,
            float,
        )

        assert isinstance(
            transition.done,
            bool,
        )

        assert transition.log_prob is not None
        assert transition.entropy is not None

    # ==================================================================
    # PHASE ANALYSIS
    # ==================================================================

    phases = [
        transition.phase
        for transition in transitions
    ]

    unique_phases = list(
        dict.fromkeys(phases)
    )

    print()
    print("Phases encountered:")

    for phase in unique_phases:
        print(
            f"  {phase}"
        )

    # ==================================================================
    # VALIDITY CHECK
    # ==================================================================

    validity_reached = any(
        transition.next_state.valid
        for transition in transitions
    )

    print()
    print(
        f"Validity reached : "
        f"{validity_reached}"
    )

    if validity_reached:

        print()
        print(
            "✓ VALIDITY WAS REACHED"
        )

        minimization_reached = any(
            transition.phase.name == "MINIMIZATION"
            for transition in transitions
        )

        print(
            f"✓ Minimization reached : "
            f"{minimization_reached}"
        )

    else:

        print()
        print(
            "⚠ Validity was NOT reached."
        )

        print(
            "This is acceptable for an untrained policy."
        )

    # ==================================================================
    # ROLLOUT BUFFER CHECK
    # ==================================================================

    print()
    print("ROLLOUT CHECK")
    print("-" * 70)

    for transition in transitions:

        assert not transition.log_prob.requires_grad
        assert not transition.entropy.requires_grad

    print(
        "✓ log_prob detached"
    )

    print(
        "✓ entropy detached"
    )

    # ==================================================================
    # FINAL INTEGRITY
    # ==================================================================

    assert (
        final_state.candidate is not None
    )

    assert (
        final_state.cut_size
        == final_state.candidate.num_cut_edges
    )

    assert (
        final_state.selected_session_count
        == final_state.candidate.num_selected_sessions
    )

    print()
    print("=" * 70)
    print("TINY PPO EPISODE TEST PASSED")
    print("=" * 70)


# ======================================================================
# PYTEST ENTRY POINT
# ======================================================================

def test_full_ppo_episode():

    # Create output directory.
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Capture the complete test output.
    # --------------------------------------------------------------

    buffer = io.StringIO()

    class Tee:

        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    original_stdout = sys.stdout

    tee = Tee(
        original_stdout,
        buffer,
    )

    try:

        with redirect_stdout(tee):

            _run_test()

    finally:

        # Always write whatever was captured,
        # even if the test fails.
        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                buffer.getvalue()
            )

        sys.stdout = original_stdout

    print()
    print(
        f"Output saved to: {OUTPUT_FILE}"
    )