from dataclasses import dataclass
from typing import Optional

from src.controller.controller import (
    SCBController,
    ControllerPhase,
)
from src.environment.actions import Action, ActionType
from src.environment.candidate import Candidate
from src.environment.state import SCBState


# ============================================================
# DUMMY SCB EPISODE
# ============================================================
#
# This test does NOT use neural policies.
#
# Instead, scripted policies imitate what we expect the learned
# policies to eventually do:
#
#   VALIDITY POLICY
#       -> build a valid cut
#
#   MINIMIZATION POLICY
#       -> improve SCB
#
#   LOCAL RETRY
#       -> attempt a small local improvement
#
#   GLOBAL EXPLORATION
#       -> discover a better SCB
#
# The purpose is to verify the COMPLETE episode logic.
# ============================================================


@dataclass
class DummyStep:
    action: Action
    state: SCBState


class DummySCBEpisode:
    """
    Scripted SCB episode used to test the architecture.

    The neural networks are deliberately replaced by scripted
    actions/states so that we can verify the control flow.
    """

    def __init__(self):
        self.controller = SCBController()

        self.step_number = 0

        # ----------------------------------------------------
        # Dummy candidate
        # ----------------------------------------------------

        self.candidate = Candidate(
            selected_sessions={0},
            cut_edges=set(),
        )

    # ========================================================
    # STATE FACTORY
    # ========================================================

    def make_state(
        self,
        *,
        valid: bool,
        scb: Optional[float],
        cut_size: int,
        separated_sessions: int,
    ) -> SCBState:

        return SCBState(
            candidate=self.candidate.copy(),
            valid=valid,
            scb=scb,
            cut_size=cut_size,
            selected_session_count=len(
                self.candidate.selected_sessions
            ),
            separated_sessions=separated_sessions,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        print("\n")
        print("=" * 70)
        print("DUMMY SCB EPISODE")
        print("=" * 70)

        # ----------------------------------------------------
        # INITIAL STATE
        # ----------------------------------------------------
        #
        # Candidate contains a session but no cut edges.
        #
        # Therefore the session is NOT separated.
        #
        # Candidate = INVALID
        # ----------------------------------------------------

        state = self.make_state(
            valid=False,
            scb=None,
            cut_size=0,
            separated_sessions=0,
        )

        self.controller.start(state)

        self.print_state(
            state,
            action="START",
        )

        # ====================================================
        # PHASE 1 — VALIDITY
        # ====================================================

        self.validity_phase()

        # ====================================================
        # PHASE 2 — MINIMIZATION
        # ====================================================

        self.minimization_phase()

        # ====================================================
        # PHASE 3 — LOCAL RETRY
        # ====================================================

        self.local_retry_phase()

        # ====================================================
        # PHASE 4 — GLOBAL EXPLORATION
        # ====================================================

        self.global_exploration_phase()

        # ====================================================
        # FINAL
        # ====================================================

        print("\n")
        print("=" * 70)
        print("DUMMY EPISODE COMPLETE")
        print("=" * 70)

        print(
            f"Final phase       : "
            f"{self.controller.current_policy().value}"
        )

        print(
            f"Best SCB          : "
            f"{self.controller.best_scb}"
        )

        print(
            f"Global best SCB   : "
            f"{self.controller.global_best_scb}"
        )

        print("=" * 70)

    # ========================================================
    # VALIDITY POLICY
    # ========================================================

    def validity_phase(self):

        print("\n")
        print("-" * 70)
        print("PHASE: VALIDITY POLICY")
        print("-" * 70)

        # ----------------------------------------------------
        # Action 1:
        #
        # Add first cut edge.
        #
        # Candidate is STILL invalid.
        # ----------------------------------------------------

        edge_1 = ("v1", "v2")

        self.candidate.add_edge(edge_1)

        state = self.make_state(
            valid=False,
            scb=None,
            cut_size=1,
            separated_sessions=0,
        )

        action = Action(
            ActionType.ADD_EDGE,
            edge_1,
        )

        self.apply_controller_step(
            state,
            action,
        )

        # ----------------------------------------------------
        # Action 2:
        #
        # Add second cut edge.
        #
        # This creates a VALID candidate.
        # ----------------------------------------------------

        edge_2 = ("v2", "v3")

        self.candidate.add_edge(edge_2)

        state = self.make_state(
            valid=True,
            scb=2.0,
            cut_size=2,
            separated_sessions=1,
        )

        action = Action(
            ActionType.ADD_EDGE,
            edge_2,
        )

        self.apply_controller_step(
            state,
            action,
        )

        assert (
            self.controller.current_policy()
            == ControllerPhase.MINIMIZATION
        )

        print(
            "\nVALIDITY SUCCESS → "
            "MINIMIZATION"
        )

    # ========================================================
    # MINIMIZATION POLICY
    # ========================================================

    def minimization_phase(self):

        print("\n")
        print("-" * 70)
        print("PHASE: MINIMIZATION POLICY")
        print("-" * 70)

        # ----------------------------------------------------
        # Improvement 1
        #
        # SCB: 2.0 → 1.5
        # ----------------------------------------------------

        state = self.make_state(
            valid=True,
            scb=1.5,
            cut_size=3,
            separated_sessions=2,
        )

        action = Action(
            ActionType.ADD_EDGE,
            ("v3", "v4"),
        )

        self.candidate.add_edge(
            ("v3", "v4")
        )

        self.apply_controller_step(
            state,
            action,
        )

        assert self.controller.best_scb == 1.5

        # ----------------------------------------------------
        # Improvement 2
        #
        # SCB: 1.5 → 1.0
        # ----------------------------------------------------

        state = self.make_state(
            valid=True,
            scb=1.0,
            cut_size=4,
            separated_sessions=4,
        )

        action = Action(
            ActionType.ADD_EDGE,
            ("v4", "v5"),
        )

        self.candidate.add_edge(
            ("v4", "v5")
        )

        self.apply_controller_step(
            state,
            action,
        )

        assert self.controller.best_scb == 1.0

        print(
            "\nCurrent best SCB:",
            self.controller.best_scb,
        )

        # ----------------------------------------------------
        # Now deliberately stall.
        #
        # 20 steps with no improvement.
        # ----------------------------------------------------

        for i in range(
            self.controller.N_STALL
        ):

            state = self.make_state(
                valid=True,
                scb=1.0,
                cut_size=4,
                separated_sessions=4,
            )

            action = Action(
                ActionType.REMOVE_EDGE,
                ("v4", "v5"),
            )

            self.apply_controller_step(
                state,
                action,
                print_step=False,
            )

        assert (
            self.controller.current_policy()
            == ControllerPhase.LOCAL_RETRY
        )

        print(
            "\n20-step stall detected → "
            "LOCAL RETRY"
        )

    # ========================================================
    # LOCAL RETRY
    # ========================================================

    def local_retry_phase(self):

        print("\n")
        print("-" * 70)
        print("PHASE: LOCAL RETRY")
        print("-" * 70)

        # ----------------------------------------------------
        # Retry 1
        #
        # Small improvement:
        #
        # 1.0 → 0.95
        #
        # Threshold = 0.10
        #
        # 0.05 is NOT promising.
        # ----------------------------------------------------

        state = self.make_state(
            valid=True,
            scb=0.95,
            cut_size=5,
            separated_sessions=5,
        )

        action = Action(
            ActionType.ADD_EDGE,
            ("v5", "v6"),
        )

        self.candidate.add_edge(
            ("v5", "v6")
        )

        self.apply_controller_step(
            state,
            action,
        )

        assert (
            self.controller.current_policy()
            == ControllerPhase.LOCAL_RETRY
        )

        # ----------------------------------------------------
        # Retry 2
        #
        # No further improvement.
        #
        # This exhausts local retries.
        # ----------------------------------------------------

        state = self.make_state(
            valid=True,
            scb=0.95,
            cut_size=5,
            separated_sessions=5,
        )

        action = Action(
            ActionType.REMOVE_EDGE,
            ("v5", "v6"),
        )

        self.apply_controller_step(
            state,
            action,
        )

        assert (
            self.controller.current_policy()
            == ControllerPhase.GLOBAL_EXPLORATION
        )

        print(
            "\nLocal retries exhausted → "
            "GLOBAL EXPLORATION"
        )

    # ========================================================
    # GLOBAL EXPLORATION
    # ========================================================

    def global_exploration_phase(self):

        print("\n")
        print("-" * 70)
        print("PHASE: GLOBAL EXPLORATION")
        print("-" * 70)

        # ----------------------------------------------------
        # Global exploration discovers:
        #
        # SCB: 0.95 → 0.70
        #
        # This is a meaningful global improvement.
        # ----------------------------------------------------

        state = self.make_state(
            valid=True,
            scb=0.70,
            cut_size=7,
            separated_sessions=10,
        )

        action = Action(
            ActionType.ADD_EDGE,
            ("v6", "v7"),
        )

        self.candidate.add_edge(
            ("v6", "v7")
        )

        self.apply_controller_step(
            state,
            action,
        )

        assert (
            self.controller.current_policy()
            == ControllerPhase.MINIMIZATION
        )

        assert (
            self.controller.global_best_scb
            == 0.70
        )

        print(
            "\nGLOBAL IMPROVEMENT FOUND!"
        )

        print(
            "0.95 → 0.70"
        )

        print(
            "GLOBAL EXPLORATION → "
            "MINIMIZATION"
        )

    # ========================================================
    # CONTROLLER STEP
    # ========================================================

    def apply_controller_step(
        self,
        state: SCBState,
        action: Action,
        print_step=True,
    ):

        self.step_number += 1

        old_phase = (
            self.controller.current_policy()
        )

        new_phase = self.controller.update(
            state
        )

        if print_step:
            print(
                f"\nStep {self.step_number}"
            )

            print(
                f"  Phase     : "
                f"{old_phase.value}"
            )

            print(
                f"  Action    : "
                f"{action.action_type.name}"
            )

            print(
                f"  Target    : "
                f"{action.target}"
            )

            print(
                f"  Valid     : "
                f"{state.valid}"
            )

            print(
                f"  SCB       : "
                f"{state.scb}"
            )

            print(
                f"  Cut size  : "
                f"{state.cut_size}"
            )

            print(
                f"  Separated : "
                f"{state.separated_sessions}"
            )

            print(
                f"  Next phase: "
                f"{new_phase.value}"
            )

    # ========================================================
    # STATE PRINTER
    # ========================================================

    @staticmethod
    def print_state(
        state: SCBState,
        action: str,
    ):

        print("\nInitial candidate")
        print(
            f"  Action    : {action}"
        )
        print(
            f"  Valid     : {state.valid}"
        )
        print(
            f"  SCB       : {state.scb}"
        )
        print(
            f"  Cut size  : {state.cut_size}"
        )
        print(
            f"  Sessions  : "
            f"{state.selected_session_count}"
        )
        print(
            f"  Separated : "
            f"{state.separated_sessions}"
        )


# ============================================================
# PYTEST
# ============================================================

def test_dummy_scb_episode():

    episode = DummySCBEpisode()

    episode.run()

    assert (
        episode.controller.current_policy()
        == ControllerPhase.MINIMIZATION
    )

    assert (
        episode.controller.global_best_scb
        == 0.70
    )

    assert (
        episode.controller.best_scb
        == 0.70
    )