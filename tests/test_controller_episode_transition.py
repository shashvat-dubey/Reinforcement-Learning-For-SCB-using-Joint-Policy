from dataclasses import dataclass
from typing import Optional

import pytest

from src.controller.controller import (
    SCBController,
    ControllerPhase,
)
from src.environment.actions import Action, ActionType
from src.environment.candidate import Candidate
from src.environment.state import SCBState


# ============================================================
# TEST HELPERS
# ============================================================

def make_state(
    *,
    valid: bool,
    scb: Optional[float],
    step: int = 0,
) -> SCBState:
    """
    Create a minimal SCBState for controller integration testing.

    This is deliberately independent of the real graph evaluator.
    We are testing controller/episode phase transitions here,
    not whether the neural policy can discover a valid cut.
    """
    candidate = Candidate(
        selected_sessions={0},
        cut_edges=set(),
    )

    return SCBState(
        candidate=candidate,
        valid=valid,
        scb=scb,
        cut_size=0,
        selected_session_count=1,
        separated_sessions=1 if valid else 0,
    )


# ============================================================
# SCRIPTED CONTROLLER TRANSITION TEST
# ============================================================

def test_controller_episode_phase_transitions():
    """
    Drive the controller through:

        VALIDITY
            ↓
        MINIMIZATION
            ↓
        LOCAL_RETRY
            ↓
        GLOBAL_EXPLORATION
            ↓
        MINIMIZATION

    No neural policy is involved.

    This test verifies that the episode/controller integration
    behaves according to the designed phase machine.
    """

    controller = SCBController()

    # --------------------------------------------------------
    # STEP 0
    # Initial state is INVALID.
    # Controller must start in VALIDITY.
    # --------------------------------------------------------

    initial_state = make_state(
        valid=False,
        scb=None,
    )

    controller.start(initial_state)

    assert controller.current_policy() == ControllerPhase.VALIDITY

    print("\n==============================================")
    print("CONTROLLER EPISODE TRANSITION TEST")
    print("==============================================")
    print(
        f"Initial state: "
        f"valid={initial_state.valid}, "
        f"SCB={initial_state.scb}"
    )
    print(f"Initial phase: {controller.current_policy().value}")

    # --------------------------------------------------------
    # STEP 1
    # Validity policy successfully finds a valid candidate.
    #
    # INVALID → VALID
    # VALIDITY → MINIMIZATION
    # --------------------------------------------------------

    valid_state = make_state(
        valid=True,
        scb=10.0,
    )

    phase = controller.update(valid_state)

    print(
        f"\nStep 1:"
        f" valid={valid_state.valid},"
        f" SCB={valid_state.scb},"
        f" phase={phase.value}"
    )

    assert phase == ControllerPhase.MINIMIZATION
    assert controller.best_scb == 10.0
    assert controller.global_best_scb == 10.0

    # --------------------------------------------------------
    # STEP 2
    # Minimization improves SCB.
    #
    # 10.0 → 9.0
    # Remains MINIMIZATION.
    # --------------------------------------------------------

    improved_state = make_state(
        valid=True,
        scb=9.0,
    )

    phase = controller.update(improved_state)

    print(
        f"\nStep 2:"
        f" valid={improved_state.valid},"
        f" SCB={improved_state.scb},"
        f" phase={phase.value}"
    )

    assert phase == ControllerPhase.MINIMIZATION
    assert controller.best_scb == 9.0
    assert controller.global_best_scb == 9.0
    assert controller.stall_count == 0

    # --------------------------------------------------------
    # STEP 3 → N_STALL + 1
    #
    # Feed unchanged SCB values until the controller enters
    # LOCAL_RETRY.
    # --------------------------------------------------------

    stalled_state = make_state(
        valid=True,
        scb=9.0,
    )

    for i in range(controller.N_STALL):
        phase = controller.update(stalled_state)

        print(
            f"\nStall step {i + 1}/{controller.N_STALL}:"
            f" valid={stalled_state.valid},"
            f" SCB={stalled_state.scb},"
            f" stall_count={controller.stall_count},"
            f" phase={phase.value}"
        )

    assert phase == ControllerPhase.LOCAL_RETRY

    # --------------------------------------------------------
    # STEP 4
    # LOCAL RETRY produces a small improvement.
    #
    # 9.0 → 8.95
    #
    # Improvement = 0.05
    # Threshold = 0.10
    #
    # Therefore this is NOT considered sufficiently promising.
    # The controller should consume a local retry.
    # --------------------------------------------------------

    local_retry_state = make_state(
        valid=True,
        scb=8.95,
    )

    phase = controller.update(local_retry_state)

    print(
        f"\nLocal retry 1:"
        f" valid={local_retry_state.valid},"
        f" SCB={local_retry_state.scb},"
        f" retry_count={controller.local_retry_count},"
        f" phase={phase.value}"
    )

    # The controller remains in LOCAL_RETRY because
    # the improvement was below the promising threshold.
    assert phase == ControllerPhase.LOCAL_RETRY

    # --------------------------------------------------------
    # STEP 5
    # Second local retry fails to improve.
    #
    # This exhausts MAX_LOCAL_RETRIES.
    #
    # LOCAL_RETRY → GLOBAL_EXPLORATION
    # --------------------------------------------------------

    retry_fail_state = make_state(
        valid=True,
        scb=8.95,
    )

    phase = controller.update(retry_fail_state)

    print(
        f"\nLocal retry 2:"
        f" valid={retry_fail_state.valid},"
        f" SCB={retry_fail_state.scb},"
        f" retry_count={controller.local_retry_count},"
        f" phase={phase.value}"
    )

    assert phase == ControllerPhase.GLOBAL_EXPLORATION
    assert controller.local_retry_count == 0

    # --------------------------------------------------------
    # STEP 6
    # Global exploration discovers a new global best.
    #
    # 9.0 → 8.0
    #
    # GLOBAL_EXPLORATION → MINIMIZATION
    # --------------------------------------------------------

    global_improvement_state = make_state(
        valid=True,
        scb=8.0,
    )

    phase = controller.update(global_improvement_state)

    print(
        f"\nGlobal exploration:"
        f" valid={global_improvement_state.valid},"
        f" SCB={global_improvement_state.scb},"
        f" global_best={controller.global_best_scb},"
        f" phase={phase.value}"
    )

    assert phase == ControllerPhase.MINIMIZATION
    assert controller.global_best_scb == 8.0
    assert controller.best_scb == 8.0
    assert controller.stall_count == 0

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n==============================================")
    print("TRANSITION TEST PASSED")
    print("==============================================")
    print(
        "VALIDITY"
        " → MINIMIZATION"
        " → LOCAL_RETRY"
        " → GLOBAL_EXPLORATION"
        " → MINIMIZATION"
    )
    print(f"Final best SCB: {controller.best_scb}")
    print(f"Final global SCB: {controller.global_best_scb}")
    print(f"Final phase: {controller.current_policy().value}")

    assert controller.current_policy() == ControllerPhase.MINIMIZATION