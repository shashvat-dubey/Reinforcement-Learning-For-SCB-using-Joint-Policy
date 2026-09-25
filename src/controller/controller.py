from enum import Enum
from typing import Optional

from src.environment.state import SCBState


class ControllerPhase(Enum):
    VALIDITY = "validity"
    MINIMIZATION = "minimization"
    LOCAL_RETRY = "local_retry"
    GLOBAL_EXPLORATION = "global_exploration"
    DONE = "done"


class SCBController:
    """
    Deterministic controller for the joint SCB RL system.

    The controller decides which policy should currently be active.

    It does NOT:
        - select actions
        - run a neural network
        - calculate PPO losses
        - use GA results
        - modify rewards

    Responsibilities:
        - Manage policy phases
        - Track SCB improvements
        - Detect minimization stalls
        - Manage local retries
        - Trigger global exploration
    """

    N_STALL = 20
    LOCAL_PROMISING_THRESHOLD = 0.10
    MAX_LOCAL_RETRIES = 2

    def __init__(self):
        self.phase = ControllerPhase.VALIDITY

        self.stall_count = 0
        self.local_retry_count = 0

        self.best_scb: Optional[float] = None
        self.global_best_scb: Optional[float] = None

        self.previous_state: Optional[SCBState] = None
        self.local_retry_start_scb: Optional[float] = None

    def reset(self):
        """Reset controller for a new episode."""

        self.phase = ControllerPhase.VALIDITY

        self.stall_count = 0
        self.local_retry_count = 0

        self.best_scb = None
        self.global_best_scb = None

        self.previous_state = None
        self.local_retry_start_scb = None

    def start(self, state: SCBState):
        """Start controller using the initial environment state."""

        self.reset()

        self.previous_state = state.copy()

        if state.valid and state.scb is not None:
            self.phase = ControllerPhase.MINIMIZATION
            self.best_scb = state.scb
            self.global_best_scb = state.scb
        else:
            self.phase = ControllerPhase.VALIDITY

    def current_policy(self) -> ControllerPhase:
        """Return the policy currently controlling the agent."""

        return self.phase

    def update(self, new_state: SCBState) -> ControllerPhase:
        """
        Update controller after an environment transition.

        Returns:
            Policy/phase that should control the next action.
        """

        if self.previous_state is None:
            raise RuntimeError(
                "Controller must be started before calling update()."
            )

        # ---------------------------------------------------------
        # VALIDITY
        # ---------------------------------------------------------

        if self.phase == ControllerPhase.VALIDITY:

            if new_state.valid:

                if new_state.scb is not None:
                    self.best_scb = new_state.scb

                    if (
                        self.global_best_scb is None
                        or new_state.scb < self.global_best_scb
                    ):
                        self.global_best_scb = new_state.scb

                self.stall_count = 0
                self.local_retry_count = 0

                self.phase = ControllerPhase.MINIMIZATION

            self.previous_state = new_state.copy()
            return self.phase

        # ---------------------------------------------------------
        # MINIMIZATION
        # ---------------------------------------------------------

        if self.phase == ControllerPhase.MINIMIZATION:

            if not new_state.valid or new_state.scb is None:
                self.phase = ControllerPhase.VALIDITY
                self.stall_count = 0

                self.previous_state = new_state.copy()
                return self.phase

            improved = (
                self.best_scb is None
                or new_state.scb < self.best_scb
            )

            if improved:

                self.best_scb = new_state.scb
                self.stall_count = 0

                if (
                    self.global_best_scb is None
                    or new_state.scb < self.global_best_scb
                ):
                    self.global_best_scb = new_state.scb

            else:
                self.stall_count += 1

            if self.stall_count >= self.N_STALL:
                self.stall_count = 0
                self.local_retry_start_scb = self.best_scb
                self.phase = ControllerPhase.LOCAL_RETRY

            self.previous_state = new_state.copy()
            return self.phase

        # ---------------------------------------------------------
        # LOCAL RETRY
        # ---------------------------------------------------------

        if self.phase == ControllerPhase.LOCAL_RETRY:

            if not new_state.valid or new_state.scb is None:
                self.phase = ControllerPhase.VALIDITY
                self.stall_count = 0

                self.previous_state = new_state.copy()
                return self.phase

            # Improvement relative to the state that existed when
            # local retry began.
            if (
                self.local_retry_start_scb is not None
                and new_state.scb < self.local_retry_start_scb
            ):
                improvement = (
                    self.local_retry_start_scb - new_state.scb
                )

                self.best_scb = new_state.scb
                self.global_best_scb = min(
                    self.global_best_scb
                    if self.global_best_scb is not None
                    else new_state.scb,
                    new_state.scb,
                )

                self.stall_count = 0
                self.local_retry_count = 0
                self.local_retry_start_scb = None

                # A sufficiently promising local improvement sends
                # us back into ordinary minimization.
                if improvement >= self.LOCAL_PROMISING_THRESHOLD:
                    self.phase = ControllerPhase.MINIMIZATION
                else:
                    # Small improvement still counts as progress,
                    # but does not meet the promising threshold.
                    self.local_retry_count += 1

                    if self.local_retry_count >= self.MAX_LOCAL_RETRIES:
                        self.local_retry_count = 0
                        self.local_retry_start_scb = None
                        self.phase = ControllerPhase.GLOBAL_EXPLORATION
                    else:
                        self.local_retry_start_scb = new_state.scb

            else:
                # No meaningful improvement during this retry.
                self.local_retry_count += 1
                self.stall_count = 0

                if self.local_retry_count >= self.MAX_LOCAL_RETRIES:
                    self.local_retry_count = 0
                    self.local_retry_start_scb = None
                    self.phase = ControllerPhase.GLOBAL_EXPLORATION

            self.previous_state = new_state.copy()
            return self.phase

        # ---------------------------------------------------------
        # GLOBAL EXPLORATION
        # ---------------------------------------------------------

        if self.phase == ControllerPhase.GLOBAL_EXPLORATION:

            if not new_state.valid or new_state.scb is None:
                self.phase = ControllerPhase.VALIDITY
                self.stall_count = 0

                self.previous_state = new_state.copy()
                return self.phase

            if (
                self.global_best_scb is None
                or new_state.scb < self.global_best_scb
            ):
                self.global_best_scb = new_state.scb
                self.best_scb = new_state.scb

                self.stall_count = 0
                self.local_retry_count = 0

                self.phase = ControllerPhase.MINIMIZATION

            else:
                self.stall_count += 1

            self.previous_state = new_state.copy()
            return self.phase

        # ---------------------------------------------------------
        # DONE
        # ---------------------------------------------------------

        if self.phase == ControllerPhase.DONE:
            self.previous_state = new_state.copy()
            return self.phase

        raise RuntimeError(
            f"Unknown controller phase: {self.phase}"
        )

    def should_stop(self) -> bool:
        """Return whether the controller is in a terminal phase."""

        return self.phase == ControllerPhase.DONE