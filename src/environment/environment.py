from typing import Optional

from src.environment.candidate import Candidate
from src.environment.actions import Action, ActionType
from src.environment.state import SCBState

from src.scb.problem import SCBProblem
from src.scb.evaluator import evaluate_candidate


class SCBEnvironment:
    """
    Deterministic environment for joint SCB search.

    Handles:
        - Current candidate (I', E')
        - Candidate evaluation
        - Validity
        - SCB
        - Rewards

    Controller logic is intentionally not included yet.
    """

    VALIDITY_MODE = "validity"
    MINIMIZATION_MODE = "minimization"

    def __init__(
        self,
        problem: SCBProblem,
        mode: str = VALIDITY_MODE,
    ):
        self.problem = problem
        self.mode = mode

        self.candidate: Optional[Candidate] = None
        self.state: Optional[SCBState] = None

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, selected_sessions=None) -> SCBState:
        """
        Start a new search.

        Default:
            I' = all sessions
            E' = empty
        """

        if selected_sessions is None:
            selected_sessions = set(self.problem.sessions)

        selected_sessions = set(selected_sessions)

        if not selected_sessions:
            raise ValueError(
                "Initial candidate must contain at least one selected session."
            )

        self.candidate = Candidate(
            selected_sessions=selected_sessions,
            cut_edges=set(),
        )

        self.state = self._evaluate_candidate(self.candidate)

        return self.state.copy()

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action: Action):
        """
        Apply one action and return:

            next_state
            reward
            done
            info
        """

        if self.candidate is None or self.state is None:
            raise RuntimeError(
                "Environment must be reset before calling step()."
            )

        old_state = self.state.copy()

        # STOP ends the current episode.
        if action.action_type == ActionType.STOP:
            return (
                self.state.copy(),
                0.0,
                True,
                {"action": "STOP"},
            )

        new_candidate = self.candidate.copy()

        self._apply_action(new_candidate, action)

        new_state = self._evaluate_candidate(new_candidate)

        reward = self._calculate_reward(
            old_state,
            new_state,
        )

        self.candidate = new_candidate
        self.state = new_state

        return (
            new_state.copy(),
            reward,
            False,
            {
                "action": action.action_type.name,
            },
        )

    # ------------------------------------------------------------------
    # ACTION APPLICATION
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        candidate: Candidate,
        action: Action,
    ):
        """Apply an action to a candidate."""

        action_type = action.action_type
        target = action.target

        if action_type == ActionType.ADD_EDGE:

            if target is None:
                raise ValueError(
                    "ADD_EDGE requires an edge target."
                )

            candidate.add_edge(target)

        elif action_type == ActionType.REMOVE_EDGE:

            if target is None:
                raise ValueError(
                    "REMOVE_EDGE requires an edge target."
                )

            candidate.remove_edge(target)

        elif action_type == ActionType.ADD_SESSION:

            if target is None:
                raise ValueError(
                    "ADD_SESSION requires a session target."
                )

            candidate.add_session(target)

        elif action_type == ActionType.REMOVE_SESSION:

            if target is None:
                raise ValueError(
                    "REMOVE_SESSION requires a session target."
                )

            candidate.remove_session(target)

        else:
            raise ValueError(
                f"Unsupported action type: {action_type}"
            )

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        candidate: Candidate,
    ) -> SCBState:
        """Evaluate a candidate using the SCB mathematical evaluator."""

        result = evaluate_candidate(
            self.problem,
            selected_sessions=candidate.selected_sessions,
            cut_edges=candidate.cut_edges,
        )

        return SCBState(
            candidate=candidate.copy(),
            valid=result.valid,
            scb=result.scb,
            cut_size=result.cut_size,
            selected_session_count=result.selected_session_count,
            separated_sessions=result.separated_sessions,
        )

    # ------------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        old_state: SCBState,
        new_state: SCBState,
    ) -> float:
        """
        Calculate reward.

        Validity mode:

            INVALID -> VALID = +1
            otherwise = 0

        Minimization mode:

            valid -> valid:
                reward = old_scb - new_scb

            otherwise:
                reward = 0
        """

        # --------------------------------------------------------------
        # VALIDITY
        # --------------------------------------------------------------

        if self.mode == self.VALIDITY_MODE:

            if not old_state.valid and new_state.valid:
                return 1.0

            return 0.0

        # --------------------------------------------------------------
        # MINIMIZATION
        # --------------------------------------------------------------

        if self.mode == self.MINIMIZATION_MODE:

            if old_state.valid and new_state.valid:

                if old_state.scb is None:
                    return 0.0

                if new_state.scb is None:
                    return 0.0

                return old_state.scb - new_state.scb

            return 0.0

        raise ValueError(
            f"Unknown environment mode: {self.mode}"
        )

    # ------------------------------------------------------------------
    # MODE
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        """Change the reward mode."""

        if mode not in (
            self.VALIDITY_MODE,
            self.MINIMIZATION_MODE,
        ):
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        self.mode = mode