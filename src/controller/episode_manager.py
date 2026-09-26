from dataclasses import dataclass
from typing import Optional

from src.controller.controller import (
    SCBController,
    ControllerPhase,
)
from src.controller.policy_orchestrator import (
    PolicyOrchestrator,
    OrchestratorOutput,
)
from src.environment.environment import SCBEnvironment
from src.environment.state import SCBState
from src.models.scb_encoder import SCBEncoder
from src.scb.problem import SCBProblem


@dataclass
class EpisodeStep:
    step_number: int
    phase: ControllerPhase
    policy_name: str
    action: object
    reward: float
    state: SCBState
    done: bool


@dataclass
class EpisodeResult:
    steps: list[EpisodeStep]
    initial_state: SCBState
    final_state: SCBState
    total_reward: float
    best_scb: Optional[float]
    final_phase: ControllerPhase


class SCBEpisodeManager:
    """
    Runs one complete SCB RL episode.

    Responsibilities:
        - Reset environment/controller.
        - Encode the current state.
        - Ask the orchestrator for an action.
        - Apply the action through the environment.
        - Update the controller.
        - Track episode statistics.
        - Optionally print human-readable debugging information.

    Does NOT:
        - Calculate rewards.
        - Select actions.
        - Calculate PPO losses.
        - Use GA information.
    """

    def __init__(
        self,
        environment: SCBEnvironment,
        controller: SCBController,
        encoder: SCBEncoder,
        orchestrator: PolicyOrchestrator,
        max_steps: int = 100,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.environment = environment
        self.controller = controller
        self.encoder = encoder
        self.orchestrator = orchestrator
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Environment mode
    # ------------------------------------------------------------------

    def _set_environment_mode(
        self,
        phase: ControllerPhase,
    ):
        if phase == ControllerPhase.VALIDITY:
            self.environment.set_mode(
                SCBEnvironment.VALIDITY_MODE
            )

        elif phase in (
            ControllerPhase.MINIMIZATION,
            ControllerPhase.LOCAL_RETRY,
        ):
            self.environment.set_mode(
                SCBEnvironment.MINIMIZATION_MODE
            )

        elif phase == ControllerPhase.GLOBAL_EXPLORATION:
            self.environment.set_mode(
                SCBEnvironment.GLOBAL_EXPLORATION_MODE
            )

        elif phase == ControllerPhase.DONE:
            return

        else:
            raise ValueError(
                f"Unsupported controller phase: {phase}"
            )

    # ------------------------------------------------------------------
    # Debug printing
    # ------------------------------------------------------------------

    @staticmethod
    def _format_action(action) -> str:
        action_type = action.action_type.name

        if action.target is None:
            return action_type

        return f"{action_type} {action.target}"

    @staticmethod
    def _print_initial_state(
        state: SCBState,
        phase: ControllerPhase,
    ):
        print()
        print("=" * 60)
        print("EPISODE START")
        print("=" * 60)

        print("Initial State")
        print(f"  Valid             : {state.valid}")
        print(f"  SCB               : {state.scb}")
        print(
            f"  Sessions selected : "
            f"{state.selected_session_count}"
        )
        print(f"  Cut edges         : {state.cut_size}")
        print(
            f"  Separated         : "
            f"{state.separated_sessions}"
        )
        print(f"  Phase             : {phase.value}")

        print("=" * 60)

    @staticmethod
    def _print_step(
        step_number: int,
        phase: ControllerPhase,
        policy_name: str,
        orchestrator_output: OrchestratorOutput,
        reward: float,
        state: SCBState,
        done: bool,
    ):
        print()
        print(f"Step {step_number:03d}")

        print(f"  Phase             : {phase.value}")
        print(f"  Policy            : {policy_name}")
        print(
            f"  Action            : "
            f"{SCBEpisodeManager._format_action(orchestrator_output.action)}"
        )
        print(f"  Reward            : {reward:+.6f}")
        print(f"  Valid             : {state.valid}")
        print(f"  SCB               : {state.scb}")
        print(f"  Cut size          : {state.cut_size}")
        print(
            f"  Separated         : "
            f"{state.separated_sessions}"
        )
        print(f"  Done              : {done}")

    @staticmethod
    def _print_phase_transition(
        old_phase: ControllerPhase,
        new_phase: ControllerPhase,
    ):
        if old_phase != new_phase:
            print()
            print(
                f"[CONTROLLER] "
                f"{old_phase.value.upper()} "
                f"→ "
                f"{new_phase.value.upper()}"
            )

    @staticmethod
    def _print_episode_summary(
        result: EpisodeResult,
    ):
        print()
        print("=" * 60)
        print("EPISODE COMPLETE")
        print("=" * 60)

        print(f"  Steps             : {len(result.steps)}")
        print(
            f"  Final phase       : "
            f"{result.final_phase.value}"
        )
        print(f"  Final valid       : {result.final_state.valid}")
        print(f"  Final SCB         : {result.final_state.scb}")
        print(f"  Best SCB          : {result.best_scb}")
        print(f"  Total reward      : {result.total_reward:.6f}")

        print("=" * 60)

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def run(
        self,
        selected_sessions=None,
        verbose: bool = True,
        deterministic: bool = False,
    ) -> EpisodeResult:

        # --------------------------------------------------------------
        # Reset
        # --------------------------------------------------------------

        initial_state = self.environment.reset(
            selected_sessions=selected_sessions
        )

        self.controller.start(initial_state)

        current_state = initial_state.copy()

        phase = self.controller.current_policy()

        if verbose:
            self._print_initial_state(
                current_state,
                phase,
            )

        steps: list[EpisodeStep] = []

        total_reward = 0.0

        best_scb = (
            current_state.scb
            if current_state.valid
            else None
        )

        # --------------------------------------------------------------
        # Main loop
        # --------------------------------------------------------------

        for step_number in range(1, self.max_steps + 1):

            phase = self.controller.current_policy()

            if phase == ControllerPhase.DONE:
                break

            # ----------------------------------------------------------
            # Set reward mode according to controller phase
            # ----------------------------------------------------------

            self._set_environment_mode(phase)

            # ----------------------------------------------------------
            # Encode current state
            # ----------------------------------------------------------

            encoder_output = self.encoder(
                self.environment.problem,
                current_state,
            )

            # ----------------------------------------------------------
            # Policy chooses action
            # ----------------------------------------------------------

            orchestrator_output = self.orchestrator.act(
                phase=phase,
                encoder_output=encoder_output,
                problem=self.environment.problem,
                state=current_state,
                deterministic=deterministic,
            )

            # ----------------------------------------------------------
            # Environment applies action
            # ----------------------------------------------------------

            new_state, reward, done, _info = self.environment.step(
                orchestrator_output.action
            )

            total_reward += reward

            # ----------------------------------------------------------
            # Track best SCB
            # ----------------------------------------------------------

            if (
                new_state.valid
                and new_state.scb is not None
                and (
                    best_scb is None
                    or new_state.scb < best_scb
                )
            ):
                best_scb = new_state.scb

            # ----------------------------------------------------------
            # Controller update
            # ----------------------------------------------------------

            old_phase = phase

            if not done:
                new_phase = self.controller.update(
                    new_state
                )
            else:
                new_phase = self.controller.current_policy()

            # ----------------------------------------------------------
            # Record step
            # ----------------------------------------------------------

            episode_step = EpisodeStep(
                step_number=step_number,
                phase=old_phase,
                policy_name=type(
                    orchestrator_output.policy
                ).__name__,
                action=orchestrator_output.action,
                reward=reward,
                state=new_state.copy(),
                done=done,
            )

            steps.append(episode_step)

            # ----------------------------------------------------------
            # Debug output
            # ----------------------------------------------------------

            if verbose:
                self._print_step(
                    step_number=step_number,
                    phase=old_phase,
                    policy_name=episode_step.policy_name,
                    orchestrator_output=orchestrator_output,
                    reward=reward,
                    state=new_state,
                    done=done,
                )

                self._print_phase_transition(
                    old_phase,
                    new_phase,
                )

            # ----------------------------------------------------------
            # Advance
            # ----------------------------------------------------------

            current_state = new_state.copy()

            if done:
                break

        # --------------------------------------------------------------
        # Build result
        # --------------------------------------------------------------

        result = EpisodeResult(
            steps=steps,
            initial_state=initial_state,
            final_state=current_state,
            total_reward=total_reward,
            best_scb=best_scb,
            final_phase=self.controller.current_policy(),
        )

        if verbose:
            self._print_episode_summary(result)

        return result