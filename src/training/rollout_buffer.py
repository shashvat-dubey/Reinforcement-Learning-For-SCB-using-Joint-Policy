from dataclasses import dataclass
from typing import Optional

import torch

from src.controller.controller import ControllerPhase
from src.environment.actions import Action
from src.environment.state import SCBState
from src.models.scb_encoder import SCBEncoderOutput


@dataclass
class RolloutTransition:
    """
    One transition collected during an RL episode.

    This contains everything PPO needs from the interaction phase.
    """

    state: SCBState
    action: Action
    action_index: int

    reward: float

    next_state: SCBState

    done: bool

    log_prob: torch.Tensor
    entropy: torch.Tensor

    phase: ControllerPhase

    # Keep the encoder output associated with this state.
    #
    # This is useful during the initial implementation/debugging
    # stage and makes the collected trajectory self-contained.
    encoder_output: Optional[SCBEncoderOutput] = None


class RolloutBuffer:
    """
    Stores transitions collected during RL interaction.

    The buffer is deliberately policy-aware because the system
    contains three separate policies:

        VALIDITY
        MINIMIZATION
        GLOBAL_EXPLORATION

    The buffer does NOT:
        - calculate rewards
        - calculate advantages
        - calculate returns
        - perform PPO updates

    Those belong to the PPO training layer.
    """

    def __init__(self):
        self.transitions: list[RolloutTransition] = []

    # ---------------------------------------------------------
    # ADD
    # ---------------------------------------------------------

    def add(
        self,
        *,
        state: SCBState,
        action: Action,
        action_index: int,
        reward: float,
        next_state: SCBState,
        done: bool,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        phase: ControllerPhase,
        encoder_output: Optional[SCBEncoderOutput] = None,
    ) -> None:

        transition = RolloutTransition(
            state=state.copy(),
            action=action,
            action_index=int(action_index),
            reward=float(reward),
            next_state=next_state.copy(),
            done=bool(done),
            log_prob=log_prob.detach().clone(),
            entropy=entropy.detach().clone(),
            phase=phase,
            encoder_output=encoder_output,
        )

        self.transitions.append(transition)

    # ---------------------------------------------------------
    # BASIC ACCESS
    # ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> RolloutTransition:
        return self.transitions[index]

    # ---------------------------------------------------------
    # CLEAR
    # ---------------------------------------------------------

    def clear(self) -> None:
        self.transitions.clear()

    # ---------------------------------------------------------
    # POLICY FILTERING
    # ---------------------------------------------------------

    def by_phase(
        self,
        phase: ControllerPhase,
    ) -> list[RolloutTransition]:

        return [
            transition
            for transition in self.transitions
            if transition.phase == phase
        ]

    def validity_transitions(
        self,
    ) -> list[RolloutTransition]:

        return self.by_phase(
            ControllerPhase.VALIDITY
        )

    def minimization_transitions(
        self,
    ) -> list[RolloutTransition]:

        transitions = []

        for transition in self.transitions:

            if transition.phase in (
                ControllerPhase.MINIMIZATION,
                ControllerPhase.LOCAL_RETRY,
            ):
                transitions.append(
                    transition
                )

        return transitions

    def global_exploration_transitions(
        self,
    ) -> list[RolloutTransition]:

        return self.by_phase(
            ControllerPhase.GLOBAL_EXPLORATION
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    def summary(self) -> dict:

        return {
            "total": len(self.transitions),
            "validity": len(
                self.validity_transitions()
            ),
            "minimization": len(
                self.minimization_transitions()
            ),
            "global_exploration": len(
                self.global_exploration_transitions()
            ),
        }