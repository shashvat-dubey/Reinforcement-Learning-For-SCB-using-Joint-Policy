from dataclasses import dataclass

from src.controller.controller import ControllerPhase
from src.models.policies import (
    BaseSCBPolicy,
    ValidityPolicy,
    MinimizationPolicy,
    GlobalExplorationPolicy,
    PolicyOutput,
)
from src.environment.actions import Action
from src.environment.environment import SCBEnvironment
from src.environment.state import SCBState
from src.models.scb_encoder import SCBEncoderOutput
from src.scb.problem import SCBProblem


@dataclass
class OrchestratorOutput:
    """
    Output of the controller-policy orchestration step.
    """

    phase: ControllerPhase
    policy: BaseSCBPolicy
    policy_output: PolicyOutput

    @property
    def action(self) -> Action:
        return self.policy_output.action

    @property
    def action_index(self) -> int:
        return self.policy_output.action_index

    @property
    def log_prob(self):
        return self.policy_output.log_prob

    @property
    def entropy(self):
        return self.policy_output.entropy


class PolicyOrchestrator:
    """
    Connects the controller's current phase to the appropriate policy.

    Responsibilities:
        - Select the policy associated with the controller phase.
        - Execute that policy.
        - Return the policy output together with the active phase.

    Does NOT:
        - Choose actions itself.
        - Modify the environment.
        - Calculate rewards.
        - Calculate PPO losses.
        - Use GA information.
        - Modify controller state.
    """

    def __init__(
        self,
        validity_policy: ValidityPolicy,
        minimization_policy: MinimizationPolicy,
        global_exploration_policy: GlobalExplorationPolicy,
    ):
        self.validity_policy = validity_policy
        self.minimization_policy = minimization_policy
        self.global_exploration_policy = global_exploration_policy

    def get_policy(
        self,
        phase: ControllerPhase,
    ) -> BaseSCBPolicy:
        """
        Return the policy associated with a controller phase.
        """

        if phase == ControllerPhase.VALIDITY:
            return self.validity_policy

        if phase == ControllerPhase.MINIMIZATION:
            return self.minimization_policy

        if phase == ControllerPhase.LOCAL_RETRY:
            # Local retry deliberately reuses minimization.
            return self.minimization_policy

        if phase == ControllerPhase.GLOBAL_EXPLORATION:
            return self.global_exploration_policy

        if phase == ControllerPhase.DONE:
            raise RuntimeError(
                "Cannot select a policy when controller phase is DONE."
            )

        raise ValueError(
            f"Unsupported controller phase: {phase}"
        )

    def act(
        self,
        phase: ControllerPhase,
        encoder_output: SCBEncoderOutput,
        problem: SCBProblem,
        state: SCBState,
        deterministic: bool = False,
    ) -> OrchestratorOutput:
        """
        Execute the policy associated with the supplied controller phase.
        """

        policy = self.get_policy(phase)

        policy_output = policy(
            encoder_output,
            problem,
            state,
            deterministic=deterministic,
        )

        return OrchestratorOutput(
            phase=phase,
            policy=policy,
            policy_output=policy_output,
        )