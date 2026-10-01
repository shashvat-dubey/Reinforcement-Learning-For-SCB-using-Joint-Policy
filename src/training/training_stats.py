from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import math

from src.controller.controller import ControllerPhase
from src.environment.actions import ActionType
from src.training.rollout_buffer import RolloutTransition


@dataclass
class EpisodeStats:
    """Statistics collected from one complete RL episode."""

    episode: int

    steps: int = 0

    initial_valid: bool = False
    final_valid: bool = False

    initial_scb: Optional[float] = None
    final_scb: Optional[float] = None
    best_scb: Optional[float] = None

    scb_improvement: Optional[float] = None

    cut_size: int = 0
    selected_sessions: int = 0
    separated_sessions: int = 0

    validity_steps: int = 0
    minimization_steps: int = 0
    local_retry_steps: int = 0
    global_exploration_steps: int = 0

    invalid_steps: int = 0
    valid_steps: int = 0

    stop_actions: int = 0

    add_edge_actions: int = 0
    remove_edge_actions: int = 0

    add_session_actions: int = 0
    remove_session_actions: int = 0

    reward_sum: float = 0.0
    reward_mean: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    mean_entropy: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "steps": self.steps,
            "initial_valid": self.initial_valid,
            "final_valid": self.final_valid,
            "initial_scb": self.initial_scb,
            "final_scb": self.final_scb,
            "best_scb": self.best_scb,
            "scb_improvement": self.scb_improvement,
            "cut_size": self.cut_size,
            "selected_sessions": self.selected_sessions,
            "separated_sessions": self.separated_sessions,
            "validity_steps": self.validity_steps,
            "minimization_steps": self.minimization_steps,
            "local_retry_steps": self.local_retry_steps,
            "global_exploration_steps": self.global_exploration_steps,
            "invalid_steps": self.invalid_steps,
            "valid_steps": self.valid_steps,
            "stop_actions": self.stop_actions,
            "add_edge_actions": self.add_edge_actions,
            "remove_edge_actions": self.remove_edge_actions,
            "add_session_actions": self.add_session_actions,
            "remove_session_actions": self.remove_session_actions,
            "reward_sum": self.reward_sum,
            "reward_mean": self.reward_mean,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "mean_entropy": self.mean_entropy,
        }


@dataclass
class TrainingStats:
    """
    Aggregates episode statistics across PPO training.

    This class is deliberately independent of PPO optimization.
    It only observes the rollout and reports what happened.
    """

    history: list[EpisodeStats] = field(default_factory=list)

    # ---------------------------------------------------------
    # EPISODE
    # ---------------------------------------------------------

    def record_episode(
        self,
        episode: int,
        transitions: list[RolloutTransition],
    ) -> EpisodeStats:

        if not transitions:
            stats = EpisodeStats(
                episode=episode,
            )

            self.history.append(stats)

            return stats

        first = transitions[0]
        last = transitions[-1]

        stats = EpisodeStats(
            episode=episode,
            steps=len(transitions),
            initial_valid=bool(first.state.valid),
            final_valid=bool(last.next_state.valid),
            initial_scb=first.state.scb,
            final_scb=last.next_state.scb,
        )

        # -----------------------------------------------------
        # SCB
        # -----------------------------------------------------

        scbs = []

        for transition in transitions:

            if transition.state.scb is not None:
                scbs.append(
                    float(transition.state.scb)
                )

            if transition.next_state.scb is not None:
                scbs.append(
                    float(transition.next_state.scb)
                )

        if scbs:
            stats.best_scb = min(scbs)

        if (
            stats.initial_scb is not None
            and stats.best_scb is not None
        ):
            stats.scb_improvement = (
                stats.initial_scb
                - stats.best_scb
            )

        # -----------------------------------------------------
        # PHASES
        # -----------------------------------------------------

        for transition in transitions:

            if transition.phase == ControllerPhase.VALIDITY:
                stats.validity_steps += 1

            elif transition.phase == ControllerPhase.MINIMIZATION:
                stats.minimization_steps += 1

            elif transition.phase == ControllerPhase.LOCAL_RETRY:
                stats.local_retry_steps += 1

            elif transition.phase == ControllerPhase.GLOBAL_EXPLORATION:
                stats.global_exploration_steps += 1

        # -----------------------------------------------------
        # VALIDITY
        # -----------------------------------------------------

        stats.valid_steps = sum(
            int(t.next_state.valid)
            for t in transitions
        )

        stats.invalid_steps = (
            stats.steps
            - stats.valid_steps
        )

        # -----------------------------------------------------
        # FINAL CANDIDATE
        # -----------------------------------------------------

        final_state = last.next_state

        stats.cut_size = (
            final_state.candidate.num_cut_edges
        )

        stats.selected_sessions = (
            final_state.candidate.num_selected_sessions
        )

        separated = final_state.separated_sessions

        if isinstance(separated, int):
            stats.separated_sessions = separated
        else:
            stats.separated_sessions = len(
                separated
            )

        # -----------------------------------------------------
        # ACTIONS
        # -----------------------------------------------------

        for transition in transitions:

            action_type = (
                transition.action.action_type
            )

            if action_type == ActionType.ADD_EDGE:
                stats.add_edge_actions += 1

            elif action_type == ActionType.REMOVE_EDGE:
                stats.remove_edge_actions += 1

            elif action_type == ActionType.ADD_SESSION:
                stats.add_session_actions += 1

            elif action_type == ActionType.REMOVE_SESSION:
                stats.remove_session_actions += 1

            elif action_type == ActionType.STOP:
                stats.stop_actions += 1

        # -----------------------------------------------------
        # REWARDS
        # -----------------------------------------------------

        rewards = [
            float(t.reward)
            for t in transitions
        ]

        if rewards:
            stats.reward_sum = sum(rewards)
            stats.reward_mean = (
                stats.reward_sum / len(rewards)
            )
            stats.reward_min = min(rewards)
            stats.reward_max = max(rewards)

        # -----------------------------------------------------
        # ENTROPY
        # -----------------------------------------------------

        entropies = []

        for transition in transitions:

            if transition.entropy is None:
                continue

            value = float(
                transition.entropy.detach().cpu().item()
            )

            if math.isfinite(value):
                entropies.append(value)

        if entropies:
            stats.mean_entropy = (
                sum(entropies)
                / len(entropies)
            )

        self.history.append(stats)

        return stats

    # ---------------------------------------------------------
    # RECENT
    # ---------------------------------------------------------

    def recent(
        self,
        n: int = 10,
    ) -> list[EpisodeStats]:

        if n <= 0:
            return []

        return self.history[-n:]

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    def summary(
        self,
        n: Optional[int] = None,
    ) -> dict[str, float]:

        episodes = (
            self.history
            if n is None
            else self.recent(n)
        )

        if not episodes:
            return {}

        def mean(
            values: list[float],
        ) -> float:

            if not values:
                return 0.0

            return sum(values) / len(values)

        improvements = [
            float(x.scb_improvement)
            for x in episodes
            if x.scb_improvement is not None
        ]

        best_scbs = [
            float(x.best_scb)
            for x in episodes
            if x.best_scb is not None
        ]

        final_scbs = [
            float(x.final_scb)
            for x in episodes
            if x.final_scb is not None
        ]

        return {
            "episodes": float(len(episodes)),

            "validity_rate": mean([
                float(x.final_valid)
                for x in episodes
            ]),

            "mean_steps": mean([
                float(x.steps)
                for x in episodes
            ]),

            "mean_best_scb": mean(best_scbs),

            "mean_final_scb": mean(final_scbs),

            "mean_scb_improvement": mean(
                improvements
            ),

            "mean_cut_size": mean([
                float(x.cut_size)
                for x in episodes
            ]),

            "mean_reward": mean([
                x.reward_mean
                for x in episodes
            ]),

            "mean_entropy": mean([
                x.mean_entropy
                for x in episodes
            ]),
        }

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    @staticmethod
    def print_episode(
        stats: EpisodeStats,
    ) -> None:

        def fmt(value):
            if value is None:
                return "--"

            if isinstance(value, float):
                return f"{value:.4f}"

            return str(value)

        print()
        print(
            f"EPISODE {stats.episode}"
        )
        print("─" * 54)

        print(
            f"Steps        : {stats.steps}"
        )

        print(
            f"Phase        : "
            f"V={stats.validity_steps} "
            f"M={stats.minimization_steps} "
            f"L={stats.local_retry_steps} "
            f"G={stats.global_exploration_steps}"
        )

        print(
            f"Valid        : "
            f"{stats.initial_valid} → "
            f"{stats.final_valid}"
        )

        print(
            f"SCB          : "
            f"{fmt(stats.initial_scb)} → "
            f"{fmt(stats.best_scb)} → "
            f"{fmt(stats.final_scb)}"
        )

        print(
            f"Improvement  : "
            f"{fmt(stats.scb_improvement)}"
        )

        print(
            f"Cut          : "
            f"{stats.cut_size} edges"
        )

        print(
            f"Sessions     : "
            f"{stats.selected_sessions} selected, "
            f"{stats.separated_sessions} separated"
        )

        print(
            f"Actions      : "
            f"+E={stats.add_edge_actions} "
            f"-E={stats.remove_edge_actions} "
            f"+S={stats.add_session_actions} "
            f"-S={stats.remove_session_actions} "
            f"STOP={stats.stop_actions}"
        )

        print(
            f"Reward       : "
            f"sum={stats.reward_sum:.4f} "
            f"mean={stats.reward_mean:.4f}"
        )

        print(
            f"Entropy      : "
            f"{stats.mean_entropy:.4f}"
        )

        print("─" * 54)

    def print_summary(
        self,
        n: Optional[int] = None,
    ) -> None:

        summary = self.summary(n)

        if not summary:
            print("No training statistics recorded.")
            return

        print()
        print("PPO TRAINING SUMMARY")
        print("═" * 54)

        print(
            f"Episodes       : "
            f"{int(summary['episodes'])}"
        )

        print(
            f"Validity rate  : "
            f"{summary['validity_rate'] * 100:.2f}%"
        )

        print(
            f"Mean steps     : "
            f"{summary['mean_steps']:.2f}"
        )

        print(
            f"Mean best SCB  : "
            f"{summary['mean_best_scb']:.4f}"
        )

        print(
            f"Mean final SCB : "
            f"{summary['mean_final_scb']:.4f}"
        )

        print(
            f"Mean ΔSCB      : "
            f"{summary['mean_scb_improvement']:.4f}"
        )

        print(
            f"Mean cut size  : "
            f"{summary['mean_cut_size']:.2f}"
        )

        print(
            f"Mean reward    : "
            f"{summary['mean_reward']:.4f}"
        )

        print(
            f"Mean entropy   : "
            f"{summary['mean_entropy']:.4f}"
        )

        print("═" * 54)