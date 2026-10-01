import torch

from src.controller.controller import ControllerPhase
from src.environment.actions import Action, ActionType
from src.environment.candidate import Candidate
from src.environment.state import SCBState
from src.training.rollout_buffer import RolloutTransition
from src.training.training_stats import TrainingStats


def make_state(
    *,
    valid: bool,
    scb,
    cut_edges=(),
    selected_sessions=(),
    separated_sessions=(),
):
    candidate = Candidate.from_sets(
        selected_sessions=selected_sessions,
        cut_edges=cut_edges,
    )

    return SCBState(
        candidate=candidate,
        valid=valid,
        scb=scb,
        cut_size=len(cut_edges),
        selected_session_count=len(
            selected_sessions
        ),
        separated_sessions=separated_sessions,
    )


def make_transition(
    state,
    next_state,
    action_type,
    target=None,
    phase=ControllerPhase.MINIMIZATION,
    reward=0.0,
    entropy=1.0,
    action_index=0,
    done=False,
):
    return RolloutTransition(
        state=state,
        action=Action(
            action_type=action_type,
            target=target,
        ),
        action_index=action_index,
        reward=reward,
        next_state=next_state,
        done=done,
        log_prob=torch.tensor(-0.5),
        entropy=torch.tensor(entropy),
        phase=phase,
    )


def test_episode_statistics():

    s0 = make_state(
        valid=False,
        scb=None,
        selected_sessions={
            ("v1", "v2"),
        },
        separated_sessions=(),
    )

    s1 = make_state(
        valid=True,
        scb=10.0,
        selected_sessions={
            ("v1", "v2"),
        },
        cut_edges={
            ("v1", "v3"),
        },
        separated_sessions=1,
    )

    s2 = make_state(
        valid=True,
        scb=8.0,
        selected_sessions={
            ("v1", "v2"),
        },
        cut_edges={
            ("v1", "v3"),
            ("v2", "v4"),
        },
        separated_sessions=2,
    )

    transitions = [
        make_transition(
            s0,
            s1,
            ActionType.ADD_EDGE,
            target=("v1", "v3"),
            phase=ControllerPhase.VALIDITY,
            reward=1.0,
        ),
        make_transition(
            s1,
            s2,
            ActionType.ADD_EDGE,
            target=("v2", "v4"),
            phase=ControllerPhase.MINIMIZATION,
            reward=2.0,
        ),
        make_transition(
            s2,
            s2,
            ActionType.STOP,
            phase=ControllerPhase.MINIMIZATION,
            reward=0.5,
            done=True,
        ),
    ]

    stats_collector = TrainingStats()

    stats = stats_collector.record_episode(
        episode=1,
        transitions=transitions,
    )

    assert stats.steps == 3

    assert stats.initial_valid is False
    assert stats.final_valid is True

    assert stats.initial_scb is None
    assert stats.best_scb == 8.0
    assert stats.final_scb == 8.0

    assert stats.validity_steps == 1
    assert stats.minimization_steps == 2

    assert stats.valid_steps == 3
    assert stats.invalid_steps == 0

    assert stats.add_edge_actions == 2
    assert stats.stop_actions == 1

    assert stats.cut_size == 2
    assert stats.separated_sessions == 2

    assert stats.reward_sum == 3.5
    assert stats.reward_mean == 3.5 / 3

    assert stats.mean_entropy == 1.0

    assert len(stats_collector.history) == 1


def test_training_summary():

    collector = TrainingStats()

    for episode in range(1, 4):

        state = make_state(
            valid=True,
            scb=10.0,
            selected_sessions={
                ("v1", "v2"),
            },
            separated_sessions=1,
        )

        next_state = make_state(
            valid=True,
            scb=float(10 - episode),
            selected_sessions={
                ("v1", "v2"),
            },
            cut_edges={
                ("v1", "v3"),
            },
            separated_sessions=1,
        )

        transition = make_transition(
            state,
            next_state,
            ActionType.ADD_EDGE,
            target=("v1", "v3"),
            phase=ControllerPhase.MINIMIZATION,
            reward=float(episode),
        )

        collector.record_episode(
            episode=episode,
            transitions=[transition],
        )

    summary = collector.summary()

    assert summary["episodes"] == 3
    assert summary["validity_rate"] == 1.0

    assert summary["mean_best_scb"] == 8.0
    assert summary["mean_scb_improvement"] == 2.0

    assert summary["mean_reward"] == 2.0


def test_recent_history():

    collector = TrainingStats()

    for episode in range(1, 6):

        state = make_state(
            valid=True,
            scb=float(episode),
            selected_sessions=(),
            separated_sessions=0,
        )

        collector.record_episode(
            episode=episode,
            transitions=[],
        )

    recent = collector.recent(2)

    assert len(recent) == 2
    assert recent[0].episode == 4
    assert recent[1].episode == 5


def test_empty_summary():

    collector = TrainingStats()

    assert collector.summary() == {}