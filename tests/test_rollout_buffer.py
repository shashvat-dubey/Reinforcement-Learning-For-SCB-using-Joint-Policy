import torch

from src.controller.controller import ControllerPhase
from src.environment.actions import Action, ActionType
from src.environment.candidate import Candidate
from src.environment.state import SCBState
from src.training.rollout_buffer import (
    RolloutBuffer,
    RolloutTransition,
)


# ============================================================
# HELPERS
# ============================================================

def make_state(
    *,
    valid=False,
    scb=None,
):
    candidate = Candidate(
        selected_sessions={
            ("v1", "v2")
        },
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


def make_transition(
    phase,
    reward=0.0,
):
    state = make_state(
        valid=True,
        scb=1.0,
    )

    next_state = make_state(
        valid=True,
        scb=0.9,
    )

    action = Action(
        ActionType.ADD_EDGE,
        ("v1", "v2"),
    )

    return {
        "state": state,
        "action": action,
        "action_index": 0,
        "reward": reward,
        "next_state": next_state,
        "done": False,
        "log_prob": torch.tensor(-0.5),
        "entropy": torch.tensor(0.8),
        "phase": phase,
    }


# ============================================================
# 1. EMPTY BUFFER
# ============================================================

def test_rollout_buffer_starts_empty():

    buffer = RolloutBuffer()

    assert len(buffer) == 0
    assert buffer.summary() == {
        "total": 0,
        "validity": 0,
        "minimization": 0,
        "global_exploration": 0,
    }


# ============================================================
# 2. ADD TRANSITION
# ============================================================

def test_add_transition():

    buffer = RolloutBuffer()

    transition = make_transition(
        ControllerPhase.VALIDITY,
        reward=1.0,
    )

    buffer.add(**transition)

    assert len(buffer) == 1

    stored = buffer[0]

    assert isinstance(
        stored,
        RolloutTransition,
    )

    assert stored.reward == 1.0
    assert stored.action_index == 0
    assert stored.phase == ControllerPhase.VALIDITY
    assert stored.done is False


# ============================================================
# 3. TRANSITION STATE IS COPIED
# ============================================================

def test_transition_state_is_copied():

    buffer = RolloutBuffer()

    state = make_state(
        valid=True,
        scb=1.0,
    )

    next_state = make_state(
        valid=True,
        scb=0.9,
    )

    action = Action(
        ActionType.ADD_EDGE,
        ("v1", "v2"),
    )

    buffer.add(
        state=state,
        action=action,
        action_index=0,
        reward=1.0,
        next_state=next_state,
        done=False,
        log_prob=torch.tensor(-0.5),
        entropy=torch.tensor(0.8),
        phase=ControllerPhase.MINIMIZATION,
    )

    # Mutate the original state.
    state.candidate.add_edge(
        ("v2", "v3")
    )

    stored = buffer[0]

    assert (
        ("v2", "v3")
        not in stored.state.candidate.cut_edges
    )


# ============================================================
# 4. LOG PROB / ENTROPY ARE DETACHED
# ============================================================

def test_log_prob_and_entropy_are_detached():

    buffer = RolloutBuffer()

    log_prob = torch.tensor(
        -0.5,
        requires_grad=True,
    )

    entropy = torch.tensor(
        0.8,
        requires_grad=True,
    )

    transition = make_transition(
        ControllerPhase.MINIMIZATION
    )

    # Replace the values already present in the
    # transition dictionary instead of passing duplicates.
    transition["log_prob"] = log_prob
    transition["entropy"] = entropy

    buffer.add(**transition)

    stored = buffer[0]

    assert stored.log_prob.requires_grad is False
    assert stored.entropy.requires_grad is False

    assert torch.isclose(
        stored.log_prob,
        torch.tensor(-0.5),
    )

    assert torch.isclose(
        stored.entropy,
        torch.tensor(0.8),
    )

# ============================================================
# 5. PHASE FILTERING
# ============================================================

def test_phase_filtering():

    buffer = RolloutBuffer()

    buffer.add(
        **make_transition(
            ControllerPhase.VALIDITY,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.MINIMIZATION,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.LOCAL_RETRY,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.GLOBAL_EXPLORATION,
        )
    )

    assert len(
        buffer.validity_transitions()
    ) == 1

    # MINIMIZATION + LOCAL_RETRY belong
    # to the minimization policy.
    assert len(
        buffer.minimization_transitions()
    ) == 2

    assert len(
        buffer.global_exploration_transitions()
    ) == 1


# ============================================================
# 6. SUMMARY
# ============================================================

def test_summary():

    buffer = RolloutBuffer()

    buffer.add(
        **make_transition(
            ControllerPhase.VALIDITY,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.MINIMIZATION,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.MINIMIZATION,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.LOCAL_RETRY,
        )
    )

    buffer.add(
        **make_transition(
            ControllerPhase.GLOBAL_EXPLORATION,
        )
    )

    assert buffer.summary() == {
        "total": 5,
        "validity": 1,
        "minimization": 3,
        "global_exploration": 1,
    }


# ============================================================
# 7. CLEAR
# ============================================================

def test_clear():

    buffer = RolloutBuffer()

    buffer.add(
        **make_transition(
            ControllerPhase.VALIDITY,
        )
    )

    assert len(buffer) == 1

    buffer.clear()

    assert len(buffer) == 0


# ============================================================
# 8. DONE TRANSITION
# ============================================================

def test_done_transition_is_preserved():

    buffer = RolloutBuffer()

    transition = make_transition(
        ControllerPhase.MINIMIZATION,
    )

    transition["done"] = True

    buffer.add(**transition)

    assert buffer[0].done is True