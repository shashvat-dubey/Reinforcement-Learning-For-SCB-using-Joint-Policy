from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.actions import (
    Action,
    ActionType,
    actions_for_policy,
)


EDGE = ("a", "b")
SESSION = ("a", "c")


def test_validity_policy_has_only_edge_actions():
    assert actions_for_policy("validity") == frozenset({
        ActionType.ADD_EDGE,
        ActionType.REMOVE_EDGE,
    })


def test_search_policies_have_full_action_space():
    expected = frozenset({
        ActionType.ADD_EDGE,
        ActionType.REMOVE_EDGE,
        ActionType.ADD_SESSION,
        ActionType.REMOVE_SESSION,
        ActionType.STOP,
    })

    assert actions_for_policy("minimization") == expected
    assert actions_for_policy("exploration") == expected


def test_actions_store_targets():
    assert Action(ActionType.ADD_EDGE, EDGE).target == EDGE
    assert Action(ActionType.ADD_SESSION, SESSION).target == SESSION
    assert Action(ActionType.STOP).target is None


def test_stop_cannot_have_target():
    with pytest.raises(ValueError):
        Action(ActionType.STOP, EDGE)


def test_non_stop_action_requires_target():
    with pytest.raises(ValueError):
        Action(ActionType.ADD_EDGE)
