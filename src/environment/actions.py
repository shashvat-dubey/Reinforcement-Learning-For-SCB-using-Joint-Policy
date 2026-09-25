"""Action definitions for the three SCB search policies."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Primitive modifications available to the joint search."""

    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_SESSION = "add_session"
    REMOVE_SESSION = "remove_session"
    STOP = "stop"


VALIDITY_ACTIONS = frozenset({
    ActionType.ADD_EDGE,
    ActionType.REMOVE_EDGE,
})

SEARCH_ACTIONS = frozenset({
    ActionType.ADD_EDGE,
    ActionType.REMOVE_EDGE,
    ActionType.ADD_SESSION,
    ActionType.REMOVE_SESSION,
    ActionType.STOP,
})


@dataclass(frozen=True)
class Action:
    """One policy action and its optional target.

    EDGE actions target an edge tuple. SESSION actions target a session tuple.
    STOP has no target.
    """

    action_type: ActionType
    target: Any = None

    def __post_init__(self) -> None:
        if self.action_type is ActionType.STOP and self.target is not None:
            raise ValueError("STOP action cannot have a target")
        if self.action_type is not ActionType.STOP and self.target is None:
            raise ValueError(f"{self.action_type.value} requires a target")


def actions_for_policy(policy: str) -> frozenset[ActionType]:
    """Return the primitive action types allowed for a policy."""

    if policy == "validity":
        return VALIDITY_ACTIONS
    if policy in {"minimization", "exploration"}:
        return SEARCH_ACTIONS
    raise ValueError(f"Unknown policy: {policy}")
