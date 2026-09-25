from dataclasses import dataclass
from typing import Optional

from .candidate import Candidate


@dataclass
class SCBState:
    """
    Represents the current state of the SCB search.

    The state is intentionally kept small for now.
    Controller-specific information such as stall counters
    and retry counters will be added later.
    """

    candidate: Candidate
    valid: bool
    scb: Optional[float]
    cut_size: int
    selected_session_count: int
    separated_sessions: int

    def copy(self) -> "SCBState":
        """Return an independent copy of the state."""
        return SCBState(
            candidate=self.candidate.copy(),
            valid=self.valid,
            scb=self.scb,
            cut_size=self.cut_size,
            selected_session_count=self.selected_session_count,
            separated_sessions=self.separated_sessions,
        )