"""Sparsest Cut Bound calculations."""


def calculate_scb(cut_size: int, selected_session_count: int) -> float:
    """Calculate |E'| / |I'| for a valid non-empty candidate."""
    if selected_session_count <= 0:
        raise ValueError("SCB is undefined for an empty selected-session set")
    if cut_size < 0:
        raise ValueError("cut_size cannot be negative")
    return cut_size / selected_session_count
