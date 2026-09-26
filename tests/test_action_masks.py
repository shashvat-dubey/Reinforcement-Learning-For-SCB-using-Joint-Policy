import torch

from src.environment.actions import ActionType
from src.environment.candidate import Candidate
from src.environment.state import SCBState
from src.models.policies import (
    ValidityPolicy,
    MinimizationPolicy,
    GlobalExplorationPolicy,
)
from src.models.scb_encoder import SCBEncoder
from src.scb.problem import SCBProblem


# ============================================================
# HELPERS
# ============================================================

def make_problem():
    """
    Small deterministic SCB problem used only for mask testing.
    """

    nodes = [
        "v1",
        "v2",
        "v3",
    ]

    edges = [
        ("v1", "v2"),
        ("v2", "v3"),
        ("v1", "v3"),
    ]

    sessions = [
        ("v1", "v3"),
        ("v2", "v3"),
    ]

    return SCBProblem(
        nodes=nodes,
        edges=edges,
        sessions=sessions,
    )


def make_state(
    problem,
    selected_sessions,
    cut_edges,
    valid=False,
    scb=None,
    separated_sessions=0,
):
    candidate = Candidate(
        selected_sessions=set(selected_sessions),
        cut_edges=set(cut_edges),
    )

    return SCBState(
        candidate=candidate,
        valid=valid,
        scb=scb,
        cut_size=len(cut_edges),
        selected_session_count=len(selected_sessions),
        separated_sessions=separated_sessions,
    )


def get_mask(policy, problem, state):
    """
    Call the actual policy mask implementation.
    """

    return policy._build_action_mask(
        problem,
        state,
    )


def action_ranges(problem):
    E = problem.num_edges
    S = problem.num_sessions

    return {
        "add_edge": range(0, E),
        "remove_edge": range(E, 2 * E),
        "add_session": range(
            2 * E,
            2 * E + S,
        ),
        "remove_session": range(
            2 * E + S,
            2 * E + 2 * S,
        ),
        "stop": range(
            2 * E + 2 * S,
            2 * E + 2 * S + 1,
        ),
    }


# ============================================================
# 1. VALIDITY POLICY ACTION SPACE
# ============================================================

def test_validity_policy_only_allows_edge_actions():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
            problem.sessions[1],
        },
        cut_edges=set(),
    )

    policy = ValidityPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    ranges = action_ranges(problem)

    # ADD_EDGE must have executable actions.
    assert mask[
        list(ranges["add_edge"])
    ].any()

    # REMOVE_EDGE is masked because no edge is cut.
    assert not mask[
        list(ranges["remove_edge"])
    ].any()

    # Validity policy cannot modify sessions.
    assert not mask[
        list(ranges["add_session"])
    ].any()

    assert not mask[
        list(ranges["remove_session"])
    ].any()

    # STOP is not available to validity policy.
    assert not mask[
        list(ranges["stop"])
    ].any()


# ============================================================
# 2. MINIMIZATION POLICY FULL ACTION SPACE
# ============================================================

def test_minimization_policy_allows_full_action_space():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges={
            problem.edges[0],
        },
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    ranges = action_ranges(problem)

    assert mask[
        list(ranges["add_edge"])
    ].any()

    assert mask[
        list(ranges["remove_edge"])
    ].any()

    assert mask[
        list(ranges["add_session"])
    ].any()

    assert mask[
        list(ranges["remove_session"])
    ].any()

    assert mask[
        list(ranges["stop"])
    ].any()


# ============================================================
# 3. GLOBAL EXPLORATION POLICY FULL ACTION SPACE
# ============================================================

def test_global_policy_allows_full_action_space():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges=set(),
    )

    policy = GlobalExplorationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    ranges = action_ranges(problem)

    # At least one edge can be added.
    assert mask[
        list(ranges["add_edge"])
    ].any()

    # No edges are currently cut.
    assert not mask[
        list(ranges["remove_edge"])
    ].any()

    # Session 0 is selected, session 1 is not.
    assert mask[
        list(ranges["add_session"])
    ].any()

    assert mask[
        list(ranges["remove_session"])
    ].any()

    # STOP is always available for global exploration.
    assert mask[
        list(ranges["stop"])
    ].any()


# ============================================================
# 4. ADD_EDGE MASK
# ============================================================

def test_add_edge_is_masked_when_edge_is_already_cut():

    problem = make_problem()

    cut_edge = problem.edges[0]

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges={
            cut_edge,
        },
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    E = problem.num_edges

    # Edge 0 is already cut.
    # Therefore ADD_EDGE(edge 0) is invalid.
    assert mask[0].item() is False

    # Edge 1 is not cut.
    # Therefore ADD_EDGE(edge 1) is executable.
    assert mask[1].item() is True

    # REMOVE_EDGE(edge 0) is executable.
    assert mask[E].item() is True


# ============================================================
# 5. REMOVE_EDGE MASK
# ============================================================

def test_remove_edge_is_masked_when_edge_is_not_cut():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges=set(),
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    E = problem.num_edges

    # No edges are cut.
    # Therefore every REMOVE_EDGE action is masked.
    assert not mask[
        E:2 * E
    ].any()

    # Every edge can currently be added.
    assert mask[
        0:E
    ].all()


# ============================================================
# 6. ADD_SESSION MASK
# ============================================================

def test_add_session_is_masked_for_selected_sessions():

    problem = make_problem()

    selected_session = problem.sessions[0]

    state = make_state(
        problem,
        selected_sessions={
            selected_session,
        },
        cut_edges=set(),
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    E = problem.num_edges

    add_session_start = 2 * E

    # Session 0 is already selected.
    assert mask[
        add_session_start
    ].item() is False

    # Session 1 is not selected.
    assert mask[
        add_session_start + 1
    ].item() is True


# ============================================================
# 7. REMOVE_SESSION MASK
# ============================================================

def test_remove_session_is_masked_for_unselected_sessions():

    problem = make_problem()

    selected_session = problem.sessions[0]

    state = make_state(
        problem,
        selected_sessions={
            selected_session,
        },
        cut_edges=set(),
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    E = problem.num_edges
    S = problem.num_sessions

    remove_session_start = (
        2 * E + S
    )

    # Session 0 is selected.
    assert mask[
        remove_session_start
    ].item() is True

    # Session 1 is not selected.
    assert mask[
        remove_session_start + 1
    ].item() is False


# ============================================================
# 8. STOP MASK
# ============================================================

def test_stop_is_always_available_for_minimization_and_global():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges=set(),
    )

    stop_index = (
        2 * problem.num_edges
        + 2 * problem.num_sessions
    )

    for policy in (
        MinimizationPolicy(),
        GlobalExplorationPolicy(),
    ):

        mask = get_mask(
            policy,
            problem,
            state,
        )

        assert mask[
            stop_index
        ].item() is True


# ============================================================
# 9. MASKED LOGITS MUST BE -INF
# ============================================================

def test_masked_logits_are_negative_infinity():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges=set(),
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    logits = torch.zeros(
        mask.shape,
        dtype=torch.float32,
    )

    masked_logits = policy._apply_mask(
        logits,
        mask,
    )

    # Every masked action must become -inf.
    assert torch.isneginf(
        masked_logits[~mask]
    ).all()

    # Every executable action must remain finite.
    assert torch.isfinite(
        masked_logits[mask]
    ).all()


# ============================================================
# 10. MASKED ACTIONS HAVE ZERO PROBABILITY
# ============================================================

def test_masked_actions_have_zero_probability():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges=set(),
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    logits = torch.zeros(
        mask.shape,
        dtype=torch.float32,
    )

    masked_logits = policy._apply_mask(
        logits,
        mask,
    )

    probabilities = torch.softmax(
        masked_logits,
        dim=-1,
    )

    # Masked actions must have exactly zero probability.
    assert torch.all(
        probabilities[~mask] == 0
    )

    # Executable action probabilities must sum to 1.
    assert torch.isclose(
        probabilities[mask].sum(),
        torch.tensor(1.0),
    )


# ============================================================
# 11. MASK MUST MATCH THE CANDIDATE STATE
# ============================================================

def test_mask_matches_candidate_state():

    problem = make_problem()

    cut_edge = problem.edges[1]
    selected_session = problem.sessions[0]

    state = make_state(
        problem,
        selected_sessions={
            selected_session,
        },
        cut_edges={
            cut_edge,
        },
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    E = problem.num_edges
    S = problem.num_sessions

    # --------------------------------------------------------
    # ADD_EDGE
    # --------------------------------------------------------

    for edge_idx, edge in enumerate(problem.edges):

        expected = (
            edge
            not in state.candidate.cut_edges
        )

        assert (
            mask[edge_idx].item()
            == expected
        )

    # --------------------------------------------------------
    # REMOVE_EDGE
    # --------------------------------------------------------

    for edge_idx, edge in enumerate(problem.edges):

        expected = (
            edge
            in state.candidate.cut_edges
        )

        index = E + edge_idx

        assert (
            mask[index].item()
            == expected
        )

    # --------------------------------------------------------
    # ADD_SESSION
    # --------------------------------------------------------

    for session_idx, session in enumerate(
        problem.sessions
    ):

        expected = (
            session
            not in state.candidate.selected_sessions
        )

        index = (
            2 * E
            + session_idx
        )

        assert (
            mask[index].item()
            == expected
        )

    # --------------------------------------------------------
    # REMOVE_SESSION
    # --------------------------------------------------------

    for session_idx, session in enumerate(
        problem.sessions
    ):

        expected = (
            session
            in state.candidate.selected_sessions
        )

        index = (
            2 * E
            + S
            + session_idx
        )

        assert (
            mask[index].item()
            == expected
        )


# ============================================================
# 12. POLICY NEVER SAMPLES A MASKED ACTION
# ============================================================

def test_policy_never_samples_masked_action():

    problem = make_problem()

    state = make_state(
        problem,
        selected_sessions={
            problem.sessions[0],
        },
        cut_edges={
            problem.edges[0],
        },
    )

    # --------------------------------------------------------
    # Real encoder output.
    # --------------------------------------------------------

    encoder = SCBEncoder()

    encoder_output = encoder(
        problem,
        state,
    )

    policy = MinimizationPolicy()

    mask = get_mask(
        policy,
        problem,
        state,
    )

    # --------------------------------------------------------
    # Sample repeatedly.
    # --------------------------------------------------------

    for _ in range(100):

        output = policy(
            encoder_output=encoder_output,
            problem=problem,
            state=state,
            deterministic=False,
        )

        # The sampled action MUST be executable according
        # to the current candidate state.
        assert mask[
            output.action_index
        ].item() is True