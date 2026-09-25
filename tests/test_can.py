from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.candidate import Candidate


SESSION_A = ("a", "d")
SESSION_B = ("b", "c")
EDGE_A = ("a", "b")
EDGE_B = ("c", "d")


def test_candidate_stores_joint_solution():
    candidate = Candidate.from_sets({SESSION_A}, {EDGE_A})

    assert candidate.selected_sessions == {SESSION_A}
    assert candidate.cut_edges == {EDGE_A}
    assert candidate.num_selected_sessions == 1
    assert candidate.num_cut_edges == 1


def test_candidate_can_modify_sessions_and_edges():
    candidate = Candidate()

    candidate.add_session(SESSION_A)
    candidate.add_edge(EDGE_A)
    candidate.add_edge(EDGE_B)
    candidate.remove_edge(EDGE_A)
    candidate.remove_session(SESSION_A)

    assert candidate.selected_sessions == set()
    assert candidate.cut_edges == {EDGE_B}


def test_candidate_copy_is_independent():
    original = Candidate.from_sets({SESSION_A}, {EDGE_A})
    copied = original.copy()

    copied.add_session(SESSION_B)
    copied.add_edge(EDGE_B)

    assert SESSION_B not in original.selected_sessions
    assert EDGE_B not in original.cut_edges
