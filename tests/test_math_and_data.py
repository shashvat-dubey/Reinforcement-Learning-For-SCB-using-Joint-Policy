from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.graph_loader import GraphDataset
from src.scb.problem import SCBProblem
from src.scb.evaluator import evaluate_candidate
from src.scb.scb import calculate_scb


DATASET = ROOT / "data" / "raw" / "labelled_dataset.pkl"


def test_dataset_has_1000_graphs():
    dataset = GraphDataset(DATASET)
    assert len(dataset) == 1000


def test_graph_and_benchmark_are_separate():
    dataset = GraphDataset(DATASET)
    graph, benchmark = dataset.graph_and_benchmark(0)

    assert graph.graph_id == benchmark.graph_id == 0
    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert graph.num_sessions > 0
    assert benchmark.scb == dataset[0]["ga_scb"]


def test_scb_formula():
    assert calculate_scb(6, 3) == 2.0


def test_simple_candidate_is_valid():
    problem = SCBProblem(
        nodes=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c")],
        sessions=[("a", "c")],
    )

    result = evaluate_candidate(
        problem,
        selected_sessions={("a", "c")},
        cut_edges={("a", "b")},
    )

    assert result.valid
    assert result.scb == 1.0


def test_unseparated_candidate_is_invalid():
    problem = SCBProblem(
        nodes=["a", "b", "c"],
        edges=[("a", "b"), ("b", "c")],
        sessions=[("a", "c")],
    )

    result = evaluate_candidate(
        problem,
        selected_sessions={("a", "c")},
        cut_edges=set(),
    )

    assert not result.valid
    assert result.scb is None
