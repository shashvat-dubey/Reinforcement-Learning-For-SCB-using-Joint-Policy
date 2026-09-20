from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.graph_loader import GraphDataset


dataset = GraphDataset(ROOT / "data" / "raw" / "labelled_dataset.pkl")
print(f"Graphs: {len(dataset)}")

for index in [0, len(dataset) // 2, len(dataset) - 1]:
    graph, benchmark = dataset.graph_and_benchmark(index)
    print(
        f"Graph {graph.graph_id}: "
        f"nodes={graph.num_nodes}, edges={graph.num_edges}, "
        f"sessions={graph.num_sessions}, GA_SCB={benchmark.scb}"
    )
