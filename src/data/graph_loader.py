"""Loader for the existing labelled_dataset.pkl."""

import pickle
from pathlib import Path

from .graph_schema import GABenchmark, GraphInstance


class GraphDataset:
    """Loads the existing 1,000-graph dataset without modifying it."""

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        with self.dataset_path.open("rb") as f:
            self.records = pickle.load(f)

        if not isinstance(self.records, list):
            raise TypeError("Expected labelled dataset to contain a list of graph records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

    def graph(self, index: int) -> GraphInstance:
        return GraphInstance.from_record(self.records[index])

    def benchmark(self, index: int) -> GABenchmark:
        return GABenchmark.from_record(self.records[index])

    def graph_and_benchmark(self, index: int) -> tuple[GraphInstance, GABenchmark]:
        record = self.records[index]
        return GraphInstance.from_record(record), GABenchmark.from_record(record)
