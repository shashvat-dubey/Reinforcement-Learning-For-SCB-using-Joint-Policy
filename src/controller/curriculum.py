from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from src.data.graph_loader import GraphDataset


class CurriculumLevel(IntEnum):
    """
    Difficulty levels for curriculum learning.

    The dataset naturally divides into four approximately
    equal-sized node bands.
    """

    L0 = 0  # 5-8 nodes
    L1 = 1  # 9-12 nodes
    L2 = 2  # 13-16 nodes
    L3 = 3  # 17-20 nodes


@dataclass(frozen=True)
class CurriculumSpec:
    """Definition of one curriculum level."""

    level: CurriculumLevel
    min_nodes: int
    max_nodes: int


CURRICULUM_SPECS = {
    CurriculumLevel.L0: CurriculumSpec(
        level=CurriculumLevel.L0,
        min_nodes=5,
        max_nodes=8,
    ),
    CurriculumLevel.L1: CurriculumSpec(
        level=CurriculumLevel.L1,
        min_nodes=9,
        max_nodes=12,
    ),
    CurriculumLevel.L2: CurriculumSpec(
        level=CurriculumLevel.L2,
        min_nodes=13,
        max_nodes=16,
    ),
    CurriculumLevel.L3: CurriculumSpec(
        level=CurriculumLevel.L3,
        min_nodes=17,
        max_nodes=20,
    ),
}


@dataclass
class CurriculumEvaluation:
    """
    Result of evaluating the RL agent on one curriculum level.
    """

    level: CurriculumLevel
    total_graphs: int
    evaluated_graphs: int
    passed_graphs: int
    failed_graphs: int

    @property
    def all_passed(self) -> bool:
        """
        Curriculum mastery condition.

        The level is mastered only when every evaluation graph passes.
        """
        return (
            self.evaluated_graphs > 0
            and self.passed_graphs == self.evaluated_graphs
        )

    @property
    def pass_rate(self) -> float:
        if self.evaluated_graphs == 0:
            return 0.0

        return self.passed_graphs / self.evaluated_graphs


class CurriculumManager:
    """
    Controls graph difficulty during RL training.

    Responsibilities:
        - Assign graphs to curriculum levels.
        - Split each level into training/evaluation graphs.
        - Track the current curriculum level.
        - Compare RL SCB against GA SCB.
        - Decide whether the current level has been mastered.

    It does NOT:
        - calculate RL rewards,
        - select actions,
        - run policies,
        - run PPO,
        - modify GA results,
        - control the environment.
    """

    def __init__(
        self,
        dataset: GraphDataset,
        evaluation_fraction: float = 0.20,
    ):
        if not 0.0 < evaluation_fraction < 1.0:
            raise ValueError(
                "evaluation_fraction must be between 0 and 1."
            )

        self.dataset = dataset
        self.evaluation_fraction = evaluation_fraction

        self.current_level = CurriculumLevel.L0

        self._level_indices = self._build_level_indices()

        (
            self._training_indices,
            self._evaluation_indices,
        ) = self._build_splits()

    # ------------------------------------------------------------------
    # Curriculum structure
    # ------------------------------------------------------------------

    def _build_level_indices(self) -> dict[CurriculumLevel, list[int]]:
        """
        Assign every dataset graph to exactly one curriculum level.
        """

        level_indices = {
            level: []
            for level in CurriculumLevel
        }

        for index in range(len(self.dataset)):
            graph = self.dataset.graph(index)

            node_count = len(graph.nodes)

            for level, spec in CURRICULUM_SPECS.items():
                if spec.min_nodes <= node_count <= spec.max_nodes:
                    level_indices[level].append(index)
                    break
            else:
                raise ValueError(
                    f"Graph at dataset index {index} has "
                    f"{node_count} nodes and does not belong to any "
                    f"curriculum level."
                )

        return level_indices

    def _build_splits(
        self,
    ) -> tuple[
        dict[CurriculumLevel, list[int]],
        dict[CurriculumLevel, list[int]],
    ]:
        """
        Deterministically split every curriculum level.

        The dataset order is preserved. The first portion is used for
        training and the final portion is held out for evaluation.

        No randomness is used here so experiments are reproducible.
        """

        training_indices = {}
        evaluation_indices = {}

        for level in CurriculumLevel:
            indices = self._level_indices[level]

            if not indices:
                raise ValueError(
                    f"Curriculum level {level.name} contains no graphs."
                )

            evaluation_count = max(
                1,
                int(len(indices) * self.evaluation_fraction),
            )

            split_index = len(indices) - evaluation_count

            training_indices[level] = indices[:split_index]
            evaluation_indices[level] = indices[split_index:]

        return training_indices, evaluation_indices

    # ------------------------------------------------------------------
    # Current level
    # ------------------------------------------------------------------

    def get_current_level(self) -> CurriculumLevel:
        return self.current_level

    def get_current_spec(self) -> CurriculumSpec:
        return CURRICULUM_SPECS[self.current_level]

    def is_final_level(self) -> bool:
        return self.current_level == CurriculumLevel.L3

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    def get_training_indices(
        self,
        level: Optional[CurriculumLevel] = None,
    ) -> list[int]:
        """
        Return graph indices available for training.
        """

        if level is None:
            level = self.current_level

        return list(self._training_indices[level])

    def get_evaluation_indices(
        self,
        level: Optional[CurriculumLevel] = None,
    ) -> list[int]:
        """
        Return held-out graph indices used for curriculum evaluation.
        """

        if level is None:
            level = self.current_level

        return list(self._evaluation_indices[level])

    def get_training_graph(
        self,
        position: int,
        level: Optional[CurriculumLevel] = None,
    ):
        indices = self.get_training_indices(level)

        if position < 0 or position >= len(indices):
            raise IndexError(
                f"Training position {position} is out of range."
            )

        return self.dataset.graph(indices[position])

    def get_evaluation_graph(
        self,
        position: int,
        level: Optional[CurriculumLevel] = None,
    ):
        indices = self.get_evaluation_indices(level)

        if position < 0 or position >= len(indices):
            raise IndexError(
                f"Evaluation position {position} is out of range."
            )

        return self.dataset.graph(indices[position])

    # ------------------------------------------------------------------
    # GA benchmark
    # ------------------------------------------------------------------

    def get_ga_scb(self, dataset_index: int) -> float:
        """
        Return the stored GA SCB benchmark for a graph.
        """

        record = self.dataset[dataset_index]

        ga_scb = record.get("ga_scb")

        if ga_scb is None:
            raise ValueError(
                f"Graph {dataset_index} does not contain a GA SCB benchmark."
            )

        return float(ga_scb)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def graph_passes(
        self,
        dataset_index: int,
        rl_scb: float,
    ) -> bool:
        """
        Determine whether the RL result matches or beats the GA result.

        SCB is a minimization objective:

            RL SCB <= GA SCB

        is considered a pass.
        """

        if rl_scb is None:
            return False

        ga_scb = self.get_ga_scb(dataset_index)

        return float(rl_scb) <= ga_scb

    def evaluate_level(
        self,
        rl_results: dict[int, float],
        level: Optional[CurriculumLevel] = None,
    ) -> CurriculumEvaluation:
        """
        Evaluate RL SCB results against GA SCB for every held-out
        evaluation graph in a curriculum level.

        Parameters
        ----------
        rl_results:
            Mapping:

                dataset_index -> RL-produced SCB

            Every evaluation graph must have an entry.

        level:
            Level to evaluate. Defaults to the current level.
        """

        if level is None:
            level = self.current_level

        evaluation_indices = self.get_evaluation_indices(level)

        passed = 0
        failed = 0

        for index in evaluation_indices:
            if index not in rl_results:
                raise ValueError(
                    f"Missing RL result for evaluation graph {index}."
                )

            rl_scb = rl_results[index]

            if self.graph_passes(index, rl_scb):
                passed += 1
            else:
                failed += 1

        return CurriculumEvaluation(
            level=level,
            total_graphs=len(self._level_indices[level]),
            evaluated_graphs=len(evaluation_indices),
            passed_graphs=passed,
            failed_graphs=failed,
        )

    # ------------------------------------------------------------------
    # Advancement
    # ------------------------------------------------------------------

    def should_advance(
        self,
        evaluation: CurriculumEvaluation,
    ) -> bool:
        """
        Determine whether the current curriculum level has been mastered.

        Mastery requires:

            RL SCB <= GA SCB

        on EVERY held-out evaluation graph.
        """

        if evaluation.level != self.current_level:
            raise ValueError(
                "Evaluation result does not belong to the current "
                "curriculum level."
            )

        if self.is_final_level():
            return False

        return evaluation.all_passed

    def advance(self) -> CurriculumLevel:
        """
        Move to the next curriculum level.

        Advancement is explicit. The caller should first call
        should_advance().
        """

        if self.is_final_level():
            raise RuntimeError(
                "Cannot advance beyond the final curriculum level."
            )

        self.current_level = CurriculumLevel(
            self.current_level + 1
        )

        return self.current_level

    def reset(self):
        """
        Reset curriculum progression back to L0.
        """

        self.current_level = CurriculumLevel.L0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a compact description of the curriculum structure.
        """

        result = {
            "current_level": self.current_level.name,
            "levels": {},
        }

        for level in CurriculumLevel:
            result["levels"][level.name] = {
                "min_nodes": CURRICULUM_SPECS[level].min_nodes,
                "max_nodes": CURRICULUM_SPECS[level].max_nodes,
                "total_graphs": len(self._level_indices[level]),
                "training_graphs": len(self._training_indices[level]),
                "evaluation_graphs": len(self._evaluation_indices[level]),
            }

        return result
