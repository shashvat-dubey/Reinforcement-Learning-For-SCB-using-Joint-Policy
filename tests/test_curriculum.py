import pytest

from src.controller.curriculum import (
    CurriculumLevel,
    CurriculumManager,
)


DATASET_PATH = "data/raw/labelled_dataset.pkl"


@pytest.fixture
def curriculum():
    from src.data.graph_loader import GraphDataset

    dataset = GraphDataset(DATASET_PATH)

    return CurriculumManager(
        dataset,
        evaluation_fraction=0.20,
    )


def test_curriculum_has_four_levels(curriculum):
    assert list(CurriculumLevel) == [
        CurriculumLevel.L0,
        CurriculumLevel.L1,
        CurriculumLevel.L2,
        CurriculumLevel.L3,
    ]


def test_current_level_starts_at_l0(curriculum):
    assert curriculum.get_current_level() == CurriculumLevel.L0


def test_all_graphs_are_assigned(curriculum):
    total = sum(
        len(curriculum.get_training_indices(level))
        + len(curriculum.get_evaluation_indices(level))
        for level in CurriculumLevel
    )

    assert total == 1000


def test_expected_level_sizes(curriculum):
    assert len(curriculum.get_training_indices(CurriculumLevel.L0)) == 206
    assert len(curriculum.get_training_indices(CurriculumLevel.L1)) == 197
    assert len(curriculum.get_training_indices(CurriculumLevel.L2)) == 202
    assert len(curriculum.get_training_indices(CurriculumLevel.L3)) == 196

    assert len(curriculum.get_evaluation_indices(CurriculumLevel.L0)) == 51
    assert len(curriculum.get_evaluation_indices(CurriculumLevel.L1)) == 49
    assert len(curriculum.get_evaluation_indices(CurriculumLevel.L2)) == 50
    assert len(curriculum.get_evaluation_indices(CurriculumLevel.L3)) == 49


def test_training_and_evaluation_sets_do_not_overlap(curriculum):
    for level in CurriculumLevel:
        training = set(curriculum.get_training_indices(level))
        evaluation = set(curriculum.get_evaluation_indices(level))

        assert training.isdisjoint(evaluation)


def test_each_graph_belongs_to_exactly_one_level(curriculum):
    memberships = {}

    for level in CurriculumLevel:
        indices = (
            curriculum.get_training_indices(level)
            + curriculum.get_evaluation_indices(level)
        )

        for index in indices:
            memberships[index] = memberships.get(index, 0) + 1

    assert len(memberships) == 1000
    assert all(count == 1 for count in memberships.values())


def test_ga_benchmark_is_available(curriculum):
    evaluation_indices = curriculum.get_evaluation_indices(
        CurriculumLevel.L0
    )

    graph_index = evaluation_indices[0]

    ga_scb = curriculum.get_ga_scb(graph_index)

    assert isinstance(ga_scb, float)
    assert ga_scb > 0.0


def test_equal_to_ga_passes(curriculum):
    graph_index = curriculum.get_evaluation_indices(
        CurriculumLevel.L0
    )[0]

    ga_scb = curriculum.get_ga_scb(graph_index)

    assert curriculum.graph_passes(
        graph_index,
        ga_scb,
    )


def test_better_than_ga_passes(curriculum):
    graph_index = curriculum.get_evaluation_indices(
        CurriculumLevel.L0
    )[0]

    ga_scb = curriculum.get_ga_scb(graph_index)

    better_scb = ga_scb * 0.99

    assert curriculum.graph_passes(
        graph_index,
        better_scb,
    )


def test_worse_than_ga_fails(curriculum):
    graph_index = curriculum.get_evaluation_indices(
        CurriculumLevel.L0
    )[0]

    ga_scb = curriculum.get_ga_scb(graph_index)

    worse_scb = ga_scb * 1.01

    assert not curriculum.graph_passes(
        graph_index,
        worse_scb,
    )


def test_all_graphs_must_pass(curriculum):
    level = CurriculumLevel.L0
    evaluation_indices = curriculum.get_evaluation_indices(level)

    ga_results = {
        index: curriculum.get_ga_scb(index)
        for index in evaluation_indices
    }

    evaluation = curriculum.evaluate_level(
        ga_results,
        level,
    )

    assert evaluation.evaluated_graphs == len(evaluation_indices)
    assert evaluation.passed_graphs == len(evaluation_indices)
    assert evaluation.failed_graphs == 0
    assert evaluation.all_passed


def test_one_failure_prevents_advancement(curriculum):
    level = CurriculumLevel.L0
    evaluation_indices = curriculum.get_evaluation_indices(level)

    rl_results = {
        index: curriculum.get_ga_scb(index)
        for index in evaluation_indices
    }

    failing_index = evaluation_indices[0]
    rl_results[failing_index] *= 1.01

    evaluation = curriculum.evaluate_level(
        rl_results,
        level,
    )

    assert evaluation.failed_graphs == 1
    assert not evaluation.all_passed
    assert not curriculum.should_advance(evaluation)


def test_successful_level_advances(curriculum):
    level = CurriculumLevel.L0
    evaluation_indices = curriculum.get_evaluation_indices(level)

    rl_results = {
        index: curriculum.get_ga_scb(index)
        for index in evaluation_indices
    }

    evaluation = curriculum.evaluate_level(
        rl_results,
        level,
    )

    assert curriculum.should_advance(evaluation)

    new_level = curriculum.advance()

    assert new_level == CurriculumLevel.L1
    assert curriculum.get_current_level() == CurriculumLevel.L1


def test_final_level_cannot_advance(curriculum):
    curriculum.current_level = CurriculumLevel.L3

    evaluation_indices = curriculum.get_evaluation_indices(
        CurriculumLevel.L3
    )

    rl_results = {
        index: curriculum.get_ga_scb(index)
        for index in evaluation_indices
    }

    evaluation = curriculum.evaluate_level(
        rl_results,
        CurriculumLevel.L3,
    )

    assert evaluation.all_passed
    assert not curriculum.should_advance(evaluation)

    with pytest.raises(RuntimeError):
        curriculum.advance()


def test_missing_evaluation_result_fails_loudly(curriculum):
    level = CurriculumLevel.L0

    with pytest.raises(ValueError):
        curriculum.evaluate_level(
            {},
            level,
        )


def test_reset_returns_to_l0(curriculum):
    curriculum.advance()

    assert curriculum.get_current_level() == CurriculumLevel.L1

    curriculum.reset()

    assert curriculum.get_current_level() == CurriculumLevel.L0


def test_summary(curriculum):
    summary = curriculum.summary()

    assert summary["current_level"] == "L0"

    assert summary["levels"]["L0"]["total_graphs"] == 257
    assert summary["levels"]["L1"]["total_graphs"] == 246
    assert summary["levels"]["L2"]["total_graphs"] == 252
    assert summary["levels"]["L3"]["total_graphs"] == 245
