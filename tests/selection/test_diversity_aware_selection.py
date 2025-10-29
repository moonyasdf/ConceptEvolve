import math

import numpy as np
import pytest

from src.concepts import AlgorithmicConcept
from src.selection import DiversityAwareSelection


def make_concept(identifier: str, combined_score: float, embedding):
    return AlgorithmicConcept(
        id=identifier,
        title=f"Concept {identifier}",
        combined_score=combined_score,
        embedding=list(embedding),
    )


def test_calculate_diversity_score_handles_missing_embeddings():
    selector = DiversityAwareSelection()
    concept = make_concept("a", 5.0, [])
    population = [concept]

    assert selector.calculate_diversity_score(concept, population) == 0.0


def test_calculate_diversity_score_singleton_population_returns_one():
    selector = DiversityAwareSelection()
    embedding = np.array([1.0, 0.0])
    concept = make_concept("solo", 5.0, embedding)
    population = [concept]

    assert selector.calculate_diversity_score(concept, population) == 1.0


def test_pareto_rank_balances_fitness_and_diversity():
    selector = DiversityAwareSelection()

    c1 = make_concept("c1", 9.0, np.array([1.0, 0.0]))
    c2 = make_concept("c2", 7.0, np.array([0.0, 1.0]))
    diag = 1 / math.sqrt(2)
    c3 = make_concept("c3", 9.0, np.array([diag, diag]))

    population = [c1, c2, c3]
    ranks = selector.pareto_rank(population)

    assert ranks == [0, 1, 1]


def test_calculate_diversity_matches_expected_average_similarity():
    selector = DiversityAwareSelection()

    c1 = make_concept("c1", 9.0, np.array([1.0, 0.0]))
    c2 = make_concept("c2", 9.0, np.array([0.0, 1.0]))
    diag = 1 / math.sqrt(2)
    c3 = make_concept("c3", 9.0, np.array([diag, diag]))

    population = [c1, c2, c3]
    diversity_c1 = selector.calculate_diversity_score(c1, population)

    avg_similarity = np.mean(
        [
            np.dot(c1.embedding, c2.embedding),
            np.dot(c1.embedding, c3.embedding),
        ]
    )
    expected = max(0.0, min(1.0, 1.0 - avg_similarity))

    assert diversity_c1 == pytest.approx(expected, rel=1e-6)


def test_select_parents_fallback_when_probabilities_fail(monkeypatch):
    selector = DiversityAwareSelection()

    c1 = make_concept("c1", 6.0, np.array([1.0, 0.0]))
    c2 = make_concept("c2", 9.0, np.array([0.0, 1.0]))
    c3 = make_concept("c3", 4.0, np.array([0.0, -1.0]))
    population = [c1, c2, c3]

    def raising_choice(*args, **kwargs):
        raise ValueError("Forced failure")

    monkeypatch.setattr("src.selection.np.random.choice", raising_choice)

    selected = selector.select_parents(population, n_parents=2)

    assert selected == [c2, c1]
