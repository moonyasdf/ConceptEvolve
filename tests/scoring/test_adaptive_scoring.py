import numpy as np
import pytest

from src.scoring import AdaptiveScoring
from src.concepts import ConceptScores


def test_normalize_scores_returns_original_with_short_history():
    scorer = AdaptiveScoring()

    history_values = [4.0, 5.0, 6.5, 7.0]
    for value in history_values:
        scorer.update_history(
            ConceptScores(
                novelty=value,
                potential=value + 0.5,
                sophistication=value + 1.0,
                feasibility=value + 1.5,
            )
        )

    scores = ConceptScores(novelty=9.0, potential=8.5, sophistication=8.0, feasibility=7.5)
    normalized = scorer.normalize_scores(scores)

    assert normalized.novelty == scores.novelty
    assert normalized.potential == scores.potential
    assert normalized.sophistication == scores.sophistication
    assert normalized.feasibility == scores.feasibility


def test_normalize_scores_uses_zscore_with_long_history():
    scorer = AdaptiveScoring()

    history_values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for value in history_values:
        scorer.update_history(
            ConceptScores(
                novelty=value,
                potential=value,
                sophistication=value,
                feasibility=value,
            )
        )

    scores = ConceptScores(novelty=9.0, potential=9.0, sophistication=9.0, feasibility=9.0)
    normalized = scorer.normalize_scores(scores)

    mean = np.mean(history_values)
    std = np.std(history_values) + 1e-6
    expected = max(1.0, min(10.0, 5.5 + ((9.0 - mean) / std) * 1.5))

    assert normalized.novelty == pytest.approx(expected, rel=1e-6)
    assert normalized.potential == pytest.approx(expected, rel=1e-6)
    assert normalized.sophistication == pytest.approx(expected, rel=1e-6)
    assert normalized.feasibility == pytest.approx(expected, rel=1e-6)


def test_combined_score_weights_shift_with_generation():
    scorer = AdaptiveScoring()

    scores = ConceptScores(novelty=10.0, potential=5.0, sophistication=5.0, feasibility=1.0)

    early_generation = scorer.calculate_combined_score(scores, generation=0, max_generations=10)
    late_generation = scorer.calculate_combined_score(scores, generation=10, max_generations=10)

    expected_early = 0.40 * 10.0 + 0.40 * 5.0 + 0.10 * 5.0 + 0.10 * 1.0
    expected_late = 0.30 * 10.0 + 0.40 * 5.0 + 0.10 * 5.0 + 0.20 * 1.0

    assert early_generation == pytest.approx(expected_early, rel=1e-6)
    assert late_generation == pytest.approx(expected_late, rel=1e-6)
    assert late_generation < early_generation

    penalized_alignment = scorer.calculate_combined_score(
        scores,
        generation=0,
        max_generations=10,
        alignment_score=2.0,
    )

    alignment_multiplier = 0.6 + (0.4 * (2.0 / 10.0))
    assert penalized_alignment == pytest.approx(expected_early * alignment_multiplier, rel=1e-6)
