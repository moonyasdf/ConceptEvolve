"""Parent selection strategies for the concept evolution database.

The module exposes two concrete parent sampling strategies that balance
exploration and exploitation when spawning new concepts:

* ``WeightedSamplingStrategy`` converts concept scores into probabilities via
  a shifted logistic curve.  The ``parent_selection_lambda`` config knob
  controls how sharply the distribution favours above-median scores.
* ``PowerLawSamplingStrategy`` orders concepts by score and samples their rank
  using a power-law distribution.  ``exploitation_ratio`` biases the strategy
  toward the virtual archive of top-scoring concepts while ``exploitation_alpha``
  controls the steepness of the power-law tail.

Both strategies can be scoped to a single island (``island_idx``) to respect
speciation boundaries.  When no eligible parents are found a deterministic
fallback returns the globally best concept.  The ``CombinedParentSelector``
wrapper dispatches to the configured strategy via ``parent_selection_strategy``.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, List

import numpy as np

logger = logging.getLogger(__name__)


def stable_sigmoid(x: float) -> float:
    """Return a numerically stable logistic transform of ``x``.

    The helper mirrors ``1 / (1 + exp(-x))`` but protects against overflow for
    large-magnitude inputs.  It is used to convert centred scores into weights
    while keeping probabilities in ``(0, 1)``.
    """
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def sample_with_powerlaw(items: List[Any], alpha: float) -> int:
    """Sample an index from ``items`` with a power-law over ranks.

    Args:
        items: Ordered population to draw from.  Only the length is used.
        alpha: Exponent controlling the heaviness of the tail.  ``alpha <= 0``
            degenerates to a uniform distribution, while larger values skew the
            draw toward lower ranks (i.e. higher-scoring concepts).

    Returns:
        Index of the selected element according to the computed probabilities.
    """
    if not items:
        raise ValueError("Cannot sample from an empty population.")
    if alpha <= 0:
        probabilities = np.ones(len(items), dtype=float) / len(items)
    else:
        ranks = np.arange(1, len(items) + 1, dtype=float)
        weights = np.power(ranks, -alpha)
        probabilities = weights / weights.sum()
    return int(np.random.choice(len(items), p=probabilities))


class ParentSamplingStrategy(ABC):
    """Abstract interface for parent sampling policies.

    All strategies operate over the ``concepts`` table using the provided
    ``sqlite3`` cursor and respect the optional island scope.  The helper
    callbacks allow strategies to lazily materialise the selected concept and to
    recover the global incumbent when a fallback is needed.
    """

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_concept_func: Callable[[str], Any],
        get_best_concept_func: Callable[[], Any],
        island_idx: Optional[int] = None,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.get_concept = get_concept_func
        self.get_best_concept = get_best_concept_func
        self.island_idx = island_idx

    @abstractmethod
    def sample_parent(self) -> Any:
        """Return a parent concept instance according to the strategy."""


class WeightedSamplingStrategy(ParentSamplingStrategy):
    """Sample parents by converting scores into soft probabilities.

    Scores are centred around the median to keep probabilities well behaved
    even when the scale drifts.  ``parent_selection_lambda`` controls how
    aggressively the sigmoid distinguishes between above- and below-median
    candidates.  If all weights collapse to zero a uniform draw is used.
    """

    def sample_parent(self) -> Any:
        """Return a parent concept proportional to its weighted score."""
        where_clause = "WHERE island_idx = ?" if self.island_idx is not None else ""
        params = (self.island_idx,) if self.island_idx is not None else ()

        # Restrict the candidate set to the island if desired.
        self.cursor.execute(
            f"SELECT id, combined_score FROM concepts {where_clause}", params
        )
        rows = self.cursor.fetchall()

        if not rows:
            return self.get_best_concept()  # Deterministic fallback when empty.

        concepts = [
            {"id": r["id"], "score": r["combined_score"] or 0}
            for r in rows
        ]
        scores = np.array([c["score"] for c in concepts])

        lambda_ = self.config.parent_selection_lambda
        median_score = np.median(scores)

        # Shift scores by the median so typical candidates map near 0.5.
        weights = np.array(
            [stable_sigmoid(lambda_ * (s - median_score)) for s in scores]
        )

        probabilities = None if weights.sum() == 0 else weights / weights.sum()

        selected_idx = np.random.choice(len(concepts), p=probabilities)
        selected_id = concepts[selected_idx]["id"]
        return self.get_concept(selected_id)


class PowerLawSamplingStrategy(ParentSamplingStrategy):
    """Sample parents using a power-law over score-ranked concepts.

    ``exploitation_ratio`` specifies the probability of drawing from the
    virtual archive consisting of the top ``archive_size`` concepts.  Otherwise
    the entire ranked population is considered.  ``exploitation_alpha`` controls
    how quickly probability mass decays with rank.
    """

    def sample_parent(self) -> Any:
        """Return a parent concept using archive-biased power-law sampling."""
        where_clause = "WHERE island_idx = ?" if self.island_idx is not None else ""
        params = (self.island_idx,) if self.island_idx is not None else ()

        if np.random.random() < self.config.exploitation_ratio:
            # Exploitation path: restrict to the virtual archive of top scorers.
            self.cursor.execute(
                f"SELECT id FROM concepts {where_clause} "
                "ORDER BY combined_score DESC LIMIT ?",
                params + (self.config.archive_size,),
            )
        else:
            # Exploration path: traverse the full ordered population.
            self.cursor.execute(
                f"SELECT id FROM concepts {where_clause} ORDER BY combined_score DESC",
                params,
            )

        rows = self.cursor.fetchall()
        if not rows:
            return self.get_best_concept()

        alpha = self.config.exploitation_alpha
        sampled_idx = sample_with_powerlaw([r["id"] for r in rows], alpha)
        return self.get_concept(rows[sampled_idx]["id"])


class CombinedParentSelector:
    """Factory that instantiates and executes the configured parent strategy."""

    def __init__(self, cursor, conn, config, get_concept_func, get_best_concept_func):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.get_concept = get_concept_func
        self.get_best_concept = get_best_concept_func

    def sample_parent(self, island_idx: Optional[int] = None) -> Any:
        """Return a parent concept for the requested island (if any).

        The concrete strategy is selected via ``config.parent_selection_strategy``
        and inherits the caller-provided ``island_idx`` to keep sampling
        consistent with the evolutionary island model.
        """
        strategy_name = self.config.parent_selection_strategy
        if strategy_name == "weighted":
            strategy = WeightedSamplingStrategy(
                self.cursor,
                self.conn,
                self.config,
                self.get_concept,
                self.get_best_concept,
                island_idx,
            )
        elif strategy_name == "power_law":
            strategy = PowerLawSamplingStrategy(
                self.cursor,
                self.conn,
                self.config,
                self.get_concept,
                self.get_best_concept,
                island_idx,
            )
        else:
            raise ValueError(f"Unknown parent selection strategy: {strategy_name}")

        parent = strategy.sample_parent()
        return parent or self.get_best_concept()  # Final fallback for safety.
