"""Context selection helpers for creating inspiration sets.

The module mirrors the multi-context sampling behaviour used during concept
variation.  Two complementary pools are produced:

* A *virtual archive* built from the highest-scoring concepts, optionally
  restricted to the parent's island when ``enforce_island_separation`` is True.
  A random subset of size ``num_archive`` is sampled from this pool to preserve
  diversity.
* A *top-k* shortlist containing the overall best concepts outside of the
  archive draw and the parent itself.

The ``CombinedContextSelector`` exposes a single entry point that returns both
lists so downstream agents can reason about elite exemplars versus broader
high-quality inspiration.
"""

import random
import logging
import sqlite3
from typing import Optional, Callable, Any, List, Set, Tuple

logger = logging.getLogger(__name__)


class CombinedContextSelector:
    """Select complementary inspiration contexts for a given parent concept.

    Args:
        cursor: Database cursor used to fetch candidate rows.
        conn: Database connection (present for API parity; not used directly).
        config: Hydra config carrying knobs such as ``archive_size`` and
            ``enforce_island_separation``.
        get_concept_func: Callback that returns a concept instance given an ID.
        get_island_idx_func: Optional helper returning the island index for a
            concept.  Required when island separation is enforced.
        concept_from_row_func: Optional deserialiser from ``sqlite3.Row`` to the
            in-memory concept model expected by downstream components.
    """

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        get_concept_func: Callable[[str], Any],
        get_island_idx_func: Optional[Callable[[str], Optional[int]]] = None,
        concept_from_row_func: Optional[Callable[[sqlite3.Row], Any]] = None,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.get_concept = get_concept_func
        self.get_island_idx = get_island_idx_func
        self.concept_from_row = concept_from_row_func

    def sample_context(
        self, parent: Any, num_archive: int, num_topk: int
    ) -> Tuple[List[Any], List[Any]]:
        """Return archive and top-k inspiration pools for ``parent``.

        Args:
            parent: Concept for which inspirations are being gathered.
            num_archive: Number of items to draw uniformly at random from the
                virtual archive.  Values ``<= 0`` skip archive sampling.
            num_topk: Number of global top concepts to return after removing the
                parent and archive draws.  Values ``<= 0`` skip the shortlist.

        Returns:
            ``(archive_inspirations, top_k_inspirations)`` where each list
            contains concept instances as produced by ``concept_from_row``.  The
            archive component embodies stochastic sampling, whereas the top-k
            component is deterministic given the underlying ranking.
        """

        archive_inspirations = self._sample_archive_inspirations(parent, num_archive)

        top_k_inspirations = self._sample_top_k_inspirations(
            parent, archive_inspirations, num_topk
        )

        return archive_inspirations, top_k_inspirations

    def _sample_archive_inspirations(self, parent: Any, n: int) -> List[Any]:
        """Simulate an elite archive by sampling high-scoring neighbours.

        The query limits the candidate pool to ``config.archive_size`` entries
        ranked by ``combined_score``.  When island separation is enabled only
        concepts from the parent's island are considered.  A random subset of
        size ``n`` is chosen without replacement to avoid repeatedly surfacing
        the same exemplars.
        """
        if n <= 0:
            return []

        island_idx = self.get_island_idx(parent.id) if self.get_island_idx else None

        query = "SELECT * FROM concepts WHERE id != ?"
        params = [parent.id]

        if island_idx is not None and self.config.enforce_island_separation:
            query += " AND island_idx = ?"
            params.append(island_idx)

        query += " ORDER BY combined_score DESC LIMIT ?"
        params.append(self.config.archive_size)

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        if not rows or not self.concept_from_row:
            return []

        candidates = [self.concept_from_row(row) for row in rows]

        if len(candidates) <= n:
            # All candidates fit into the archive sample; no randomness needed.
            return candidates

        return random.sample(candidates, n)

    def _sample_top_k_inspirations(
        self, parent: Any, excluded_concepts: List[Any], k: int
    ) -> List[Any]:
        """Return the top-``k`` scoring concepts excluding known inspirations.

        Args:
            parent: Parent concept used to seed exclusion rules.
            excluded_concepts: Concepts already surfaced (archive sample) that
                must be removed to keep the shortlist distinct.
            k: Number of concepts to retrieve in descending score order.

        Returns:
            Deterministic list of up to ``k`` concept instances sorted by score.
        """
        if k <= 0 or not self.concept_from_row:
            return []

        excluded_ids: Set[str] = {parent.id}
        excluded_ids.update(c.id for c in excluded_concepts)

        placeholders = ",".join("?" * len(excluded_ids))
        query = (
            f"SELECT * FROM concepts WHERE id NOT IN ({placeholders}) "
            "ORDER BY combined_score DESC LIMIT ?"
        )

        params = list(excluded_ids) + [k]

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        if not rows:
            return []

        return [self.concept_from_row(row) for row in rows]
