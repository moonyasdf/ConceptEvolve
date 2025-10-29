"""Island management utilities for the evolutionary database layer.

The island model partitions the concept population into semi-isolated subgroups
that occasionally exchange migrants.  This module is responsible for assigning
new concepts to islands and orchestrating migrations according to the configured
policy parameters:

* ``num_islands`` controls the number of available subpopulations.
* ``migration_rate`` dictates what fraction of each island's population crosses
  over during a migration event.
* ``island_elitism`` optionally protects the top concept when selecting
  migrants.

Assignments favour inheriting the parent's island to preserve lineage cohesion
while migrations use stochastic destinations to encourage cross-island mixing.
"""

import logging
import random
import sqlite3
from typing import Optional, Any, List

logger = logging.getLogger(__name__)


class CombinedIslandManager:
    """Manage island assignment and migration within the concept database."""

    def __init__(self, cursor: sqlite3.Cursor, conn: sqlite3.Connection, config: Any):
        self.cursor = cursor
        self.conn = conn
        self.config = config

    def assign_island(self, concept: Any) -> None:
        """Assign ``concept`` to an island based on lineage and configuration.

        New concepts inherit their parent's island when available so offspring
        remain within the same subpopulation.  Concepts without a parent are
        placed uniformly at random across the configured islands.  When
        ``num_islands`` is zero or negative the system degenerates to a single
        island (index ``0``).
        """
        num_islands = self.config.num_islands
        if num_islands <= 0:
            concept.island_idx = 0
            return

        if concept.parent_id:
            parent_island = self.get_island_idx(concept.parent_id)
            if parent_island is not None:
                concept.island_idx = parent_island
                return

        # No parent information: randomly distribute to maintain diversity.
        concept.island_idx = random.randint(0, num_islands - 1)

    def get_island_idx(self, concept_id: str) -> Optional[int]:
        """Lookup the stored island index for ``concept_id``."""
        self.cursor.execute("SELECT island_idx FROM concepts WHERE id = ?", (concept_id,))
        row = self.cursor.fetchone()
        return row["island_idx"] if row else None

    def perform_migration(self, population: List[Any], current_generation: int, get_concept_func):
        """Move a slice of each island's population according to ``migration_rate``.

        Args:
            population: Current list of concept instances participating in the
                generation.  Only concepts with an ``island_idx`` are considered.
            current_generation: Generation counter used for logging purposes.
            get_concept_func: Present for compatibility with upstream code; not
                used directly by the migration routine.

        The routine exits early when fewer than two islands exist or when the
        migration rate is non-positive.  For each island the method selects
        ``max(1, floor(len(pop)*migration_rate))`` migrants.  When elitism is
        enabled the highest-scoring concept is protected from migration.
        Destination islands are chosen uniformly at random from the remaining
        islands, ensuring probabilistic mixing across the archipelago.
        """
        num_islands = self.config.num_islands
        migration_rate = self.config.migration_rate
        if num_islands < 2 or migration_rate <= 0:
            return

        print(f"  🏝️  Realizando migración en la generación {current_generation}...")

        population_by_island = {i: [] for i in range(num_islands)}
        for concept in population:
            if concept.island_idx is not None:
                population_by_island[concept.island_idx].append(concept)

        for source_idx in range(num_islands):
            island_pop = population_by_island[source_idx]
            if len(island_pop) <= 1:
                continue

            num_migrants = max(1, int(len(island_pop) * migration_rate))

            # Sort in descending score so we can optionally skip the elite.
            island_pop.sort(key=lambda c: c.combined_score, reverse=True)

            start_index = 1 if self.config.island_elitism and len(island_pop) > 1 else 0
            migrants = island_pop[start_index : start_index + num_migrants]

            dest_islands = [i for i in range(num_islands) if i != source_idx]
            if not dest_islands:
                continue

            for migrant in migrants:
                dest_idx = random.choice(dest_islands)
                self._migrate_concept(migrant.id, source_idx, dest_idx)

        self.conn.commit()

    def _migrate_concept(self, concept_id: str, from_island: int, to_island: int):
        """Persist the island change for ``concept_id`` and log the migration."""
        self.cursor.execute(
            "UPDATE concepts SET island_idx = ? WHERE id = ?", (to_island, concept_id)
        )
        print(
            f"    - Concepto {concept_id[:8]}... migró de la isla {from_island} a la {to_island}"
        )
