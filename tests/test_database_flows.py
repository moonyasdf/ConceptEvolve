from dataclasses import dataclass
from typing import Optional
import random

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.database import parents as parent_module
from src.database.parents import CombinedParentSelector
from src.database.inspiration import CombinedContextSelector
from src.database.islands import CombinedIslandManager
from src.database.dbase import DatabaseConfig


@dataclass
class StubConcept:
    id: str
    combined_score: float
    generation: int = 0
    island_idx: Optional[int] = None
    parent_id: Optional[str] = None
    title: str = ""


def insert_concept(cursor, concept: StubConcept) -> None:
    cursor.execute(
        "INSERT INTO concepts (id, title, combined_score, generation, island_idx, parent_id)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            concept.id,
            concept.title,
            concept.combined_score,
            concept.generation,
            concept.island_idx,
            concept.parent_id,
        ),
    )


def build_config_kwargs(**overrides):
    base = {
        "db_path": "test.db",
        "num_islands": 3,
        "archive_size": 5,
        "migration_interval": 5,
        "migration_rate": 0.1,
        "island_elitism": True,
        "parent_selection_strategy": "weighted",
        "parent_selection_lambda": 5.0,
        "exploitation_alpha": 1.0,
        "exploitation_ratio": 0.5,
        "num_archive_inspirations": 2,
        "num_top_k_inspirations": 2,
        "enforce_island_separation": False,
    }
    base.update(overrides)
    return base


def make_config(**overrides):
    return DatabaseConfig(**build_config_kwargs(**overrides))


def test_weighted_sampling_prefers_high_scores(monkeypatch, in_memory_db):
    conn, cursor = in_memory_db
    concepts = [
        StubConcept("c1", combined_score=0.2, island_idx=0, title="Concept 1"),
        StubConcept("c2", combined_score=0.5, island_idx=0, title="Concept 2"),
        StubConcept("c3", combined_score=0.9, island_idx=0, title="Concept 3"),
    ]
    for concept in concepts:
        insert_concept(cursor, concept)
    conn.commit()

    config = make_config(parent_selection_strategy="weighted", parent_selection_lambda=3.0)
    concept_map = {c.id: c for c in concepts}

    captured = {}

    def fake_choice(size, p=None):
        if p is not None:
            captured["probabilities"] = np.array(p, dtype=float)
            return int(np.argmax(p))
        captured["probabilities"] = None
        return 0

    monkeypatch.setattr(np.random, "choice", fake_choice)

    selector = CombinedParentSelector(
        cursor, conn, config, concept_map.__getitem__, lambda: concept_map["c3"]
    )
    parent = selector.sample_parent()

    assert parent.id == "c3"
    probs = captured.get("probabilities")
    assert probs is not None
    assert pytest.approx(float(probs.sum())) == 1.0
    cursor.execute("SELECT id FROM concepts")
    ordered_ids = [row["id"] for row in cursor.fetchall()]
    assert ordered_ids[np.argmax(probs)] == "c3"


def test_weighted_sampling_returns_best_when_population_empty(in_memory_db):
    conn, cursor = in_memory_db
    config = make_config(parent_selection_strategy="weighted")
    best = StubConcept("best", combined_score=1.0)

    selector = CombinedParentSelector(cursor, conn, config, lambda _: None, lambda: best)
    parent = selector.sample_parent()

    assert parent is best


def test_power_law_sampling_uses_archive_window(monkeypatch, in_memory_db):
    conn, cursor = in_memory_db
    concepts = [
        StubConcept("p3", combined_score=0.9, island_idx=0),
        StubConcept("p2", combined_score=0.8, island_idx=0),
        StubConcept("p1", combined_score=0.7, island_idx=0),
    ]
    for concept in concepts:
        insert_concept(cursor, concept)
    conn.commit()

    config = make_config(
        parent_selection_strategy="power_law",
        exploitation_ratio=0.5,
        archive_size=2,
        exploitation_alpha=1.4,
    )
    concept_map = {c.id: c for c in concepts}
    best = concept_map["p3"]

    monkeypatch.setattr(np.random, "random", lambda: 0.1)
    captured = {}

    def fake_powerlaw(ids, alpha):
        captured["ids"] = list(ids)
        captured["alpha"] = alpha
        return 1  # select the second element in the archive slice

    monkeypatch.setattr(parent_module, "sample_with_powerlaw", fake_powerlaw)

    selector = CombinedParentSelector(cursor, conn, config, concept_map.__getitem__, lambda: best)
    parent = selector.sample_parent()

    assert captured["ids"] == ["p3", "p2"]
    assert captured["alpha"] == pytest.approx(1.4)
    assert parent.id == "p2"


def test_power_law_sampling_returns_best_when_population_empty(in_memory_db):
    conn, cursor = in_memory_db
    config = make_config(parent_selection_strategy="power_law")
    best = StubConcept("best", combined_score=1.0)

    selector = CombinedParentSelector(cursor, conn, config, lambda _: None, lambda: best)
    parent = selector.sample_parent()

    assert parent is best


def test_context_selector_returns_archive_and_top_k(monkeypatch, in_memory_db):
    conn, cursor = in_memory_db
    parent = StubConcept("parent", combined_score=0.9, island_idx=0, generation=3, title="Parent")
    insert_concept(cursor, parent)

    archive_candidates = [
        StubConcept("a1", combined_score=0.88, island_idx=0, generation=2, title="A1"),
        StubConcept("a2", combined_score=0.85, island_idx=0, generation=4, title="A2"),
        StubConcept("a3", combined_score=0.75, island_idx=0, generation=5, title="A3"),
        StubConcept("b1", combined_score=0.92, island_idx=1, generation=6, title="B1"),
    ]
    for concept in archive_candidates:
        insert_concept(cursor, concept)
    conn.commit()

    config = make_config(enforce_island_separation=True, archive_size=3)

    def get_island_idx(concept_id: str) -> Optional[int]:
        cursor.execute("SELECT island_idx FROM concepts WHERE id = ?", (concept_id,))
        row = cursor.fetchone()
        return row["island_idx"] if row else None

    def concept_from_row(row) -> StubConcept:
        return StubConcept(
            id=row["id"],
            combined_score=row["combined_score"] or 0.0,
            generation=row["generation"] or 0,
            island_idx=row["island_idx"],
            title=row["title"] or "",
        )

    monkeypatch.setattr(random, "sample", lambda seq, n: list(seq)[:n])

    selector = CombinedContextSelector(
        cursor,
        conn,
        config,
        lambda cid: None,
        get_island_idx,
        concept_from_row,
    )

    archive, top_k = selector.sample_context(parent, num_archive=2, num_topk=2)

    assert {c.id for c in archive} == {"a1", "a2"}
    assert all(c.island_idx == parent.island_idx for c in archive)
    assert {c.id for c in top_k} == {"a3", "b1"}
    assert not ({c.id for c in archive} & {c.id for c in top_k})
    assert parent.id not in {c.id for c in top_k}


def test_context_selector_handles_empty_tables(in_memory_db):
    conn, cursor = in_memory_db
    parent = StubConcept("parent", combined_score=0.5, island_idx=0)
    config = make_config(enforce_island_separation=True, archive_size=3)

    selector = CombinedContextSelector(
        cursor,
        conn,
        config,
        lambda _: None,
        lambda _: None,
        None,
    )

    archive, top_k = selector.sample_context(parent, num_archive=2, num_topk=3)

    assert archive == []
    assert top_k == []


def test_island_migration_moves_non_elite_concepts(monkeypatch, in_memory_db):
    conn, cursor = in_memory_db
    config = make_config(num_islands=2, migration_rate=0.4, island_elitism=True)
    manager = CombinedIslandManager(cursor, conn, config)

    population = [
        StubConcept("i0_best", combined_score=0.9, island_idx=0),
        StubConcept("i0_mid", combined_score=0.7, island_idx=0),
        StubConcept("i0_low", combined_score=0.4, island_idx=0),
        StubConcept("i1_best", combined_score=0.95, island_idx=1),
        StubConcept("i1_mid", combined_score=0.8, island_idx=1),
        StubConcept("i1_low", combined_score=0.5, island_idx=1),
    ]
    for concept in population:
        insert_concept(cursor, concept)
    conn.commit()

    monkeypatch.setattr(random, "choice", lambda options: options[0])

    manager.perform_migration(population, current_generation=10, get_concept_func=lambda cid: None)

    cursor.execute(
        "SELECT id, island_idx FROM concepts WHERE id IN (?, ?, ?, ?, ?)",
        ("i0_best", "i0_mid", "i0_low", "i1_best", "i1_mid"),
    )
    island_map = {row["id"]: row["island_idx"] for row in cursor.fetchall()}

    assert island_map["i0_best"] == 0
    assert island_map["i0_mid"] == 1
    assert island_map["i1_best"] == 1
    assert island_map["i1_mid"] == 0


def test_island_migration_skips_when_rate_zero(monkeypatch, in_memory_db):
    conn, cursor = in_memory_db
    config = make_config(num_islands=2, migration_rate=0.0, island_elitism=False)
    manager = CombinedIslandManager(cursor, conn, config)

    population = [
        StubConcept("j0", combined_score=0.6, island_idx=0),
        StubConcept("j1", combined_score=0.7, island_idx=1),
    ]
    for concept in population:
        insert_concept(cursor, concept)
    conn.commit()

    calls = []

    def fake_choice(options):
        calls.append(options)
        return options[0]

    monkeypatch.setattr(random, "choice", fake_choice)

    manager.perform_migration(population, current_generation=5, get_concept_func=lambda cid: None)

    assert calls == []
    cursor.execute("SELECT island_idx FROM concepts WHERE id = ?", ("j0",))
    assert cursor.fetchone()["island_idx"] == 0


def test_database_config_validation_accepts_valid_values():
    cfg = make_config(
        archive_size=10,
        migration_interval=2,
        migration_rate=0.2,
        parent_selection_lambda=4.0,
        exploitation_alpha=1.2,
        exploitation_ratio=0.3,
        num_archive_inspirations=1,
        num_top_k_inspirations=2,
    )

    assert cfg.archive_size == 10
    assert cfg.migration_rate == pytest.approx(0.2)


def test_database_config_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="migration_rate"):
        make_config(migration_rate=1.5)

    with pytest.raises(ValueError, match="archive_size"):
        make_config(archive_size=0)

    with pytest.raises(ValueError, match="parent_selection_strategy"):
        make_config(parent_selection_strategy="invalid")

    with pytest.raises(ValueError, match="parent_selection_lambda"):
        make_config(parent_selection_lambda="not-a-number")


def test_database_config_from_omegaconf_parses_strings():
    hydra_cfg = OmegaConf.create(
        {
            "db_path": "relative.db",
            "num_islands": "6",
            "archive_size": "12",
            "migration_interval": "4",
            "migration_rate": "0.35",
            "island_elitism": "false",
            "parent_selection_strategy": "Power_Law",
            "parent_selection_lambda": "9.5",
            "exploitation_alpha": "1.5",
            "exploitation_ratio": "0.45",
            "num_archive_inspirations": "3",
            "num_top_k_inspirations": "1",
            "enforce_island_separation": "true",
        }
    )

    cfg = DatabaseConfig.from_omegaconf(hydra_cfg)

    assert cfg.db_path == "relative.db"
    assert cfg.num_islands == 6
    assert cfg.archive_size == 12
    assert cfg.parent_selection_strategy == "power_law"
    assert cfg.island_elitism is False
    assert cfg.enforce_island_separation is True
    assert cfg.migration_rate == pytest.approx(0.35)


def test_database_config_from_omegaconf_requires_fields():
    hydra_cfg = OmegaConf.create(
        {
            "db_path": "relative.db",
            "archive_size": 10,
            "migration_rate": 0.1,
        }
    )

    with pytest.raises(ValueError, match="Missing required parameters"):
        DatabaseConfig.from_omegaconf(hydra_cfg)
