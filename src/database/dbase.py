# MEJORA: #11 (Base de Datos Persistente) - Adaptado de ShinkaEvolve para ConceptEvolve
# Gestiona AlgorithmicConcept en una base de datos SQLite, implementando la lógica
# de almacenamiento, recuperación y muestreo avanzado.

import json
import logging
import numbers
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from src.concepts import AlgorithmicConcept, ConceptScores, SystemRequirements
from .islands import CombinedIslandManager
from .parents import CombinedParentSelector
from .inspiration import CombinedContextSelector

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuración validada para la base de datos, compatible con Hydra."""

    FIELD_META: Dict[str, Dict[str, Any]] = {
        "db_path": {"type": "str", "default": "evolution.db", "allow_blank": False},
        "num_islands": {"type": "int", "min": 0},
        "archive_size": {"type": "int", "min": 1},
        "migration_interval": {"type": "int", "min": 1},
        "migration_rate": {"type": "float", "min": 0.0, "max": 1.0},
        "island_elitism": {"type": "bool"},
        "parent_selection_strategy": {
            "type": "str",
            "choices": {"weighted", "power_law"},
            "normalize": "lower",
        },
        "parent_selection_lambda": {"type": "float", "min_exclusive": 0.0},
        "exploitation_alpha": {"type": "float", "min_exclusive": 0.0},
        "exploitation_ratio": {"type": "float", "min": 0.0, "max": 1.0},
        "num_archive_inspirations": {"type": "int", "min": 0, "default": 2},
        "num_top_k_inspirations": {"type": "int", "min": 0, "default": 1},
        "enforce_island_separation": {"type": "bool", "default": False},
    }

    TYPE_LABELS = {
        "int": "an integer",
        "float": "a float",
        "bool": "a boolean",
        "str": "a string",
    }

    def __init__(self, **kwargs: Any):
        normalized = self._normalized_values(kwargs, require_all=True)
        for key, value in normalized.items():
            setattr(self, key, value)

    @classmethod
    def from_omegaconf(cls, config: Union[DictConfig, Mapping[str, Any]]) -> "DatabaseConfig":
        """Crea una instancia desde un objeto DictConfig o un mapeo estándar."""
        if config is None:
            raise ValueError("Database configuration cannot be None.")
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config, Mapping):
            raise TypeError(
                "Database configuration must be a mapping or DictConfig-compatible object."
            )
        return cls(**config)

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELD_META}

    def validate(self) -> None:
        """Revalida el estado actual de la configuración."""
        self._normalized_values(self.as_dict(), require_all=True)

    def apply_overrides(self, **overrides: Any) -> None:
        """Aplica overrides validados sobre la instancia actual."""
        unknown = set(overrides) - set(self.FIELD_META)
        if unknown:
            raise KeyError(
                "Unknown database configuration parameters: "
                + ", ".join(sorted(unknown))
            )
        merged = self.as_dict()
        merged.update(overrides)
        normalized = self._normalized_values(merged, require_all=True)
        for key, value in normalized.items():
            setattr(self, key, value)

    @classmethod
    def _normalized_values(
        cls, overrides: Mapping[str, Any], *, require_all: bool
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        errors: List[str] = []
        missing: List[str] = []

        for name, meta in cls.FIELD_META.items():
            if name in overrides:
                raw_value = overrides[name]
            elif "default" in meta:
                raw_value = meta["default"]
            else:
                if require_all:
                    missing.append(name)
                continue

            try:
                value = cls._cast_value(name, raw_value, meta)
            except TypeError as exc:
                errors.append(str(exc))
                continue

            try:
                cls._validate_constraints(name, value, meta)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            data[name] = value

        if require_all and missing:
            errors.insert(0, f"Missing required parameters: {', '.join(sorted(missing))}")

        if errors:
            raise ValueError("Invalid database configuration: " + "; ".join(errors))

        return data

    @classmethod
    def _cast_value(cls, name: str, value: Any, meta: Dict[str, Any]) -> Any:
        if value is None:
            raise TypeError(f"Parameter '{name}' cannot be null.")

        type_name = meta["type"]
        if type_name == "int":
            return cls._cast_int(name, value)
        if type_name == "float":
            return cls._cast_float(name, value)
        if type_name == "bool":
            return cls._cast_bool(name, value)
        if type_name == "str":
            return cls._cast_str(name, value, meta)
        raise TypeError(f"Unsupported type declaration for '{name}'.")

    @classmethod
    def _cast_int(cls, name: str, value: Any) -> int:
        if isinstance(value, bool):
            raise cls._type_error(name, "int", value)
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            converted = float(value)
            if converted.is_integer():
                return int(converted)
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                raise cls._type_error(name, "int", value)
            try:
                converted = float(text) if "." in text else int(text, 10)
            except ValueError as exc:
                raise cls._type_error(name, "int", value) from exc
            if isinstance(converted, float):
                if converted.is_integer():
                    return int(converted)
                raise cls._type_error(name, "int", value)
            return int(converted)
        raise cls._type_error(name, "int", value)

    @classmethod
    def _cast_float(cls, name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise cls._type_error(name, "float", value)
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                raise cls._type_error(name, "float", value)
            try:
                return float(text)
            except ValueError as exc:
                raise cls._type_error(name, "float", value) from exc
        raise cls._type_error(name, "float", value)

    @classmethod
    def _cast_bool(cls, name: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "on"}:
                return True
            if normalized in {"false", "no", "0", "off"}:
                return False
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            if value in (0, 1):
                return bool(value)
        raise cls._type_error(name, "bool", value)

    @classmethod
    def _cast_str(cls, name: str, value: Any, meta: Dict[str, Any]) -> str:
        if isinstance(value, bool):
            raise cls._type_error(name, "str", value)
        text = str(value)
        if meta.get("strip", True):
            text = text.strip()
        if meta.get("normalize") == "lower":
            text = text.lower()
        return text

    @classmethod
    def _type_error(cls, name: str, type_name: str, value: Any) -> TypeError:
        label = cls.TYPE_LABELS.get(type_name, type_name)
        return TypeError(
            f"Parameter '{name}' must be {label} (received {value!r} of type {type(value).__name__})."
        )

    @staticmethod
    def _validate_constraints(name: str, value: Any, meta: Dict[str, Any]) -> None:
        if meta.get("allow_blank") is False and isinstance(value, str) and value == "":
            raise ValueError(f"Parameter '{name}' cannot be empty.")

        choices = meta.get("choices")
        if choices and value not in choices:
            allowed = ", ".join(sorted(choices))
            raise ValueError(
                f"Parameter '{name}' must be one of: {allowed} (received {value!r})."
            )

        min_value = meta.get("min")
        if min_value is not None and value < min_value:
            raise ValueError(
                f"Parameter '{name}' must be greater than or equal to {min_value} (received {value})."
            )

        min_exclusive = meta.get("min_exclusive")
        if min_exclusive is not None and value <= min_exclusive:
            raise ValueError(
                f"Parameter '{name}' must be greater than {min_exclusive} (received {value})."
            )

        max_value = meta.get("max")
        if max_value is not None and value > max_value:
            raise ValueError(
                f"Parameter '{name}' must be less than or equal to {max_value} (received {value})."
            )

        max_exclusive = meta.get("max_exclusive")
        if max_exclusive is not None and value >= max_exclusive:
            raise ValueError(
                f"Parameter '{name}' must be less than {max_exclusive} (received {value})."
            )


class ConceptDatabase:
    """Base de datos para almacenar y gestionar AlgorithmicConcepts."""

    def __init__(self, config: DatabaseConfig, read_only: bool = False):
        if not isinstance(config, DatabaseConfig):
            raise TypeError("ConceptDatabase expects a DatabaseConfig instance.")

        self.config = config
        self.db_path = Path(config.db_path).resolve()
        self.read_only = read_only
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database file not found: {self.db_path}")
            db_uri = f"file:{self.db_path}?mode=ro"
            self.conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        if not read_only:
            self._create_tables()

        self.island_manager = CombinedIslandManager(self.cursor, self.conn, self.config)
        self.best_program_id: Optional[str] = None
        self._schedule_migration = False
        self.last_iteration = self.get_max_generation()

    def _create_tables(self) -> None:
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA busy_timeout = 30000;")
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                draft_history TEXT,
                critique_history TEXT,
                system_requirements TEXT,
                generation INTEGER,
                parent_id TEXT,
                inspiration_ids TEXT,
                embedding TEXT,
                scores TEXT,
                combined_score REAL,
                island_idx INTEGER,
                timestamp REAL
            )
            """
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_concepts_generation ON concepts(generation)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_concepts_island ON concepts(island_idx)"
        )
        self.conn.commit()

    def add(self, concept: AlgorithmicConcept) -> None:
        if self.read_only:
            raise PermissionError("Database is in read-only mode.")

        self.island_manager.assign_island(concept)

        self.cursor.execute(
            """
            INSERT OR IGNORE INTO concepts (
                id, title, description, draft_history, critique_history,
                system_requirements, generation, parent_id, inspiration_ids,
                embedding, scores, combined_score, island_idx, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept.id,
                concept.title,
                concept.description,
                json.dumps(concept.draft_history),
                json.dumps(concept.critique_history),
                concept.system_requirements.model_dump_json(),
                concept.generation,
                concept.parent_id,
                json.dumps(concept.inspiration_ids),
                json.dumps(concept.embedding),
                concept.scores.model_dump_json() if concept.scores else None,
                concept.combined_score,
                concept.island_idx,
                time.time(),
            ),
        )
        self.conn.commit()
        if concept.generation > self.last_iteration:
            self.last_iteration = concept.generation

    def update(self, concept: AlgorithmicConcept) -> None:
        self.cursor.execute(
            """
            UPDATE concepts SET
                title = ?, description = ?, draft_history = ?, critique_history = ?,
                system_requirements = ?, embedding = ?, scores = ?, combined_score = ?
            WHERE id = ?
            """,
            (
                concept.title,
                concept.description,
                json.dumps(concept.draft_history),
                json.dumps(concept.critique_history),
                concept.system_requirements.model_dump_json(),
                json.dumps(concept.embedding),
                concept.scores.model_dump_json() if concept.scores else None,
                concept.combined_score,
                concept.id,
            ),
        )
        self.conn.commit()

    def _concept_from_row(self, row: sqlite3.Row) -> Optional[AlgorithmicConcept]:
        if not row:
            return None

        data = dict(row)
        data["draft_history"] = json.loads(row["draft_history"] or "[]")
        data["critique_history"] = json.loads(row["critique_history"] or "[]")
        data["system_requirements"] = SystemRequirements.model_validate_json(
            row["system_requirements"] or "{}"
        )
        data["inspiration_ids"] = json.loads(row["inspiration_ids"] or "[]")
        data["embedding"] = json.loads(row["embedding"] or "[]")
        data["scores"] = (
            ConceptScores.model_validate_json(row["scores"] or "{}")
            if row["scores"]
            else None
        )

        # Pydantic model creará un ID si no está presente, lo cual está bien.
        return AlgorithmicConcept.model_validate(data)

    def get(self, concept_id: str) -> Optional[AlgorithmicConcept]:
        self.cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
        return self._concept_from_row(self.cursor.fetchone())

    def get_all_programs(self) -> List[AlgorithmicConcept]:
        self.cursor.execute("SELECT * FROM concepts")
        return [self._concept_from_row(row) for row in self.cursor.fetchall() if row]

    def count_programs(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM concepts")
        return self.cursor.fetchone()[0]

    def get_max_generation(self) -> int:
        self.cursor.execute("SELECT MAX(generation) FROM concepts")
        result = self.cursor.fetchone()[0]
        return result if result is not None else 0

    def get_best_program(self) -> Optional[AlgorithmicConcept]:
        """Obtiene el concepto con el mayor combined_score."""
        self.cursor.execute(
            "SELECT * FROM concepts ORDER BY combined_score DESC LIMIT 1"
        )
        return self._concept_from_row(self.cursor.fetchone())

    def sample(
        self,
    ) -> Tuple[
        Optional[AlgorithmicConcept],
        List[AlgorithmicConcept],
        List[AlgorithmicConcept],
    ]:
        parent_selector = CombinedParentSelector(
            self.cursor, self.conn, self.config, self.get, self.get_best_program
        )
        parent = parent_selector.sample_parent()
        if not parent:
            return None, [], []

        context_selector = CombinedContextSelector(
            self.cursor,
            self.conn,
            self.config,
            self.get,
            self.island_manager.get_island_idx,
            self._concept_from_row,
        )
        archive_insp, top_k_insp = context_selector.sample_context(
            parent,
            self.config.num_archive_inspirations,
            self.config.num_top_k_inspirations,
        )
        return parent, archive_insp, top_k_insp

    def close(self) -> None:
        if self.conn:
            self.conn.close()
