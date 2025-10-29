# src/evolution.py

# MEJORA: Refactorizado para usar la nueva arquitectura de base de datos, islas,
# Hydra y para pasar la configuración del modelo a los agentes.

import os
import math
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
from omegaconf import DictConfig

from src.database.dbase import ConceptDatabase, DatabaseConfig
from src.agents import (
    IdeaGenerator,
    ConceptCritic,
    ConceptEvaluator,
    NoveltyJudge,
    RequirementsExtractor,
    AlignmentValidator,
)
from src.concepts import AlgorithmicConcept, ConceptScores
from src.llm_utils import EmbeddingClient
from src.vector_index import ConceptIndex
from src.scoring import AdaptiveScoring
from src.selection import DiversityAwareSelection
from src.checkpoint import CheckpointManager
from src.meta_learning import MetaSummarizer

logger = logging.getLogger(__name__)


class ConceptEvolution:
    def __init__(self, problem_description: str, cfg: DictConfig):
        self.problem_description = problem_description
        self.cfg = cfg
        self.current_generation = 0

        self.idea_generator = IdeaGenerator()
        self.critic = ConceptCritic()
        self.evaluator = ConceptEvaluator()
        self.novelty_judge = NoveltyJudge()
        self.req_extractor = RequirementsExtractor()
        self.alignment_validator = AlignmentValidator()

        # MEJORA: Pasamos el nombre del modelo de embedding desde la config de Hydra
        self.embedding_client = EmbeddingClient(model_name=cfg.model.embedding_model)

        self.scoring_system = AdaptiveScoring()
        self.selector = DiversityAwareSelection()

        self.db_config = self._build_database_config(cfg)
        self.db = ConceptDatabase(self.db_config)

        self.embedding_dimension = self._determine_embedding_dimension()
        try:
            self.concept_index = ConceptIndex(dimension=self.embedding_dimension)
        except Exception as exc:
            logger.warning(
                "No se pudo inicializar ConceptIndex con dimensión %s: %s. Se usará 768 por defecto.",
                self.embedding_dimension,
                exc,
            )
            self.embedding_dimension = 768
            self.concept_index = ConceptIndex(dimension=self.embedding_dimension)

        self.checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

    def _build_database_config(self, cfg: DictConfig) -> DatabaseConfig:
        if not hasattr(cfg, "database") or cfg.database is None:
            raise ValueError(
                "La configuración de Hydra debe incluir una sección 'database' con los parámetros requeridos."
            )

        try:
            db_config = DatabaseConfig.from_omegaconf(cfg.database)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Database configuration failed validation: {exc}") from exc

        db_path_value = Path(db_config.db_path)
        if not db_path_value.is_absolute():
            db_path_value = Path(os.getcwd()) / db_path_value

        try:
            db_config.apply_overrides(db_path=str(db_path_value))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Database configuration failed validation: {exc}") from exc

        db_config.validate()
        return db_config

    def _get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not text:
            return []
        # MEJORA: `get_embedding` ahora puede manejar lotes.
        return self.embedding_client.get_embedding(text)

    def _infer_dimension_from_concepts(self, concepts: List[AlgorithmicConcept]) -> Optional[int]:
        for concept in concepts or []:
            if concept and concept.embedding:
                length = len(concept.embedding)
                if length:
                    return length
        return None

    def _determine_embedding_dimension(self) -> int:
        configured = (
            getattr(self.cfg.model, "embedding_dimension", None)
            if hasattr(self.cfg, "model")
            else None
        )
        if configured is not None:
            try:
                configured_value = int(configured)
                if configured_value > 0:
                    return configured_value
                logger.warning(
                    "La dimensión de embedding configurada debe ser positiva (recibido %s).",
                    configured,
                )
            except (TypeError, ValueError):
                logger.warning(
                    "La dimensión de embedding configurada no es válida: %s",
                    configured,
                )

        existing_concepts = self._safe_get_all_concepts()
        inferred = self._infer_dimension_from_concepts(existing_concepts)
        if inferred:
            return inferred

        logger.warning(
            "No se pudo determinar la dimensión de embeddings. Se usará 768 por defecto. Especifique `model.embedding_dimension` en su config.",
        )
        return 768

    def _rebuild_concept_index(
        self, existing_concepts: Optional[List[AlgorithmicConcept]] = None
    ) -> None:
        concepts = existing_concepts if existing_concepts is not None else self._safe_get_all_concepts()
        inferred = self._infer_dimension_from_concepts(concepts)
        if inferred and inferred != self.embedding_dimension:
            logger.info(
                "Actualizando dimensión del índice de embeddings de %s a %s basada en datos existentes.",
                self.embedding_dimension,
                inferred,
            )
            self.embedding_dimension = inferred
        try:
            self.concept_index = ConceptIndex(dimension=self.embedding_dimension)
        except Exception as exc:
            logger.warning(
                "Fallo al reinstanciar ConceptIndex con dimensión %s: %s. Se usará 768.",
                self.embedding_dimension,
                exc,
            )
            self.embedding_dimension = 768
            self.concept_index = ConceptIndex(dimension=self.embedding_dimension)
        for concept in concepts:
            self._add_concept_to_index(concept)

    def _scores_need_recomputation(self, concept: Optional[AlgorithmicConcept]) -> bool:
        if concept is None:
            return False
        scores = concept.scores
        if scores is None or not isinstance(scores, ConceptScores):
            return True
        try:
            values = scores.model_dump()
        except Exception:
            return True
        for value in values.values():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return True
            if not math.isfinite(numeric):
                return True
        try:
            combined = float(concept.combined_score)
        except (TypeError, ValueError):
            return True
        return not math.isfinite(combined)

    def _safe_add_to_db(self, concept: AlgorithmicConcept) -> bool:
        try:
            self.db.add(concept)
            return True
        except Exception as exc:
            logger.warning(
                "No se pudo añadir el concepto %s a la base de datos: %s",
                concept.id,
                exc,
            )
            return False

    def _safe_update_in_db(self, concept: AlgorithmicConcept) -> bool:
        try:
            self.db.update(concept)
            return True
        except Exception as exc:
            logger.warning(
                "No se pudo actualizar el concepto %s en la base de datos: %s",
                concept.id,
                exc,
            )
            return False

    def _safe_get_all_concepts(self) -> List[AlgorithmicConcept]:
        try:
            concepts = self.db.get_all_programs()
            return [c for c in concepts if c]
        except Exception as exc:
            logger.warning("Error al recuperar los conceptos de la base de datos: %s", exc)
            return []

    def _safe_get_concept(self, concept_id: str) -> Optional[AlgorithmicConcept]:
        try:
            return self.db.get(concept_id)
        except Exception as exc:
            logger.warning(
                "Error al consultar el concepto %s en la base de datos: %s",
                concept_id,
                exc,
            )
            return None

    def _safe_sample(self) -> Tuple[Optional[AlgorithmicConcept], List[AlgorithmicConcept], List[AlgorithmicConcept]]:
        try:
            return self.db.sample()
        except Exception as exc:
            logger.warning("Error al muestrear población desde la base de datos: %s", exc)
            return None, [], []

    def _count_programs(self) -> int:
        try:
            return self.db.count_programs()
        except Exception as exc:
            logger.warning("No se pudo contar los conceptos en la base de datos: %s", exc)
            return 0

    def _index_size(self) -> int:
        try:
            return self.concept_index.size()
        except Exception as exc:
            logger.warning("No se pudo obtener el tamaño del índice FAISS: %s", exc)
            return 0

    def _add_concept_to_index(self, concept: Optional[AlgorithmicConcept]) -> None:
        if not concept or not concept.embedding:
            return
        if len(concept.embedding) != self.embedding_dimension:
            logger.warning(
                "Se omite el concepto %s al reconstruir el índice: dimensión esperada %s, recibida %s.",
                concept.id,
                self.embedding_dimension,
                len(concept.embedding),
            )
            return
        try:
            self.concept_index.add_concept(concept.id, concept.embedding)
        except Exception as exc:
            logger.warning("No se pudo indexar el concepto %s en FAISS: %s", concept.id, exc)

    def _initialize_population(self):
        print("🧬 Inicializando población de conceptos...")
        max_attempts = self.cfg.evolution.population_size * 3
        attempts = 0
        with tqdm(
            total=self.cfg.evolution.population_size,
            desc="Generando conceptos iniciales",
        ) as pbar:
            while attempts < max_attempts:
                current_size = self._count_programs()
                if current_size >= self.cfg.evolution.population_size:
                    break
                
                concept = self.idea_generator.generate_initial_concept(
                    self.problem_description, generation=0, model_cfg=self.cfg.model
                )
                
                if concept and self._safe_add_to_db(concept):
                    pbar.update(1)
                attempts += 1
        print("  Evaluando población inicial...")
        self._evaluate_population()

    def _evaluate_population(self):
        concepts = self._safe_get_all_concepts()
        concepts_to_evaluate = [
            c for c in concepts if self._scores_need_recomputation(c)
        ]
        print(f"  Encontrados {len(concepts_to_evaluate)} conceptos para evaluar.")
        index_dirty = False
        inferred_dimension = None
        
        texts_to_embed = [c.get_full_prompt_text() for c in concepts_to_evaluate]
        if texts_to_embed:
            print("  Generando embeddings en lote...")
            all_embeddings = self._get_embedding(texts_to_embed)
            for concept, embedding in zip(concepts_to_evaluate, all_embeddings):
                concept.embedding = embedding
        
        for concept in tqdm(concepts_to_evaluate, desc="Evaluando conceptos"):
            try:
                self._evaluate_single_concept(concept, persist_embedding=False)
                if concept.embedding:
                    inferred_dimension = inferred_dimension or len(concept.embedding)
                index_dirty = True
            except Exception as exc:
                logger.warning("No se pudo evaluar el concepto %s: %s", getattr(concept, "id", "desconocido"), exc)

        if inferred_dimension and inferred_dimension != self.embedding_dimension:
            logger.info(
                "Actualizando dimensión de embeddings tras reevaluación: %s -> %s",
                self.embedding_dimension,
                inferred_dimension,
            )
            self.embedding_dimension = inferred_dimension
        if index_dirty:
            self._rebuild_concept_index()

    def _evaluate_single_concept(
        self, concept: AlgorithmicConcept, *, persist: bool = True, persist_embedding: bool = True
    ) -> None:
        if persist_embedding:
            concept.embedding = self._get_embedding(concept.get_full_prompt_text())

        try:
            raw_scores = self.evaluator.run(concept, model_cfg=self.cfg.model)
        except Exception as exc:
            logger.warning("Error al evaluar el concepto %s: %s", concept.id, exc)
            raw_scores = None

        if raw_scores and not isinstance(raw_scores, ConceptScores):
            try:
                raw_scores = ConceptScores.model_validate(raw_scores)
            except Exception as exc:
                logger.warning(
                    "Scores inválidos para el concepto %s: %s",
                    concept.id,
                    exc,
                )
                raw_scores = None
                
        if raw_scores:
            self.scoring_system.update_history(raw_scores)
            concept.scores = self.scoring_system.normalize_scores(raw_scores)
            try:
                is_aligned, _, alignment_score = self.alignment_validator.run(
                    concept, self.problem_description, model_cfg=self.cfg.model
                )
            except Exception as exc:
                logger.warning("Error en validador de alineación para el concepto %s: %s", concept.id, exc)
                is_aligned = True
                alignment_score = 0.0
            
            if not is_aligned:
                alignment_score = 0.0
            
            concept.combined_score = self.scoring_system.calculate_combined_score(
                concept.scores, concept.generation, self.cfg.evolution.num_generations, alignment_score
            )
        
        if persist:
            self._safe_update_in_db(concept)

    def _check_novelty_with_faiss(self, draft_concept: AlgorithmicConcept) -> bool:
        if not draft_concept.embedding:
            return True
        if self._index_size() == 0:
            return True
        try:
            similar_concepts_ids = self.concept_index.find_similar(
                draft_concept.embedding,
                k=1,
                threshold=self.cfg.evolution.novelty_threshold,
            )
        except Exception as exc:
            logger.warning(
                "Error al consultar FAISS para el concepto %s: %s",
                draft_concept.id,
                exc,
            )
            return True
        if not similar_concepts_ids:
            return True
        candidate = similar_concepts_ids[0]
        try:
            candidate_id = candidate[0]
        except (TypeError, IndexError):
            logger.warning("Resultado inesperado del índice FAISS: %s", candidate)
            return True
        most_similar_concept = self._safe_get_concept(candidate_id)
        if not most_similar_concept:
            logger.info(
                "El índice FAISS devolvió el concepto %s, pero no se encontró en la base de datos.",
                candidate_id,
            )
            return True
        print(f"  🤔 Concepto similar detectado: '{most_similar_concept.title}'")
        try:
            decision = self.novelty_judge.run(draft_concept, most_similar_concept, model_cfg=self.cfg.model)
        except Exception as exc:
            logger.warning(
                "Error al ejecutar el juez de novedad para concepto %s: %s",
                draft_concept.id,
                exc,
            )
            return True
        if decision and not decision.is_novel:
            print(f"  ❌ Descartado por duplicación: {decision.explanation}")
            return False
        return True

    def _refine_concept(self, draft_concept: AlgorithmicConcept) -> AlgorithmicConcept:
        current_description = draft_concept.description
        draft_concept.draft_history.append(current_description)
        addressed_points = []
        for i in range(self.cfg.evolution.refinement_steps):
            print(
                f"  🔍 Ronda de refinamiento {i + 1}/{self.cfg.evolution.refinement_steps}..."
            )
            critiques = self.critic.run(draft_concept, self.problem_description, model_cfg=self.cfg.model)
            draft_concept.critique_history.append(critiques)
            current_description, addressed_points = self.idea_generator.refine(
                draft_concept,
                draft_concept.critique_history,
                addressed_points,
                self.problem_description,
                model_cfg=self.cfg.model
            )
            draft_concept.description = current_description
            draft_concept.draft_history.append(current_description)
        return draft_concept

    def run(self):
        start_gen = 0
        if self.cfg.resume:
            latest_checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint:
                pop, gen, _ = latest_checkpoint
                if pop:
                    for concept in pop:
                        if not self._safe_get_concept(concept.id):
                            self._safe_add_to_db(concept)
                    start_gen = gen
                    print(f"  ✅ Reanudando desde la generación {start_gen}")

        if self._count_programs() == 0:
            self._initialize_population()

        all_concepts = self._safe_get_all_concepts()
        self._rebuild_concept_index(existing_concepts=all_concepts)

        for gen in range(start_gen + 1, self.cfg.evolution.num_generations + 1):
            self.current_generation = gen
            print(
                f"\n{'=' * 70}\n{'GENERACIÓN ' + str(gen) + '/' + str(self.cfg.evolution.num_generations):^70}\n{'=' * 70}\n"
            )

            parent, inspirations, _ = self._safe_sample()
            if not parent:
                print("  ⚠️ No se pudo seleccionar padre. Saltando generación.")
                continue

            print(
                f"👨‍👩‍👧 Padre seleccionado: '{parent.title}' (Score: {parent.combined_score:.2f})"
            )

            try:
                new_concept = self.idea_generator.mutate_or_crossover(
                    parent, inspirations, gen, self.problem_description, model_cfg=self.cfg.model
                )
            except Exception as exc:
                logger.warning(
                    "Error al mutar o cruzar el concepto %s: %s",
                    parent.id,
                    exc,
                )
                continue
            if not new_concept:
                continue

            new_concept = self._refine_concept(new_concept)
            new_concept.embedding = self._get_embedding(
                new_concept.get_full_prompt_text()
            )

            if not self._check_novelty_with_faiss(new_concept):
                continue

            self._evaluate_single_concept(new_concept, persist=False, persist_embedding=False)
            try:
                new_concept.system_requirements = self.req_extractor.run(new_concept, model_cfg=self.cfg.model)
            except Exception as exc:
                logger.warning(
                    "No se pudieron extraer los requisitos del concepto %s: %s",
                    new_concept.id,
                    exc,
                )

            if not self._safe_add_to_db(new_concept):
                continue

            if new_concept.embedding and len(new_concept.embedding) != self.embedding_dimension:
                logger.warning(
                    "Dimensión de embedding distinta detectada (esperado %s, recibido %s). Reconstruyendo índice.",
                    self.embedding_dimension,
                    len(new_concept.embedding),
                )
                self.embedding_dimension = len(new_concept.embedding)
                self._rebuild_concept_index()
            else:
                self._add_concept_to_index(new_concept)

            print(
                f"  ✅ Concepto añadido: '{new_concept.title}' (Score: {new_concept.combined_score:.2f})"
            )

            if gen > 0 and gen % self.db.config.migration_interval == 0:
                print("\n🏝️ Realizando migración entre islas...")
                try:
                    population_snapshot = self._safe_get_all_concepts()
                    self.db.island_manager.perform_migration(
                        population_snapshot, gen, self.db.get
                    )
                except Exception as exc:
                    logger.warning("Error durante la migración entre islas: %s", exc)

            if gen % self.cfg.evolution.checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    self._safe_get_all_concepts(), gen, self.problem_description
                )

        print("\n✅ Evolución completada.")
        try:
            self.db.close()
        except Exception as exc:
            logger.warning("Error al cerrar la base de datos: %s", exc)