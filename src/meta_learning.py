# src/meta_learning.py

# MEJORA: Sistema de Meta-Aprendizaje adaptado de ShinkaEvolve.
# Este agente analiza la historia de la evolución para extraer lecciones y
# generar recomendaciones que guíen a las futuras generaciones de conceptos.

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig

from src.concepts import AlgorithmicConcept
from src.llm_utils import query_gemini_unstructured
from src.prompts_meta import (
    META_STEP1_SYSTEM_MSG, META_STEP1_USER_MSG,
    META_STEP2_SYSTEM_MSG, META_STEP2_USER_MSG,
    META_STEP3_SYSTEM_MSG, META_STEP3_USER_MSG
)

logger = logging.getLogger(__name__)

def construct_concept_summary_for_meta(concept: AlgorithmicConcept) -> str:
    """Construye un resumen de un concepto para el prompt de meta-aprendizaje."""
    score_str = f"Combined Score: {concept.combined_score:.2f}"
    if concept.scores:
        scores = concept.scores
        score_str += (
            f" (N: {scores.novelty:.1f}, P: {scores.potential:.1f}, "
            f"S: {scores.sophistication:.1f}, F: {scores.feasibility:.1f})"
        )
    
    critique_summary = " ".join(concept.critique_history[-1].split()[:50]) + "..." if concept.critique_history else "N/A"

    return (
        f"**Title:** {concept.title}\n"
        f"**Description:** {concept.description[:400]}...\n"
        f"**Performance:** {score_str}\n"
        f"**Final Critique Summary:** {critique_summary}"
    )

class MetaSummarizer:
    """Gestiona el meta-aprendizaje, resumiendo la evolución y generando recomendaciones."""

    def __init__(self, model_cfg: DictConfig, max_recommendations: int = 5):
        self.model_cfg = model_cfg
        self.max_recommendations = max_recommendations
        self.meta_summary: Optional[str] = None
        self.meta_scratch_pad: Optional[str] = None
        self.meta_recommendations: Optional[str] = None
        self.evaluated_since_last_meta: List[AlgorithmicConcept] = []
        self.total_concepts_processed = 0

    def add_evaluated_concept(self, concept: AlgorithmicConcept):
        """Añade un concepto recién evaluado para el próximo ciclo de meta-análisis."""
        self.evaluated_since_last_meta.append(concept)

    def should_update_meta(self, meta_interval: Optional[int]) -> bool:
        """Determina si es momento de ejecutar un ciclo de meta-análisis."""
        if meta_interval is None or meta_interval <= 0:
            return False
        return len(self.evaluated_since_last_meta) >= meta_interval

    def update_meta_memory(self, best_concept: Optional[AlgorithmicConcept] = None) -> Tuple[Optional[str], float]:
        """Ejecuta el ciclo de meta-análisis de 3 pasos."""
        if not self.evaluated_since_last_meta:
            return None, 0.0

        print(f"  🧠 Ejecutando ciclo de meta-aprendizaje con {len(self.evaluated_since_last_meta)} nuevos conceptos...")
        
        # Paso 1: Resumir cada concepto nuevo
        individual_summaries, cost1 = self._step1_individual_summaries()
        if not individual_summaries:
            logger.error("Meta-aprendizaje (Paso 1) falló: no se generaron resúmenes.")
            return None, 0.0

        # Paso 2: Sintetizar ideas globales
        global_insights, cost2 = self._step2_global_insights(individual_summaries, best_concept)
        if not global_insights:
            logger.error("Meta-aprendizaje (Paso 2) falló: no se generaron ideas globales.")
            return None, cost1
        
        # Paso 3: Generar nuevas recomendaciones
        recommendations, cost3 = self._step3_generate_recommendations(global_insights, best_concept)
        if not recommendations:
            logger.error("Meta-aprendizaje (Paso 3) falló: no se generaron recomendaciones.")
            return None, cost1 + cost2

        # Actualizar estado interno
        self.meta_summary = individual_summaries if self.meta_summary is None else f"{self.meta_summary}\n\n{individual_summaries}"
        self.meta_scratch_pad = global_insights
        self.meta_recommendations = recommendations
        
        num_processed = len(self.evaluated_since_last_meta)
        self.total_concepts_processed += num_processed
        self.evaluated_since_last_meta = []
        
        total_cost = cost1 + cost2 + cost3
        print(f"  ✅ Ciclo de meta-aprendizaje completado. Coste: ${total_cost:.4f}")
        return self.meta_recommendations, total_cost

    def _step1_individual_summaries(self) -> Tuple[Optional[str], float]:
        """Paso 1: Crea resúmenes individuales para cada concepto nuevo."""
        summaries = []
        total_cost = 0.0
        # NOTA: ShinkaEvolve usa un batch query aquí. Por simplicidad y para evitar
        # la complejidad de multiprocessing, lo hacemos secuencial. Se puede optimizar a futuro.
        for concept in self.evaluated_since_last_meta:
            prompt = META_STEP1_USER_MSG.format(concept_summary=construct_concept_summary_for_meta(concept))
            summary = query_gemini_unstructured(prompt, META_STEP1_SYSTEM_MSG, self.model_cfg, self.model_cfg.temp_evaluation)
            if summary:
                summaries.append(summary)
            # El coste no está implementado en la función de llm_utils, así que lo omitimos por ahora.
        
        return "\n\n".join(summaries), total_cost

    def _step2_global_insights(self, summaries: str, best_concept: Optional[AlgorithmicConcept]) -> Tuple[Optional[str], float]:
        """Paso 2: Genera ideas globales a partir de los resúmenes."""
        prompt = META_STEP2_USER_MSG.format(
            individual_summaries=summaries,
            previous_insights=self.meta_scratch_pad or "Ninguna.",
            best_concept_info=construct_concept_summary_for_meta(best_concept) if best_concept else "Ninguno."
        )
        insights = query_gemini_unstructured(prompt, META_STEP2_SYSTEM_MSG, self.model_cfg, self.model_cfg.temp_evaluation)
        return insights, 0.0

    def _step3_generate_recommendations(self, insights: str, best_concept: Optional[AlgorithmicConcept]) -> Tuple[Optional[str], float]:
        """Paso 3: Genera recomendaciones a partir de las ideas globales."""
        prompt = META_STEP3_USER_MSG.format(
            global_insights=insights,
            previous_recommendations=self.meta_recommendations or "Ninguna.",
            max_recommendations=self.max_recommendations,
            best_concept_info=construct_concept_summary_for_meta(best_concept) if best_concept else "Ninguno."
        )
        recommendations = query_gemini_unstructured(prompt, META_STEP3_SYSTEM_MSG, self.model_cfg, self.model_cfg.temp_evaluation)
        return recommendations, 0.0

    def get_current_recommendations(self) -> Optional[str]:
        return self.meta_recommendations

    def save_state(self, filepath: str):
        """Guarda el estado del meta-aprendizaje en un fichero JSON."""
        state = {
            "meta_summary": self.meta_summary,
            "meta_scratch_pad": self.meta_scratch_pad,
            "meta_recommendations": self.meta_recommendations,
            "total_concepts_processed": self.total_concepts_processed,
            "evaluated_since_last_meta": [c.model_dump() for c in self.evaluated_since_last_meta]
        }
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"No se pudo guardar el estado de meta-aprendizaje: {e}")

    def load_state(self, filepath: str):
        """Carga el estado del meta-aprendizaje desde un fichero JSON."""
        if not Path(filepath).exists():
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.meta_summary = state.get("meta_summary")
            self.meta_scratch_pad = state.get("meta_scratch_pad")
            self.meta_recommendations = state.get("meta_recommendations")
            self.total_concepts_processed = state.get("total_concepts_processed", 0)
            self.evaluated_since_last_meta = [AlgorithmicConcept(**c) for c in state.get("evaluated_since_last_meta", [])]
            print("  🧠 Estado de meta-aprendizaje cargado.")
        except Exception as e:
            logger.error(f"No se pudo cargar el estado de meta-aprendizaje: {e}")