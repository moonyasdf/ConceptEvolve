# Fichero: src/scoring.py
# MEJORA #5: Sistema de puntuación adaptativo y normalizado

import numpy as np
from typing import List
from src.concepts import ConceptScores

class AdaptiveScoring:
    """
    Sistema de scoring adaptativo que normaliza puntuaciones y ajusta pesos
    según el progreso de la evolución.

    Normalización:
        * Cada métrica acumula historial y solo se normaliza cuando existen al
          menos cinco muestras para reducir la varianza inicial. Este umbral se
          refleja en `normalize_scores` y evita distorsiones en las primeras
          generaciones.
        * Una vez alcanzado el historial mínimo, los valores se estandarizan con
          z-score y se reescalan al rango [1, 10] colocando el centro en 5.5 y
          usando un factor de 1.5. El resultado final se recorta para mantener
          compatibilidad con las escalas de los agentes evaluadores.

    Pesos adaptativos:
        * El progreso se estima con `generation / max_generations`, donde
          `max_generations` se alimenta desde `cfg.evolution.num_generations`
          (ver `configs/evolution/default.yaml`).
        * `w_novelty` decrece linealmente de 0.40 a 0.30 mientras
          `w_feasibility` aumenta de 0.10 a 0.20, manteniendo constantes los
          pesos de `potential` y `sophistication`.
        * El score combinado se multiplica por un factor de alineación en el
          rango [0.6, 1.0] derivado del `alignment_score`, lo que permite ajustar
          la influencia de `AlignmentValidator` (configurable mediante Hydra si
          se expone un campo específico).
    """
    
    def __init__(self, stagnation_threshold: int = 3, novelty_boost: float = 1.2):
        self.score_history = {
            'novelty': [], 'potential': [],
            'sophistication': [], 'feasibility': []
        }
        # MEJORA: Historial para detectar estancamiento
        self.best_score_history: List[float] = []
        self.stagnation_threshold = stagnation_threshold
        self.novelty_boost = novelty_boost
    
    def update_history(self, scores: ConceptScores):
        """Actualiza el historial de scores para normalización futura"""
        self.score_history['novelty'].append(scores.novelty)
        self.score_history['potential'].append(scores.potential)
        self.score_history['sophistication'].append(scores.sophistication)
        self.score_history['feasibility'].append(scores.feasibility)
    
    def normalize_scores(self, scores: ConceptScores) -> ConceptScores:
        """
        Aplica una normalización por z-score cuando existe historial suficiente.
        Las primeras generaciones (historial < 5) devuelven los valores crudos
        para evitar amplificar ruido. Al activar la normalización se calcula:

            z = (x - μ) / (σ + 1e-6)
            score_normalizado = clip(5.5 + 1.5 * z, 1.0, 10.0)

        De esta forma se conserva la escala [1, 10] usada por los prompts de los
        agentes. El umbral de cinco muestras puede ajustarse sobrescribiendo la
        clase cuando se trabaje con configuraciones Hydra que generen poblaciones
        más pequeñas.
        """
        if len(self.score_history['novelty']) < 5:
            # No hay suficiente historial, devolver scores originales
            return scores
        
        normalized = {}
        
        for metric in ['novelty', 'potential', 'sophistication', 'feasibility']:
            values = self.score_history[metric]
            mean = np.mean(values)
            std = np.std(values) + 1e-6  # Evitar división por cero
            
            raw_score = getattr(scores, metric)
            
            # Calcular z-score
            z_score = (raw_score - mean) / std
            
            # Normalizar a rango [1, 10] centrado en 5.5
            normalized_score = 5.5 + (z_score * 1.5)
            
            # Limitar al rango [1, 10]
            normalized[metric] = max(1.0, min(10.0, normalized_score))
        
        return ConceptScores(**normalized)
    
    def update_best_score_history(self, best_score_this_gen: float):
        """Añade el mejor score de la generación actual al historial."""
        self.best_score_history.append(best_score_this_gen)

    def _is_stagnated(self) -> bool:
        """Comprueba si el score máximo no ha mejorado en las últimas N generaciones."""
        if len(self.best_score_history) < self.stagnation_threshold:
            return False
        
        last_scores = self.best_score_history[-self.stagnation_threshold:]
        # Estancamiento si el score más reciente no es estrictamente mayor que los anteriores
        if last_scores[-1] <= max(last_scores[:-1]):
            print("  📉 ¡Estancamiento detectado! Aumentando peso de la novedad...")
            return True
            
        return False

    def calculate_combined_score(
        self, 
        scores: ConceptScores, 
        generation: int, 
        max_generations: int,
        alignment_score: float = 10.0
    ) -> float:
        """
        Calcula el score compuesto aplicando un calendario de pesos dependiente
        del progreso evolutivo y un multiplicador de alineación.

        Fórmulas clave:
            progreso = generation / max(max_generations, 1)
            w_novelty = 0.40 - 0.10 * progreso
            w_potential = 0.40
            w_sophistication = 0.10
            w_feasibility = 0.10 + 0.10 * progreso
            alignment_multiplier = 0.6 + 0.4 * (alignment_score / 10)

        El parámetro `max_generations` normalmente proviene de
        `cfg.evolution.num_generations`, lo que permite modificar la pendiente
        del schedule desde `configs/evolution/default.yaml` o mediante overrides.
        Alineaciones bajas reducen el score final hasta un 40 %, mientras que una
        alineación perfecta mantiene la suma ponderada intacta. Todos los valores
        se recortan a [0, 10] para mantener compatibilidad con los prompts.

        Args:
            scores: Scores del concepto
            generation: Generación actual
            max_generations: Total de generaciones configuradas via Hydra
            alignment_score: Score de alineación con problema (0-10)
        
        Returns:
            Score combinado (0-10)
        """
        # Progreso de evolución (0.0 a 1.0)
        progress = generation / max(max_generations, 1)
        
        # Pesos adaptativos
        # Novelty decrece, feasibility crece con el progreso
        w_novelty = 0.40 - (0.10 * progress)       # 0.40 -> 0.30
        w_potential = 0.40                          # Constante
        w_sophistication = 0.10                     # Constante
        w_feasibility = 0.10 + (0.10 * progress)   # 0.10 -> 0.20
        
        # MEJORA: Aumentar la novedad si hay estancamiento
        if self._is_stagnated():
            w_novelty *= self.novelty_boost
        
        # Re-normalizar pesos para que sumen 1
        total_weight = w_novelty + w_potential + w_sophistication + w_feasibility
        w_novelty /= total_weight
        w_potential /= total_weight
        w_sophistication /= total_weight
        w_feasibility /= total_weight

        # Score base
        base_score = (
            w_novelty * scores.novelty +
            w_potential * scores.potential +
            w_sophistication * scores.sophistication +
            w_feasibility * scores.feasibility
        )
        
        # Penalizar por desalineación (multiplicador de 0.6 a 1.0)
        alignment_multiplier = 0.6 + (0.4 * (alignment_score / 10.0))
        
        final_score = base_score * alignment_multiplier
        
        return max(0.0, min(10.0, final_score))
    
    def get_score_statistics(self) -> dict:
        """Retorna estadísticas del historial de scores"""
        stats = {}
        
        for metric, values in self.score_history.items():
            if len(values) > 0:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                stats[metric] = None
        
        return stats