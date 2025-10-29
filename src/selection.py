# Fichero: src/selection.py
# MEJORA #6: Selección multiobjetivo inspirada en NSGA-II

import numpy as np
from typing import List
from src.concepts import AlgorithmicConcept

class DiversityAwareSelection:
    """
    Balancea fitness y diversidad inspirándose en NSGA-II y operadores de
    muestreo probabilístico.

    * La diversidad se calcula como \(1 - \overline{\cos(\theta)}\) sobre los
      embeddings de `EmbeddingClient`, por lo que depende indirectamente de la
      configuración del modelo (`cfg.model.embedding_model`).
    * Las puntuaciones de fitness provienen del score combinado generado por
      `AdaptiveScoring`, cuyos pesos responden a `cfg.evolution.num_generations`.
    * Los parámetros `n_parents` y `diversity_weight` se pueden mapear a claves
      Hydra (p. ej. incorporando `evolution.diversity_weight` en los overrides)
      para ajustar la explotación/exploración sin cambiar el código fuente.
    """
    
    def calculate_diversity_score(
        self, 
        concept: AlgorithmicConcept, 
        population: List[AlgorithmicConcept]
    ) -> float:
        """
        Evalúa la diversidad como \(1 - \overline{\cos(\theta)}\) respecto al resto
        de la población.

        Para cada par se calcula la similitud coseno usando los embeddings
        normalizados (la normalización depende del modelo elegido en
        `cfg.model.embedding_model`). Si el concepto es único o el resto carece
        de embeddings, se devuelve 1.0 para favorecer exploración en configuraciones
        con poblaciones pequeñas (`cfg.evolution.population_size`).

        Returns:
            Score de diversidad (0.0 a 1.0), donde 1.0 = muy diverso
        """
        if not concept.embedding or len(concept.embedding) == 0:
            return 0.0
        
        similarities = []
        
        for other in population:
            if other.id == concept.id:
                continue
            
            if not other.embedding or len(other.embedding) == 0:
                continue
            
            # Similitud coseno (producto interno de vectores normalizados)
            similarity = float(np.dot(concept.embedding, other.embedding))
            similarities.append(similarity)
        
        if not similarities:
            return 1.0  # Único concepto, máxima diversidad
        
        # Diversidad = 1 - similitud promedio
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
    
    def pareto_rank(self, population: List[AlgorithmicConcept]) -> List[int]:
        """
        Evalúa dominancia de Pareto sobre dos objetivos: fitness (`combined_score`)
        y diversidad.

        Para cada par \(i, j\) se consideran tolerancias de ±0.01 para evitar
        problemas numéricos en poblaciones grandes (`cfg.evolution.population_size`).
        Un individuo domina a otro si mejora en al menos un objetivo y no empeora
        en el restante. El resultado es la cuenta de individuos que dominan a cada
        elemento: cuanto menor el número, mejor su posición en el frente de Pareto.

        Returns:
            Lista de conteos de dominación (menor = mejor)
        """
        n = len(population)
        
        if n == 0:
            return []
        
        # Calcular diversidad para cada concepto
        diversities = [
            self.calculate_diversity_score(c, population) 
            for c in population
        ]
        
        # Contar cuántos conceptos dominan a cada uno
        dominated_count = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # i domina a j si es mejor en al menos un objetivo
                # y no peor en ninguno
                
                i_fitness = population[i].combined_score
                j_fitness = population[j].combined_score
                i_diversity = diversities[i]
                j_diversity = diversities[j]
                
                # i es mejor en fitness
                i_better_fitness = i_fitness > j_fitness + 0.01
                i_equal_fitness = abs(i_fitness - j_fitness) <= 0.01
                
                # i es mejor en diversidad
                i_better_diversity = i_diversity > j_diversity + 0.01
                i_equal_diversity = abs(i_diversity - j_diversity) <= 0.01
                
                # i domina a j si:
                # - Es mejor en uno y al menos igual en el otro
                if (i_better_fitness and (i_better_diversity or i_equal_diversity)) or \
                   (i_better_diversity and (i_better_fitness or i_equal_fitness)):
                    dominated_count[j] += 1
        
        return dominated_count
    
    def select_parents(
        self, 
        population: List[AlgorithmicConcept], 
        n_parents: int = 1,
        diversity_weight: float = 0.3
    ) -> List[AlgorithmicConcept]:
        """
        Selecciona padres combinando el frente de Pareto con los scores brutos.

        Se normalizan los `combined_score` y el ranking inverso de Pareto y se
        ponderan según `selection_score = (1 - w) * fitness + w * diversidad`,
        siendo `w` el parámetro `diversity_weight`. Este valor suele obtenerse de
        la configuración Hydra (p. ej. añadiendo `evolution.diversity_weight` a los
        overrides) para ajustar el balance exploración/explotación. `n_parents`
        puede mapearse a un override como `evolution.n_parents` o derivarse de
        `population_size` para mantener consistencia con el tamaño de cohorte.

        Args:
            population: Población actual
            n_parents: Número de padres a seleccionar
            diversity_weight: Peso de diversidad vs fitness (0.0 a 1.0)
        
        Returns:
            Lista de conceptos seleccionados como padres
        """
        if len(population) == 0:
            return []
        
        if len(population) <= n_parents:
            return population
        
        # Calcular ranking de Pareto
        pareto_ranks = self.pareto_rank(population)
        
        # Scores de fitness
        fitness_scores = np.array([c.combined_score for c in population])
        
        # Scores de ranking (invertir para que menor rank = mayor score)
        max_rank = max(pareto_ranks) if pareto_ranks else 0
        rank_scores = max_rank + 1 - np.array(pareto_ranks)
        
        # Normalizar a [0, 1]
        fitness_norm = fitness_scores / (fitness_scores.max() + 1e-6)
        rank_norm = rank_scores / (rank_scores.max() + 1e-6)
        
        # Combinar fitness y Pareto ranking con diversidad
        selection_scores = (
            (1 - diversity_weight) * fitness_norm + 
            diversity_weight * rank_norm
        )
        
        # Asegurar que todos los scores sean positivos
        selection_scores = np.maximum(selection_scores, 0.01)
        
        # Convertir a probabilidades
        probabilities = selection_scores / selection_scores.sum()
        
        # Seleccionar sin reemplazo
        try:
            indices = np.random.choice(
                len(population), 
                size=min(n_parents, len(population)), 
                replace=False, 
                p=probabilities
            )
            return [population[i] for i in indices]
        except:
            # Fallback: selección por ranking simple
            sorted_pop = sorted(
                population, 
                key=lambda c: c.combined_score, 
                reverse=True
            )
            return sorted_pop[:n_parents]
    
    def select_diverse_inspirations(
        self,
        parent: AlgorithmicConcept,
        population: List[AlgorithmicConcept],
        n_inspirations: int = 2
    ) -> List[AlgorithmicConcept]:
        """
        Prioriza inspiraciones alejadas del padre midiendo `1 - coseno`.

        Si el embedding del padre no está disponible (por ejemplo, porque el
        modelo definido en `cfg.model.embedding_model` devolvió un vector vacío),
        se recurre a una muestra aleatoria sin reemplazo. En los flujos Hydra es
        habitual vincular `n_inspirations` a un override como `evolution.n_inspirations`
        o reutilizar `cfg.evolution.refinement_steps` para mantener un número
        consistente de ideas complementarias durante la fase de mutación.

        Args:
            parent: Concepto padre
            population: Población total
            n_inspirations: Número de inspiraciones a seleccionar
        
        Returns:
            Lista de conceptos inspiración diversos
        """
        if not parent.embedding or len(parent.embedding) == 0:
            # Sin embedding, selección aleatoria
            candidates = [c for c in population if c.id != parent.id]
            n = min(n_inspirations, len(candidates))
            if n == 0:
                return []
            return list(np.random.choice(candidates, n, replace=False))
        
        # Calcular distancias al padre
        candidates_with_distance = []
        
        for concept in population:
            if concept.id == parent.id:
                continue
            
            if not concept.embedding or len(concept.embedding) == 0:
                continue
            
            # Distancia = 1 - similitud coseno
            similarity = float(np.dot(parent.embedding, concept.embedding))
            distance = 1.0 - similarity
            
            candidates_with_distance.append((concept, distance))
        
        if not candidates_with_distance:
            return []
        
        # Ordenar por distancia (descendente = más diversos)
        candidates_with_distance.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar los n_inspirations más diversos
        n = min(n_inspirations, len(candidates_with_distance))
        return [c for c, _ in candidates_with_distance[:n]]