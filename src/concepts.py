# Fichero: src/concepts.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid

# --- Esquemas para Salida Estructurada de Gemini ---

class ConceptScores(BaseModel):
    """Esquema para la evaluación de fitness de un concepto."""
    novelty: float = Field(
        default=5.0,
        description="Puntuación de 1 a 10 sobre qué tan original y fuera de lo común es la idea."
    )
    potential: float = Field(
        default=5.0,
        description="Puntuación de 1 a 10 sobre el potencial de la idea para superar a las soluciones SOTA."
    )
    sophistication: float = Field(
        default=5.0,
        description="Puntuación de 1 a 10 sobre la profundidad técnica y la elegancia del concepto."
    )
    feasibility: float = Field(
        default=5.0,
        description="Puntuación de 1 a 10 sobre qué tan viable es implementar la idea con la tecnología actual."
    )

class NoveltyDecision(BaseModel):
    """Esquema para la decisión del juez de novedad."""
    is_novel: bool = Field(
        default=True,
        description="True si el concepto es significativamente diferente, False si es una variación trivial."
    )
    explanation: str = Field(
        default="",
        description="Breve justificación de la decisión de novedad."
    )

class SystemRequirements(BaseModel):
    """Esquema para los requisitos extraídos de un concepto."""
    core_components: List[str] = Field(
        default_factory=list,
        description="Componentes técnicos clave necesarios (ej: 'GPU NVIDIA A100 24GB', 'FAISS con IVF-PQ')."
    )
    discovered_sub_problems: List[str] = Field(
        default_factory=list,
        description="Nuevos desafíos o sub-problemas identificados durante el diseño del concepto."
    )

# --- Estructura de Datos Principal de la Población ---

class AlgorithmicConcept(BaseModel):
    """Representa un individuo en la población de conceptos algorítmicos."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="")
    description: str = Field(default="")
    
    # Historial de refinamiento
    draft_history: List[str] = Field(default_factory=list)
    critique_history: List[str] = Field(default_factory=list)
    
    # Salidas extraídas
    system_requirements: SystemRequirements = Field(default_factory=SystemRequirements)
    
    # Metadatos evolutivos
    generation: int = Field(default=0)
    parent_id: Optional[str] = Field(default=None)
    inspiration_ids: List[str] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)
    
    # Puntuaciones de fitness
    scores: Optional[ConceptScores] = Field(default=None)
    combined_score: float = Field(default=0.0)

    def get_full_prompt_text(self) -> str:
        """Genera un texto completo que representa el concepto para prompts."""
        return f"## Título: {self.title}\n\n### Descripción\n{self.description}"