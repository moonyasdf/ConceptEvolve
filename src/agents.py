"""Gemini-powered agent pipeline for evolving algorithmic concepts.

This module orchestrates the end-to-end lifecycle of concept evolution: idea
creation, critique, refinement, scoring, novelty filtering, requirements
extraction, and alignment validation.  Each agent wraps a Gemini prompt strategy
and, where needed, a structured response schema that feeds downstream steps in
the evolution loop.
"""

# Fichero: src/agents.py
# MEJORAS: #4 (Mejores prompts), #7 (Refinamiento contextual), #8 (Validación alineación)

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from src.concepts import AlgorithmicConcept, ConceptScores, NoveltyDecision, SystemRequirements
from src.llm_utils import query_gemini_structured, query_gemini_unstructured
from src.prompts import PromptSampler
from src.prompts import (
    IDEA_GENERATION_EXAMPLES,
    SYSTEM_MSG_IDEA_GENERATOR,
    SYSTEM_MSG_CRITIC,
    SYSTEM_MSG_EVALUATOR,
    SYSTEM_MSG_REFINEMENT,
    SYSTEM_MSG_ALIGNMENT,
    SYSTEM_MSG_NOVELTY_JUDGE
)
from src.config import model_config

class IdeaGenerator:
    """Produce seed, mutated, and refined algorithmic concepts via Gemini prompts.

    The generator is responsible for the creative side of the pipeline.  It
    bootstraps first-generation ideas with few-shot exemplars, performs
    mutation/crossover against prior concepts, and rewrites descriptions after
    critiques.  Each stage relies on structured Gemini responses that can be
    promoted directly to :class:`AlgorithmicConcept` objects for downstream
    agents.
    """

    def __init__(self):
        self.prompt_sampler = PromptSampler()

    def generate_initial_concept(
        self, 
        problem_description: str, 
        generation: int,
        model_cfg: DictConfig  # MEJORA: Recibe la configuración del modelo
    ) -> Optional[AlgorithmicConcept]:
        """Generate a seed concept from the problem description via few-shot prompting.

        The prompt concatenates ``IDEA_GENERATION_EXAMPLES`` with the user brief
        and requests a structured ``InitialIdea`` payload containing a title and
        long-form technical description.  The structured output is promoted to
        an :class:`AlgorithmicConcept`, providing the root node for downstream
        critique, scoring, and refinement agents.

        Args:
            problem_description: Natural-language description of the task the
                concept must solve.
            generation: Index used to tag the resulting concept within the
                evolutionary run.

        Returns:
            An :class:`AlgorithmicConcept` if Gemini produces a complete
            ``InitialIdea`` payload, otherwise ``None`` when generation fails.
        """
        prompt = f"""
{IDEA_GENERATION_EXAMPLES}

### 🎯 TU TURNO:

**Problema del Usuario:**
{problem_description}

**Instrucciones:**
Genera un concepto algorítmico que:
1. Sea TÉCNICAMENTE ESPECÍFICO (nombra arquitecturas, algoritmos, métricas concretas)
2. Combine al menos 2 ideas existentes de forma NO OBVIA
3. Identifique un INSIGHT CLAVE que justifique por qué funcionaría
4. Sea IMPLEMENTABLE pero NO TRIVIAL
5. Incluya componentes técnicos concretos en la descripción

**Formato requerido:**
- title: Título descriptivo y técnico del concepto
- description: Descripción detallada (mínimo 200 palabras) que incluya:
  * Explicación del enfoque principal
  * Componentes técnicos específicos
  * Cómo se integran los componentes
  * Por qué este enfoque podría funcionar (insight clave)
  * Detalles de implementación relevantes
"""
        
        class InitialIdea(BaseModel):
            title: str = Field(description="Título técnico del concepto")
            description: str = Field(description="Descripción detallada con componentes técnicos")
        
        # MEJORA: Se pasa `model_cfg` y la temperatura correcta a la función de llm_utils
        response = query_gemini_structured(
            prompt, 
            SYSTEM_MSG_IDEA_GENERATOR, 
            InitialIdea,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_generation
        )
        
        if response and response.title and response.description:
            return AlgorithmicConcept(
                title=response.title,
                description=response.description,
                generation=generation
            )
        return None

    def mutate_or_crossover(
        self,
        parent: AlgorithmicConcept,
        inspirations: List[AlgorithmicConcept],
        generation: int,
        problem_description: str,
        model_cfg: DictConfig  # MEJORA: Recibe la configuración del modelo
    ) -> Optional[AlgorithmicConcept]:
        """Mutate or crossover a parent concept with inspiration concepts.

        A dynamic mutation task sampled from :class:`PromptSampler` adds
        variability to the crossover instructions.  The prompt composes the
        parent, a set of inspirational snippets, and explicit novelty
        guardrails before requesting a structured ``MutatedIdea`` response.  The
        structured schema keeps downstream agents agnostic to whether the concept
        originated from mutation or fresh generation.

        Args:
            parent: Concept selected as the primary genome to evolve.
            inspirations: Additional concepts that supply contrasting ideas or
                building blocks for crossover.
            generation: Target generation index for the offspring.
            problem_description: Original user problem used to anchor the
                mutation instructions.

        Returns:
            A novel :class:`AlgorithmicConcept` instance or ``None`` if Gemini
            fails to produce a differentiated idea.
        """
        inspiration_texts = "\n\n".join([
            f"**Inspiración {i+1}:** {insp.title}\n{insp.description[:300]}..."
            for i, insp in enumerate(inspirations)
        ])
        mutation_task = self.prompt_sampler.sample_mutation_prompt()
        prompt = f"""
### 🎯 PROBLEMA ORIGINAL:
{problem_description}

### 👨‍👩‍👧 IDEA PADRE (A EVOLUCIONAR):
**{parent.title}**
{parent.description}

### 💡 IDEAS DE INSPIRACIÓN:
{inspiration_texts}

{mutation_task}

**REQUISITOS:**
1. Debe ser significativamente DIFERENTE al padre (no solo más detallado)
2. Debe incluir al menos UN componente técnico completamente nuevo
3. Debe mantener alineación con el problema original
4. Debe ser técnicamente específico

**Formato:**
- title: Título que refleje la novedad
- description: Descripción detallada con componentes técnicos
"""
        
        class MutatedIdea(BaseModel):
            title: str
            description: str

        # Structured response ensures mutated concepts have the same interface as fresh seeds.
        response = query_gemini_structured(
            prompt,
            SYSTEM_MSG_IDEA_GENERATOR,
            MutatedIdea,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_generation
        )

        if response and response.title and response.description:
            return AlgorithmicConcept(
                title=response.title,
                description=response.description,
                generation=generation,
                parent_id=parent.id,
                inspiration_ids=[insp.id for insp in inspirations]
            )
        return None

    def refine(
        self, 
        concept: AlgorithmicConcept,
        all_critiques: List[str], 
        addressed_points: List[str], 
        problem_description: str,
        model_cfg: DictConfig  # MEJORA: Recibe la configuración del modelo
    ) -> Tuple[str, List[str]]:
        """Refine an existing concept using accumulated critiques and history.

        The refinement prompt threads together the current draft, every critique
        the concept has received, and a list of previously addressed issues.  By
        requesting a structured ``RefinementOutput`` schema, Gemini must produce
        both a fully rewritten description and a list of critique points tackled
        in the current iteration.  The caller merges ``newly_addressed_points``
        into ``addressed_points`` to drive multi-round refinement loops.

        Args:
            concept: The concept whose description is being rewritten.
            all_critiques: Chronological critiques sourced from
                :class:`ConceptCritic`.
            addressed_points: Critique snippets already resolved in prior loops.
            problem_description: Original user prompt for context anchoring.

        Returns:
            A tuple containing the refined description and an updated list of
            addressed critique points.
        """
        # Track which critiques are already resolved so Gemini can prioritize remaining gaps.
        addressed_summary = "\n- ".join(addressed_points) if addressed_points else "Ninguno aún"
        all_critiques_text = "\n\n---\n\n".join([
            f"**Ronda de Críticas {i+1}:**\n{c}"
            for i, c in enumerate(all_critiques)
        ])
        prompt = f"""
### 🎯 PROBLEMA ORIGINAL:
{problem_description}

### 📄 BORRADOR ACTUAL:
**{concept.title}**
{concept.description}

### 🔍 TODAS LAS CRÍTICAS RECIBIDAS:
{all_critiques_text}

### ✅ PUNTOS YA ABORDADOS EN REFINAMIENTOS PREVIOS:
{addressed_summary}

### 📋 TAREA DE REFINAMIENTO:

1. **Identificar críticas pendientes:** ¿Qué críticas NO se han abordado completamente?

2. **Reescribir descripción** para:
   - Abordar las críticas pendientes ESPECÍFICAMENTE con soluciones técnicas concretas
   - Mantener y fortalecer los puntos ya mejorados
   - NO diluir la creatividad original
   - Añadir detalles técnicos donde se requiera

3. **Ejemplo de abordaje específico:**
   - ❌ Mal: "Se optimizará el rendimiento"
   - ✅ Bien: "Para reducir latencia, se implementa caché LRU con TTL adaptativo basado en frecuencia de queries, reduciendo lookups de O(n) a O(1)"

**IMPORTANTE:** La descripción refinada debe REEMPLAZAR completamente a la anterior, pero manteniendo la esencia innovadora.

**Formato de salida:**
- refined_description: Descripción completa mejorada
- newly_addressed_points: Lista de puntos específicos abordados en ESTE refinamiento
"""
        
        class RefinementOutput(BaseModel):
            refined_description: str = Field(description="Descripción completa refinada")
            newly_addressed_points: List[str] = Field(description="Puntos abordados en este refinamiento")
        
        # MEJORA: Se pasa `model_cfg` y la temperatura correcta
        response = query_gemini_structured(
            prompt, 
            SYSTEM_MSG_REFINEMENT, 
            RefinementOutput,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_refinement
        )
        
        if response:
            new_addressed = addressed_points + response.newly_addressed_points
            return response.refined_description, new_addressed
        
        return concept.description, addressed_points

class ConceptCritic:
    """Deliver structured critiques that drive the refinement loop.

    The critic asks Gemini for an unstructured but highly prescriptive review of
    the concept, covering alignment, logic, feasibility, novelty, and potential
    failure modes.  Its textual feedback is fed to :meth:`IdeaGenerator.refine`
    so the generator can resolve outstanding issues in subsequent iterations.
    """
    
    def run(self, concept: AlgorithmicConcept, problem_description: str, model_cfg: DictConfig) -> str:
        """Request a detailed critique anchored to the original problem statement.

        The prompt enumerates five lenses—alignment, logical consistency,
        viability, novelty, and implementation risks—and instructs Gemini to
        produce bullet-point guidance.  Because the output is observational text
        rather than a structured schema, ``query_gemini_unstructured`` is used
        here.  The raw critique is appended to the concept's history for later
        refinements.

        Args:
            concept: Candidate concept being stress-tested.
            problem_description: Original problem description used to validate
                alignment.

        Returns:
            A markdown-compatible critique string.
        """
        prompt = f"""
### 🎯 PROBLEMA ORIGINAL DEL USUARIO:
{problem_description}

### 📄 PROPUESTA A CRITICAR:
**{concept.title}**
{concept.description}

### 📋 TU TAREA:
Escribe una crítica DETALLADA y ESPECÍFICA identificando:

**1. Alineación con el Problema:**
- ¿Aborda TODOS los requisitos clave del problema original?
- ¿Hay desviaciones del objetivo principal?

**2. Fallos Lógicos:**
- ¿Hay inconsistencias en el razonamiento?
- ¿Hay pasos que no se siguen lógicamente?

**3. Viabilidad Técnica:**
- ¿Qué suposiciones no realistas hace?
- ¿Qué cuellos de botella computacionales existirían?
- ¿Qué componentes serían difíciles de implementar?

**4. Novedad Real vs Superficial:**
- ¿Es genuinamente novedoso o re-branding de técnicas existentes?
- Si usa técnicas conocidas, ¿la combinación es no-trivial?

**5. Potenciales Problemas de Implementación:**
- ¿Qué podría fallar en la práctica?
- ¿Qué edge cases no se consideran?

**FORMATO:** Lista de puntos específicos con justificación técnica.

**SÉ ESPECÍFICO:** No digas "podría tener problemas". Di "El componente X requeriría Y recursos, lo cual es prohibitivo porque Z".
"""
        
        # Unstructured call preserves free-form markdown critiques for human legibility.
        return query_gemini_unstructured(
            prompt, 
            SYSTEM_MSG_CRITIC,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_critique
        )

class ConceptEvaluator:
    """Score concepts across novelty, potential impact, sophistication, and viability.

    The evaluator converts Gemini's structured assessment into a
    :class:`ConceptScores` Pydantic model.  These scores feed selection logic in
    the evolution core and influence which concepts continue to future
    generations.
    """
    
    def run(self, concept: AlgorithmicConcept, model_cfg: DictConfig) -> Optional[ConceptScores]:
        """Ask Gemini to quantify multiple fitness dimensions for a concept.

        The prompt outlines concrete rubrics for novelty, potential impact,
        sophistication, and viability.  ``query_gemini_structured`` enforces the
        :class:`ConceptScores` schema so downstream selection operators receive
        normalized floats instead of free-form prose.

        Args:
            concept: Concept under evaluation.

        Returns:
            A :class:`ConceptScores` instance or ``None`` if Gemini cannot
            respect the schema.
        """
        prompt = f"""
### 📄 IDEA A EVALUAR:
**{concept.title}**
{concept.description}

### 📊 CRITERIOS DE EVALUACIÓN (1.0 a 10.0):

Evalúa esta idea según los siguientes criterios. **Prioriza creatividad e innovación.**

**Novedad (40% del peso total):**
- 9-10: Idea completamente nueva, sorprendente, no se había visto antes
- 7-8: Combinación no obvia de conceptos con twist innovador
- 5-6: Extensión interesante de trabajo existente
- 3-4: Variación menor de métodos conocidos
- 1-2: Idea ya establecida

**Potencial (40% del peso total):**
- 9-10: Si funciona, revolucionaría el campo
- 7-8: Mejora significativa probable sobre SOTA (>5%)
- 5-6: Mejora marginal probable (2-5%)
- 3-4: Beneficio incierto
- 1-2: Poco probable que mejore baselines

**Sofisticación (10% del peso total):**
- 9-10: Profundidad técnica excepcional
- 7-8: Técnicamente sólido, componentes no triviales
- 5-6: Técnicamente correcto pero straightforward
- 3-4: Técnicamente simple
- 1-2: Superficial

**Viabilidad (10% del peso total):**
- 9-10: Implementable con recursos estándar
- 7-8: Recursos razonables (4-8 GPUs, semanas)
- 5-6: Recursos significativos pero accesibles
- 3-4: Requiere recursos excepcionales
- 1-2: No implementable actualmente

**IMPORTANTE:** No castigues excesivamente ideas muy novedosas por viabilidad media-baja.
"""
        
        # Structured evaluation ensures each rubric is returned as a normalized float.
        return query_gemini_structured(
            prompt, 
            SYSTEM_MSG_EVALUATOR, 
            ConceptScores,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_evaluation
        )

class NoveltyJudge:
    """Detect semantic duplicates so the population maintains diversity.

    The judge compares two candidate concepts and returns a structured
    :class:`NoveltyDecision` describing whether they are meaningfully distinct.
    This signal helps the population manager avoid promoting near-identical
    concepts.
    """
    
    def run(
        self, 
        concept1: AlgorithmicConcept, 
        concept2: AlgorithmicConcept,
        model_cfg: DictConfig
    ) -> Optional[NoveltyDecision]:
        """Request a binary novelty decision between two concepts.

        Gemini receives both titles and descriptions plus explicit decision
        criteria and must respond with ``is_novel`` and an explanation via the
        :class:`NoveltyDecision` schema.  The low temperature favors consistent
        duplicate detection.

        Args:
            concept1: First concept to compare.
            concept2: Second concept to compare.

        Returns:
            A structured novelty verdict or ``None`` when the schema is not met.
        """
        
        prompt = f"""
### 🔍 COMPARACIÓN DE CONCEPTOS:

**Concepto 1:**
Título: {concept1.title}
Descripción: {concept1.description}

**Concepto 2:**
Título: {concept2.title}
Descripción: {concept2.description}

### ❓ PREGUNTA:
¿Son estos dos conceptos fundamentalmente LA MISMA IDEA?

**Son la MISMA idea (is_novel = false) si:**
- Mismo algoritmo core aunque diferente terminología
- Mismos componentes principales con variaciones triviales
- Uno es más detallado pero sin cambios sustanciales

**Son ideas DIFERENTES (is_novel = true) si:**
- Arquitecturas fundamentalmente distintas
- Mecanismos core diferentes
- Componentes adicionales NO TRIVIALES que cambian el enfoque

**Formato de salida:**
- is_novel: true si son DIFERENTES, false si son LA MISMA
- explanation: 2-3 oraciones justificando la decisión
"""
        
        # Structured decision keeps novelty filtering machine-readable for the FAISS index manager.
        return query_gemini_structured(
            prompt, 
            SYSTEM_MSG_NOVELTY_JUDGE, 
            NoveltyDecision,
            model_cfg=model_cfg,
            temperature=0.3
        )

class RequirementsExtractor:
    """Map concepts to concrete system requirements and research sub-problems.

    The extractor converts descriptive concepts into actionable engineering
    checklists and open research questions.  The resulting
    :class:`SystemRequirements` objects help the execution planner scope future
    work.
    """
    
    def run(self, concept: AlgorithmicConcept, model_cfg: DictConfig) -> Optional[SystemRequirements]:
        """Derive implementation components and follow-up research tasks.

        Gemini receives the concept description along with examples of the level
        of specificity expected for ``core_components`` and
        ``discovered_sub_problems``.  ``query_gemini_structured`` guarantees that
        the response fits the :class:`SystemRequirements` schema.

        Args:
            concept: Concept whose requirements are being extracted.

        Returns:
            A :class:`SystemRequirements` instance or ``None`` when Gemini cannot
            honor the schema.
        """
        
        prompt = f"""
### 📄 DESCRIPCIÓN DEL ALGORITMO:
**{concept.title}**
{concept.description}

### 📋 TAREA:
Analiza la descripción y extrae:

**1. core_components:** Lista de componentes de software/hardware ESPECÍFICOS requeridos
   Ejemplos:
   - "GPU NVIDIA con mínimo 24GB VRAM (A100 o superior)"
   - "Base de datos vectorial FAISS con índice IVF-PQ"
   - "Modelo de embeddings text-embedding-004 de Google"
   - "Framework HuggingFace Transformers >=4.30"
   - "Modelo cross-encoder ms-marco-MiniLM"

**2. discovered_sub_problems:** Lista de NUEVOS problemas/desafíos de investigación que la implementación implicaría
   Ejemplos:
   - "Diseñar métrica de fusión de contextos que balancee relevancia y diversidad"
   - "Optimizar construcción de grafo para escalar a >10M nodos sin degradación"
   - "Crear dataset de entrenamiento para modelo discriminador de coherencia lógica"
   - "Implementar algoritmo de backtracking eficiente que no re-genere pasos innecesariamente"

**IMPORTANTE:** Sé específico y técnico. No pongas "se necesita una GPU", pon "GPU con arquitectura Ampere o superior, 24GB+ VRAM".
"""
        
        # Structured extraction returns normalized component lists for scheduling and build planning.
        return query_gemini_structured(
            prompt, 
            "Eres un arquitecto de sistemas de IA experto en análisis de requisitos técnicos.", 
            SystemRequirements,
            model_cfg=model_cfg,
            temperature=0.3
        )


class AlignmentValidator:
    """Check problem-solution alignment before promoting a concept.

    The validator scores whether a concept directly addresses the original user
    problem, using Gemini's structured output to return per-dimension scores plus
    a boolean verdict.  This gate keeps the pipeline focused on relevant ideas
    despite aggressive mutation.
    """
    
    def run(
        self, 
        concept: AlgorithmicConcept, 
        problem_description: str,
        model_cfg: DictConfig
    ) -> Tuple[bool, str, float]:
        """Evaluate coverage, scope fidelity, and directness for a concept.

        The prompt instructs Gemini to reason about alignment across three axes,
        compute an average, and emit a structured payload with scores, verdict,
        and rationale.  The tuple returned here is consumed by the evolution
        loop to drop misaligned ideas early.

        Args:
            concept: Concept under validation.
            problem_description: Original problem statement to validate against.

        Returns:
            Tuple of ``(is_aligned, explanation, alignment_score)`` where
            ``alignment_score`` is the mean of the three axis scores.
        """
        prompt = f"""
### 🎯 PROBLEMA ORIGINAL:
{problem_description}

### 💡 SOLUCIÓN PROPUESTA:
**{concept.title}**
{concept.description}

### 📋 TAREA DE EVALUACIÓN:
Evalúa si la solución propuesta REALMENTE resuelve el problema original.

**Analiza en 3 dimensiones (puntúa cada una de 0.0 a 10.0):**

1. **coverage_score:** ¿Aborda TODOS los requisitos explícitos del problema?
   - 10.0: Cubre todos los requisitos completamente
   - 5.0: Cubre algunos requisitos, otros parcialmente
   - 0.0: Ignora requisitos clave

2. **scope_fidelity_score:** ¿Se mantiene en el scope del problema o se desvía?
   - 10.0: Totalmente enfocado en el problema planteado
   - 5.0: Aborda el problema pero con desviaciones significativas
   - 0.0: Resuelve un problema completamente diferente

3. **directness_score:** ¿Es una solución directa o tangencial?
   - 10.0: Solución directa al problema
   - 5.0: Solución que requiere pasos adicionales significativos
   - 0.0: Solo facilita que alguien más resuelva el problema

**Cálculo:**
- alignment_score = (coverage_score + scope_fidelity_score + directness_score) / 3
- is_aligned = true si alignment_score >= 6.0, false en caso contrario

**Formato de salida:**
- coverage_score: float
- scope_fidelity_score: float
- directness_score: float
- alignment_score: float (promedio de los 3)
- is_aligned: boolean
- explanation: 2-3 oraciones específicas justificando
"""
        
        class AlignmentResult(BaseModel):
            coverage_score: float
            scope_fidelity_score: float
            directness_score: float
            alignment_score: float
            is_aligned: bool
            explanation: str
        
        # MEJORA: Se pasa `model_cfg` y una temperatura baja y fija
        result = query_gemini_structured(
            prompt, 
            SYSTEM_MSG_ALIGNMENT, 
            AlignmentResult,
            model_cfg=model_cfg,
            temperature=0.3
        )
        
        if result:
            return result.is_aligned, result.explanation, result.alignment_score
        
        return True, "Error en validación - aceptado por defecto", 6.0