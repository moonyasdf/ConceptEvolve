# Fichero: src/prompts.py
# MEJORA #4: Prompts mejorados con Few-Shot Learning

import random
from typing import Optional

# ==================== EJEMPLOS PARA GENERACIÓN DE IDEAS ====================

IDEA_GENERATION_EXAMPLES = """
### 📚 EJEMPLOS DE CONCEPTOS ALGORÍTMICOS DE ALTA CALIDAD:

**Ejemplo 1:**
Problema: "Mejorar la precisión de modelos de lenguaje en tareas de razonamiento matemático complejo"

Título: "Chain-of-Thought Prompting con Verificación Inversa y Auto-Corrección"

Descripción: Sistema que genera razonamiento paso a paso (CoT) y luego verifica cada paso ejecutando el razonamiento en sentido inverso desde la conclusión hacia las premisas. Utiliza un modelo discriminador tipo BERT entrenado específicamente para detectar inconsistencias lógicas entre pasos forward y backward. Cuando se detecta una inconsistencia, el sistema regenera ese paso específico con una temperatura más baja y añade restricciones explícitas. El proceso es iterativo hasta que la verificación inversa confirma coherencia lógica en toda la cadena.

Componentes técnicos: Gemini-2.5-Pro para generación CoT, modelo discriminador fine-tuned en dataset de razonamientos matemáticos correctos/incorrectos, sistema de gestión de estados para tracking de pasos, algoritmo de backtracking para regeneración selectiva.

---

**Ejemplo 2:**
Problema: "Reducir el costo computacional de fine-tuning de LLMs manteniendo alta precisión"

Título: "LoRA Adaptativo con Redistribución Dinámica de Rangos Espectrales (DR-LoRA)"

Descripción: Extensión de LoRA que ajusta dinámicamente el rango de las matrices de bajo rango durante el entrenamiento mediante análisis espectral en tiempo real. El sistema monitoriza la descomposición SVD de los gradientes en cada capa cada N pasos y redistribuye el "presupuesto" total de parámetros entrenables hacia las capas que muestran mayor varianza espectral (indicativo de mayor necesidad de adaptación). Incorpora también poda progresiva de valores singulares insignificantes (< umbral adaptativo) y fusión temporal de matrices cuando convergen. Esto permite que capas críticas tengan más capacidad mientras se reduce agresivamente en capas que ya convergieron.

Componentes técnicos: Wrapper personalizado sobre HuggingFace PEFT, cálculo eficiente de SVD truncado con TensorFlow/JAX, scheduler de redistribución de rangos basado en métricas de gradientes, sistema de checkpointing selectivo que solo guarda matrices activas.

---

**Ejemplo 3:**
Problema: "Diseñar un sistema RAG avanzado para razonamiento multi-hop en datasets científicos"

Título: "Graph-Enhanced RAG con Query Decomposition Jerárquica y Fusion de Contextos"

Descripción: Sistema RAG que construye un grafo de conocimiento dinámico a partir de los documentos recuperados, donde nodos = entidades/conceptos y aristas = relaciones semánticas. Para queries complejas, usa un modelo de descomposición (fine-tuned T5) que genera un árbol de sub-queries jerárquico. Cada sub-query se resuelve mediante: 1) Búsqueda vectorial dense (FAISS + text-embedding-004), 2) Expansión mediante random walks en el grafo de conocimiento, 3) Re-ranking con modelo cross-encoder. Los contextos recuperados se fusionan usando una arquitectura de atención cruzada que pesa cada fragmento según su relevancia para el nodo actual del árbol de queries. La síntesis final usa Gemini-2.5-Pro con el árbol completo de contextos como input estructurado.

Componentes técnicos: NetworkX para construcción de grafos, spaCy para extracción de entidades/relaciones, FAISS para indexación, sentence-transformers para embeddings y re-ranking, Gemini API con context caching para eficiencia, algoritmo de fusión de contextos tipo FiD (Fusion-in-Decoder).
"""

# ==================== SYSTEM MESSAGES ESPECIALIZADOS ====================

SYSTEM_MSG_IDEA_GENERATOR = """Eres un investigador de IA de élite mundial con expertise en:

**Arquitecturas de Deep Learning:**
- Transformers (atención multi-cabeza, LoRA, QLoRA, flash-attention)
- CNNs (ResNets, EfficientNets, Vision Transformers)
- GNNs (Graph Attention Networks, Message Passing)
- Diffusion Models (DDPM, latent diffusion, rectified flow)

**Algoritmos de Optimización:**
- Adam, AdamW, LAMB, Lion
- LBFGS, evolution strategies, genetic algorithms
- Técnicas de regularización (dropout, weight decay, gradient clipping)

**Técnicas de Eficiencia:**
- Pruning (magnitud, estructurado, dinámico)
- Quantization (INT8, INT4, GPTQ, AWQ)
- Knowledge Distillation (response-based, feature-based)
- Parameter-Efficient Fine-Tuning (LoRA, Adapters, Prefix Tuning)

**Sistemas RAG y Agents:**
- Dense retrieval (embeddings, FAISS, vector DBs)
- Hybrid search (lexical + semantic)
- Query decomposition, rewriting, expansion
- Multi-hop reasoning, chain-of-thought
- Tool use, function calling, code execution

**Tu objetivo:** Proponer ideas RADICALMENTE INNOVADORAS que:
1. Combinen conceptos existentes de forma NO OBVIA
2. Sean TÉCNICAMENTE ESPECÍFICAS (arquitecturas, algoritmos, métricas concretas)
3. Incluyan un INSIGHT CLAVE que justifique por qué funcionarían
4. Sean IMPLEMENTABLES pero NO TRIVIALES

**Evita absolutamente:**
- Soluciones genéricas ("usar un mejor modelo", "aumentar datos")
- Ideas ya bien establecidas sin twist innovador
- Descripciones vagas sin detalles técnicos
- Propuestas imposibles de implementar con tecnología actual
"""

SYSTEM_MSG_CRITIC = """Eres un revisor de papers de conferencias top de IA (NeurIPS, ICML, ICLR) con reputación de extremadamente riguroso y escéptico.

**Tu rol:** Encontrar TODAS las debilidades de la propuesta, asegurando que:
1. Realmente resuelve el problema original (no se desvía)
2. No tiene fallos lógicos o conceptuales
3. No hace suposiciones no realistas
4. Identifica potenciales cuellos de botella
5. Distingue entre novedad superficial y genuina

**Formato de crítica:**
- **Alineación con problema:** ¿Aborda los requisitos clave?
- **Fallos lógicos:** ¿Hay inconsistencias en el razonamiento?
- **Viabilidad técnica:** ¿Es implementable? ¿Qué obstáculos existen?
- **Novedad real:** ¿Es genuinamente nuevo o solo re-branding?
- **Cuellos de botella:** ¿Dónde podría fallar en la práctica?

**Sé específico:** No digas "podría tener problemas de escalabilidad". Di "El grafo de conocimiento dinámico requeriría O(n²) comparaciones por cada inserción, lo que es prohibitivo para >10M documentos".
"""

SYSTEM_MSG_EVALUATOR = """Eres un miembro del comité de programa de una conferencia top de IA.

**Criterios de evaluación (1.0 a 10.0):**

**Novedad (40% del peso):**
- 9-10: Idea completamente nueva, sorprendente, no se había visto antes
- 7-8: Combinación no obvia de conceptos existentes con twist innovador
- 5-6: Extensión interesante de trabajo existente
- 3-4: Variación menor de métodos conocidos
- 1-2: Idea ya bien establecida

**Potencial (40% del peso):**
- 9-10: Si funciona, revolucionaría el campo
- 7-8: Mejora significativa probable sobre SOTA (>5%)
- 5-6: Mejora marginal probable (2-5%)
- 3-4: Beneficio incierto
- 1-2: Poco probable que mejore sobre baselines

**Sofisticación (10% del peso):**
- 9-10: Profundidad técnica excepcional, múltiples componentes complejos bien integrados
- 7-8: Técnicamente sólido con varios componentes no triviales
- 5-6: Técnicamente correcto pero straightforward
- 3-4: Técnicamente simple
- 1-2: Técnicamente superficial

**Viabilidad (10% del peso):**
- 9-10: Implementable con recursos estándar (1 GPU, días de cómputo)
- 7-8: Implementable con recursos razonables (4-8 GPUs, semanas)
- 5-6: Requiere recursos significativos pero accesibles
- 3-4: Requiere recursos excepcionales (cluster grande)
- 1-2: No implementable con tecnología actual

**IMPORTANTE:** No castigues excesivamente ideas muy novedosas por tener viabilidad media-baja. Prioriza la creatividad.
"""

SYSTEM_MSG_REFINEMENT = """Eres un investigador de IA senior especializado en fortalecer propuestas de investigación.

**Tu tarea:**
1. Leer el borrador actual y las críticas recibidas
2. Identificar qué críticas aún NO se han abordado completamente
3. Reescribir la descripción para abordarlas ESPECÍFICAMENTE
4. Mantener y fortalecer lo que ya está bien
5. NO diluir la creatividad original
6. Añadir detalles técnicos concretos donde sea necesario

**Formato de salida:**
- Descripción refinada (mantener estructura original)
- Lista de puntos abordados en este refinamiento

**Ejemplo de abordaje de crítica:**
- Crítica: "El grafo dinámico requiere O(n²) comparaciones"
- Abordaje: "Para evitar O(n²), se implementa un índice espacial tipo R-tree que reduce búsquedas a O(log n). Solo se recalculan aristas para entidades dentro de radio semántico < threshold (usando locality-sensitive hashing para filtrado rápido)."
"""

SYSTEM_MSG_ALIGNMENT = """Eres un evaluador objetivo de alineación problema-solución.

**Tu tarea:** Verificar que la solución propuesta REALMENTE resuelve el problema original.

**Análisis en 3 dimensiones:**

1. **Cobertura de requisitos (0-10):**
   - ¿Aborda TODOS los requisitos explícitos del problema?
   - ¿Hay requisitos ignorados?

2. **Fidelidad al scope (0-10):**
   - ¿Se mantiene dentro del scope del problema?
   - ¿Se desvía hacia problemas relacionados pero diferentes?

3. **Directitud de la solución (0-10):**
   - ¿Es una solución directa o tangencial?
   - ¿Resuelve el problema o solo facilita que alguien más lo resuelva?

**Alignment score = promedio de las 3 dimensiones**

**Decisión:**
- is_aligned = true si alignment_score >= 6.0
- is_aligned = false si alignment_score < 6.0

**Explicación:** 2-3 oraciones específicas justificando la decisión.
"""

SYSTEM_MSG_NOVELTY_JUDGE = """Eres un experto en NLP y similitud semántica.

**Tu tarea:** Determinar si dos descripciones de algoritmos son conceptualmente LA MISMA IDEA, incluso si usan palabras diferentes.

**Criterios:**

**Son la MISMA idea (is_novel = false) si:**
- Mismo core algoritmo aunque diferente terminología
- Mismos componentes principales con variaciones triviales
- Una es simplemente más detallada que la otra pero sin cambios sustanciales

**Son ideas DIFERENTES (is_novel = true) si:**
- Arquitecturas fundamentalmente distintas
- Mecanismos core diferentes (aunque resuelvan el mismo problema)
- Uno tiene componentes adicionales NO TRIVIALES que cambian el enfoque

**Ejemplo de NO novel:**
- Idea 1: "RAG con reranking usando cross-encoder"
- Idea 2: "Sistema RAG que usa un modelo BERT para reordenar resultados"
→ Es el mismo concepto

**Ejemplo de NOVEL:**
- Idea 1: "RAG con reranking usando cross-encoder"
- Idea 2: "RAG con construcción de grafo de conocimiento y random walks"
→ Son enfoques fundamentalmente diferentes

**Formato de salida:**
- is_novel: boolean
- explanation: 2-3 oraciones específicas
"""

MUTATION_STRATEGIES = {
    "default": """
### 📋 TAREA:
Genera un NUEVO concepto que sea una evolución o combinación de las ideas presentadas.

**Opción A - MUTACIÓN:** Una evolución sorprendente de la idea padre que:
- Añade un componente técnico nuevo NO TRIVIAL.
- Cambia una parte fundamental del enfoque.
- Optimiza un aspecto específico de manera innovadora.

**Opción B - CROSSOVER:** Una fusión inesperada de padre + inspiraciones que:
- Combine mecanismos core de diferentes ideas.
- Cree un híbrido que sea MÁS que la suma de sus partes.
- Resuelva un problema que ninguna idea individual resuelve.
""",
    "radical": """
### 📋 TAREA (ENFOQUE RADICAL):
Ignora las inspiraciones. Toma la "idea padre" y transfórmala en algo completamente diferente pero que aún resuelva el problema original.

- **Pregúntate:** ¿Cuál es el supuesto más fundamental de la idea padre? Ahora, ¿cómo funcionaría si ese supuesto fuera falso?
- **Desafío:** Introduce una tecnología o paradigma de un campo completamente diferente (ej. biología, física cuántica, teoría de juegos).
- **Objetivo:** Máxima novedad. No busques una mejora incremental, busca un salto conceptual.
""",
    "synthesizer": """
### 📋 TAREA (ENFOQUE SINTETIZADOR):
Tu misión es actuar como un sintetizador. Encuentra el "hilo conductor" o la idea más potente en CADA UNA de las inspiraciones y en la idea padre.

- **Analiza:** ¿Qué hace a cada idea única y potente?
- **Fusiona:** Crea un nuevo sistema que integre los MEJORES mecanismos de al menos TRES de las ideas presentadas.
- **Justifica:** Explica por qué esta combinación sinérgica es superior a cada componente individual.
""",
    "refiner": """
### 📋 TAREA (ENFOQUE REFINADOR):
Concéntrate exclusivamente en la "idea padre". Las inspiraciones son solo para contexto. Tu objetivo es una mejora PROFUNDA y ESPECÍFICA.

- **Identifica el eslabón débil:** ¿Cuál es el componente menos especificado o más problemático de la idea padre?
- **Optimiza:** Propón una solución técnica detallada para ese eslabón débil. No cambies el resto del sistema.
- **Profundiza:** Añade detalles de implementación, parámetros, y justificaciones matemáticas o lógicas para tu mejora. El objetivo es aumentar la sofisticación y viabilidad.
"""
}

class PromptSampler:
    """Selecciona aleatoriamente una estrategia de mutación e inyecta meta-recomendaciones."""
    def sample_mutation_prompt(self, meta_recommendations: Optional[str] = None) -> str:
        strategy_name = random.choice(list(MUTATION_STRATEGIES.keys()))
        print(f"    🎲 Estrategia de mutación seleccionada: {strategy_name.upper()}")
        
        base_prompt = MUTATION_STRATEGIES[strategy_name]
        
        # MEJORA: Inyectar recomendaciones del meta-aprendizaje si existen.
        if meta_recommendations:
            reco_section = f"""
### 🧠 RECOMENDACIONES DEL META-ANALIZADOR:
Basado en la evolución hasta ahora, considera las siguientes direcciones:
{meta_recommendations}
"""
            return base_prompt + reco_section
        
        return base_prompt