# ConceptEvolve 🧬💡

**ConceptEvolve** es un framework de "ideación aumentada" que utiliza un enfoque evolutivo impulsado por LLMs para generar, refinar y diversificar conceptos algorítmicos sofisticados. Esta versión ha sido mejorada con la arquitectura robusta de [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

En lugar de saltar directamente a la implementación de código, ConceptEvolve explora el "espacio de las ideas", produciendo un portafolio de documentos de diseño de alto nivel, creativos y robustos. Estos documentos sirven como un punto de partida de alta calidad para proyectos de desarrollo de software complejos, como los que se pueden implementar con frameworks como [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

 <!-- Reemplaza esto con una URL a un diagrama si lo creas -->

## Características Principales

-   **Generación Evolutiva de Conceptos:** Utiliza un sistema de población donde las ideas "mutan" y se "cruzan" para crear nuevas y mejores soluciones.
-   **Refinamiento Iterativo:** Cada nueva idea pasa por un ciclo de **crítica y refinamiento**, donde un agente de IA escéptico encuentra debilidades y el generador fortalece el concepto.
-   **Filtro de Novedad:** Incorpora un sistema de "muestreo por rechazo" basado en embeddings y un LLM-juez para descartar ideas redundantes y fomentar la diversidad.
-   **Evaluación de Fitness Conceptual:** Las ideas son calificadas por un "comité de programa de IA" que evalúa la novedad, el potencial, la sofisticación y la viabilidad, en lugar de solo la corrección de una ejecución.
-   **Salida Estructurada:** El resultado final no es código, sino un conjunto de **documentos de diseño** que incluyen requisitos del sistema y sub-problemas identificados.
-   **Gestión Interactiva de API Keys:** Solicita tu API key de forma segura y permite cambiarla en tiempo de ejecución si falla, evitando interrupciones.

## Requisitos

-   Python 3.9+
-   Una **API Key de Google** para el modelo Gemini.
-   Una **API Key de OpenAI** para la generación de embeddings.

## 🚀 Guía de Inicio Rápido

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/conceptevolve.git
cd conceptevolve
```

### 2. Configurar el Entorno Virtual

Se recomienda usar un entorno virtual de Python.

```bash
# Crear un entorno virtual
python -m venv .venv

# Activar el entorno
# En macOS/Linux:
source .venv/bin/activate
# En Windows (CMD):
# .venv\Scripts\activate.bat
```

### Ejecutar ConceptEvolve con Hydra

El script principal ahora se ejecuta a través de Hydra, lo que permite una configuración flexible desde la línea de comandos.

**Sintaxis:**
```bash
python src/run.py [OPCIONES_DE_HYDRA]
```

**Ejemplo Práctico:**
```bash
# Ejecutar con la configuración por defecto (definida en configs/config.yaml)
python src/run.py

# Modificar parámetros desde la línea de comandos
python src/run.py evolution.num_generations=20 database.num_islands=8

# Reanudar desde el último checkpoint
python src/run.py resume=true
```

### Validación de la configuración de base de datos

Antes de iniciar el proceso evolutivo, ConceptEvolve valida que la sección `database` de la configuración de Hydra esté completa y contenga valores válidos. Si falta algún parámetro obligatorio (por ejemplo `migration_interval`, `parent_selection_lambda` o `exploitation_ratio`) o un campo tiene un tipo fuera de rango, se lanzará un `ValueError` que indica exactamente qué parámetros deben corregirse. Asegúrate de definir todos los campos requeridos y de proporcionar números dentro de sus intervalos esperados (p.ej. tasas entre 0 y 1, tamaños positivos) cuando ajustes la configuración desde YAML o la línea de comandos.

### Visualización en Tiempo Real

Al ejecutar `src/run.py`, se iniciará automáticamente un servidor web.
- **Abre tu navegador y ve a `http://localhost:8000`** para monitorear el progreso de la evolución en tiempo real.
- El panel mostrará el árbol genealógico de ideas, y al hacer clic en un nodo, verás su descripción, puntuaciones e historial de críticas.

### 3. Instalar Dependencias

Instala todos los paquetes necesarios con un solo comando:

```bash
pip install -r requirements.txt
```

### 4. Configurar las API Keys

El sistema necesita acceso a las APIs de Google y OpenAI.

-   **API Key de Google (Gemini):** El programa te la pedirá interactivamente la primera vez que lo ejecutes. También puedes configurarla como una variable de entorno para evitar que te la pida cada vez:
    ```bash
    export GOOGLE_API_KEY="tu_api_key_de_google"
    ```
-   **API Key de OpenAI (Embeddings):** Debes configurarla como una variable de entorno.
    ```bash
    export OPENAI_API_KEY="tu_api_key_de_openai"
    ```
    Puedes añadir estas líneas a tu fichero `~/.bashrc` o `~/.zshrc` para que estén disponibles en todas tus sesiones de terminal.

### 5. Ejecutar ConceptEvolve

El script principal es `run.py`. Debes ejecutarlo desde la raíz del proyecto y proporcionarle la descripción del problema que quieres resolver.

**Sintaxis:**

```bash
python run.py --problem "DESCRIPCIÓN_DEL_PROBLEMA" [OPCIONES]
```

**Ejemplo Práctico:**

Vamos a pedirle a `ConceptEvolve` que genere ideas para un sistema avanzado de RAG (Retrieval-Augmented Generation).

```bash
python run.py \
    --problem "Diseñar un sistema de RAG de última generación para el benchmark MuSiQue, que requiere razonamiento en múltiples pasos. El sistema debe ser capaz de descomponer preguntas complejas, realizar búsquedas de información de manera iterativa y sintetizar respuestas coherentes a partir de fragmentos de evidencia distribuidos en múltiples documentos." \
    --generations 15 \
    --population 25 \
    --output_dir "musique_rag_concepts"
```

**Argumentos:**

-   `--problem` (obligatorio): La descripción del problema. Intenta ser lo más detallado posible.
-   `--generations` (opcional): Número de ciclos evolutivos a ejecutar. (Por defecto: 10)
-   `--population` (opcional): Tamaño de la población de ideas que se mantiene en cada generación. (Por defecto: 20)
-   `--output_dir` (opcional): Carpeta donde se guardarán los resultados. (Por defecto: `concept_results`)

### 6. Revisar los Resultados

Una vez que el proceso finalice, encontrarás los resultados en el directorio de salida especificado (ej. `musique_rag_concepts/`).

-   `final_population.json`: Un fichero JSON que contiene todos los conceptos generados y evaluados durante el proceso, ordenados por su puntuación final.
-   `top_1_concept_... .txt`, `top_2_concept_... .txt`, etc.: Documentos de diseño detallados para los 5 mejores conceptos, listos para ser analizados o utilizados como base para una implementación.

## ¿Cómo Funciona?

`ConceptEvolve` simula un proceso de investigación y desarrollo a nivel conceptual.

1.  **Población Inicial:** Un agente de IA genera un conjunto inicial de ideas diversas.
2.  **Bucle Evolutivo:**
    -   **Selección:** Se seleccionan las ideas "padre" más prometedoras (basado en una combinación de rendimiento y novedad).
    -   **Generación:** Se crean nuevas ideas "hijas" a través de "mutaciones" (refinamientos) y "crossovers" (combinaciones) de las ideas padre e inspiraciones.
    -   **Refinamiento:** Cada nueva idea es "criticada" por un agente de IA escéptico. El generador original refina la idea para abordar las críticas, fortaleciéndola.
    -   **Evaluación y Archivo:** Las ideas refinadas y novedosas son evaluadas, puntuadas y añadidas a la población, reemplazando a las menos prometedoras.
3.  **Salida:** El proceso se repite durante varias "generaciones", y al final se presentan los conceptos más evolucionados.


## 🚀 Nuevas Características (v2.0)

- **🚀 Arquitectura Robusta:** Potenciado por la configuración Hydra y una base de datos SQLite persistente para escalabilidad y reproducibilidad.
- **🏝️ Modelo de Islas:** Mantiene la diversidad conceptual mediante sub-poblaciones que evolucionan en paralelo y comparten ideas.
- **🎲 Estrategias de Mutación Dinámicas:** Utiliza múltiples "personalidades" de prompts para guiar al LLM hacia enfoques más variados.
- **🎨 Visualización en Tiempo Real:** Un servidor web interactivo muestra el árbol de ideas y sus detalles a medida que se generan.
- **🧠 API de Gemini Mejorada:** Utiliza `response_schema` para un parsing robusto y la función `thinking_config` de `gemini-2.5-pro` para un razonamiento de mayor calidad.

### Mejoras de Rendimiento
- **⚡ Evaluación Paralela:** Procesamiento asíncrono de conceptos (5-10x más rápido)
- **💾 Caché Inteligente:** Sistema de caché para respuestas LLM (evita llamadas redundantes)
- **🔍 Indexación FAISS:** Búsqueda vectorial O(log n) en lugar de O(n)

### Mejoras de Precisión
- **📚 Prompts Mejorados:** Few-shot learning con ejemplos de alta calidad
- **📊 Scoring Adaptativo:** Normalización z-score y pesos dinámicos por generación
- **🎯 Validación de Alineación:** Verifica que soluciones realmente aborden el problema
- **🧬 Selección Multiobjetivo:** Balance entre fitness y diversidad (NSGA-II inspired)
- **🔄 Refinamiento Contextual:** Tracking de puntos abordados en iteraciones

### Mejoras de Robustez
- **💾 Checkpointing Automático:** Guarda progreso cada N generaciones
- **🔄 Modo Reanudar:** Continúa desde último checkpoint con `--resume`
- **🛡️ Parsing Robusto:** Usa Structured Output nativo de Gemini
- **📝 Logging Detallado:** Trazabilidad completa del proceso

### Nuevos Parámetros CLI

```bash
# Reanudar desde checkpoint
python src/run.py problema.txt --resume

# Ajustar configuración de novedad
python src/run.py problema.txt --novelty-threshold 0.90 --refinement-steps 3

# Controlar frecuencia de checkpoints
python src/run.py problema.txt --checkpoint-interval 10
```

### Variables de Entorno Configurables

```bash
# Modelo a usar (gemini-2.5-pro o gemini-2.0-flash-exp)
export GEMINI_MODEL="gemini-2.5-pro"

# Temperaturas por tipo de tarea
export GEMINI_TEMP_GEN="1.0"    # Generación (máxima creatividad)
export GEMINI_TEMP_EVAL="0.3"   # Evaluación (consistencia)
export GEMINI_TEMP_CRIT="0.5"   # Crítica
export GEMINI_TEMP_REF="0.7"    # Refinamiento

# Configuración de thinking (solo gemini-2.5-pro)
export GEMINI_USE_THINKING="true"
export GEMINI_THINKING_BUDGET="-1"  # -1 = dinámico, 0 = off, 128-32768 = fijo
```


## Contribuciones

Este es un proyecto en desarrollo. Las contribuciones, reportes de errores y sugerencias son bienvenidas. Por favor, abre un "Issue" en GitHub para discutir cualquier cambio.
