# ConceptEvolve 🧬💡

**ConceptEvolve** is an "augmented ideation" framework that uses an LLM-driven evolutionary approach to generate, refine, and diversify sophisticated algorithmic concepts. This version has been enhanced with the robust architecture of [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

> Instead of jumping directly into code implementation, ConceptEvolve explores the "idea space," producing a portfolio of high-level, creative, and robust design documents. These documents serve as a high-quality starting point for complex software development projects, such as those that can be implemented with frameworks like [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

 <!-- Replace this with a URL to a diagram if you create one -->

## Key Features

-   **Evolutionary Concept Generation:** Uses a population system where ideas "mutate" and "crossover" to create new and better solutions.
-   **Iterative Refinement:** Each new idea goes through a cycle of **criticism and refinement**, where a skeptical AI agent finds weaknesses and the generator strengthens the concept.
-   **Novelty Filter:** Incorporates an embedding-based "rejection sampling" system and an LLM-judge to discard redundant ideas and foster diversity.
-   **Conceptual Fitness Evaluation:** Ideas are scored by an "AI program committee" that evaluates novelty, potential, sophistication, and viability, rather than just execution correctness.
-   **Structured Output:** The final result is not code, but a set of **design documents** that include system requirements and identified sub-problems.
-   **Interactive API Key Management:** Securely requests your API key and allows changing it at runtime if it fails, preventing interruptions.

## Requirements

-   Python 3.9+
-   A **Google API Key** for the Gemini model.
-   An **OpenAI API Key** for embedding generation.

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/tu-usuario/conceptevolve.git
cd conceptevolve
```

### 2. Configure the Virtual Environment

It is recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (CMD):
# .venv\Scripts\activate.bat
```

### Running ConceptEvolve with Hydra

The main script now runs via Hydra, allowing flexible configuration from the command line.

**Syntax:**
```bash
python src/run.py [HYDRA_OPTIONS]
```

**Practical Example:**
```bash
# Run with default configuration (defined in configs/config.yaml)
python src/run.py

# Modify parameters from the command line
python src/run.py evolution.num_generations=20 database.num_islands=8

# Resume from the last checkpoint
python src/run.py resume=true
```

### Database Configuration Validation

Before starting the evolutionary process, ConceptEvolve validates that the `database` section of the Hydra configuration is complete and contains valid values. If any required parameter is missing (e.g., `migration_interval`, `parent_selection_lambda`, or `exploitation_ratio`) or a field has an out-of-range type, a `ValueError` will be thrown indicating exactly which parameters must be corrected. Ensure you define all required fields and provide numbers within their expected ranges (e.g., rates between 0 and 1, positive sizes) when adjusting the configuration from YAML or the command line.

### Real-Time Visualization

When running `src/run.py`, a web server will automatically start.
- **Open your browser and navigate to `http://localhost:8000`** to monitor the evolution progress in real-time.
- The dashboard will show the idea genealogy tree, and clicking on a node will display its description, scores, and criticism history.

### 3. Install Dependencies

Install all necessary packages with a single command:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

The system needs access to Google and OpenAI APIs.

-   **Google API Key (Gemini):** The program will ask for it interactively the first time you run it. You can also configure it as an environment variable to prevent it from asking every time:
    ```bash
    export GOOGLE_API_KEY="your_google_api_key"
    ```
-   **OpenAI API Key (Embeddings):** You must configure this as an environment variable.
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    You can add these lines to your `~/.bashrc` or `~/.zshrc` file so they are available in all your terminal sessions.

### 5. Run ConceptEvolve

The main script is `run.py`. You must run it from the project root and provide the description of the problem you want to solve.

**Syntax:**

```bash
python run.py --problem "PROBLEM_DESCRIPTION" [OPTIONS]
```

**Practical Example:**

Let's ask `ConceptEvolve` to generate ideas for an advanced RAG (Retrieval-Augmented Generation) system.

```bash
python run.py \
    --problem "Design a next-generation RAG system for the MuSiQue benchmark, which requires multi-step reasoning. The system must be capable of decomposing complex questions, performing iterative information retrieval, and synthesizing coherent answers from evidence fragments distributed across multiple documents." \
    --generations 15 \
    --population 25 \
    --output_dir "musique_rag_concepts"
```

**Arguments:**

-   `--problem` (required): The description of the problem. Try to be as detailed as possible.
-   `--generations` (optional): Number of evolutionary cycles to run. (Default: 10)
-   `--population` (optional): Size of the idea population maintained in each generation. (Default: 20)
-   `--output_dir` (optional): Folder where results will be saved. (Default: `concept_results`)

### 6. Reviewing the Results

Once the process is complete, you will find the results in the specified output directory (e.g., `musique_rag_concepts/`).

-   `final_population.json`: A JSON file containing all concepts generated and evaluated during the process, sorted by their final score.
-   `top_1_concept_... .txt`, `top_2_concept_... .txt`, etc.: Detailed design documents for the top 5 concepts, ready to be analyzed or used as a basis for implementation.

## How It Works

`ConceptEvolve` simulates a conceptual-level research and development process.

1.  **Initial Population:** An AI agent generates an initial set of diverse ideas.
2.  **Evolutionary Loop:**
    -   **Selection:** The most promising "parent" ideas are selected (based on a combination of fitness and novelty).
    -   **Generation:** New "child" ideas are created through "mutations" (refinements) and "crossovers" (combinations) of parent ideas and inspirations.
    -   **Refinement:** Each new idea is "criticized" by a skeptical AI agent. The original generator refines the idea to address the criticisms, strengthening it.
    -   **Evaluation and Archiving:** Refined and novel ideas are evaluated, scored, and added to the population, replacing the less promising ones.
3.  **Output:** The process is repeated for several "generations," and the most evolved concepts are presented at the end.

## 🚀 New Features (v2.0)

- **🚀 Robust Architecture:** Powered by Hydra configuration and a persistent SQLite database for scalability and reproducibility.
- **🏝️ Island Model:** Maintains conceptual diversity through sub-populations that evolve in parallel and share ideas.
- **🎲 Dynamic Mutation Strategies:** Uses multiple prompt "personalities" to guide the LLM toward more varied approaches.
- **🎨 Real-Time Visualization:** An interactive web server displays the idea tree and its details as they are generated.
- **🧠 Enhanced Gemini API:** Uses `response_schema` for robust parsing and the `thinking_config` function of `gemini-2.5-pro` for higher-quality reasoning.

### Performance Improvements
- **⚡ Parallel Evaluation:** Asynchronous processing of concepts (5-10x faster)
- **💾 Smart Caching:** Caching system for LLM responses (avoids redundant calls)
- **🔍 FAISS Indexing:** O(log n) vector search instead of O(n)

### Accuracy Improvements
- **📚 Improved Prompts:** Few-shot learning with high-quality examples
- **📊 Adaptive Scoring:** Z-score normalization and dynamic weights per generation
- **🎯 Alignment Validation:** Verifies that solutions genuinely address the problem
- **🧬 Multiobjective Selection:** Balances between fitness and diversity (NSGA-II inspired)
- **🔄 Contextual Refinement:** Tracks addressed points across iterations

### Robustness Improvements
- **💾 Automatic Checkpointing:** Saves progress every N generations
- **🔄 Resume Mode:** Continues from the last checkpoint with `--resume`
- **🛡️ Robust Parsing:** Uses native Gemini Structured Output
- **📝 Detailed Logging:** Complete traceability of the process

### New CLI Parameters

```bash
# Resume from checkpoint
python src/run.py problema.txt --resume

# Adjust novelty configuration
python src/run.py problema.txt --novelty-threshold 0.90 --refinement-steps 3

# Control checkpoint frequency
python src/run.py problema.txt --checkpoint-interval 10
```

### Configurable Environment Variables

```bash
# Model to use (gemini-2.5-pro or gemini-2.0-flash-exp)
export GEMINI_MODEL="gemini-2.5-pro"

# Temperatures by task type
export GEMINI_TEMP_GEN="1.0"    # Generation (maximum creativity)
export GEMINI_TEMP_EVAL="0.3"   # Evaluation (consistency)
export GEMINI_TEMP_CRIT="0.5"   # Criticism
export GEMINI_TEMP_REF="0.7"    # Refinement

# Thinking configuration (gemini-2.5-pro only)
export GEMINI_USE_THINKING="true"
export GEMINI_THINKING_BUDGET="-1"  # -1 = dynamic, 0 = off, 128-32768 = fixed
```

## Contributions

This is a project in development. Contributions, bug reports, and suggestions are welcome. Please open an "Issue" on GitHub to discuss any changes.
