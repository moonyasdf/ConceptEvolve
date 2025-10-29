# ConceptEvolve Documentation
 
This folder contains English-language documentation for operating the experimental ConceptEvolve pipeline.
 
## Overview
 
ConceptEvolve is an evolutionary ideation framework that orchestrates multiple Gemini-powered agents to generate, critique, and curate algorithmic design concepts. Experiments are configured with [Hydra](https://hydra.cc), stored in a SQLite persistence layer, and exposed through a lightweight real-time genealogy dashboard.
 
## Prerequisites
 
- **Python**: 3.12 (create a virtual environment for isolation).
- **Google Gemini API key**: required for text generation, critique, and structured-output parsing.
- **(Optional) OpenAI API key**: only if you plan to swap the default embedding model.
- **FAISS CPU** libraries are installed automatically from `requirements.txt`.
 
## Environment Setup
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=""
# Optional, if using OpenAI embeddings
export OPENAI_API_KEY=""
```
 
> **Note:** the project relies on the official `google-genai` SDK (>= 1.46.0). Make sure the API key has access to Gemini 2.5 models.
 
## Running an Evolution Session
 
The entry point is `src/run.py`, wrapped by Hydra. Default configuration files live under `configs/`.
 
```bash
python -m src.run \
    evolution.num_generations=15 \
    evolution.population_size=25 \
    --multirun
```
 
Common overrides:
 
- `problem_file=path/to/custom_problem.txt` – supply your own prompt.
- `resume=true` – continue from the latest checkpoint in `checkpoints/`.
- `model.name=gemini-2.5-pro` – swap between compatible Gemini models.
- `database.num_islands=8` – control the number of evolutionary islands.
 
Hydra writes run artifacts to `outputs/YYYY-MM-DD/HH-MM-SS/`. Each run includes:
 
- `run.log` – streaming console output.
- `.hydra/config.yaml` – resolved configuration snapshot.
- (When applicable) concept design dumps such as `final_population.json` and `top_k_concept_*.txt`.
 
## Visualization Dashboard
 
During `src.run`, a background thread launches a local HTTP server (default port **8000**). The dashboard shows:
 
- Population genealogy tree with scoring overlays.
- Detailed critique and refinement history per concept.
- Embedding-based similarity insights when FAISS is enabled.
 
If the browser fails to open automatically, visit `http://localhost:8000` manually.
 
## Checkpointing and Resuming
 
- Automatic checkpoints are stored under the `checkpoints/` directory every `evolution.checkpoint_interval` generations.
- Interrupt the run with `Ctrl+C` to trigger an emergency checkpoint.
- Resume by passing `resume=true` in the Hydra overrides. The process restores the database, embeddings, and scoring history.
 
## Customising Prompts & Agents
 
Prompts and agent personalities live in:
 
- `src/prompts.py` and `src/prompts_meta.py` – templates for generators, critics, evaluators, and requirement extractors.
- `src/agents.py` – orchestration logic for each agent role.
 
Update these files to tailor tone, structure, or evaluation rubrics for new domains.
 
## Working with the Database
 
Concepts are persisted in a SQLite file (`evolution.db` by default). Tables store:
 
- Raw text (`title`, `description`),
- Critique/refinement history,
- Structured scores (`ConceptScores`),
- Embeddings and island metadata.
 
Use any SQLite browser or Python script to inspect results offline. Remember to add `evolution.db` to `.gitignore` (already configured) to avoid committing large artifacts.
 
## Troubleshooting
 
- **API errors**: the runtime offers interactive recovery. Choose to retry, swap API keys, or abort.
- **Embedding dimension mismatch**: ensure all previously persisted concepts share the same embedding length. Override `model.embedding_model` or wipe the database when changing embedding providers.
- **FAISS not available**: the system falls back to HNSW-style novelty checks when FAISS initialisation fails.
 
## Reproducing the QICE Case Study
 
1. Use `src/mi_problema.txt` as the problem file.
2. Run a short session (`evolution.num_generations=1`, `population_size=3`) as demonstrated in `docs/qice_session_summary.txt`.
3. Consult the generated summary file for the winning concept and reviewer feedback.
 
## Extending the Framework
 
- Create new Hydra configs under `configs/` for alternate agents or selection strategies.
- Plug in new evaluators or reward models via `src/scoring.py` and `src/selection.py`.
- Integrate additional telemetry or analytics via `src/webui/visualization.py`.
 
For deeper architectural notes, review `README.md` at the repository root and `apigemini.md` for Gemini API guidance.
