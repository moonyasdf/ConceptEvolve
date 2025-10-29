import os
from typing import List
from src.concepts import AlgorithmicConcept

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ConceptEvolve - Progreso de la Evolución</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
        h1, h2 {{ color: #2c3e50; text-align: center; }}
        #progress-bar-container {{ width: 80%; margin: 20px auto; background-color: #e0e0e0; border-radius: 10px; }}
        #progress-bar {{ width: {progress_percent}%; background-color: #4CAF50; height: 30px; border-radius: 10px; text-align: center; line-height: 30px; color: white; }}
        .container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; padding: 20px; }}
        .concept-card {{ background-color: white; border: 1px solid #ddd; border-left: 5px solid; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.3s ease; }}
        .concept-card:hover {{ transform: translateY(-5px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }}
        .concept-card h3 {{ margin-top: 0; color: #34495e; }}
        .concept-card p {{ color: #555; }}
        .scores {{ display: flex; justify-content: space-between; font-size: 0.9em; margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px; }}
        .score {{ text-align: center; }}
        .score-label {{ font-weight: bold; color: #7f8c8d; }}
        .score-value {{ font-size: 1.2em; color: #2980b9; }}
        .generation-info {{ text-align: center; margin: 20px 0; font-size: 1.1em; color: #555; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; max-width: 800px; margin: 20px auto; }}
        .stat-box {{ background: white; padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #ecf0f1; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #2980b9; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
        .combined-score {{ border-left-color: #2980b9; }}
        .generation-badge {{ float: right; background-color: #ecf0f1; color: #7f8c8d; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; }}
    </style>
    <meta http-equiv="refresh" content="15">
</head>
<body>
    <h1>ConceptEvolve - Progreso de la Evolución</h1>
    <h2>Generación {current_generation} / {total_generations}</h2>
    <div id="progress-bar-container">
        <div id="progress-bar">{progress_percent}%</div>
    </div>
    <div class="container">
        {concept_cards_html}
    </div>
</body>
</html>
"""

def create_concept_card(concept: AlgorithmicConcept) -> str:
    """Genera el HTML para una tarjeta de concepto."""
    if not concept.scores:
        return ""
    
    # Color del borde basado en el score
    score_color = f"hsl({120 * (concept.combined_score / 10)}, 70%, 50%)"

    return f"""
    <div class="concept-card" style="border-left-color: {score_color};">
        <span class="generation-badge">Gen: {concept.generation}</span>
        <h3>{concept.title}</h3>
        <p>{concept.description[:200]}...</p>
        <div class="scores">
            <div class="score">
                <div class="score-label">Novedad</div>
                <div class="score-value">{concept.scores.novelty:.1f}</div>
            </div>
            <div class="score">
                <div class="score-label">Potencial</div>
                <div class="score-value">{concept.scores.potential:.1f}</div>
            </div>
            <div class="score">
                <div class="score-label">Total</div>
                <div class="score-value" style="color: {score_color};">{concept.combined_score:.2f}</div>
            </div>
        </div>
    </div>
    """

def create_visualization(population: List[AlgorithmicConcept], total_generations: int, current_generation: int, output_path: str):
    """Crea y guarda el fichero HTML de visualización."""
    # Ordenar población por score
    sorted_population = sorted(population, key=lambda c: c.combined_score, reverse=True)
    
    cards_html = "".join([create_concept_card(c) for c in sorted_population])
    
    progress = (current_generation / total_generations) * 100 if total_generations > 0 else 0
    
    # Calcular estadísticas
    scores = [c.combined_score for c in sorted_population]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    
    novelties = [c.scores.novelty for c in sorted_population if c.scores]
    avg_novelty = sum(novelties) / len(novelties) if novelties else 0
    
    stats_html = f"""
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-value">{len(sorted_population)}</div>
            <div class="stat-label">Conceptos</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{max_score:.2f}</div>
            <div class="stat-label">Score Máximo</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_score:.2f}</div>
            <div class="stat-label">Score Promedio</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_novelty:.2f}</div>
            <div class="stat-label">Novedad Promedio</div>
        </div>
    </div>
    """
    
    final_html = HTML_TEMPLATE.format(
        current_generation=current_generation,
        total_generations=total_generations,
        progress_percent=f"{progress:.1f}",
        concept_cards_html=stats_html + cards_html
    )
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_html)
    except IOError as e:
        print(f"❌ Error al escribir el fichero de visualización: {e}")