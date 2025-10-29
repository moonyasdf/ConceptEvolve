# src/prompts_meta.py

# MEJORA: Prompts para el sistema de meta-aprendizaje, adaptados de ShinkaEvolve.
# Estos prompts guían al LLM para realizar un análisis en 3 pasos:
# 1. Resumir conceptos individuales.
# 2. Sintetizar ideas globales a partir de los resúmenes.
# 3. Generar recomendaciones accionables para futuras generaciones.

# Paso 1: Resumir conceptos individuales
META_STEP1_SYSTEM_MSG = (
    "You are an expert programming assistant analyzing an individual concept. "
    "Create a standalone summary focusing on implementation details and "
    "evaluation feedback. Consider how this specific concept performs and "
    "what implementation choices were made."
)

META_STEP1_USER_MSG = (
    "# Concept to Analyze\n"
    "{concept_summary}\n\n"
    "# Instructions\n\n"
    "Create a standalone summary for this concept using the following "
    "exact format:\n\n"
    "**Concept Name: [Short summary name of the algorithm (up to 10 words)]**\n"
    "- **Implementation**: [Key implementation details (1-2 sentences)]\n"
    "- **Performance**: [Score/metrics summary (1 sentence)]\n"
    "- **Feedback**: [Key insights from evaluation (1-2 sentences)]\n\n"
    "Focus on:\n"
    "1. What specific implementation details were done\n"
    "2. How these details affected performance\n"
    "3. Implementation details that are relevant to the approach\n\n"
    "Keep the concept summary concise but informative. Follow the format exactly."
)

# Paso 2: Generar un "Scratchpad" de ideas globales
META_STEP2_SYSTEM_MSG = (
    "You are an expert AI researcher analyzing multiple concept evaluations to extract "
    "actionable optimization insights. Focus on concrete performance data and "
    "design choices from the actual concepts that were evaluated."
)

META_STEP2_USER_MSG = (
    "# Individual Concept Summaries\n"
    "{individual_summaries}\n\n"
    "# Previous Global Insights (if any)\n"
    "{previous_insights}\n\n"
    "# Current Best Concept\n"
    "{best_concept_info}\n\n"
    "# Instructions\n\n"
    "Analyze the SPECIFIC concept evaluation results above to extract "
    "concrete design insights. Look at the actual performance scores, "
    "design details, and evaluation feedback to identify patterns. "
    "Reference specific concepts and their results.\n"
    "Make sure to incorporate the previous insights into the new insights.\n\n"
    "**CRITICAL: Pay special attention to the current best concept and how it "
    "compares to other evaluated concepts. Ensure the best concept's successful "
    "patterns are prominently featured in your analysis.**\n\n"
    "Update or create insights in these sections:\n\n"
    "## Successful Architectural Patterns\n"
    "- Identify specific design choices that led to score "
    "improvements.\n"
    "- Reference which concepts achieved these improvements and their "
    "scores.\n"
    "- **Highlight patterns from the current best concept.**\n"
    "- Note the specific techniques or approaches that worked (2-4 bullet points).\n\n"
    "## Ineffective Approaches\n"
    "- Identify specific design choices that worsened "
    "performance.\n"
    "- Reference which concepts had these issues and how scores were "
    "affected.\n"
    "- Note why these approaches failed based on evaluation feedback (2-4 bullet points).\n\n"
    "## Design Insights\n"
    "- Extract specific design patterns/techniques from the evaluated "
    "concepts.\n"
    "- **Analyze what makes the current best concept effective.**\n"
    "- Connect design details to their performance impact.\n"
    "- Reference concrete examples from the concept summaries (2-4 bullet points).\n\n"
    "## Performance Analysis\n"
    "- Analyze actual score changes and trends from the evaluated "
    "concepts.\n"
    "- **Compare other concepts' performance against the current best.**\n"
    "- Compare performance between different design "
    "approaches.\n"
    "- Identify score patterns and correlations (2-4 bullet points).\n\n"
    "CRITICAL: Base insights on the ACTUAL individual concept summaries, "
    "previous insights, and the current best concept information. "
    "Reference specific concept names, scores, and design details. "
    "Build upon previous insights with concrete evidence from the new evaluations. "
    "IMPORTANT: Make sure that the best results are not ignored and are "
    "prominently featured in your analysis. "
    "Do not make recommendations for the next steps. ONLY PERFORM THE ANALYSIS."
)

# Paso 3: Generar recomendaciones
META_STEP3_SYSTEM_MSG = (
    "You are an expert AI researcher generating actionable "
    "recommendations for future concept mutations based on successful "
    "patterns and insights."
)

META_STEP3_USER_MSG = (
    "# Global Insights\n"
    "{global_insights}\n\n"
    "# Previous Recommendations (if any)\n"
    "{previous_recommendations}\n\n"
    "# Current Best Concept\n"
    "{best_concept_info}\n\n"
    "# Instructions\n\n"
    "Based on the global insights above and the current best concept, generate {max_recommendations} "
    "actionable recommendations for future concept mutations. Each "
    "recommendation should be:\n\n"
    "1. **Specific**: Clear about what to explore or combine.\n"
    "2. **Actionable**: A concrete direction for a new concept.\n"
    "3. **Evidence-based**: Grounded in the successful patterns identified.\n"
    "4. **Diverse**: Cover different types of creative directions.\n"
    "5. **Best-concept informed**: Consider what makes the current best concept successful.\n\n"
    "Format as a numbered list:\n\n"
    "1. [Specific recommendation based on successful patterns]\n"
    "2. [Another recommendation focusing on a different aspect]\n"
    "...\n\n"
    "**CRITICAL: Prioritize recommendations that build upon or extend the successful "
    "patterns from the current best concept. Consider both incremental improvements "
    "to the best concept's approach and novel variations that could surpass it.**\n\n"
    "Focus on the most promising approaches that have shown success in "
    "recent evaluations, especially those demonstrated by the best concept. "
    "Avoid generic advice. Provide 2-3 sentences per recommendation."
)