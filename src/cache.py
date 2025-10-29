# Fichero: src/cache.py
# MEJORA #2: Sistema de caché avanzado (ya incluido en llm_utils.py)
# Este archivo existe solo como referencia, la implementación está en llm_utils.py

"""
El sistema de caché está implementado en src/llm_utils.py
para evitar imports circulares y mantener todo centralizado.

Características:
- Caché basado en hash de (modelo + temperatura + prompt + system_message)
- Almacenamiento en archivos JSON en el directorio .cache/
- Activación/desactivación por parámetro use_cache
- Compatible con respuestas estructuradas y no estructuradas
"""