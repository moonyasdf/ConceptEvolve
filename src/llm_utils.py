# src/llm_utils.py

# MEJORA: #15 (Actualización API Gemini) - Refactorización completa para usar
# la sintaxis recomendada del SDK google-genai, incluyendo salida estructurada,
# thinking_config y procesamiento por lotes para embeddings.
# Se elimina la dependencia del fichero obsoleto src/config.py.

import json
import hashlib
import os
import asyncio
import time
from typing import Type, TypeVar, Optional, List, Union
from pydantic import BaseModel
from google import genai
from google.genai import types
from omegaconf import DictConfig

from src.config import app_config  # Mantenemos app_config para la gestión del cliente

T = TypeVar("T", bound=BaseModel)

class LLMCache:
    """Clase de caché simple para las respuestas del LLM. Sin cambios."""
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    def _get_cache_key(self, prompt: str, system_message: str, model: str, temperature: float) -> str:
        content = f"{model}::{temperature:.2f}::{system_message}::{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    def get(self, prompt: str, system_message: str, model: str, temperature: float) -> Optional[str]:
        key = self._get_cache_key(prompt, system_message, model, temperature)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('response')
            except:
                return None
        return None
    def set(self, prompt: str, system_message: str, model: str, temperature: float, response: str):
        key = self._get_cache_key(prompt, system_message, model, temperature)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'response': response}, f, ensure_ascii=False)
        except Exception as e:
            print(f"  ⚠️ Error al guardar en caché: {e}")

_cache = LLMCache()

def _handle_api_error() -> bool:
    """Gestión de errores de API. Sin cambios."""
    while True:
        choice = input(
            "\n⚠️ Error con la API de Google.\n"
            "   1. Reintentar\n"
            "   2. Introducir nueva API Key\n"
            "   3. Salir\n"
            "Elige (1, 2, o 3): "
        )
        if choice == '1':
            return True
        elif choice == '2':
            app_config.api_key = None
            app_config.client = None
            app_config.get_client()
            return True
        elif choice == '3':
            print("Saliendo del programa.")
            exit()
        else:
            print("Opción no válida.")

def query_gemini_structured(
    prompt: str,
    system_message: str,
    response_schema: Type[T],
    model_cfg: DictConfig,
    use_cache: bool = True,
    max_retries: int = 3
) -> Optional[T]:
    """
    MEJORA: Versión corregida que usa la sintaxis correcta del SDK y recibe la
    configuración del modelo desde Hydra (cfg.model).
    """
    client = app_config.get_client()
    model_name = model_cfg.name
    temperature = model_cfg.temp_evaluation # Usamos una temperatura por defecto, la correcta se pasará desde el agente

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, temperature)
        if cached:
            try:
                # La caché guarda un JSON string, así que lo validamos.
                return response_schema.model_validate_json(cached)
            except Exception as e:
                print(f"  ⚠️ Error de validación en caché: {e}, se volverá a llamar a la API.")

    for attempt in range(max_retries):
        try:
            # MEJORA: Construcción de GenerationConfig según la documentación oficial
            generation_config = types.GenerationConfig(temperature=temperature)

            # MEJORA: thinking_config se añade a generation_config si está activado
            if "2.5-pro" in model_name and model_cfg.use_thinking:
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=model_cfg.thinking_budget
                )
            
            # MEJORA: La salida estructurada se configura en el diccionario `config` de la llamada principal
            response = client.models.generate_content(
                model=f"models/{model_name}",
                contents=[{'role': 'user', 'parts': [{'text': prompt}]}],
                system_instruction=system_message,
                generation_config=generation_config,
                # MEJORA: Sintaxis correcta para salida estructurada
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                }
            )

            # MEJORA: La respuesta parseada Pydantic se encuentra en `response.parsed`
            parsed_obj = response.parsed
            if not isinstance(parsed_obj, response_schema):
                 raise TypeError(f"La respuesta de la API no es del tipo esperado: {type(parsed_obj)}")

            if use_cache and parsed_obj:
                _cache.set(prompt, system_message, model_name, temperature, parsed_obj.model_dump_json())

            return parsed_obj

        except Exception as e:
            print(f"  ❌ Error en llamada a API estructurada (intento {attempt+1}/{max_retries}): {e}")
            if attempt >= max_retries - 1:
                if _handle_api_error():
                    return query_gemini_structured(prompt, system_message, response_schema, model_cfg, use_cache, max_retries)
                return None
            time.sleep(2 ** attempt)
    return None

def query_gemini_unstructured(
    prompt: str,
    system_message: str,
    model_cfg: DictConfig,
    temperature: float,
    use_cache: bool = True,
    max_retries: int = 3
) -> str:
    """
    MEJORA: Versión corregida que usa la sintaxis correcta del SDK y recibe la
    configuración del modelo desde Hydra (cfg.model).
    """
    client = app_config.get_client()
    model_name = model_cfg.name

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, temperature)
        if cached:
            return cached

    for attempt in range(max_retries):
        try:
            generation_config = types.GenerationConfig(temperature=temperature)

            if "2.5-pro" in model_name and model_cfg.use_thinking:
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=model_cfg.thinking_budget
                )
            
            # MEJORA: Sintaxis de llamada correcta a client.models.generate_content
            response = client.models.generate_content(
                model=f"models/{model_name}",
                contents=[{'role': 'user', 'parts': [{'text': prompt}]}],
                system_instruction=system_message,
                generation_config=generation_config,
            )

            result_text = response.text

            if use_cache and result_text:
                _cache.set(prompt, system_message, model_name, temperature, result_text)

            return result_text

        except Exception as e:
            print(f"  ❌ Error en llamada a API no estructurada (intento {attempt+1}/{max_retries}): {e}")
            if attempt >= max_retries - 1:
                if _handle_api_error():
                    return query_gemini_unstructured(prompt, system_message, model_cfg, temperature, use_cache, max_retries)
                return f"Error no resuelto: {e}"
            time.sleep(2 ** attempt)
    return ""

class EmbeddingClient:
    """
    MEJORA: Cliente de embeddings actualizado para usar la API correcta y
    soportar procesamiento por lotes.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        if not self.model_name.startswith('models/'):
            self.model_name = f'models/{self.model_name}'

    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not text:
            return [] if isinstance(text, str) else [[] for _ in text]
        
        client = app_config.get_client()
        is_single_item = isinstance(text, str)
        
        try:
            # MEJORA: La API `embed_content` puede manejar tanto un string como una lista
            result = client.models.embed_content(model=self.model_name, contents=text)
            
            if is_single_item:
                return result['embeddings'][0]['values']
            else:
                return [item['values'] for item in result['embeddings']]
                
        except Exception as e:
            print(f"  ❌ Error al generar embedding: {e}")
            if _handle_api_error():
                return self.get_embedding(text)
            return [] if is_single_item else [[] for _ in text]

    async def get_embedding_async(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)