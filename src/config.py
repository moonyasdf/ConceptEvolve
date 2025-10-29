# Fichero: src/config.py
# MEJORAS: #12 (Configuración mejorada de modelos)

import os
import getpass
from google import genai

class ModelConfig:
    """Configuración para el comportamiento de los modelos LLM de Gemini"""
    
    def __init__(self):
        # Modelo principal - USAR SOLO gemini-2.5-pro o gemini-2.0-flash-exp
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        
        # Configuración de generación por tipo de tarea
        self.temp_generation = float(os.getenv("GEMINI_TEMP_GEN", "1.0"))      # Máxima creatividad
        self.temp_evaluation = float(os.getenv("GEMINI_TEMP_EVAL", "0.3"))     # Consistencia
        self.temp_critique = float(os.getenv("GEMINI_TEMP_CRIT", "0.5"))       # Media
        self.temp_refinement = float(os.getenv("GEMINI_TEMP_REF", "0.7"))      # Media-alta
        
        # Configuración general
        self.top_p = float(os.getenv("GEMINI_TOP_P", "0.95"))
        self.top_k = int(os.getenv("GEMINI_TOP_K", "40"))
        
        # Thinking config (para gemini-2.5-pro)
        self.use_thinking = os.getenv("GEMINI_USE_THINKING", "true").lower() == "true"
        # -1 = dinámico, 0 = deshabilitado, 128-32768 = presupuesto fijo
        self.thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "-1"))  
        
        # Modelo de embeddings
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

class AppConfig:
    """Singleton para almacenar la configuración de la aplicación"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance.api_key = None
            cls._instance.client = None
            cls._instance.model_config = ModelConfig()
        return cls._instance

    def set_api_key(self, key: str):
        """Establece y configura la API key de Google"""
        self.api_key = key
        try:
            # Configurar con el nuevo SDK oficial
            self.client = genai.Client(api_key=self.api_key)
            print("✅ API Key de Google configurada correctamente.")
        except Exception as e:
            print(f"❌ Error al configurar la API Key: {e}")
            self.api_key = None
            self.client = None

    def get_client(self):
        """Obtiene el cliente de Gemini, solicitando API key si es necesario"""
        if self.client:
            return self.client

        key_from_env = os.getenv("GOOGLE_API_KEY")
        if key_from_env:
            print("🔑 API Key encontrada en la variable de entorno GOOGLE_API_KEY.")
            self.set_api_key(key_from_env)
            return self.client

        print("🔑 No se encontró la API Key de Google.")
        while not self.client:
            try:
                key_input = getpass.getpass("Por favor, introduce tu API Key de Google y presiona Enter: ")
                if key_input:
                    self.set_api_key(key_input)
                else:
                    print("La API Key no puede estar vacía.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperación cancelada. Saliendo.")
                exit()
        return self.client

# Instancias globales
app_config = AppConfig()
model_config = app_config.model_config