# MEJORA: #13 (Integración de Hydra) - Se reemplaza argparse por Hydra para una configuración flexible.
# MEJORA: #16 (Visualización en Tiempo Real) - Se añade el lanzamiento del servidor de visualización.

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import threading
import webbrowser
import time
from pathlib import Path

from src.evolution import ConceptEvolution
from src.config import app_config
from src.webui.visualization import start_server

def read_file_content(filepath: str) -> str:
    """Lee el contenido de un archivo de texto"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado: '{filepath}'")
        exit(1)
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")
        exit(1)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Función principal para ejecutar ConceptEvolve con configuración de Hydra.
    """
    # Imprimir configuración
    print("\n" + "="*70)
    print("🧬 CONCEPTEVOLVE - EVOLUCIÓN DE CONCEPTOS ALGORÍTMICOS")
    print("="*70)
    print("\n📊 Configuración del Experimento:")
    print(OmegaConf.to_yaml(cfg))

    # Configurar API key
    app_config.get_client()

    # Leer descripción del problema
    problem_description = read_file_content(cfg.problem_file)
    if not problem_description:
        print(f"❌ Error: El archivo de problema '{cfg.problem_file}' está vacío.")
        exit(1)
    
    print(f"\n🎯 Problema:")
    print(f"   {problem_description[:200]}{'...' if len(problem_description) > 200 else ''}")
    print("\n" + "="*70 + "\n")

    # Iniciar el servidor de visualización en un hilo separado
    # El directorio de búsqueda será el directorio de ejecución de Hydra
    search_root = os.getcwd()
    db_path_from_config = cfg.database.db_path
    
    # Construimos la ruta absoluta al fichero de la base de datos
    db_abs_path = Path(search_root) / db_path_from_config
    
    port = 8000
    server_thread = threading.Thread(
        target=start_server,
        # MEJORA: Pasamos el directorio de búsqueda y la ruta específica de la BD
        args=(port, search_root, str(db_abs_path)),
        daemon=True
    )
    server_thread.start()
    time.sleep(1)
    
    url = f"http://localhost:{port}"
    print(f"🎨 Visualización en tiempo real disponible en: {url}")
    try:
        webbrowser.open_new_tab(url)
    except Exception as e:
        print(f"  (No se pudo abrir el navegador automáticamente: {e})")

    # Crear y ejecutar proceso evolutivo
    evolution_process = ConceptEvolution(
        problem_description=problem_description,
        cfg=cfg
    )

    try:
        # MEJORA: Ejecutar el bucle principal de forma asíncrona
        evolution_process.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Proceso interrumpido por el usuario")
        print("💾 Guardando checkpoint de emergencia...")
        evolution_process.checkpoint_manager.save_checkpoint(
            evolution_process.db.get_all_programs(),
            evolution_process.current_generation,
            problem_description
        )
        print("✅ Checkpoint guardado. Puedes reanudar con --resume")
    except Exception as e:
        print(f"\n\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Intentando guardar checkpoint de emergencia...")
        try:
            evolution_process.checkpoint_manager.save_checkpoint(
                evolution_process.db.get_all_programs(),
                evolution_process.current_generation,
                problem_description
            )
            print("✅ Checkpoint de emergencia guardado")
        except:
            print("❌ No se pudo guardar checkpoint de emergencia")
        exit(1)

if __name__ == "__main__":
    main()