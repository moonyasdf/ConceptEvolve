# src/webui/visualization.py

# MEJORA: #16 (Visualización en Tiempo Real) - Servidor web adaptado de ShinkaEvolve.
# MEJORA: Se elimina la dependencia de un nombre de DB hardcodeado (`DB_NAME`).

import http.server
import json
import os
import socketserver
import sqlite3
import time
import urllib.parse
from pathlib import Path

class DatabaseRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Manejador de peticiones HTTP para la visualización."""
    
    # MEJORA: El handler ahora recibe la ruta a la base de datos
    def __init__(self, *args, db_path_to_serve=None, **kwargs):
        self.db_path_to_serve = db_path_to_serve
        # El directorio de los archivos estáticos (HTML, CSS, JS)
        webui_dir = Path(__file__).parent.resolve()
        super().__init__(*args, directory=str(webui_dir), **kwargs)

    def log_message(self, format, *args):
        # Silenciar logs para una salida más limpia
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/get_programs":
            # MEJORA: Ya no necesita buscar la BD, la tiene en self.db_path_to_serve
            return self.handle_get_programs()
        
        # Servir el HTML principal en la raíz
        if parsed_url.path == "/":
            self.path = "/viz_tree.html"

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def handle_get_programs(self):
        db_path = self.db_path_to_serve
        if not db_path or not os.path.exists(db_path):
            self.send_error(404, f"Database not found at {db_path}")
            return

        for attempt in range(5): # Intentar hasta 5 veces
            try:
                # MEJORA: Usa la ruta de la BD directamente
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM concepts")
                rows = cursor.fetchall()
                
                concepts = []
                for row in rows:
                    concept_dict = dict(row)
                    # Deserializar campos JSON
                    for key in ['draft_history', 'critique_history', 'inspiration_ids', 'embedding', 'scores', 'system_requirements']:
                        if key in concept_dict and isinstance(concept_dict[key], str):
                            try:
                                concept_dict[key] = json.loads(concept_dict[key])
                            except json.JSONDecodeError:
                                concept_dict[key] = {} if 'scores' in key or 'req' in key else []
                    concepts.append(concept_dict)
                
                conn.close()
                self.send_json_response(concepts)
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    print(f"  [WebUI] Database is locked, attempt {attempt+1}/5...")
                    time.sleep(0.5 + attempt * 0.5) # Backoff
                else:
                    self.send_error(500, f"Database error: {e}")
                    return
            except Exception as e:
                self.send_error(500, f"An unexpected error occurred: {e}")
                return
        
        self.send_error(503, "Database is busy, please try again later.")

    def send_json_response(self, data):
        payload = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

# MEJORA: El factory ahora pasa la ruta de la BD al handler
def create_handler_factory(db_path_to_serve):
    def handler_factory(*args, **kwargs):
        return DatabaseRequestHandler(*args, db_path_to_serve=db_path_to_serve, **kwargs)
    return handler_factory

# MEJORA: `start_server` ahora acepta la ruta específica de la BD a servir
def start_server(port: int, db_path_to_serve: str):
    handler_factory = create_handler_factory(db_path_to_serve)
    
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), handler_factory) as httpd:
        print(f"[*] Servidor de visualización iniciado en http://0.0.0.0:{port}")
        print(f"    Sirviendo datos de: {db_path_to_serve}")
        httpd.serve_forever()