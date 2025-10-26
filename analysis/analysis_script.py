import json
import os
from django.conf import settings

# --- CONFIGURACIÓN DE ARCHIVOS ---
# El archivo JSON debe estar en la misma carpeta o accesible a través de la ruta.
# Se usa la ruta BASE de Django para mayor robustez.
JSON_FILE_NAME = 'static_results.json'

# Inicializamos el diccionario de datos cargados
STATIC_RESULTS = None
RESOURCES_LOADED = False

def load_global_resources():
    """
    Función que carga el archivo JSON estático en memoria una sola vez al inicio.
    """
    global STATIC_RESULTS, RESOURCES_LOADED
    
    if RESOURCES_LOADED:
        print("Recursos estáticos ya cargados. Omitiendo la carga.")
        return

    try:
        # Intentamos obtener la ruta base del proyecto Django
        base_dir = settings.BASE_DIR
        file_path = os.path.join(base_dir, 'analysis', JSON_FILE_NAME)
        
        # Si la ruta no funciona, intentamos la ruta directa (por si el script está en la raíz)
        if not os.path.exists(file_path):
             file_path = os.path.join(base_dir, JSON_FILE_NAME)
             if not os.path.exists(file_path):
                  raise FileNotFoundError(f"Archivo {JSON_FILE_NAME} no encontrado en {base_dir}")

        print(f"Iniciando carga del archivo JSON estático desde: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            STATIC_RESULTS = json.load(f)
            
        RESOURCES_LOADED = True
        print("✅ Recursos estáticos de análisis cargados correctamente.")

    except Exception as e:
        print(f"❌ ERROR FATAL AL CARGAR EL JSON ESTATICO: {e}") 
        RESOURCES_LOADED = False
        STATIC_RESULTS = None # Aseguramos que sea None si hay fallo

# -------------------------------------------------------------------------
# FUNCIÓN DE EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------

def run_malware_analysis():
    
    if not RESOURCES_LOADED or STATIC_RESULTS is None:
        return {
            'error': "ERROR: El JSON estático no se pudo cargar. Revise la ruta del archivo.", 
            'accuracy': 0.0, 
            'dataframe': [],
            'grafica1_b64': '',
            'grafica3_b64': '',
            'regressionData': {},
            'status_message': '❌ Fallo en la carga de datos estáticos.'
        }

    # El script simplemente devuelve el diccionario completo cargado del archivo.
    print("STATUS: ✅ Todos los datos estáticos de análisis han sido procesados y retornados.")

    # Devolvemos el diccionario completo.
    return STATIC_RESULTS