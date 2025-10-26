import json
import os
import time # 👈 Importamos la librería time
from django.conf import settings

# --- CONFIGURACIÓN DE ARCHIVOS ---
JSON_FILE_NAME = 'static_results.json'

# --- CONFIGURACIÓN DEL RETRASO ---
# Ajusta este valor (en segundos) para simular el tiempo de cálculo.
SIMULATION_DELAY_SECONDS = 5 

STATIC_RESULTS = None
RESOURCES_LOADED = False

def load_global_resources():
    """
    Función que carga el archivo JSON estático en memoria una sola vez al inicio.
    """
    global STATIC_RESULTS, RESOURCES_LOADED
    
    if RESOURCES_LOADED:
        # print("Recursos estáticos ya cargados. Omitiendo la carga.")
        return

    try:
        # Intentamos obtener la ruta base del proyecto Django
        # Asume que el archivo static_results.json está en el directorio 'analysis' o en la raíz.
        base_dir = settings.BASE_DIR
        file_path = os.path.join(base_dir, 'analysis', JSON_FILE_NAME)
        
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
        STATIC_RESULTS = None 

# -------------------------------------------------------------------------
# FUNCIÓN DE EJECUCIÓN PRINCIPAL (CON DELAY)
# -------------------------------------------------------------------------

def run_malware_analysis():
    
    # ⏱️ SIMULACIÓN DE CÁLCULO
    print(f"STATUS: Simulando tiempo de cálculo ({SIMULATION_DELAY_SECONDS} segundos)...")
    time.sleep(SIMULATION_DELAY_SECONDS) 
    
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

    # 🟢 MENSAJE DE CONFIRMACIÓN (para el log del servidor)
    print("STATUS: ✅ Datos de análisis cargados completamente.")

    # Devolvemos el diccionario completo, añadiendo un mensaje de estado para el frontend
    if 'status_message' not in STATIC_RESULTS:
         STATIC_RESULTS['status_message'] = '✅ Datos de análisis cargados completamente.'
         
    return STATIC_RESULTS