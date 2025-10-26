import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib 
from huggingface_hub import hf_hub_download 
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
from django.conf import settings 

# CONFIGURACIÓN DE PLOT Y RUTAS (Se mantiene para la estructura de Django)
matplotlib.use('Agg')
plt.style.use('default') 

# --- CONFIGURACIÓN DE HUGGING FACE Y ARTEFACTOS ---
HF_REPO_ID = "Lia896gh/csv" 
HF_REPO_TYPE = "dataset" 
CSV_FILENAME = 'TotalFeatures-ISCXFlowMeter.csv'

# MUESTRAS SIMPLIFICADAS: 
N_ROWS_FOR_F1 = 20000 # Recomendado para evitar fallos de memoria/timeout
N_ROWS_FOR_PLOTS = 10 # Fijo en 10 para la Tabla de datos

# SOLO NECESITAMOS LOS MODELOS PARA F1-SCORE
ARTEFACTS = {
    'f1': 'model_f1.joblib', 
    'scaler': 'scaler_f1.joblib' 
}

RESOURCES_DIR = os.path.join(settings.BASE_DIR, 'hf_cache')
os.makedirs(RESOURCES_DIR, exist_ok=True)

# --- VARIABLES GLOBALES DE RECURSOS ---
GLOBAL_RESOURCES = {}
CSV_FILE_PATH = None
RESOURCES_LOADED = False

COLUMNS_NEEDED_FOR_ML = [
    'calss', 'duration', 'total_fpackets', 'total_bpktl', 
    'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 
    'mean_active', 'Init_Win_bytes_forward', 'min_flowpktl', 'flow_fin'
]
FEATURES_CLS_ALL = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 
                    'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 
                    'Init_Win_bytes_forward']
TARGET_COL_CLS = 'Class'

# --------------------------------------------------------------------
# FUNCIÓN DE CARGA GLOBAL
# --------------------------------------------------------------------

def download_hf_file(filename):
    """Descarga un archivo de Hugging Face a la caché local."""
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename, local_dir=RESOURCES_DIR, repo_type=HF_REPO_TYPE
    )

def load_global_resources():
    """Descarga modelos y CSV, y carga solo los modelos en memoria."""
    global CSV_FILE_PATH, RESOURCES_LOADED, GLOBAL_RESOURCES
    
    try:
        print("Iniciando descarga y carga optimizada de ARTEFACTOS...")
        
        # SOLO CARGA F1 Y SCALER
        for key, filename in ARTEFACTS.items():
            path = download_hf_file(filename)
            GLOBAL_RESOURCES[key] = joblib.load(path)
            print(f"✅ Cargado {key}")

        CSV_FILE_PATH = download_hf_file(CSV_FILENAME)
        
        RESOURCES_LOADED = True
        print("Recursos de ML listos para el despliegue.")

    except Exception as e:
        print(f"ERROR FATAL AL CARGAR RECURSOS: {e}") 
        RESOURCES_LOADED = False

# --------------------------------------------------------------------
# FUNCIÓN AUXILIAR VACÍA 
# --------------------------------------------------------------------

def generar_grafica_base64(fig):
    """Función de dummy para evitar fallos de Matplotlib."""
    plt.close(fig)
    return ""

# -------------------------------------------------------------------------
# FUNCIÓN DE EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------

def run_malware_analysis():
    """Usa los modelos cargados para calcular solo el F1-Score y la Tabla."""
    
    if not RESOURCES_LOADED:
        return {'error': "ERROR: Recursos de ML no cargados.", 'accuracy': 0.0, 'dataframe': []}

    df_full = None
    try:
        # Carga solo la muestra necesaria para el cálculo del F1
        df_full = pd.read_csv(
            CSV_FILE_PATH, nrows=N_ROWS_FOR_F1, usecols=COLUMNS_NEEDED_FOR_ML
        )
        
        df_full.columns = df_full.columns.str.strip()
        df_full.columns = [col.replace("calss", "Class") for col in df_full.columns] 
        for col in df_full.columns:
            if col != TARGET_COL_CLS:
                df_full[col] = pd.to_numeric(df_full[col], errors='coerce') 
        df_full.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    except Exception as e:
        return {'error': f"Fallo al cargar la muestra del CSV: {e}", 'accuracy': 0.0, 'dataframe': []}
    
    df_safe = df_full.copy()
    
    # *** MUESTRA UNIFICADA DE 10 FILAS PARA TABLA ***
    df_sample_10 = df_safe.head(N_ROWS_FOR_PLOTS).copy() 
    
    model_f1 = GLOBAL_RESOURCES['f1']
    scaler_f1 = GLOBAL_RESOURCES['scaler']

    # =========================================================================
    # PARTE A: CLASIFICACIÓN BINARIA (Cálculo de F1-Score)
    # =========================================================================
    class_counts = df_safe[TARGET_COL_CLS].value_counts()
    top_2_classes = class_counts.index[:2].tolist()
    
    # Manejo seguro si no hay al menos dos clases
    if len(top_2_classes) < 2:
        f1_rounded = 0.0
    else:
        df_filtered_cls_f1 = df_safe[df_safe[TARGET_COL_CLS].isin(top_2_classes)].copy()
        
        # Mapeo de clases a 0 y 1
        class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
        df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[TARGET_COL_CLS].map(class_map)
        
        y_cls_f1 = df_filtered_cls_f1['target_binary']
        X_cls_f1 = df_filtered_cls_f1[FEATURES_CLS_ALL].copy()
        
        # LIMPIEZA EXTREMA: Fundamental para evitar F1 = 0 por incompatibilidad del Scaler
        X_cls_f1.replace([np.inf, -np.inf, np.nan], 0, inplace=True) 
        
        # Split de datos
        X_train_f1, X_test_f1, y_train_f1, y_test_f1 = train_test_split(X_cls_f1, y_cls_f1, test_size=0.4, random_state=42)
        
        # Inferencia y cálculo de F1
        X_test_scaled_f1 = scaler_f1.transform(X_test_f1)
        y_pred_f1 = model_f1.predict(X_test_scaled_f1)
        
        # Verificación final para evitar errores de F1-Score 
        if len(y_test_f1.unique()) < 2 or len(y_pred_f1) == 0:
             f1_rounded = 0.0
        else:
             f1 = f1_score(y_test_f1, y_pred_f1, average='binary') 
             f1_rounded = round(f1, 4)

    # 3. Preparación de Salida Final (Tabla de 10 filas)
    df_sample_head = df_sample_10.to_dict('records') 

    # 4. REGRESO DE VALORES MÍNIMOS (Gráficas vacías)
    return {
        'accuracy': f1_rounded, 
        'dataframe': df_sample_head,
        'grafica1_b64': "",      # VACÍO
        'grafica3_b64': "",      # VACÍO
        'regressionData': {}     # VACÍO
    }