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

# CONFIGURACIÓN DE PLOT Y RUTAS
matplotlib.use('Agg')
plt.style.use('default') 

# --- CONFIGURACIÓN DE HUGGING FACE Y ARTEFACTOS ---
HF_REPO_ID = "Lia896gh/csv" 
HF_REPO_TYPE = "dataset" 
CSV_FILENAME = 'TotalFeatures-ISCXFlowMeter.csv'

# MUESTRAS SIMPLIFICADAS: 
N_ROWS_FOR_F1 = 20000 # Reducido para evitar problemas de memoria/timeout
N_ROWS_FOR_PLOTS = 10 

ARTEFACTS = {
    'f1': 'model_f1.joblib', 'reg': 'model_reg.joblib', 
    'clas': 'model_svm.joblib', 'le': 'le_clas.joblib',
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
        
        # Solo necesitamos los modelos f1, clas, le y scaler para este script simplificado
        for key in ['f1', 'clas', 'le', 'scaler']:
            path = download_hf_file(ARTEFACTS[key])
            GLOBAL_RESOURCES[key] = joblib.load(path)
            print(f"✅ Cargado {key}")

        CSV_FILE_PATH = download_hf_file(CSV_FILENAME)
        
        RESOURCES_LOADED = True
        print("Recursos de ML listos para el despliegue.")

    except Exception as e:
        print(f"ERROR FATAL AL CARGAR RECURSOS: {e}") 
        RESOURCES_LOADED = False

# --------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------

def generar_grafica_base64(fig):
    """Guarda una figura de Matplotlib en un buffer y la codifica a Base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white') 
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# -------------------------------------------------------------------------
# FUNCIÓN DE EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------

def run_malware_analysis():
    """Usa los modelos cargados y datos bajo demanda para generar resultados."""
    
    if not RESOURCES_LOADED:
        return {'error': "ERROR: Recursos de ML no cargados.", 'accuracy': 0.0, 'dataframe': []}

    df_full = None
    try:
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
        return {'error': f"Fallo al cargar la muestra del CSV en Railway: {e}", 'accuracy': 0.0, 'dataframe': []}
    
    df_safe = df_full.copy()
    
    # *** MUESTRA UNIFICADA DE 10 FILAS PARA VISUALIZACIÓN ***
    df_sample_10 = df_safe.head(N_ROWS_FOR_PLOTS).copy() 
    
    model_f1 = GLOBAL_RESOURCES['f1']
    model_clas = GLOBAL_RESOURCES['clas']
    le_clas = GLOBAL_RESOURCES['le']
    scaler_f1 = GLOBAL_RESOURCES['scaler']

    # =========================================================================
    # PARTE A: CLASIFICACIÓN BINARIA (F1-Score)
    # =========================================================================
    class_counts = df_safe[TARGET_COL_CLS].value_counts()
    top_2_classes = class_counts.index[:2].tolist()
    df_filtered_cls_f1 = df_safe[df_safe[TARGET_COL_CLS].isin(top_2_classes)].copy()
    class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
    df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[TARGET_COL_CLS].map(class_map)
    y_cls_f1 = df_filtered_cls_f1['target_binary']
    X_cls_f1 = df_filtered_cls_f1[FEATURES_CLS_ALL].copy()
    X_cls_f1.replace([np.inf, -np.inf], 0, inplace=True) 
    
    X_train_f1, X_test_f1, y_train_f1, y_test_f1 = train_test_split(X_cls_f1, y_cls_f1, test_size=0.4, random_state=42)
    
    X_test_scaled_f1 = scaler_f1.transform(X_test_f1)
    y_pred_f1 = model_f1.predict(X_test_scaled_f1)
    f1 = f1_score(y_test_f1, y_pred_f1, average='binary') 
    f1_rounded = round(f1, 4)

    # =========================================================================
    # PARTE B: GRÁFICA 1 - Clasificación SVM (Muestra de 10 filas)
    # =========================================================================
    top_3_classes = class_counts.index[:3].tolist()
    df_filtered_svm = df_sample_10[df_sample_10[TARGET_COL_CLS].isin(top_3_classes)].copy()
    X_clas_filt = df_filtered_svm[['min_flowpktl', 'flow_fin']].copy()
    
    X_clas_filt['min_flowpktl'] = np.log1p(X_clas_filt['min_flowpktl'])
    X_clas_filt['flow_fin'] = np.log1p(X_clas_filt['flow_fin'])
    
    try:
        y_clas_encoded = le_clas.transform(df_filtered_svm[TARGET_COL_CLS])
        class_names_svm = le_clas.classes_
    except ValueError:
        class_names_svm = ['A', 'B', 'C']
        y_clas_encoded = [0] * len(X_clas_filt)
    
    x_min, x_max = X_clas_filt.iloc[:, 0].min() - 0.1, X_clas_filt.iloc[:, 0].max() + 0.1
    y_min, y_max = X_clas_filt.iloc[:, 1].min() - 0.1, X_clas_filt.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    feature_names = X_clas_filt.columns
    grid_data_svc = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names) 

    Z = model_clas.predict(grid_data_svc) 
    Z = Z.reshape(xx.shape)

    fig1, ax1 = plt.subplots(figsize=(10, 8)) 
    ax1.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm') 
    for i in range(len(class_names_svm)):
        ax1.scatter(X_clas_filt.iloc[y_clas_encoded == i, 0], X_clas_filt.iloc[y_clas_encoded == i, 1],
                    edgecolors='k', s=60, label=f'Clase: {class_names_svm[i]}', alpha=0.8)
    ax1.set_title('Gráfica 1: Separabilidad de Datos con SVM (Log Transformación)', fontsize=14)
    ax1.set_xlabel('Característica: log(1 + min_flowpktl)', fontsize=12) 
    ax1.set_ylabel('Característica: log(1 + flow_fin)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    grafica1_b64 = generar_grafica_base64(fig1) 

    # =========================================================================
    # PARTES C & D: REGRESIÓN - ELIMINADAS.
    # Se regresan valores vacíos para que el frontend no falle.
    # =========================================================================
    grafica3_b64 = "" # Vacío
    regression_data_surface = {} # Vacío

    # 3. Preparación de Salida Final (Tabla de 10 filas)
    df_sample_head = df_sample_10.to_dict('records') 

    return {
        'accuracy': f1_rounded, 
        'dataframe': df_sample_head,
        'grafica1_b64': grafica1_b64, 
        'grafica3_b64': grafica3_b64, # Vacío
        'regressionData': regression_data_surface # Vacío
    }