import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import joblib 
from huggingface_hub import hf_hub_download 
from django.conf import settings 

# Usar backend no interactivo para servidores
matplotlib.use('Agg')
plt.style.use('default') 

# --- CONFIGURACIÓN DE HUGGING FACE ---
HF_REPO_ID = "Lia896gh/csv" 
HF_REPO_TYPE = "dataset" 
HF_SUBFOLDER = "" 

# --- NOMBRES DE ARCHIVOS EN EL REPOSITORIO DE HF ---
MODEL_F1_FILENAME = 'model_f1.joblib'
MODEL_REG_FILENAME = 'model_reg.joblib'
MODEL_SVM_FILENAME = 'model_svm.joblib'
LE_CLAS_FILENAME = 'le_clas.joblib'
CSV_FILENAME = 'TotalFeatures-ISCXFlowMeter.csv'

# Columnas mínimas necesarias
COLUMNS_NEEDED_FOR_ML = [
    'calss', 'duration', 'total_fpackets', 'total_bpktl', 
    'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 
    'mean_active', 'Init_Win_bytes_forward', 'min_flowpktl', 'flow_fin'
]

# Directorio de caché temporal
RESOURCES_DIR = os.path.join(settings.BASE_DIR, 'hf_cache')
os.makedirs(RESOURCES_DIR, exist_ok=True)


# --------------------------------------------------------------------
# *** FUNCIÓN DE OPTIMIZACIÓN DE MEMORIA DEL DATAFRAME ***
# --------------------------------------------------------------------

def optimize_dataframe_memory(df):
    """Reduce el uso de memoria de un DataFrame ajustando los tipos de datos."""
    for col in df.columns:
        if df[col].dtype != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(df[col].dtype)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)    
            
            elif str(df[col].dtype)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

# --------------------------------------------------------------------
# *** CARGA GLOBAL DE DATOS Y MODELOS DESDE HUGGING FACE ***
# --------------------------------------------------------------------

# Inicializar variables globales
GLOBAL_DF = None
GLOBAL_MODEL_F1 = None
GLOBAL_MODEL_REG = None
GLOBAL_MODEL_CLAS = None
GLOBAL_LE_CLAS = None
RESOURCES_LOADED = False

# Función auxiliar para descargar un archivo de Hugging Face
def download_hf_file(filename):
    file_path_in_repo = os.path.join(HF_SUBFOLDER, filename)
    print(f"Descargando {file_path_in_repo} de Hugging Face (ID: {HF_REPO_ID}, Tipo: {HF_REPO_TYPE})...")
    
    return hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename=file_path_in_repo,
        local_dir=RESOURCES_DIR,
        repo_type=HF_REPO_TYPE
    )

try:
    # 1. DESCARGA A CACHÉ
    print(f"Iniciando descarga y lectura optimizada desde Hugging Face: {HF_REPO_ID}")
    
    # Descargar todos los archivos
    CSV_FILE_PATH = download_hf_file(CSV_FILENAME)
    MODEL_F1_PATH = download_hf_file(MODEL_F1_FILENAME)
    MODEL_REG_PATH = download_hf_file(MODEL_REG_FILENAME)
    MODEL_SVM_PATH = download_hf_file(MODEL_SVM_FILENAME)
    LE_CLAS_PATH = download_hf_file(LE_CLAS_FILENAME)

    # 1b. Cargar CSV con optimización (Aumento de filas CRÍTICO)
    N_ROWS_TO_LOAD = 50000  # Mantener en 50,000
    
    print(f"Leyendo las primeras {N_ROWS_TO_LOAD} filas y columnas específicas del CSV descargado.")

    df_temp = pd.read_csv(
        CSV_FILE_PATH,
        nrows=N_ROWS_TO_LOAD,
        usecols=COLUMNS_NEEDED_FOR_ML 
    ) 
    
    # Preprocesamiento inicial
    df_temp.columns = df_temp.columns.str.strip()
    df_temp.columns = [col.replace("calss", "Class") for col in df_temp.columns] 
    target_col_name = 'Class' 

    # Conversión y manejo de NaNs/Infinitos
    for col in df_temp.columns:
        if col != target_col_name:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce') 
    df_temp.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    df_temp = optimize_dataframe_memory(df_temp)
    GLOBAL_DF = df_temp
    
    # 1c. Cargar Modelos 
    GLOBAL_MODEL_F1 = joblib.load(MODEL_F1_PATH)
    GLOBAL_MODEL_REG = joblib.load(MODEL_REG_PATH)
    GLOBAL_MODEL_CLAS = joblib.load(MODEL_SVM_PATH)
    GLOBAL_LE_CLAS = joblib.load(LE_CLAS_PATH)
    
    RESOURCES_LOADED = True
    print("Recursos de ML cargados exitosamente desde Hugging Face y optimizados.")

except Exception as e:
    print(f"ERROR FATAL: Fallo al cargar recursos (Timeout/Memoria/Ruta de Archivo) debido a: {e}")
    

# Función auxiliar para convertir gráficas a base64
def generar_grafica_base64(fig):
    """Convierte un objeto Matplotlib figure a una cadena base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

# -------------------------------------------------------------------------
# FUNCIÓN DE EJECUCIÓN (USADA POR DJANGO EN RENDER)
# -------------------------------------------------------------------------

def run_malware_analysis():
    """Usa los modelos y datos cargados globalmente y genera resultados."""
    
    # 1. Comprobación de recursos globales
    if GLOBAL_DF is None or GLOBAL_MODEL_F1 is None or not RESOURCES_LOADED:
        return { 
            'error': "ERROR: Recursos de ML no cargados. El servidor falló en la inicialización o Hugging Face no fue accesible. Verifique los logs.", 
            'accuracy': 0.0, 
            'dataframe': [] 
        }

    # Usar los recursos globales
    df_safe = GLOBAL_DF.copy()
    model_f1 = GLOBAL_MODEL_F1
    model_reg = GLOBAL_MODEL_REG
    model_clas = GLOBAL_MODEL_CLAS
    le_clas = GLOBAL_LE_CLAS

    # 2. Preprocesamiento de datos (solo de variables globales)
    target_col_cls = 'Class' 
    features_cls_all = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 'Init_Win_bytes_forward']
    
    # Muestreo al 10% de las filas cargadas
    df_sample = df_safe.sample(frac=0.1, random_state=42)
    class_counts = df_safe[target_col_cls].value_counts()
    f1_rounded = 0.0 
    
    # Inicializar variables de salida
    grafica1_b64 = None
    grafica3_b64 = None
    regression_data_surface = {}
    
    # =========================================================================
    # PARTE A: CLASIFICACIÓN BINARIA (Calculo F1-Score)
    # =========================================================================
    
    if len(class_counts) < 2:
        return { 
            'error': f"Error ML: La muestra cargada ({len(df_safe)} filas) solo tiene {len(class_counts)} clases únicas. Se requieren al menos 2 para F1.", 
            'accuracy': 0.0, 
            'dataframe': df_safe.head(10).to_dict('records') 
        }
        
    top_2_classes = class_counts.index[:2].tolist()
    df_filtered_cls_f1 = df_safe[df_safe[target_col_cls].isin(top_2_classes)].copy()
    class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
    df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[target_col_cls].map(class_map)
    y_cls_f1 = df_filtered_cls_f1['target_binary']
    X_cls_f1 = df_filtered_cls_f1[features_cls_all].copy()
    X_cls_f1.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Split
    if len(X_cls_f1) < 2:
        return {'error': "Error ML: Después de filtrar, no quedan suficientes muestras para el split de F1.", 'accuracy': 0.0, 'dataframe': [] }
        
    X_train_f1, X_test_f1, y_train_f1, y_test_f1 = train_test_split(X_cls_f1, y_cls_f1, test_size=0.4, random_state=42)
    
    # Escalar y Predecir
    scaler_f1 = StandardScaler()
    if X_train_f1.empty:
         print("ADVERTENCIA: X_train_f1 está vacío, saltando cálculo de F1.")
    else:
        scaler_f1.fit(X_train_f1) 
        X_test_scaled_f1 = scaler_f1.transform(X_test_f1)
        
        y_pred_f1 = model_f1.predict(X_test_scaled_f1) 
        f1 = f1_score(y_test_f1, y_pred_f1, average='binary') 
        f1_rounded = round(f1, 4)

    # =========================================================================
    # PARTE B: GRÁFICA 1 - Clasificación SVM
    # =========================================================================
    
    required_svm_features = ['min_flowpktl', 'flow_fin']
    
    if not all(f in df_safe.columns for f in required_svm_features) or len(class_counts) < 3:
        print("ADVERTENCIA: Saltando Gráfica 1 (SVM) por insuficiencia de datos/clases/columnas.")
    else:
        top_3_classes = class_counts.index[:3].tolist()
        df_filtered_svm = df_sample[df_sample[target_col_cls].isin(top_3_classes)].copy()
        X_clas_filt = df_filtered_svm[required_svm_features].copy()
        
        X_clas_filt['min_flowpktl'] = np.log1p(X_clas_filt['min_flowpktl'])
        X_clas_filt['flow_fin'] = np.log1p(X_clas_filt['flow_fin'])
        
        try:
             y_clas_encoded = le_clas.transform(df_filtered_svm[target_col_cls]) 
        except ValueError:
             print("ADVERTENCIA: Saltando Gráfica 1, LabelEncoder no reconoce las clases en la muestra.")
        else:
             x_min, x_max = X_clas_filt.iloc[:, 0].min() - 0.1, X_clas_filt.iloc[:, 0].max() + 0.1
             y_min, y_max = X_clas_filt.iloc[:, 1].min() - 0.1, X_clas_filt.iloc[:, 1].max() + 0.1
             
             # CORRECCIÓN CRÍTICA: Reducción de la malla de predicción a 30x30
             xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30)) 
             
             feature_names = X_clas_filt.columns
             grid_data_svc = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names) 
             grid_data_svc.replace([np.inf, -np.inf, np.nan], 0, inplace=True) 

             Z = model_clas.predict(grid_data_svc) 
             Z = Z.reshape(xx.shape)
             
             fig1, ax1 = plt.subplots(figsize=(10, 8))
             ax1.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm') 
             class_names_svm = le_clas.classes_
             for i, class_name in enumerate(class_names_svm):
                 ax1.scatter(X_clas_filt.iloc[y_clas_encoded == i, 0], X_clas_filt.iloc[y_clas_encoded == i, 1],
                             edgecolors='k', s=60, label=f'Clase: {class_name}', alpha=0.8)
             ax1.set_title('Gráfica 1: Separabilidad de Datos con SVM (Log Transformación)', fontsize=14)
             ax1.set_xlabel('Característica: log(1 + min_flowpktl)', fontsize=12) 
             ax1.set_ylabel('Característica: log(1 + flow_fin)', fontsize=12)
             ax1.legend(loc='upper right', fontsize=10)
             ax1.grid(True, linestyle='--', alpha=0.6)
             grafica1_b64 = generar_grafica_base64(fig1)

    # =========================================================================
    # PARTES C & D: REGRESIÓN (Gráficas 2 y 3)
    # =========================================================================
    
    y_reg_original = df_sample['Init_Win_bytes_forward'].copy()
    y_reg_original[y_reg_original < 0] = 0
    y_reg_original.replace([np.inf, -np.inf, np.nan], 0, inplace=True) 
    y_reg_transformed = np.log1p(y_reg_original)
    X_reg = df_sample.drop(['Init_Win_bytes_forward', target_col_cls], axis=1, errors='ignore')
    X_reg.replace([np.inf, -np.inf], 0, inplace=True) 
    
    if len(X_reg.columns) < 2:
        print("ADVERTENCIA: Saltando Gráfica 2 y 3, no hay suficientes columnas para X_reg.")
    else:
        # Usar feature importances o las primeras 2
        if hasattr(model_reg, 'feature_importances_'):
            feature_importances = pd.Series(model_reg.feature_importances_, index=X_reg.columns)
            top_2_features = feature_importances.nlargest(2).index.tolist()
        else:
            top_2_features = X_reg.columns[:2].tolist()
        
        if len(top_2_features) < 2:
            top_2_features = X_reg.columns[:2].tolist()

        X_reg_top = X_reg[top_2_features]
        X_reg_top.replace([np.inf, -np.inf], 0, inplace=True)
        
        X_train_reg, X_test_reg, y_train_reg_transf, y_test_reg_transf = train_test_split(
            X_reg_top, y_reg_transformed, test_size=0.3, random_state=42
        )
        
        x_min_r, x_max_r = X_reg_top.iloc[:, 0].min() - 0.5, X_reg_top.iloc[:, 0].max() + 0.5
        y_min_r, y_max_r = X_reg_top.iloc[:, 1].min() - 0.5, X_reg_top.iloc[:, 1].max() + 0.5
        
        # CORRECCIÓN CRÍTICA: Reducción de la malla de predicción a 10x10
        xx_r, yy_r = np.meshgrid(np.linspace(x_min_r, x_max_r, 10), np.linspace(y_min_r, y_max_r, 10))
        
        grid_data = pd.DataFrame(np.c_[xx_r.ravel(), yy_r.ravel()], columns=top_2_features)
        grid_data.replace([np.inf, -np.inf], 0, inplace=True) 

        Z_reg = model_reg.predict(grid_data) 

        regression_data_surface = {
            'x_feature': top_2_features[0], 'y_feature': top_2_features[1],
            'x_line': xx_r.flatten().tolist(), 'y_line': yy_r.flatten().tolist(),           
            'z_line': Z_reg.flatten().tolist(), 'x_data': X_reg_top.iloc[:, 0].tolist(), 
            'y_data': X_reg_top.iloc[:, 1].tolist(), 'y_data_class': y_reg_transformed.tolist()
        }
        
        y_pred_reg_transf = model_reg.predict(X_test_reg)
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.scatter(y_test_reg_transf, y_pred_reg_transf, alpha=0.6, color='#5B21B6') 
        min_val = min(y_test_reg_transf.min(), y_pred_reg_transf.min())
        max_val = max(y_test_reg_transf.max(), y_pred_reg_transf.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax3.set_xlabel("Valores Reales (log transformados)", fontsize=12)
        ax3.set_ylabel("Valores Predichos (log transformados)", fontsize=12)
        ax3.set_title("Gráfica 3: Valores reales vs. Predicciones (Log Transformados)", fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.6)
        grafica3_b64 = generar_grafica_base64(fig3)

    # 3. Preparación de Salida Final
    df_sample_head = df_safe.head(10).to_dict('records') 

    return {
        'accuracy': f1_rounded, 
        'dataframe': df_sample_head,
        'grafica1_b64': grafica1_b64, 
        'grafica3_b64': grafica3_b64, 
        'regressionData': regression_data_surface
    }


if __name__ == '__main__':
    pass