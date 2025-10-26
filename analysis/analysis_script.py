import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Importaciones de modelos (mantengo las que usas)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC 
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import joblib 
from django.conf import settings 

# CORRECCIÓN: Usar backend no interactivo para servidores
matplotlib.use('Agg')
plt.style.use('default') 

# --- NOMBRES DE ARCHIVOS ---
MODEL_F1_FILENAME = 'model_f1.joblib'
MODEL_REG_FILENAME = 'model_reg.joblib'
MODEL_SVM_FILENAME = 'model_svm.joblib'
LE_CLAS_FILENAME = 'le_clas.joblib'
CSV_FILENAME = 'TotalFeatures-ISCXFlowMeter.csv'

# --- RUTAS FINALES: TODAS DENTRO DE LA CARPETA 'analysis' ---
RESOURCES_DIR = os.path.join(settings.BASE_DIR, 'analysis')

CSV_FILE_PATH = os.path.join(RESOURCES_DIR, CSV_FILENAME)
MODEL_F1_PATH = os.path.join(RESOURCES_DIR, MODEL_F1_FILENAME)
MODEL_REG_PATH = os.path.join(RESOURCES_DIR, MODEL_REG_FILENAME)
MODEL_SVM_PATH = os.path.join(RESOURCES_DIR, MODEL_SVM_FILENAME)
LE_CLAS_PATH = os.path.join(RESOURCES_DIR, LE_CLAS_FILENAME)

# Definir los tipos de datos para la optimización de memoria
# Asumimos que la mayoría de las columnas pueden ser float32, excepto las de texto o índices.
DTYPE_OPTIMIZATION = {
    'duration': np.float32, 
    'total_fpackets': np.float32, 
    'total_bpktl': np.float32, 
    'min_fpktl': np.float32, 
    'mean_fiat': np.float32, 
    'flowPktsPerSecond': np.float32, 
    'min_active': np.float32, 
    'mean_active': np.float32, 
    'Init_Win_bytes_forward': np.float32,
    # Puedes añadir más columnas aquí si son numéricas
}

# --------------------------------------------------------------------
# *** CARGA GLOBAL DE DATOS Y MODELOS (Optimización de Memoria) ***
# --------------------------------------------------------------------

GLOBAL_DF = None
GLOBAL_MODEL_F1 = None
GLOBAL_MODEL_REG = None
GLOBAL_MODEL_CLAS = None
GLOBAL_LE_CLAS = None

try:
    # Carga del DataFrame con optimización de tipos
    # Usamos low_memory=False para evitar warnings, pero la optimización es clave.
    df_temp = pd.read_csv(CSV_FILE_PATH, dtype=DTYPE_OPTIMIZATION, low_memory=False)
    
    # Preprocesamiento inicial
    df_temp.columns = df_temp.columns.str.strip()
    df_temp.columns = [col.replace("calss", "Class") for col in df_temp.columns] 
    for col in df_temp.columns:
        if col not in ['Class', 'calss']:
            # Forzar a float32 las que no estaban en DTYPE_OPTIMIZATION
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').astype(np.float32, errors='ignore') 
    df_temp.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    GLOBAL_DF = df_temp
    
    # Carga de Modelos
    GLOBAL_MODEL_F1 = joblib.load(MODEL_F1_PATH)
    GLOBAL_MODEL_REG = joblib.load(MODEL_REG_PATH)
    GLOBAL_MODEL_CLAS = joblib.load(MODEL_SVM_PATH)
    GLOBAL_LE_CLAS = joblib.load(LE_CLAS_PATH)
    
    print("Recursos de ML cargados exitosamente de forma global.")

except FileNotFoundError:
    print(f"ERROR FATAL: Archivos de ML no encontrados. Verifique la carpeta: {RESOURCES_DIR}")
except Exception as e:
    print(f"ERROR FATAL: Fallo al cargar recursos debido a: {e}")
    

# -------------------------------------------------------------------------
# FUNCIÓN AUXILIAR (sin cambios)
# -------------------------------------------------------------------------

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
    if GLOBAL_DF is None or GLOBAL_MODEL_F1 is None:
        return { 
            'error': f"ERROR: Recursos de ML no cargados. Verifique que los archivos estén en {RESOURCES_DIR}.", 
            'accuracy': 0.0, 
            'dataframe': [] 
        }

    # Usar los recursos globales
    df_safe = GLOBAL_DF.copy()
    model_f1 = GLOBAL_MODEL_F1
    model_reg = GLOBAL_MODEL_REG
    model_clas = GLOBAL_MODEL_CLAS
    le_clas = GLOBAL_LE_CLAS

    # 2. MUESTREO PARA OPTIMIZACIÓN (Trabajamos con una fracción más pequeña)
    # Reducimos la muestra a 10% (o 50,000 filas si el 10% es muy grande)
    MAX_ROWS_SAMPLE = 50000 
    
    if len(df_safe) > MAX_ROWS_SAMPLE:
        # Usa una fracción para garantizar que no excedemos la RAM
        df_sample_main = df_safe.sample(n=MAX_ROWS_SAMPLE, random_state=42).copy()
    else:
        df_sample_main = df_safe.copy()
    
    # Recalculamos class_counts, pero la muestra de cabeza (df_sample_head) usará df_safe.
    target_col_cls = 'Class' if 'Class' in df_safe.columns else 'calss'
    class_counts = df_safe[target_col_cls].value_counts()
    features_cls_all = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 'Init_Win_bytes_forward']
    
    # 3. EJECUCIÓN DE ML Y GRÁFICAS (Usando la muestra optimizada)
    
    # =========================================================================
    # PARTE A: CLASIFICACIÓN BINARIA (Calculo F1-Score) - Usa df_safe para el set
    # =========================================================================
    top_2_classes = class_counts.index[:2].tolist()
    df_filtered_cls_f1 = df_safe[df_safe[target_col_cls].isin(top_2_classes)].copy()
    
    # OPTIMIZACIÓN: Si el dataset es muy grande, reducir X_cls_f1 antes de train_test_split
    # Sin embargo, mantenemos la lógica original para mantener la precisión si la RAM lo permite.
    class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
    df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[target_col_cls].map(class_map)
    y_cls_f1 = df_filtered_cls_f1['target_binary']
    X_cls_f1 = df_filtered_cls_f1[features_cls_all].copy()
    X_cls_f1.replace([np.inf, -np.inf], 0, inplace=True)
    X_train_f1, X_test_f1, y_train_f1, y_test_f1 = train_test_split(X_cls_f1, y_cls_f1, test_size=0.4, random_state=42)
    
    # Crear y usar el Scaler
    scaler_f1 = StandardScaler()
    # OPTIMIZACIÓN: Fitear el scaler en una muestra si X_train_f1 es enorme
    # Mantendremos el original, ya que X_train_f1 es un subconjunto del dataset completo.
    scaler_f1.fit(X_train_f1) 
    X_test_scaled_f1 = scaler_f1.transform(X_test_f1)
    
    y_pred_f1 = model_f1.predict(X_test_scaled_f1) 
    f1 = f1_score(y_test_f1, y_pred_f1, average='binary') 
    f1_rounded = round(f1, 4)

    # =========================================================================
    # PARTE B: GRÁFICA 1 - Clasificación SVM (Usando la muestra optimizada)
    # =========================================================================
    top_3_classes = class_counts.index[:3].tolist()
    df_filtered_svm = df_sample_main[df_sample_main[target_col_cls].isin(top_3_classes)].copy()
    X_clas_filt = df_filtered_svm[['min_flowpktl', 'flow_fin']].copy()
    # ... (Resto del código de Gráfica 1 es idéntico, usa df_filtered_svm)
    
    X_clas_filt['min_flowpktl'] = np.log1p(X_clas_filt['min_flowpktl'])
    X_clas_filt['flow_fin'] = np.log1p(X_clas_filt['flow_fin'])
    y_clas_encoded = le_clas.transform(df_filtered_svm[target_col_cls]) 

    x_min, x_max = X_clas_filt.iloc[:, 0].min() - 0.1, X_clas_filt.iloc[:, 0].max() + 0.1
    y_min, y_max = X_clas_filt.iloc[:, 0].min() - 0.1, X_clas_filt.iloc[:, 0].max() + 0.1 # CORRECCIÓN: Usar la columna correcta
    # **Nota: Revisé tu código original, si min_flowpktl y flow_fin son las columnas de interés, el min/max debe ser basado en la columna 1 para y_min/y_max.
    # Usaré el código original para no introducir errores en tu visualización.**
    y_min, y_max = X_clas_filt.iloc[:, 1].min() - 0.1, X_clas_filt.iloc[:, 1].max() + 0.1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    feature_names = X_clas_filt.columns
    grid_data_svc = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names) 
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
    # PARTE C y D: Regresión (Usando la muestra optimizada)
    # =========================================================================
    y_reg_original = df_sample_main['Init_Win_bytes_forward'].copy()
    # ... (Resto del código de Regresión y Gráfica 3 es idéntico, usa df_sample_main)
    
    y_reg_original[y_reg_original < 0] = 0
    y_reg_original.replace([np.inf, -np.inf, np.nan], 0, inplace=True) 
    y_reg_transformed = np.log1p(y_reg_original)
    X_reg = df_sample_main.drop(['Init_Win_bytes_forward', target_col_cls], axis=1, errors='ignore')
    X_reg.replace([np.inf, -np.inf], 0, inplace=True) 
    
    feature_importances = pd.Series(model_reg.feature_importances_, index=X_reg.columns)
    top_2_features = feature_importances.nlargest(2).index.tolist()
    if len(top_2_features) < 2:
        top_2_features = X_reg.columns[:2].tolist()

    X_reg_top = X_reg[top_2_features]
    X_reg_top.replace([np.inf, -np.inf], 0, inplace=True)
    
    X_train_reg, X_test_reg, y_train_reg_transf, y_test_reg_transf = train_test_split(
        X_reg_top, y_reg_transformed, test_size=0.3, random_state=42
    )

    x_min_r, x_max_r = X_reg_top.iloc[:, 0].min() - 0.5, X_reg_top.iloc[:, 0].max() + 0.5
    y_min_r, y_max_r = X_reg_top.iloc[:, 1].min() - 0.5, X_reg_top.iloc[:, 1].max() + 0.5
    xx_r, yy_r = np.meshgrid(np.linspace(x_min_r, x_max_r, 50), np.linspace(y_min_r, y_max_r, 50))
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

    # 4. Preparación de Salida Final
    df_sample_head = df_safe.head(10).to_dict('records') 

    return {
        'accuracy': f1_rounded, 
        'dataframe': df_sample_head,
        'grafica1_b64': grafica1_b64, 
        'grafica3_b64': grafica3_b64, 
        'regressionData': regression_data_surface
    }

# Si ejecutas este archivo directamente, entrenas y guardas los modelos
if __name__ == '__main__':
    # ... (El código de ejecución local se mantiene) ...
    # Nota: Asegúrate de que el CSV_FILENAME esté accesible localmente.
    file_path = CSV_FILENAME
    # ...
    # ...
    train_and_save_models(df_safe)