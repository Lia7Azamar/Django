import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import joblib 
# ¡IMPORTANTE! Necesitamos settings de Django para la ruta absoluta
from django.conf import settings 

# CORRECCIÓN: Usar backend no interactivo para servidores
matplotlib.use('Agg')
plt.style.use('default') 

# Nombres de archivos. IMPORTANTE: Asumimos que están en la raíz del proyecto.
# Si los moviste a la carpeta 'analysis', ajusta la ruta en la función run_malware_analysis.
MODEL_F1_FILENAME = 'model_f1.joblib'
MODEL_REG_FILENAME = 'model_reg.joblib'
MODEL_SVM_FILENAME = 'model_svm.joblib'
LE_CLAS_FILENAME = 'le_clas.joblib'
CSV_FILENAME = 'TotalFeatures-ISCXFlowMeter.csv'

# -------------------------------------------------------------------------
# FUNCIÓN DE ENTRENAMIENTO Y GUARDADO (EJECUCIÓN LOCAL SOLAMENTE)
# -------------------------------------------------------------------------

# NOTA: Los paths en esta función SÓLO son relevantes si entrenas localmente.
# Asumo que train_and_save_models se ejecuta desde la raíz o que los paths
# de guardado ('model_f1.joblib', etc.) están configurados para guardar
# en la raíz (junto a manage.py) para que Render los encuentre.

def train_and_save_models(df_safe):
    """Entrena y guarda los modelos necesarios para la aplicación."""
    # ... (El código de entrenamiento sigue igual) ...
    # Asegúrate que las líneas joblib.dump usen solo el nombre del archivo
    # si se supone que se guardan en la raíz desde donde se ejecuta el script.
    
    # Ejemplo de guardado:
    joblib.dump(model_f1, MODEL_F1_FILENAME)
    print(f"Modelo F1 guardado en {MODEL_F1_FILENAME}")
    
    # ... (resto de train_and_save_models) ...


# -------------------------------------------------------------------------
# FUNCIÓN DE EJECUCIÓN (USADA POR DJANGO EN RENDER)
# -------------------------------------------------------------------------

# Función auxiliar para convertir gráficas a base64 (Asumo que existe)
def generar_grafica_base64(fig):
    """Convierte un objeto Matplotlib figure a una cadena base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def run_malware_analysis():
    """Carga modelos y genera los datos y gráficas requeridos."""
    
    # 1. Carga de Datos y Preprocesamiento
    
    # *** CORRECCIÓN CRUCIAL: Usar settings.BASE_DIR para ruta absoluta ***
    # Esto garantiza que Python encuentre el archivo CSV descargado por Git LFS.
    csv_file_path = os.path.join(settings.BASE_DIR, 'analysis', CSV_FILENAME)

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return { 
            'error': f"Archivo no encontrado: {csv_file_path}. Verifique la ubicación y el estado de Git LFS.", 
            'accuracy': 0.0, 
            'dataframe': [] 
        }

    # ... (Preprocesamiento y limpieza del DataFrame, sin cambios)
    df.columns = df.columns.str.strip()
    df.columns = [col.replace("calss", "Class") for col in df.columns] 
    df_safe = df.copy()
    for col in df_safe.columns:
        if col != 'Class' and col != 'calss':
            df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce') 
    df_safe.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    target_col_cls = 'Class' if 'Class' in df_safe.columns else 'calss'
    features_cls_all = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 'Init_Win_bytes_forward']
    df_sample = df_safe.sample(frac=0.1, random_state=42)
    class_counts = df_safe[target_col_cls].value_counts()


    # 2. Carga de Modelos y Ejecución de ML

    try:
        # *** CORRECCIÓN CRUCIAL: Cargar modelos usando settings.BASE_DIR ***
        model_f1 = joblib.load(os.path.join(settings.BASE_DIR, MODEL_F1_FILENAME))
        model_reg = joblib.load(os.path.join(settings.BASE_DIR, MODEL_REG_FILENAME))
        model_clas = joblib.load(os.path.join(settings.BASE_DIR, MODEL_SVM_FILENAME))
        le_clas = joblib.load(os.path.join(settings.BASE_DIR, LE_CLAS_FILENAME))
    except FileNotFoundError:
        # Devuelve la ruta donde se buscó el primer modelo para el debugging
        model_f1 = joblib.load(os.path.join(settings.BASE_DIR, 'analysis', MODEL_F1_FILENAME))
        return { 
            'error': f"Modelos ML no encontrados. Buscados en: {debug_path}", 
            'accuracy': 0.0, 
            'dataframe': [] 
        }


    # =========================================================================
    # PARTE A: CLASIFICACIÓN BINARIA (Calculo F1-Score)
    # =========================================================================
    # ... (Todo el código de ML y generación de resultados/gráficas es el mismo)
    top_2_classes = class_counts.index[:2].tolist()
    df_filtered_cls_f1 = df_safe[df_safe[target_col_cls].isin(top_2_classes)].copy()
    class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
    df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[target_col_cls].map(class_map)
    y_cls_f1 = df_filtered_cls_f1['target_binary']
    X_cls_f1 = df_filtered_cls_f1[features_cls_all].copy()
    X_cls_f1.replace([np.inf, -np.inf], 0, inplace=True)
    X_train_f1, X_test_f1, y_train_f1, y_test_f1 = train_test_split(X_cls_f1, y_cls_f1, test_size=0.4, random_state=42)
    
    # Crear y usar el Scaler
    scaler_f1 = StandardScaler()
    scaler_f1.fit(X_train_f1) # Volvemos a fitear solo el scaler
    X_test_scaled_f1 = scaler_f1.transform(X_test_f1)
    
    y_pred_f1 = model_f1.predict(X_test_scaled_f1) # Usar modelo CARGADO
    f1 = f1_score(y_test_f1, y_pred_f1, average='binary') 
    f1_rounded = round(f1, 4)

    # =========================================================================
    # PARTE B: GRÁFICA 1 - Clasificación SVM
    # =========================================================================
    top_3_classes = class_counts.index[:3].tolist()
    df_filtered_svm = df_sample[df_sample[target_col_cls].isin(top_3_classes)].copy()
    X_clas_filt = df_filtered_svm[['min_flowpktl', 'flow_fin']].copy()
    X_clas_filt['min_flowpktl'] = np.log1p(X_clas_filt['min_flowpktl'])
    X_clas_filt['flow_fin'] = np.log1p(X_clas_filt['flow_fin'])
    y_clas_encoded = le_clas.transform(df_filtered_svm[target_col_cls]) # Usar transform

    x_min, x_max = X_clas_filt.iloc[:, 0].min() - 0.1, X_clas_filt.iloc[:, 0].max() + 0.1
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
    # PARTE C: GRÁFICA 2 - Superficie de Predicción 
    # =========================================================================
    y_reg_original = df_sample['Init_Win_bytes_forward'].copy()
    y_reg_original[y_reg_original < 0] = 0
    y_reg_original.replace([np.inf, -np.inf, np.nan], 0, inplace=True) 
    y_reg_transformed = np.log1p(y_reg_original)
    X_reg = df_sample.drop(['Init_Win_bytes_forward', target_col_cls], axis=1, errors='ignore')
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

    Z_reg = model_reg.predict(grid_data) # Usar modelo CARGADO

    regression_data_surface = {
        'x_feature': top_2_features[0], 'y_feature': top_2_features[1],
        'x_line': xx_r.flatten().tolist(), 'y_line': yy_r.flatten().tolist(),           
        'z_line': Z_reg.flatten().tolist(), 'x_data': X_reg_top.iloc[:, 0].tolist(), 
        'y_data': X_reg_top.iloc[:, 1].tolist(), 'y_data_class': y_reg_transformed.tolist()
    }
    
    # =========================================================================
    # PARTE D: GRÁFICA 3 - Reales vs Predichos
    # =========================================================================
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

# Si ejecutas este archivo directamente, entrenas y guardas los modelos
if __name__ == '__main__':
    # Carga de Datos inicial para el entrenamiento local
    file_path = CSV_FILENAME # No es necesario el path absoluto para la ejecución local
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df.columns = [col.replace("calss", "Class") for col in df.columns] 
    df_safe = df.copy()
    
    for col in df_safe.columns:
        if col != 'Class' and col != 'calss':
            df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce') 
    df_safe.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    train_and_save_models(df_safe)