import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

# CORRECCIÓN 2: Usar backend no interactivo para servidores (soluciona el warning GUI/Qt)
matplotlib.use('Agg')

# CONFIGURACIÓN GLOBAL DE MATPLOTLIB
plt.style.use('default') 


def generar_grafica_base64(fig):
    """Guarda una figura de Matplotlib en un buffer y la codifica a Base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white') 
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_malware_analysis():
    """
    Carga datos, aplica preprocesamiento, entrena modelos y genera los datos 
    y gráficas (Base64) requeridos para el dashboard, aplicando transformación 
    logarítmica para dispersar los datos.
    """
    # 1. Carga de Datos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'TotalFeatures-ISCXFlowMeter.csv')

    try:
        if not os.path.exists(file_path):
            file_path = 'TotalFeatures-ISCXFlowMeter.csv' 
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return {
            'error': f"Archivo 'TotalFeatures-ISCXFlowMeter.csv' no encontrado. Asegúrate de que exista en el path: {file_path}",
            'accuracy': 0.0,
            'dataframe': [],
            'regressionData': {},
            'grafica1_b64': '',
            'grafica3_b64': ''
        }

    df.columns = df.columns.str.strip()
    df.columns = [col.replace("calss", "Class") for col in df.columns] 
    df_safe = df.copy()
    
    # 2. Preprocesamiento General
    
    # CORRECCIÓN FINAL: Forzar todas las columnas (excepto la clase) a numérico.
    for col in df_safe.columns:
        if col != 'Class' and col != 'calss':
            df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce') 

    # Limpieza inicial para todo el DataFrame
    df_safe.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    # Definiciones
    target_col_cls = 'Class' 
    features_cls_all = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 'Init_Win_bytes_forward']
    if target_col_cls not in df_safe.columns:
         target_col_cls = 'calss' 
    
    # Muestra del 10% para las gráficas de Regresión/Clasificación 2D
    df_sample = df_safe.sample(frac=0.1, random_state=42)
    
    # =========================================================================
    # PARTE A: CLASIFICACIÓN BINARIA (Tu código F1-Score)
    # =========================================================================
    class_counts = df_safe[target_col_cls].value_counts()
    top_2_classes = class_counts.index[:2].tolist()
    df_filtered_cls_f1 = df_safe[df_safe[target_col_cls].isin(top_2_classes)].copy()
    class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
    df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[target_col_cls].map(class_map)
    y_cls_f1 = df_filtered_cls_f1['target_binary']
    X_cls_f1 = df_filtered_cls_f1[features_cls_all].copy()
    X_cls_f1.replace([np.inf, -np.inf], 0, inplace=True) 
    
    X_train_f1, X_test_f1, y_train_f1, y_test_f1 = train_test_split(X_cls_f1, y_cls_f1, test_size=0.4, random_state=42)
    scaler_f1 = StandardScaler()
    X_train_scaled_f1 = scaler_f1.fit_transform(X_train_f1)
    X_test_scaled_f1 = scaler_f1.transform(X_test_f1)
    model_f1 = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    model_f1.fit(X_train_scaled_f1, y_train_f1)
    y_pred_f1 = model_f1.predict(X_test_scaled_f1)
    f1 = f1_score(y_test_f1, y_pred_f1, average='binary') 
    f1_rounded = round(f1, 4)

    # =========================================================================
    # PARTE B: GRÁFICA 1 - Clasificación SVM (Base64) - LOG-TRANSFORMADO
    # =========================================================================
    
    top_3_classes = class_counts.index[:3].tolist()
    df_filtered_svm = df_sample[df_sample[target_col_cls].isin(top_3_classes)].copy()
    
    X_clas_filt = df_filtered_svm[['min_flowpktl', 'flow_fin']].copy()
    
    # APLICAR TRANSFORMACIÓN LOGARÍTMICA
    X_clas_filt['min_flowpktl'] = np.log1p(X_clas_filt['min_flowpktl'])
    X_clas_filt['flow_fin'] = np.log1p(X_clas_filt['flow_fin'])
    
    le_clas = LabelEncoder() 
    
    y_clas_encoded = le_clas.fit_transform(df_filtered_svm[target_col_cls])
    class_names_svm = le_clas.classes_

    # Entrenar SVM con datos transformados
    model_clas = SVC(kernel='rbf', C=10, gamma=0.1)
    model_clas.fit(X_clas_filt, y_clas_encoded)

    # Generar la cuadrícula
    x_min, x_max = X_clas_filt.iloc[:, 0].min() - 0.1, X_clas_filt.iloc[:, 0].max() + 0.1
    y_min, y_max = X_clas_filt.iloc[:, 1].min() - 0.1, X_clas_filt.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # CORRECCIÓN 1: Crear DataFrame con nombres de características
    feature_names = X_clas_filt.columns
    grid_data_svc = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names) 

    Z = model_clas.predict(grid_data_svc) # Usar el DataFrame con nombres
    Z = Z.reshape(xx.shape)

    # Plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm') 

    # Plotear los puntos de datos
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
    # PARTE C: GRÁFICA 2 - Superficie de Predicción (Random Forest Regresión - Datos)
    # =========================================================================
    y_reg_original = df_sample['Init_Win_bytes_forward'].copy()
    
    # CORRECCIÓN 3: Truncar valores negativos a cero (evita problemas con log1p)
    y_reg_original[y_reg_original < 0] = 0
    
    # Limpieza de 'Y'
    y_reg_original.replace([np.inf, -np.inf, np.nan], 0, inplace=True) 

    # TRANSFORMACIÓN LOGARÍTMICA a la variable objetivo
    y_reg_transformed = np.log1p(y_reg_original)

    # Excluir la variable objetivo y la variable de clase
    X_reg = df_sample.drop(['Init_Win_bytes_forward', target_col_cls], axis=1, errors='ignore')
    
    # Limpieza de 'X'
    X_reg.replace([np.inf, -np.inf], 0, inplace=True) 

    # Usamos todas las features para importarancia
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_full.fit(X_reg, y_reg_transformed) # Entrenar con Y transformada
    feature_importances = pd.Series(rf_full.feature_importances_, index=X_reg.columns)
    
    # Seleccionar las 2 features más importantes para el plot 2D
    top_2_features = feature_importances.nlargest(2).index.tolist()
    if len(top_2_features) < 2:
        top_2_features = X_reg.columns[:2].tolist()

    X_reg_top = X_reg[top_2_features]
    
    # Limpieza final de las 2 features seleccionadas (X)
    X_reg_top.replace([np.inf, -np.inf], 0, inplace=True)
    
    # División de datos (usando Y transformada)
    X_train_reg, X_test_reg, y_train_reg_transf, y_test_reg_transf = train_test_split(
        X_reg_top, y_reg_transformed, test_size=0.3, random_state=42
    )

    model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_reg.fit(X_train_reg, y_train_reg_transf) # Entrenar con Y transformada

    # Generar datos para la superficie de predicción (usando predicciones transformadas)
    x_min_r, x_max_r = X_reg_top.iloc[:, 0].min() - 0.5, X_reg_top.iloc[:, 0].max() + 0.5
    y_min_r, y_max_r = X_reg_top.iloc[:, 1].min() - 0.5, X_reg_top.iloc[:, 1].max() + 0.5
    xx_r, yy_r = np.meshgrid(np.linspace(x_min_r, x_max_r, 50), np.linspace(y_min_r, y_max_r, 50))
    
    # Aplicar la misma limpieza a la cuadrícula de predicción
    grid_data = pd.DataFrame(np.c_[xx_r.ravel(), yy_r.ravel()], columns=top_2_features)
    grid_data.replace([np.inf, -np.inf], 0, inplace=True) 

    Z_reg = model_reg.predict(grid_data) # Predicción transformada

    # Preparación de datos para la Gráfica 2 (Superficie de Predicción)
    regression_data_surface = {
        'x_feature': top_2_features[0],
        'y_feature': top_2_features[1],
        'x_line': xx_r.flatten().tolist(), 
        'y_line': yy_r.flatten().tolist(),           
        'z_line': Z_reg.flatten().tolist(), # Valores predichos (transformados)
        'x_data': X_reg_top.iloc[:, 0].tolist(), 
        'y_data': X_reg_top.iloc[:, 1].tolist(), 
        'y_data_class': y_reg_transformed.tolist() # Valores reales (transformados)
    }
    
    # =========================================================================
    # PARTE D: GRÁFICA 3 - Reales vs Predichos (Base64) - LOG-TRANSFORMADO
    # =========================================================================
    y_pred_reg_transf = model_reg.predict(X_test_reg)
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    # USAR VALORES TRANSFORMADOS para que los puntos se dispersen
    ax3.scatter(y_test_reg_transf, y_pred_reg_transf, alpha=0.6, color='#5B21B6') 
    
    # Línea de predicción ideal (transformada)
    min_val = min(y_test_reg_transf.min(), y_pred_reg_transf.min())
    max_val = max(y_test_reg_transf.max(), y_pred_reg_transf.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # ETIQUETAS MODIFICADAS
    ax3.set_xlabel("Valores Reales (log transformados)", fontsize=12)
    ax3.set_ylabel("Valores Predichos (log transformados)", fontsize=12)
    ax3.set_title("Gráfica 3: Valores reales vs. Predicciones (Log Transformados)", fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    grafica3_b64 = generar_grafica_base64(fig3)

    # 8. Preparación de Salida Final (Tu código original)
    df_sample_head = df_safe.head(10).to_dict('records') 

    return {
        'accuracy': f1_rounded, 
        'dataframe': df_sample_head,
        'grafica1_b64': grafica1_b64, 
        'grafica3_b64': grafica3_b64, 
        'regressionData': regression_data_surface
    }