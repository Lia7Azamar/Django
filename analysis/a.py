# reentrenar_modelos.py (Ejecutar en tu PC LOCAL)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC # Importar el modelo SVM
import joblib
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
CSV_PATH = 'TotalFeatures-ISCXFlowMeter.csv' 
FEATURES_CLS_ALL = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 'Init_Win_bytes_forward']
REQUIRED_SVM_FEATURES = ['min_flowpktl', 'flow_fin'] # Necesario para la Gráfica 1
TARGET_COL_CLS = 'Class' 
RANDOM_SEED = 42
N_ROWS_TO_TRAIN = 50000 # Usar un número grande para el entrenamiento

# --- 1. CARGA DEL DATASET COMPLETO ---
print(f"Cargando dataset completo desde: {CSV_PATH}...")
try:
    df_full = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR FATAL: Archivo no encontrado en la ruta: {CSV_PATH}")
    exit()

# Preprocesamiento general:
df_full.columns = df_full.columns.str.strip()
df_full.columns = [col.replace("calss", "Class") for col in df_full.columns]
df_full.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
print("Carga y preprocesamiento general de datos listos.")


# --------------------------------------------------------------------------
# PARTE A: REENTRENAMIENTO DEL MODELO F1 (CLASIFICADOR)
# --------------------------------------------------------------------------

# Preparación de datos (filtrando a las 2 clases principales)
class_counts = df_full[TARGET_COL_CLS].value_counts()
top_2_classes = class_counts.index[:2].tolist()
df_filtered_cls_f1 = df_full[df_full[TARGET_COL_CLS].isin(top_2_classes)].copy()

# Mapeo a binario (0 y 1)
class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[TARGET_COL_CLS].map(class_map)

X_f1 = df_filtered_cls_f1[FEATURES_CLS_ALL]
y_f1 = df_filtered_cls_f1['target_binary']

# Usar el set de entrenamiento (3/5 partes si separamos 40%)
X_train_f1, _, y_train_f1, _ = train_test_split(X_f1, y_f1, test_size=0.4, random_state=RANDOM_SEED)

# Escalar los datos de entrenamiento
scaler_f1 = StandardScaler()
X_train_f1_scaled = scaler_f1.fit_transform(X_train_f1)

print("\n--- INICIANDO REENTRENAMIENTO CLASIFICADOR (model_f1.joblib) ---")

# ** PARÁMETROS DE REDUCCIÓN DE TAMAÑO DEL MODELO **
model_f1_small = RandomForestClassifier(
    n_estimators=30,      # Reducido
    max_depth=12,         # Limitada
    min_samples_leaf=5,   # Aumentada
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model_f1_small.fit(X_train_f1_scaled, y_train_f1)

# Guardar el nuevo modelo con MÁXIMA COMPRESIÓN (compress=9)
joblib.dump(model_f1_small, 'model_f1.joblib', compress=9)
print("✅ Nuevo modelo 'model_f1.joblib' generado con compresión MÁXIMA.")


# --------------------------------------------------------------------------
# PARTE B: REENTRENAMIENTO DEL MODELO REGRESOR (model_reg.joblib)
# --------------------------------------------------------------------------

# Preparación de datos (Regresión: target Init_Win_bytes_forward)
X_reg = df_full.drop(['Init_Win_bytes_forward', TARGET_COL_CLS], axis=1, errors='ignore')
y_reg_original = df_full['Init_Win_bytes_forward'].copy()
y_reg_original[y_reg_original < 0] = 0
y_reg_transformed = np.log1p(y_reg_original) # Aplicar log transformado

# Usar el set de entrenamiento (70% de los datos)
X_train_reg, _, y_train_reg_transf, _ = train_test_split(
    X_reg, y_reg_transformed, test_size=0.3, random_state=RANDOM_SEED
)

print("\n--- INICIANDO REENTRENAMIENTO REGRESOR (model_reg.joblib) ---")

# ** PARÁMETROS DE REDUCCIÓN DE TAMAÑO DEL MODELO **
model_reg_small = RandomForestRegressor(
    n_estimators=30,      # Reducido
    max_depth=12,         # Limitada
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model_reg_small.fit(X_train_reg, y_train_reg_transf)

# Guardar el nuevo modelo con MÁXIMA COMPRESIÓN (compress=9)
joblib.dump(model_reg_small, 'model_reg.joblib', compress=9)
print("✅ Nuevo modelo 'model_reg.joblib' generado con compresión MÁXIMA.")


# --------------------------------------------------------------------------
# PARTE C: REENTRENAMIENTO DEL MODELO SVM y LABEL ENCODER (para Gráfica 1)
# --------------------------------------------------------------------------

# Preparación de datos (filtrando a las 3 clases principales para SVM)
top_3_classes = class_counts.index[:3].tolist()
df_filtered_svm = df_full[df_full[TARGET_COL_CLS].isin(top_3_classes)].copy()
X_svm = df_filtered_svm[REQUIRED_SVM_FEATURES].copy()
y_svm = df_filtered_svm[TARGET_COL_CLS]

# 1. Entrenar y guardar el LabelEncoder (CRÍTICO para el análisis)
le_clas = LabelEncoder()
y_svm_encoded = le_clas.fit_transform(y_svm)
joblib.dump(le_clas, 'le_clas.joblib', compress=9)
print("✅ 'le_clas.joblib' (LabelEncoder) generado.")

# 2. Preprocesamiento de datos para SVM (Log Transformación, como en el análisis)
X_svm_transf = X_svm.copy()
X_svm_transf['min_flowpktl'] = np.log1p(X_svm_transf['min_flowpktl'])
X_svm_transf['flow_fin'] = np.log1p(X_svm_transf['flow_fin'])

# ** PARÁMETROS DE REDUCCIÓN DE TAMAÑO DEL MODELO SVM **
# Reducir el tamaño de entrenamiento para que el modelo sea más pequeño
X_svm_train, _, y_svm_train, _ = train_test_split(
    X_svm_transf, y_svm_encoded, test_size=0.95, random_state=RANDOM_SEED
) # Usamos solo el 5% para que el modelo sea PEQUEÑO

print(f"\n--- INICIANDO REENTRENAMIENTO SVM (model_svm.joblib) con {len(X_svm_train)} muestras ---")

model_svm_small = SVC(
    kernel='rbf',       
    C=1.0,              # Regularización baja
    gamma='scale',
    random_state=RANDOM_SEED,
)

model_svm_small.fit(X_svm_train, y_svm_train)

# Guardar el nuevo modelo con MÁXIMA COMPRESIÓN (compress=9)
joblib.dump(model_svm_small, 'model_svm.joblib', compress=9)
print("✅ Nuevo modelo 'model_svm.joblib' generado con compresión MÁXIMA.")

print("\nProceso de generación de modelos SMALL y COMPRIMIDOS completado.")
print("\nPASOS A SEGUIR:")
print("1. Sube los 4 archivos .joblib generados a tu repositorio de Hugging Face.")
print("2. Despliega la última versión de analysis_script.py en Railway.")