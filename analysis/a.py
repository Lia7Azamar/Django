# reentrenar_modelos.py (Ejecutar en tu PC LOCAL)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
CSV_PATH = 'TotalFeatures-ISCXFlowMeter.csv' 
FEATURES_CLS_ALL = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 'Init_Win_bytes_forward']
TARGET_COL_CLS = 'Class' 
RANDOM_SEED = 42

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

# El Scaler debe ser ajustado (fit) también para guardar la lógica de transformación
scaler_f1 = StandardScaler()
X_train_f1_scaled = scaler_f1.fit_transform(X_train_f1)
# Aunque tu modelo solo usa predict, entrenarlo con el scaler es buena práctica.

print("\n--- INICIANDO REENTRENAMIENTO CLASIFICADOR (model_f1.joblib) ---")

# ** PARÁMETROS DE REDUCCIÓN DE TAMAÑO **
model_f1_small = RandomForestClassifier(
    n_estimators=30,      # Número de árboles: ¡REDUCIDO!
    max_depth=12,         # Profundidad máxima: ¡LIMITADA!
    min_samples_leaf=5,   # Mínimo de muestras por hoja: AUMENTADA
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model_f1_small.fit(X_train_f1_scaled, y_train_f1) # Entrenar con datos escalados

# Guardar el nuevo modelo
joblib.dump(model_f1_small, 'model_f1.joblib')
print("✅ Nuevo modelo 'model_f1.joblib' generado. ¡Verifica su tamaño!")


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

# ** PARÁMETROS DE REDUCCIÓN DE TAMAÑO **
model_reg_small = RandomForestRegressor(
    n_estimators=30,      # Reducido
    max_depth=12,         # Limitada
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model_reg_small.fit(X_train_reg, y_train_reg_transf)

# Guardar el nuevo modelo
joblib.dump(model_reg_small, 'model_reg.joblib')
print("✅ Nuevo modelo 'model_reg.joblib' generado. ¡Verifica su tamaño!")

print("\nProceso de generación de modelos pequeños completado.")