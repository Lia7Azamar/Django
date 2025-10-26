import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
# Asegúrate de que tu archivo CSV esté en el mismo directorio.
CSV_PATH = 'TotalFeatures-ISCXFlowMeter.csv' 
TARGET_COL_CLS = 'Class' 
RANDOM_SEED = 42
FEATURES_CLS_ALL = ['duration', 'total_fpackets', 'total_bpktl', 'min_fpktl', 
                    'mean_fiat', 'flowPktsPerSecond', 'min_active', 'mean_active', 
                    'Init_Win_bytes_forward']
REQUIRED_SVM_FEATURES = ['min_flowpktl', 'flow_fin']

# --- 1. CARGA DEL DATASET COMPLETO ---
print(f"Cargando dataset completo desde: {CSV_PATH}...")
try:
    df_full = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR FATAL: Archivo no encontrado en la ruta: {CSV_PATH}")
    exit()

# Preprocesamiento inicial (replicando la lógica del original)
df_full.columns = df_full.columns.str.strip()
df_full.columns = [col.replace("calss", "Class") for col in df_full.columns]
for col in df_full.columns:
    if col != TARGET_COL_CLS:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce') 
df_full.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
print(f"Dataset de {len(df_full)} filas cargado y preprocesado.")

# --------------------------------------------------------------------------
# PARTE A: ENTRENAMIENTO CLASIFICADOR (model_f1.joblib)
# --------------------------------------------------------------------------

class_counts = df_full[TARGET_COL_CLS].value_counts()
top_2_classes = class_counts.index[:2].tolist()
df_filtered_cls_f1 = df_full[df_full[TARGET_COL_CLS].isin(top_2_classes)].copy()
class_map = {top_2_classes[0]: 0, top_2_classes[1]: 1} 
df_filtered_cls_f1['target_binary'] = df_filtered_cls_f1[TARGET_COL_CLS].map(class_map)

X_f1 = df_filtered_cls_f1[FEATURES_CLS_ALL]
y_f1 = df_filtered_cls_f1['target_binary']

# Usar el 60% para entrenamiento (test_size=0.4)
X_train_f1, _, y_train_f1, _ = train_test_split(X_f1, y_f1, test_size=0.4, random_state=RANDOM_SEED)

# 1. Entrenar y guardar StandardScaler (CRÍTICO)
scaler_f1 = StandardScaler()
X_train_f1_scaled = scaler_f1.fit_transform(X_train_f1)
joblib.dump(scaler_f1, 'scaler_f1.joblib', compress=9)
print("✅ 'scaler_f1.joblib' (StandardScaler) generado.")

# 2. Entrenar y guardar el modelo F1
print("\n--- INICIANDO ENTRENAMIENTO CLASIFICADOR (model_f1.joblib) ---")
# Usar los parámetros ORIGINALES (para mantener el mismo F1-Score)
model_f1 = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1)
model_f1.fit(X_train_f1_scaled, y_train_f1)
joblib.dump(model_f1, 'model_f1.joblib', compress=9)
print("✅ Nuevo modelo 'model_f1.joblib' generado con compresión MÁXIMA.")


# --------------------------------------------------------------------------
# PARTE B: ENTRENAMIENTO REGRESOR (model_reg.joblib)
# --------------------------------------------------------------------------

# Usar la muestra del 10% para replicar la lógica original de entrenamiento (para RF Regressor)
df_sample = df_full.sample(frac=0.1, random_state=RANDOM_SEED)

X_reg = df_sample.drop(['Init_Win_bytes_forward', TARGET_COL_CLS], axis=1, errors='ignore')
y_reg_original = df_sample['Init_Win_bytes_forward'].copy()
y_reg_original[y_reg_original < 0] = 0
y_reg_transformed = np.log1p(y_reg_original)

# Usar el 70% para entrenamiento (test_size=0.3)
X_train_reg, _, y_train_reg_transf, _ = train_test_split(
    X_reg, y_reg_transformed, test_size=0.3, random_state=RANDOM_SEED
)

print("\n--- INICIANDO ENTRENAMIENTO REGRESOR (model_reg.joblib) ---")
# Usar los parámetros ORIGINALES (para mantener la misma superficie de predicción)
model_reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
model_reg.fit(X_train_reg, y_train_reg_transf)
joblib.dump(model_reg, 'model_reg.joblib', compress=9)
print("✅ Nuevo modelo 'model_reg.joblib' generado con compresión MÁXIMA.")


# --------------------------------------------------------------------------
# PARTE C: ENTRENAMIENTO SVM y LABEL ENCODER (para Gráfica 1)
# --------------------------------------------------------------------------

top_3_classes = class_counts.index[:3].tolist()
df_filtered_svm = df_sample[df_sample[TARGET_COL_CLS].isin(top_3_classes)].copy()
X_svm = df_filtered_svm[REQUIRED_SVM_FEATURES].copy()
y_svm = df_filtered_svm[TARGET_COL_CLS]

# 1. Entrenar y guardar el LabelEncoder (CRÍTICO)
le_clas = LabelEncoder()
y_svm_encoded = le_clas.fit_transform(y_svm)
joblib.dump(le_clas, 'le_clas.joblib', compress=9)
print("✅ 'le_clas.joblib' (LabelEncoder) generado.")

# 2. Entrenar y guardar el modelo SVM
X_svm_transf = X_svm.copy()
X_svm_transf['min_flowpktl'] = np.log1p(X_svm_transf['min_flowpktl'])
X_svm_transf['flow_fin'] = np.log1p(X_svm_transf['flow_fin'])

print("\n--- INICIANDO ENTRENAMIENTO SVM (model_svm.joblib) ---")
# Usar los parámetros ORIGINALES
model_svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=RANDOM_SEED)
model_svm.fit(X_svm_transf, y_svm_encoded)
joblib.dump(model_svm, 'model_svm.joblib', compress=9)
print("✅ Nuevo modelo 'model_svm.joblib' generado con compresión MÁXIMA.")

print("\n---------------------------------------------------------")
print("PROCESO FINALIZADO.")
print("Los 5 archivos .joblib deben ser SUBIDOS a tu repositorio de Hugging Face.")