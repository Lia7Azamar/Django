import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC # Importar el modelo SVC para clasificación

# --- 1. Lectura y preparación del dataset ---
df = pd.read_csv("/home/lia/Documentos/Simulacion/datasets/datasets/TotalFeatures-ISCXFlowMeter.csv")
df.columns = [col.replace(" ", "_") for col in df.columns]
df.dropna(inplace=True)
df_sample = df.sample(frac=0.1, random_state=42)

# --- 2. Preparación y generación de la 1ª Gráfica (SVM para suavizar la frontera) ---
X_clas = df_sample[['min_flowpktl', 'flow_fin']]
y_clas = df_sample['calss']
le = LabelEncoder()
y_clas_encoded = le.fit_transform(y_clas)
class_names = le.classes_

# CAMBIO CLAVE: Usar SVC con un kernel 'rbf' para fronteras de decisión redondeadas
# El parámetro 'C' y 'gamma' se ajustan para un mejor rendimiento
model_clas = SVC(kernel='rbf', C=10, gamma=0.1)
model_clas.fit(X_clas, y_clas_encoded)

plt.figure(figsize=(12, 10))
x_min, x_max = X_clas.iloc[:, 0].min() - 0.5, X_clas.iloc[:, 0].max() + 0.5
y_min, y_max = X_clas.iloc[:, 1].min() - 0.5, X_clas.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model_clas.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5, cmap='jet')

for i, class_name in enumerate(class_names):
    plt.scatter(X_clas.iloc[y_clas_encoded == i, 0], X_clas.iloc[y_clas_encoded == i, 1],
                edgecolors='white', s=60, label=f'Clase: {class_name}')

plt.title('Gráfica 1: Separabilidad de datos con SVM (Frontera suavizada)', fontsize=16, fontweight='bold')
plt.xlabel('Característica: min_flowpktl', fontsize=12)
plt.ylabel('Característica: flow_fin', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.show()

# --- 3. Las gráficas de regresión (sin cambios) ---
y_reg = df_sample['Init_Win_bytes_forward']
X_reg = df_sample.drop(['Init_Win_bytes_forward', 'calss'], axis=1)

if len(X_reg) > 2000:
    X_reg_sub = X_reg.sample(n=2000, random_state=42)
    y_reg_sub = y_reg.loc[X_reg_sub.index]
else:
    X_reg_sub = X_reg
    y_reg_sub = y_reg

rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(X_reg_sub, y_reg_sub)

feature_importances = pd.Series(rf_full.feature_importances_, index=X_reg.columns)
top_2_features = feature_importances.nlargest(2).index.tolist()

X_reg_top = X_reg[top_2_features]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_top, y_reg, test_size=0.3, random_state=42
)

model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_train_reg, y_train_reg)

plt.figure(figsize=(10, 8))
x_min, x_max = X_reg_top.iloc[:, 0].min() - 0.5, X_reg_top.iloc[:, 0].max() + 0.5
y_min, y_max = X_reg_top.iloc[:, 1].min() - 0.5, X_reg_top.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=20, cmap='plasma', alpha=0.7)
plt.colorbar(label='Valor Predicho de la variable objetivo')
plt.scatter(X_reg_top.iloc[:, 0], X_reg_top.iloc[:, 1], c=y_reg, edgecolors='k', cmap='plasma', label='Valores Reales')
plt.title('Gráfica 2: Superficie de predicción (Regresión)')
plt.xlabel(f'Característica: {top_2_features}')
plt.ylabel(f'Característica: {top_2_features}')
plt.legend()
plt.show()

y_pred_reg = model_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print("\n--- Métricas del modelo de regresión ---")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")
print("-" * 50)
plt.figure(figsize=(10, 8))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, color='b')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Gráfica 3: Valores reales vs. Predicciones (Línea de predicción ideal)")
plt.grid(True)
plt.show()
