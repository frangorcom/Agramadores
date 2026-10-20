# 1_rf_bruto.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("--- FASE 1: ENTRENAMIENTO RANDOM FOREST BRUTO ---")

# 1. Cargar datos SIN escalar
df = pd.read_csv('Tabla_Maestra_GDD_FusionPCA.csv')

# Validar y extraer el mapeo de conjuntos desde la tabla MinMax si es necesario
if 'CONJUNTO' not in df.columns:
    print("Columna 'CONJUNTO' faltante. Importando partición desde la tabla MinMax...")
    df_ref = pd.read_csv('Tabla_Maestra_Mensual_Escalada_PCA_MinMax.csv')
    df = df.merge(df_ref[['ID_POLIGONO', 'CONJUNTO']], on='ID_POLIGONO', how='left')

df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()

# 2. Configurar variables y pesos
# Lista base de columnas a excluir
cols_excluir = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA']

# Eliminar columnas base si existen (manejando posibles sufijos del merge)
cols_presentes = [col for col in df_train.columns if col in cols_excluir or 'CONJUNTO' in col]
X_train_temp = df_train.drop(columns=cols_presentes)

# FILTRO DE SEGURIDAD ESTRICTO: Conservar exclusivamente columnas numéricas (int, float)
X_train = X_train_temp.select_dtypes(include=[np.number])

y_train = df_train['RENDIMIENTO_T_HA']

# Penalización para corregir sobreestimación en rendimientos bajos
pesos_train = np.where(y_train < 3.0, 2.5, 1.0)

print(f"Dimensiones de entrenamiento predictivo: {X_train.shape[1]} variables numéricas.")

# 3. Optimización con GridSearchCV
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'n_estimators': [100, 250],
    'max_depth': [3, 5, 7],
    'max_features': ['sqrt', 'log2'] # Crítico para dimensionalidad extrema
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_strategy, 
                    scoring='neg_mean_squared_error', n_jobs=-1)

print("Buscando hiperparámetros...")
grid.fit(X_train, y_train, sample_weight=pesos_train)
best_model = grid.best_estimator_

# 4. Evaluación Out-Of-Fold
y_pred_cv = np.zeros(len(y_train))
for train_idx, val_idx in cv_strategy.split(X_train):
    best_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], sample_weight=pesos_train[train_idx])
    y_pred_cv[val_idx] = best_model.predict(X_train.iloc[val_idx])

# Reentrenamiento final
best_model.fit(X_train, y_train, sample_weight=pesos_train)

rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
mae = mean_absolute_error(y_train, y_pred_cv)
r2 = r2_score(y_train, y_pred_cv)

print(f"\n[RF Bruto] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# 5. Guardar modelo y graficar importancia
joblib.dump(best_model, 'modelo_rf_bruto.pkl')

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]
top_features = [X_train.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.barh(range(15), importances[indices][::-1], color='purple')
plt.yticks(range(15), top_features[::-1])
plt.xlabel('Importancia Relativa (Gini)')
plt.title('Top 15 Variables - Random Forest Bruto')
plt.tight_layout()
plt.savefig('importancia_rf_bruto.png')
print("Pipeline bruto finalizado. Modelo y gráfica guardados.")