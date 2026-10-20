# 3_rf_lasso_completo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("--- FASE 3: ENTRENAMIENTO RANDOM FOREST CON VARIABLES LASSO ---")

# ==========================================
# 1. CARGA DE DATOS Y CRUCE DE PARTICIONES
# ==========================================
# Cargar datos SIN escalar para mantener interpretabilidad en los árboles
df = pd.read_csv('Tabla_Maestra_GDD_FusionPCA.csv') 

try:
    variables_lasso = joblib.load('variables_seleccionadas_lasso.pkl')
except FileNotFoundError:
    raise Exception("Error: Ejecuta primero el script de filtrado Lasso (fase 2) para generar el archivo .pkl")

# Validar y extraer el mapeo de conjuntos desde la tabla MinMax si la tabla actual no lo tiene
if 'CONJUNTO' not in df.columns:
    print("Columna 'CONJUNTO' faltante. Importando partición desde la tabla MinMax...")
    df_ref = pd.read_csv('Tabla_Maestra_Mensual_Escalada_PCA_MinMax.csv')
    df = df.merge(df_ref[['ID_POLIGONO', 'CONJUNTO']], on='ID_POLIGONO', how='left')

df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()

# ==========================================
# 2. FILTRADO ESTRICTO DE VARIABLES Y PESOS
# ==========================================
# Filtrar solo las columnas seleccionadas por Lasso
X_train_temp = df_train[variables_lasso]
X_pred_temp = df_pred[variables_lasso]

# FILTRO DE SEGURIDAD ESTRICTO: Conservar exclusivamente columnas numéricas para evitar errores de string
X_train = X_train_temp.select_dtypes(include=[np.number])
X_pred = X_pred_temp.select_dtypes(include=[np.number])

y_train = df_train['RENDIMIENTO_T_HA']
id_pred = df_pred['ID_POLIGONO']

# Penalización para corregir sobreestimación en rendimientos bajos (< 3 t/ha)
pesos_train = np.where(y_train < 3.0, 2.5, 1.0)

print(f"Dimensiones de entrenamiento: {X_train.shape[1]} variables numéricas optimizadas.")

# ==========================================
# 3. CONFIGURACIÓN Y OPTIMIZACIÓN (GRIDSEARCH)
# ==========================================
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'n_estimators': [100, 250],
    'max_depth': [5, 7, 10], 
    'max_features': ['sqrt', 1.0] # 1.0 permite evaluar todas las variables seleccionadas
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_strategy, 
                    scoring='neg_mean_squared_error', n_jobs=-1)

print("Buscando hiperparámetros óptimos...")
grid.fit(X_train, y_train, sample_weight=pesos_train)
best_model = grid.best_estimator_
print(f"Mejores parámetros encontrados: {grid.best_params_}")

# ==========================================
# 4. EVALUACIÓN OUT-OF-FOLD (OOF)
# ==========================================
y_pred_cv = np.zeros(len(y_train))

for train_idx, val_idx in cv_strategy.split(X_train):
    # Entrenar fold aplicando los pesos específicos de esa partición
    best_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], sample_weight=pesos_train[train_idx])
    y_pred_cv[val_idx] = best_model.predict(X_train.iloc[val_idx])

# Reentrenamiento final con el 100% de los datos de entrenamiento
best_model.fit(X_train, y_train, sample_weight=pesos_train)

# Cálculo de métricas
rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
mae = mean_absolute_error(y_train, y_pred_cv)
r2 = r2_score(y_train, y_pred_cv)

print(f"\n[RF Lasso] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# ==========================================
# 5. VISUALIZACIÓN 1: REAL VS PREDICHO
# ==========================================
plt.figure(figsize=(9, 7))

# Puntos de dispersión (Real vs Predicho)
plt.scatter(y_train, y_pred_cv, alpha=0.7, color='teal', edgecolors='k', label='Predicciones (OOF)')

# Línea de referencia ideal (y = x)
min_val = min(y_train.min(), y_pred_cv.min())
max_val = max(y_train.max(), y_pred_cv.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Predicción Perfecta (y=x)')

# Línea del umbral crítico definido para la cebada
plt.axvline(x=3.0, color='gray', linestyle=':', lw=2, label='Umbral Bajo Rendimiento (< 3 t/ha)')

# Anotación de métricas
texto_metricas = f"R²: {r2:.3f}\nRMSE: {rmse:.3f} t/ha\nMAE: {mae:.3f} t/ha"
plt.text(0.05, 0.95, texto_metricas, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel('Rendimiento Real de Cebada (t/ha)', fontsize=12)
plt.ylabel('Rendimiento Predicho por RF (t/ha)', fontsize=12)
plt.title('Random Forest (Filtro Lasso): Rendimiento Real vs Predicho', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('dispersion_rf_lasso.png', dpi=300)
plt.show()

# ==========================================
# 6. VISUALIZACIÓN 2: IMPORTANCIA DE VARIABLES
# ==========================================
importances = best_model.feature_importances_
top_n = min(15, len(variables_lasso))
indices = np.argsort(importances)[::-1][:top_n]
top_features = [X_train.columns[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices][::-1], color='teal')
plt.yticks(range(top_n), top_features[::-1])
plt.xlabel('Importancia Relativa (Gini)')
plt.title(f'Top {top_n} Variables - Random Forest Lasso')
plt.tight_layout()
plt.savefig('importancia_rf_lasso.png', dpi=300)
plt.show()

# ==========================================
# 7. EXPORTACIÓN DE PREDICCIONES Y MODELO
# ==========================================
pred_final = best_model.predict(X_pred)
df_pred_final = pd.DataFrame({'ID_POLIGONO': id_pred, 'RENDIMIENTO_PREDICHO_T_HA': pred_final})
df_pred_final.to_csv('Predicciones_Finales_RF_Lasso.csv', index=False)

joblib.dump(best_model, 'modelo_rf_lasso.pkl')
print("\n¡Pipeline Lasso finalizado exitosamente! Modelo y predicciones exportados.")