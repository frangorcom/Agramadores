import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer  # Importación añadida para manejar NaNs
import joblib

# Carga de datos
df = pd.read_csv('Tabla_Maestra_Estandarizada.csv')

# Separar el conjunto de entrenamiento y el de predicción final ciega
df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()

# Definir variables objetivo e identificadores
target_col = 'RENDIMIENTO_T_HA'
cols_to_drop = ['ID_POLIGONO', 'CONJUNTO', target_col]

# Separar features (X) y target (y)
X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
y_train = df_train[target_col]

X_pred = df_pred.drop(columns=cols_to_drop, errors='ignore')
id_pred = df_pred['ID_POLIGONO']

# ==========================================
# CORRECCIÓN 1: Forzar solo columnas numéricas 
# ==========================================
cols_numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train = X_train[cols_numericas]
X_pred = X_pred[cols_numericas]

# ==========================================
# CORRECCIÓN 2: Nombre de la variable corregido a cv_strategy
# ==========================================
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# CORRECCIÓN 3: Añadido SimpleImputer a los Pipelines
# ==========================================
# Modelo 1: Regresión Lineal Regularizada (Lasso)
pipeline_lasso = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('lasso', Lasso(random_state=42, max_iter=10000))
])
param_grid_lasso = {
    'lasso__alpha': [0.001, 0.01, 0.1, 1, 10.0]
}

# Modelo 2: Regresión Polinomial (Grado 2)
pipeline_poly = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('ridge', Ridge(random_state=42, max_iter=10000))
])
param_grid_poly = {
    'ridge__alpha': [0.01, 0.1, 1, 10.0, 100.0]
}

# Modelo 3: Support Vector Regression (SVR)
pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('svr', SVR(kernel='rbf'))
])
param_grid_svr = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto', 0.1, 1],
    'svr__epsilon': [0.01, 0.1, 0.5]
}

# ==========================================
# ENTRENAMIENTO Y OPTIMIZACIÓN
# ==========================================
def train_and_evaluate_model(pipeline, param_grid, model_name):
    print(f"--- Entrenando y evaluando modelo: {model_name} ---")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy, # Ahora sí coincide con la variable definida arriba
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    
    # Evaluación con Cross-Validation sobre el mejor modelo
    # CORRECCIÓN 4: Usar np.zeros para evitar problemas de índices con Pandas Series
    y_pred_cv = np.zeros(len(y_train)) 
    
    for train_idx, val_idx in cv_strategy.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t = y_train.iloc[train_idx]
        
        best_model.fit(X_t, y_t)
        y_pred_cv[val_idx] = best_model.predict(X_v)
    
    # Entrenar modelo final con todo el dataset de entrenamiento
    best_model.fit(X_train, y_train)
    
    # Calcular métricas OOF (Out-Of-Fold)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    mae = mean_absolute_error(y_train, y_pred_cv)
    r2 = r2_score(y_train, y_pred_cv)
    
    print(f"Métricas Out-Of-Fold (Más realistas que train directo):")
    print(f"RMSE: {rmse:.4f} t/ha | MAE: {mae:.4f} t/ha | R2: {r2:.4f}\n")
    
    return best_model, y_pred_cv

# Ejecución de entrenamiento y evaluación para cada modelo
model_lasso, pred_cv_lasso = train_and_evaluate_model(pipeline_lasso, param_grid_lasso, "Regresión Lineal Regularizada (Lasso)")
model_poly, pred_cv_poly = train_and_evaluate_model(pipeline_poly, param_grid_poly, "Regresión Polinomial + Ridge")
model_svr, pred_cv_svr = train_and_evaluate_model(pipeline_svr, param_grid_svr, "SVR (Kernel RBF)")

# ==========================================
# VISUALIZACIÓN DE RESULTADOS
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(y_train, pred_cv_svr, alpha=0.7, color='blue', edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Rendimiento Real (t/ha)')
plt.ylabel('Rendimiento Predicho (SVR) (t/ha)')
plt.title('Predicción vs Realidad (Validación Cruzada)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==========================================
# GENERACIÓN DE PREDICCIONES FINALES PARA EL CONJUNTO CIEGO
# ==========================================
pred_final_svr = model_svr.predict(X_pred)
df_pred_final = pd.DataFrame({
    'ID_POLIGONO': id_pred, 
    'RENDIMIENTO_PREDICHO_T_HA': pred_final_svr
})

# Guardar las predicciones finales en un CSV
df_pred_final.to_csv('Predicciones_Finales_SVR.csv', index=False)
print("Predicciones finales guardadas en 'Predicciones_Finales_SVR.csv'")

# Guardar el modelo final entrenado para futuras predicciones
joblib.dump(model_svr, 'modelo_final_svr.pkl')
print("Modelo final SVR guardado como 'modelo_final_svr.pkl' para futuras predicciones.")

# ==========================================
# 1. GUARDAR EL MODELO LASSO EN DISCO
# ==========================================
joblib.dump(model_lasso, 'modelo_lasso_cebada.pkl')
print("Modelo Lasso guardado como 'modelo_lasso_cebada.pkl'")

# ==========================================
# 2. ANÁLISIS Y VISUALIZACIÓN DE VARIABLES (LASSO)
# ==========================================
print("\n--- Extrayendo importancia de variables (Lasso) ---")
# Extraemos los coeficientes del modelo que ya está en la memoria
coeficientes_lasso = model_lasso.named_steps['lasso'].coef_
nombres_variables = X_train.columns

# Crear DataFrame para organizar los resultados
df_coef = pd.DataFrame({
    'Variable': nombres_variables,
    'Coeficiente': coeficientes_lasso,
    'Importancia_Abs': np.abs(coeficientes_lasso)
})

# Separar variables descartadas y retenidas
variables_descartadas = df_coef[df_coef['Coeficiente'] == 0]
df_activas = df_coef[df_coef['Coeficiente'] != 0].sort_values(by='Importancia_Abs', ascending=True)

print(f"Total de variables originales: {len(df_coef)}")
print(f"Variables anuladas (ruido): {len(variables_descartadas)}")
print(f"Variables retenidas (activas): {len(df_activas)}\n")

# Top 5 variables más importantes
print("Top 5 variables con mayor impacto en el modelo:")
print(df_activas.sort_values(by='Importancia_Abs', ascending=False).head(5)[['Variable', 'Coeficiente']])

# Graficar
plt.figure(figsize=(10, 8))
colores = ['#2ca02c' if c > 0 else '#d62728' for c in df_activas['Coeficiente']]

plt.barh(df_activas['Variable'], df_activas['Coeficiente'], color=colores, edgecolor='black', alpha=0.8)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.xlabel('Impacto en Rendimiento (t/ha) por cada Desviación Estándar', fontsize=11)
plt.ylabel('Variables Predictoras', fontsize=11)
plt.title('Importancia de Variables (Modelo Lasso)', fontsize=14, pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()

# ==========================================
# REENTRENAMIENTO SVR CON VARIABLES SELECCIONADAS POR LASSO
# ==========================================

# 1. Recuperar las 19 variables óptimas directamente del modelo Lasso
coeficientes_lasso = model_lasso.named_steps['lasso'].coef_
mascara_activas = coeficientes_lasso != 0
variables_seleccionadas = X_train.columns[mascara_activas].tolist()

print(f"\n--- Preparando reentrenamiento SVR con {len(variables_seleccionadas)} variables ---")

# 2. Reducir los datasets (Eliminamos las 34 variables ruidosas)
X_train_opt = X_train[variables_seleccionadas]
X_pred_opt = X_pred[variables_seleccionadas]

# 3. Configurar el Pipeline del SVR Optimizado
pipeline_svr_opt = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('svr', SVR(kernel='rbf'))
])

# Ampliamos el grid de hiperparámetros. Al tener menos variables, 
# el modelo puede explorar combinaciones más complejas sin sobreajustarse ni tardar demasiado.
param_grid_svr_opt = {
    'svr__C': [0.1, 1, 10, 50, 100],
    'svr__gamma': ['scale', 'auto', 0.01, 0.05, 0.1],
    'svr__epsilon': [0.01, 0.05, 0.1, 0.2]
}

# 4. Entrenamiento con Validación Cruzada
print("Iniciando Grid Search para SVR Optimizado...")
grid_search_opt = GridSearchCV(
    estimator=pipeline_svr_opt,
    param_grid=param_grid_svr_opt,
    cv=cv_strategy,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search_opt.fit(X_train_opt, y_train)
best_svr_opt = grid_search_opt.best_estimator_

print(f"\nMejores hiperparámetros SVR (Reducido): {grid_search_opt.best_params_}")

# Evaluación rigurosa (Out-Of-Fold)
y_pred_cv_opt = np.zeros(len(y_train))

for train_idx, val_idx in cv_strategy.split(X_train_opt):
    X_t, X_v = X_train_opt.iloc[train_idx], X_train_opt.iloc[val_idx]
    y_t = y_train.iloc[train_idx]
    
    best_svr_opt.fit(X_t, y_t)
    y_pred_cv_opt[val_idx] = best_svr_opt.predict(X_v)

# Entrenar con el 100% de los datos reducidos para la entrega final
best_svr_opt.fit(X_train_opt, y_train)

rmse_opt = np.sqrt(mean_squared_error(y_train, y_pred_cv_opt))
mae_opt = mean_absolute_error(y_train, y_pred_cv_opt)
r2_opt = r2_score(y_train, y_pred_cv_opt)

print("\nMétricas Out-Of-Fold (SVR con 19 variables):")
print(f"RMSE: {rmse_opt:.4f} t/ha | MAE: {mae_opt:.4f} t/ha | R2: {r2_opt:.4f}")

# ==========================================
# 5. VISUALIZACIÓN DEL MODELO OPTIMIZADO
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_cv_opt, alpha=0.7, color='green', edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Rendimiento Real (t/ha)')
plt.ylabel('Rendimiento Predicho SVR (t/ha)')
plt.title('SVR Optimizado vs Realidad (Filtro Lasso Aplicado)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================
# 6. GENERACIÓN DE PREDICCIONES CIEGAS (ENTREGABLE)
# ==========================================
pred_final_svr_opt = best_svr_opt.predict(X_pred_opt)

df_pred_final_opt = pd.DataFrame({
    'ID_POLIGONO': id_pred, 
    'RENDIMIENTO_PREDICHO_T_HA': pred_final_svr_opt
})

df_pred_final_opt.to_csv('Predicciones_Finales_SVR_Opt.csv', index=False)
joblib.dump(best_svr_opt, 'modelo_final_svr_opt.pkl')

print("\nPredicciones finales guardadas en 'Predicciones_Finales_SVR_Opt.csv'")