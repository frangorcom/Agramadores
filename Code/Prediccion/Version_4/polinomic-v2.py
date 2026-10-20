import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

print("--- FASE 3: EVALUACIÓN DE MODELOS POLINÓMICOS (GRADOS 2, 3 Y 4) ---")

# 1. Cargar datos y variables seleccionadas
df = pd.read_csv('Tabla_Maestra_Mensual_Escalada_PCA_MinMax.csv')
variables_optimas = joblib.load('variables_seleccionadas_lasso.pkl')

df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()

X_train_opt = df_train[variables_optimas]
y_train = df_train['RENDIMIENTO_T_HA']

X_pred_opt = df_pred[variables_optimas]
id_pred = df_pred['ID_POLIGONO']

print(f"Entrenando con {X_train_opt.shape[1]} variables optimizadas base...")

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Función de entrenamiento adaptada para RMSE, MAE y R2
def train_and_evaluate_poly(pipeline, param_grid, degree_name):
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, 
                        scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_opt, y_train)
    best_model = grid.best_estimator_
    
    y_pred_cv = np.zeros(len(y_train))
    for train_idx, val_idx in cv_strategy.split(X_train_opt):
        best_model.fit(X_train_opt.iloc[train_idx], y_train.iloc[train_idx])
        y_pred_cv[val_idx] = best_model.predict(X_train_opt.iloc[val_idx])
    
    # Reentrenar en el dataset completo para generar la ecuación final
    best_model.fit(X_train_opt, y_train)
    
    # Cálculo de las 3 métricas solicitadas
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    mae = mean_absolute_error(y_train, y_pred_cv)
    r2 = r2_score(y_train, y_pred_cv)
    
    print(f"[{degree_name}] RMSE OOF: {rmse:.4f} | MAE OOF: {mae:.4f} | R2: {r2:.4f}")
    return best_model, y_pred_cv

# 3. Construcción de los 3 Pipelines Polinómicos
# Usamos Ridge para estabilizar el modelo ante la explosión de dimensionalidad
param_grid_ridge = {'ridge__alpha': [0.01, 0.1, 1, 10.0, 100.0]}

pipe_poly2 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), 
    ('ridge', Ridge(random_state=42, max_iter=20000))
])

pipe_poly3 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('poly', PolynomialFeatures(degree=3, include_bias=False)), 
    ('ridge', Ridge(random_state=42, max_iter=20000))
])

pipe_poly4 = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('poly', PolynomialFeatures(degree=4, include_bias=False)), 
    ('ridge', Ridge(random_state=42, max_iter=20000))
])

print("\nBuscando hiperparámetros y evaluando modelos OOF (Out-Of-Fold)...")
model_poly2, cv_poly2 = train_and_evaluate_poly(pipe_poly2, param_grid_ridge, "Poly Grado 2")
model_poly3, cv_poly3 = train_and_evaluate_poly(pipe_poly3, param_grid_ridge, "Poly Grado 3")
model_poly4, cv_poly4 = train_and_evaluate_poly(pipe_poly4, param_grid_ridge, "Poly Grado 4")

# 4. Visualización comparativa de los 3 modelos
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
modelos_cv = [("Grado 2", cv_poly2, 'blue'), 
              ("Grado 3", cv_poly3, 'orange'), 
              ("Grado 4", cv_poly4, 'green')]

for i, (nombre, cv_preds, color) in enumerate(modelos_cv):
    axes[i].scatter(y_train, cv_preds, alpha=0.7, color=color, edgecolors='k')
    axes[i].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[i].set_xlabel('Rendimiento Real (t/ha)')
    if i == 0: axes[i].set_ylabel('Rendimiento Predicho (t/ha)')
    axes[i].set_title(f'Polinomio {nombre} vs Realidad')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# 5. Función para extraer y exportar las ecuaciones generadas
def print_and_save_equation(model, degree_name, base_features):
    poly = model.named_steps['poly']
    ridge = model.named_steps['ridge']
    
    # Obtener el nombre algebraico de todas las características generadas
    poly_features = poly.get_feature_names_out(base_features)
    coefs = ridge.coef_
    intercept = ridge.intercept_
    
    # Filtrar coeficientes estadísticamente nulos (menores a 1e-5) para limpieza
    terms = [f"{intercept:.5f} (Intercepto)"]
    for coef, feat in zip(coefs, poly_features):
        if abs(coef) > 1e-5: 
            terms.append(f"{coef:+.5f} * ({feat})")
            
    ecuacion_str = " \n".join(terms)
    
    # Mostrar resumen en consola
    print(f"\n--- ECUACIÓN: {degree_name} ---")
    print(f"La expansión creó {len(poly_features)} variables. Términos activos: {len(terms)-1}")
    print(f"y = {terms[0]} \n {terms[1]} \n {terms[2]} \n ... (truncado)")
    
    # Guardar ecuación completa en archivo de texto
    filename = f'Ecuacion_{degree_name.replace(" ", "_")}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Ecuación Analítica para {degree_name}\n")
        f.write(f"Total de términos generados: {len(poly_features)}\n")
        f.write("="*50 + "\n\n")
        f.write(f"y = \n{ecuacion_str}\n")
    print(f"-> Ecuación completa guardada en: {filename}")

base_features = list(X_train_opt.columns)
print_and_save_equation(model_poly2, "Poly Grado 2", base_features)
print_and_save_equation(model_poly3, "Poly Grado 3", base_features)
print_and_save_equation(model_poly4, "Poly Grado 4", base_features)

# 6. Guardar predicciones finales (Se parametriza con Grado 2 como baseline conservador, 
# pero puedes cambiar model_poly2 por el que obtenga mejor R2)
mejor_modelo = model_poly2 
pred_final = mejor_modelo.predict(X_pred_opt)

df_pred_final = pd.DataFrame({'ID_POLIGONO': id_pred, 'RENDIMIENTO_PREDICHO_T_HA': pred_final})
df_pred_final.to_csv('Predicciones_Finales_Poly.csv', index=False)
joblib.dump(mejor_modelo, 'modelo_final_poly.pkl')
print("\n¡Pipeline finalizado! Modelo y predicciones listos para el dashboard.")