import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

print("--- FASE 3: ENTRENAMIENTO DE MODELOS OPTIMIZADOS (FILTRO LASSO) ---")

# 1. Cargar datos y variables seleccionadas
df = pd.read_csv('Tabla_Maestra_Mensual_Escalada_MinMax.csv')
variables_optimas = joblib.load('variables_seleccionadas_lasso.pkl')

df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()

# 2. Filtrar X utilizando SOLO las variables seleccionadas por Lasso
X_train_opt = df_train[variables_optimas]
y_train = df_train['RENDIMIENTO_T_HA']

X_pred_opt = df_pred[variables_optimas]
id_pred = df_pred['ID_POLIGONO']

print(f"Entrenando con {X_train_opt.shape[1]} variables optimizadas...")

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Función de entrenamiento (reutilizada)
def train_and_evaluate_opt(pipeline, param_grid, model_name):
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, 
                        scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_opt, y_train)
    best_model = grid.best_estimator_
    
    y_pred_cv = np.zeros(len(y_train))
    for train_idx, val_idx in cv_strategy.split(X_train_opt):
        best_model.fit(X_train_opt.iloc[train_idx], y_train.iloc[train_idx])
        y_pred_cv[val_idx] = best_model.predict(X_train_opt.iloc[val_idx])
    
    best_model.fit(X_train_opt, y_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    r2 = r2_score(y_train, y_pred_cv)
    print(f"[{model_name} Optimizado] RMSE OOF: {rmse:.4f} | R2: {r2:.4f}")
    return best_model, y_pred_cv

# 4. Pipelines (El SVR ahora puede explorar parámetros más agresivos al tener menos ruido)
# Definición de Pipelines
pipe_lasso = Pipeline([('imputer', SimpleImputer(strategy='median')), ('lasso', Lasso(random_state=42, max_iter=10000))])
pipe_poly = Pipeline([('imputer', SimpleImputer(strategy='median')), ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)), ('ridge', Ridge(random_state=42, max_iter=10000))])
pipe_svr = Pipeline([('imputer', SimpleImputer(strategy='median')), ('svr', SVR(kernel='rbf'))])

# 1. Recuperar las predicciones OOF (reemplazar el '_' por variables explícitas)
model_lasso_opt, cv_lasso_opt = train_and_evaluate_opt(pipe_lasso, {'lasso__alpha': [0.001, 0.01, 0.1, 1]}, "Lasso")
model_poly_opt, cv_poly_opt = train_and_evaluate_opt(pipe_poly, {'ridge__alpha': [0.01, 0.1, 1, 10.0]}, "Poly+Ridge")
model_svr_opt, cv_svr_opt = train_and_evaluate_opt(pipe_svr, {'svr__C': [0.1, 1, 10, 50, 100], 'svr__gamma': ['scale', 'auto', 0.01, 0.05], 'svr__epsilon': [0.01, 0.05, 0.1]}, "SVR")

# 5. Visualización comparativa de los 3 modelos
# Crear un lienzo con 1 fila y 3 columnas (compartiendo el eje Y para facilitar la comparación)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Gráfica 1: Lasso
axes[0].scatter(y_train, cv_lasso_opt, alpha=0.7, color='blue', edgecolors='k')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Rendimiento Real (t/ha)')
axes[0].set_ylabel('Rendimiento Predicho (t/ha)')
axes[0].set_title('Lasso vs Realidad')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Gráfica 2: Poly + Ridge
axes[1].scatter(y_train, cv_poly_opt, alpha=0.7, color='orange', edgecolors='k')
axes[1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1].set_xlabel('Rendimiento Real (t/ha)')
axes[1].set_title('Poly+Ridge vs Realidad')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Gráfica 3: SVR
axes[2].scatter(y_train, cv_svr_opt, alpha=0.7, color='green', edgecolors='k')
axes[2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[2].set_xlabel('Rendimiento Real (t/ha)')
axes[2].set_title('SVR vs Realidad')
axes[2].grid(True, linestyle='--', alpha=0.6)

# Ajustar el espaciado para que no se superpongan los textos y mostrar
plt.tight_layout()
plt.show()
# 6. Guardar predicciones finales de la competencia
pred_final_svr_opt = model_svr_opt.predict(X_pred_opt)
df_pred_final_opt = pd.DataFrame({'ID_POLIGONO': id_pred, 'RENDIMIENTO_PREDICHO_T_HA': pred_final_svr_opt})
df_pred_final_opt.to_csv('Predicciones_Finales_SVR_Opt.csv', index=False)
joblib.dump(model_svr_opt, 'modelo_final_svr_opt.pkl')
print("\n¡Pipeline finalizado! Predicciones guardadas en 'Predicciones_Finales_SVR_Opt.csv'")