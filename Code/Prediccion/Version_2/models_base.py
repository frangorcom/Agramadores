import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

print("--- FASE 1: ENTRENAMIENTO DE MODELOS BASE (CON TODAS LAS VARIABLES) ---")

# 1. Carga de datos escalados
df = pd.read_csv('Tabla_Maestra_Mensual_Escalada_MinMax.csv')

# 2. Separar conjuntos
df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()

target_col = 'RENDIMIENTO_T_HA'
cols_to_drop = ['ID_POLIGONO', 'CONJUNTO', target_col, 'AREA_HA'] # Aseguramos no usar AREA_HA

X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
y_train = df_train[target_col]
X_pred = df_pred.drop(columns=cols_to_drop, errors='ignore')

# Forzar columnas numéricas por seguridad
cols_numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
X_train = X_train[cols_numericas]
X_pred = X_pred[cols_numericas]

# 3. Configuración de Validación Cruzada
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Definición de Pipelines y Grids
pipeline_lasso = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('lasso', Lasso(random_state=42, max_iter=10000))
])
param_grid_lasso = {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10.0]}


pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('svr', SVR(kernel='rbf'))
])
param_grid_svr = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 'auto', 0.1, 1],
    'svr__epsilon': [0.01, 0.1, 0.5]
}

# 5. Función de Entrenamiento
def train_and_evaluate(pipeline, param_grid, model_name):
    print(f"\nEntrenando: {model_name}...")
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, 
                        scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_pred_cv = np.zeros(len(y_train))
    for train_idx, val_idx in cv_strategy.split(X_train):
        best_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred_cv[val_idx] = best_model.predict(X_train.iloc[val_idx])
    
    best_model.fit(X_train, y_train) # Reentrenar con todo
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    r2 = r2_score(y_train, y_pred_cv)
    print(f"[{model_name}] RMSE OOF: {rmse:.4f} | R2: {r2:.4f}")
    return best_model

# 6. Ejecución y Guardado del modelo Lasso (Necesario para el Script 2)
model_lasso = train_and_evaluate(pipeline_lasso, param_grid_lasso, "Lasso")
# model_poly = train_and_evaluate(pipeline_poly, param_grid_poly, "Poly+Ridge")
model_svr = train_and_evaluate(pipeline_svr, param_grid_svr, "SVR Base")

joblib.dump(model_lasso, 'modelo_lasso_base.pkl')
print("\nModelos base entrenados. 'modelo_lasso_base.pkl' guardado para extracción de variables.")