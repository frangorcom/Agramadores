import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

print("--- FASE 1: ENTRENAMIENTO DE MODELOS BASE ---")

versiones = ['Lineal', 'PCA']
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

for version in versiones:
    print(f"\n{'='*40}\nPROCESANDO VERSIÓN: {version}\n{'='*40}")
    
    # 1. Carga de datos escalados
    df = pd.read_csv(f'Tabla_Maestra_Mensual_Escalada_{version}_MinMax.csv')
    
    df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
    target_col = 'RENDIMIENTO_T_HA'
    cols_to_drop = ['ID_POLIGONO', 'CONJUNTO', target_col, 'AREA_HA']
    
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    y_train = df_train[target_col]
    
    cols_numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train = X_train[cols_numericas].astype(np.float32) # Ahorro RAM
    y_train = y_train.astype(np.float32)

    # 2. Definición de Pipelines
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

    # 2. Función de entrenamiento (n_jobs=2 para cuidar memoria)
    def train_and_evaluate_opt(pipeline, param_grid, model_name):
        print(f"Entrenando: {model_name}...")
        
        # Configurar búsqueda en malla con validación cruzada
        grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv_strategy, 
                            scoring='neg_mean_squared_error', n_jobs=2)
        grid.fit(X_train_opt, y_train)
        best_model = grid.best_estimator_
        
        # Generar predicciones Out-Of-Fold
        y_pred_cv = np.zeros(len(y_train))
        for train_idx, val_idx in cv_strategy.split(X_train_opt):
            best_model.fit(X_train_opt.iloc[train_idx], y_train.iloc[train_idx])
            y_pred_cv[val_idx] = best_model.predict(X_train_opt.iloc[val_idx])
        
        # Reentrenar el mejor modelo con el 100% de los datos de entrenamiento
        best_model.fit(X_train_opt, y_train)
        
        # --- CÁLCULO DE MÉTRICAS ---
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
        r2 = r2_score(y_train, y_pred_cv)
        
        # Imprimir resultados en consola
        print(f"[{model_name}] RMSE OOF: {rmse:.4f} | R2: {r2:.4f}")
        
        return best_model, y_pred_cv

    # 4. Ejecución
    model_lasso = train_and_evaluate(pipeline_lasso, param_grid_lasso, "Lasso")
    model_svr = train_and_evaluate(pipeline_svr, param_grid_svr, "SVR Base")
    
    joblib.dump(model_lasso, f'modelo_lasso_base_{version}.pkl')
    print(f"Modelo guardado: modelo_lasso_base_{version}.pkl")