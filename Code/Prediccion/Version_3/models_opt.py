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

versiones = ['Lineal', 'PCA']
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

for version in versiones:
    print(f"\n{'='*50}\nOPTIMIZANDO Y PREDICIENDO: VERSIÓN {version}\n{'='*50}")
    
    # 1. Cargar datos
    df = pd.read_csv(f'Tabla_Maestra_Mensual_Escalada_{version}_MinMax.csv')
    variables_optimas = joblib.load(f'variables_seleccionadas_lasso_{version}.pkl')
    
    df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
    df_pred = df[df['CONJUNTO'] == 'PREDICCION'].copy()
    
    X_train_opt = df_train[variables_optimas].astype(np.float32)
    y_train = df_train['RENDIMIENTO_T_HA'].astype(np.float32)
    X_pred_opt = df_pred[variables_optimas].astype(np.float32)
    id_pred = df_pred['ID_POLIGONO']
    
    print(f"Entrenando con {X_train_opt.shape[1]} variables optimizadas...")
    
    # 2. Función de entrenamiento (n_jobs=2)
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

    # 3. Pipelines
    pipe_lasso = Pipeline([('imputer', SimpleImputer(strategy='median')), ('lasso', Lasso(random_state=42, max_iter=10000))])
    pipe_poly = Pipeline([('imputer', SimpleImputer(strategy='median')), ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)), ('ridge', Ridge(random_state=42, max_iter=10000))])
    pipe_svr = Pipeline([('imputer', SimpleImputer(strategy='median')), ('svr', SVR(kernel='rbf'))])

    model_lasso_opt, cv_lasso_opt = train_and_evaluate_opt(pipe_lasso, {'lasso__alpha': [0.001, 0.01, 0.1, 1]}, "Lasso")
    model_poly_opt, cv_poly_opt = train_and_evaluate_opt(pipe_poly, {'ridge__alpha': [0.01, 0.1, 1, 10.0]}, "Poly+Ridge")
    model_svr_opt, cv_svr_opt = train_and_evaluate_opt(pipe_svr, {'svr__C': [0.1, 1, 10, 50], 'svr__gamma': ['scale', 'auto', 0.01, 0.05], 'svr__epsilon': [0.01, 0.05]}, "SVR")

    # 4. Visualización Comparativa
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Comparativa de Modelos Optimizados - Versión {version}', fontsize=16)

    axes[0].scatter(y_train, cv_lasso_opt, alpha=0.7, color='blue', edgecolors='k')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Real (t/ha)')
    axes[0].set_ylabel('Predicción (t/ha)')
    axes[0].set_title('Lasso')

    axes[1].scatter(y_train, cv_poly_opt, alpha=0.7, color='orange', edgecolors='k')
    axes[1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[1].set_xlabel('Real (t/ha)')
    axes[1].set_title('Poly+Ridge')

    axes[2].scatter(y_train, cv_svr_opt, alpha=0.7, color='green', edgecolors='k')
    axes[2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[2].set_xlabel('Real (t/ha)')
    axes[2].set_title('SVR')

    plt.tight_layout()
    plt.show()

    # 5. Predicción Final y Guardado
    pred_final_svr_opt = model_svr_opt.predict(X_pred_opt)
    df_pred_final_opt = pd.DataFrame({'ID_POLIGONO': id_pred, 'RENDIMIENTO_PREDICHO_T_HA': pred_final_svr_opt})
    
    nombre_csv = f'Predicciones_Finales_SVR_{version}.csv'
    df_pred_final_opt.to_csv(nombre_csv, index=False)
    joblib.dump(model_svr_opt, f'modelo_final_svr_{version}.pkl')
    print(f"Predicciones guardadas en '{nombre_csv}'")