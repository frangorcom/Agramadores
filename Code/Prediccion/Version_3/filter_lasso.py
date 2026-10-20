import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

print("--- FASE 2: SELECCIÓN DE VARIABLES CON LASSO ---")
versiones = ['Lineal', 'PCA']

for version in versiones:
    print(f"\nExtrayendo variables para versión: {version}...")
    
    # 1. Cargar datos y modelo
    df_train = pd.read_csv(f'Tabla_Maestra_Mensual_Escalada_{version}_MinMax.csv')
    df_train = df_train[df_train['CONJUNTO'] == 'ENTRENAMIENTO']
    
    cols_to_drop = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA']
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    model_lasso = joblib.load(f'modelo_lasso_base_{version}.pkl')
    
    # 2. Extraer coeficientes
    coeficientes_lasso = model_lasso.named_steps['lasso'].coef_
    df_coef = pd.DataFrame({
        'Variable': X_train.columns,
        'Coeficiente': coeficientes_lasso,
        'Importancia_Abs': np.abs(coeficientes_lasso)
    })
    
    # 3. Filtrar activas
    df_activas = df_coef[df_coef['Coeficiente'] != 0].sort_values(by='Importancia_Abs', ascending=True)
    print(f"Variables predictivas retenidas ({version}): {len(df_activas)}")
    
    joblib.dump(df_activas['Variable'].tolist(), f'variables_seleccionadas_lasso_{version}.pkl')
    
    # 4. Visualización
    plt.figure(figsize=(10, 8))
    colores = ['#2ca02c' if c > 0 else '#d62728' for c in df_activas['Coeficiente']]
    top_n = min(25, len(df_activas))
    df_plot = df_activas.tail(top_n) 
    
    plt.barh(df_plot['Variable'], df_plot['Coeficiente'], color=colores, edgecolor='black', alpha=0.8)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    plt.xlabel('Impacto en Rendimiento (t/ha)', fontsize=11)
    plt.title(f'Top {top_n} Variables - Lasso ({version})', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()