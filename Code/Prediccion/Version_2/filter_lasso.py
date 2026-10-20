import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

print("--- FASE 2: SELECCIÓN DE VARIABLES CON LASSO ---")

# 1. Cargar datos y modelo
df_train = pd.read_csv('Tabla_Maestra_Mensual_Escalada_MinMax.csv')
df_train = df_train[df_train['CONJUNTO'] == 'ENTRENAMIENTO']

# Extraer columnas exactamente como entraron al modelo
cols_to_drop = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA']
X_train = df_train.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])

model_lasso = joblib.load('modelo_lasso_base.pkl')

# 2. Extraer coeficientes del Pipeline
coeficientes_lasso = model_lasso.named_steps['lasso'].coef_
nombres_variables = X_train.columns

df_coef = pd.DataFrame({
    'Variable': nombres_variables,
    'Coeficiente': coeficientes_lasso,
    'Importancia_Abs': np.abs(coeficientes_lasso)
})

# 3. Filtrar variables activas (distintas de 0)
variables_descartadas = df_coef[df_coef['Coeficiente'] == 0]
df_activas = df_coef[df_coef['Coeficiente'] != 0].sort_values(by='Importancia_Abs', ascending=True)

print(f"Total variables originales: {len(df_coef)}")
print(f"Ruido anulado por Lasso: {len(variables_descartadas)}")
print(f"Variables predictivas retenidas: {len(df_activas)}\n")

# Guardar la lista de variables retenidas para el Script 3
variables_seleccionadas = df_activas['Variable'].tolist()
joblib.dump(variables_seleccionadas, 'variables_seleccionadas_lasso.pkl')
print("Lista de variables óptimas guardada en 'variables_seleccionadas_lasso.pkl'")

# 4. Visualización
plt.figure(figsize=(10, 8))
colores = ['#2ca02c' if c > 0 else '#d62728' for c in df_activas['Coeficiente']]

# Si hay muchas variables, graficamos solo el Top 25
top_n = min(25, len(df_activas))
df_plot = df_activas.tail(top_n) 

plt.barh(df_plot['Variable'], df_plot['Coeficiente'], color=colores, edgecolor='black', alpha=0.8)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.xlabel('Impacto en Rendimiento (t/ha)', fontsize=11)
plt.title(f'Top {top_n} Variables - Selección Lasso', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()