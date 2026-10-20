import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Cargar los datos para recuperar los nombres exactos de las columnas
df = pd.read_csv('Tabla_Maestra_Estandarizada.csv')
df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
cols_to_drop = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA']
X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
X_train = X_train.select_dtypes(include=[np.number]) # Filtrar numéricas

# 2. Cargar el modelo desde el disco
model_lasso = joblib.load('modelo_lasso_cebada.pkl')

# 3. Extraer coeficientes (ya no dará error)
coeficientes_lasso = model_lasso.named_steps['lasso'].coef_
nombres_variables = X_train.columns

# Crear un DataFrame para organizar los resultados
df_coef = pd.DataFrame({
    'Variable': nombres_variables,
    'Coeficiente': coeficientes_lasso,
    'Importancia_Abs': np.abs(coeficientes_lasso)
})
# ==========================================
# ANÁLISIS DE VARIABLES DESCARTADAS VS ACTIVAS
# ==========================================
# Lasso penaliza llevando a 0 exacto las variables redundantes o ruidosas
variables_descartadas = df_coef[df_coef['Coeficiente'] == 0]
df_activas = df_coef[df_coef['Coeficiente'] != 0].sort_values(by='Importancia_Abs', ascending=True)

print(f"Total de variables originales: {len(df_coef)}")
print(f"Variables anuladas por el modelo (Coeficiente = 0): {len(variables_descartadas)}")
print(f"Variables retenidas (Activas): {len(df_activas)}\n")

if len(variables_descartadas) > 0:
    print("Variables descartadas (Ejemplos):")
    print(variables_descartadas['Variable'].head(10).tolist())
    print("\n")

# ==========================================
# VISUALIZACIÓN DE LA IMPORTANCIA DE VARIABLES
# ==========================================
plt.figure(figsize=(10, 8))

# Asignar verde a impactos positivos y rojo a impactos negativos
colores = ['#2ca02c' if c > 0 else '#d62728' for c in df_activas['Coeficiente']]

# Crear gráfico de barras horizontales
plt.barh(df_activas['Variable'], df_activas['Coeficiente'], color=colores, edgecolor='black', alpha=0.8)

# Configuraciones estéticas del gráfico
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.xlabel('Impacto en Rendimiento (t/ha) por cada Desviación Estándar', fontsize=11)
plt.ylabel('Variables Predictoras', fontsize=11)
plt.title('Importancia de Variables (Modelo Lasso)', fontsize=14, pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()

# ==========================================
# TOP 5 VARIABLES MÁS INFLUYENTES
# ==========================================
print("Top 5 variables que más definen el rendimiento:")
print(df_activas.sort_values(by='Importancia_Abs', ascending=False).head(5)[['Variable', 'Coeficiente']])