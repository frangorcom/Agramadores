# 2_filtro_lasso.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("--- FASE 2: SELECCIÓN DE VARIABLES CON LASSO ---")

# 1. Cargar datos ESCALADOS (Estrictamente necesario para Lasso)
df = pd.read_csv('Tabla_Maestra_Mensual_Escalada_PCA_MinMax.csv') 

df_train = df[df['CONJUNTO'] == 'ENTRENAMIENTO'].copy()

# 2. Configurar variables con filtrado de seguridad estricto
cols_excluir = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA']
cols_presentes = [col for col in df_train.columns if col in cols_excluir or 'CONJUNTO' in col]

X_train_temp = df_train.drop(columns=cols_presentes)
# Filtro de seguridad: Conservar exclusivamente columnas numéricas
X_train = X_train_temp.select_dtypes(include=[np.number])
y_train = df_train['RENDIMIENTO_T_HA']

# 3. Imputación de nulos
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# 4. Entrenamiento de Lasso
print("Calculando penalización L1 para reducción de dimensionalidad...")
lasso = LassoCV(cv=5, random_state=42, max_iter=20000, n_jobs=-1)
lasso.fit(X_train_imputed, y_train)

# 5. Extracción de variables activas
coeficientes = lasso.coef_
columnas = X_train.columns
variables_seleccionadas = columnas[coeficientes != 0].tolist()

print(f"\nProceso finalizado. El mejor Alpha fue: {lasso.alpha_:.5f}")
print(f"De {len(columnas)} variables originales, Lasso conservó {len(variables_seleccionadas)}.")

# Exportar la lista de variables
joblib.dump(variables_seleccionadas, 'variables_seleccionadas_lasso.pkl')
print("Lista de variables exportada como 'variables_seleccionadas_lasso.pkl'")