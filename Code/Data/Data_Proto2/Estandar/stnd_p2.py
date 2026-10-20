import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib # Para guardar el escalador y usarlo a futuro

# 1. Cargar la Tabla Maestra
print("1. Cargando la Tabla Maestra...")
df_maestra = pd.read_csv('Tabla_Maestra_AgroCebada_Completa.csv')

# 2. Separar el dataset en ENTRENAMIENTO y PREDICCION
# NUNCA se debe escalar usando los datos de prueba, eso causaría "Data Leakage"
df_train = df_maestra[df_maestra['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_test = df_maestra[df_maestra['CONJUNTO'] == 'PREDICCION'].copy()

# 3. Identificar columnas a escalar (todas las numéricas predictivas)
# Excluimos IDs, variables categóricas o la variable objetivo
columnas_excluidas = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA']
columnas_predictivas = [col for col in df_train.columns if col not in columnas_excluidas]

# Asegurarse de que todas las columnas predictivas sean numéricas
df_train[columnas_predictivas] = df_train[columnas_predictivas].apply(pd.to_numeric, errors='coerce')
df_test[columnas_predictivas] = df_test[columnas_predictivas].apply(pd.to_numeric, errors='coerce')

# 4. Imputación de nulos residuales
# Algunos modelos como SVR no toleran NaN. Rellenaremos con la mediana de ENTRENAMIENTO.
print("2. Imputando valores faltantes residuales...")
mediana_train = df_train[columnas_predictivas].median()
df_train[columnas_predictivas] = df_train[columnas_predictivas].fillna(mediana_train)
df_test[columnas_predictivas] = df_test[columnas_predictivas].fillna(mediana_train) # ¡Usamos la mediana del train!

# 5. Inicializar y aplicar el MinMaxScaler
print("3. Aplicando MinMaxScaler...")
scaler = MinMaxScaler()

# FIT_TRANSFORM en Train: El escalador "aprende" el mínimo y máximo solo de las parcelas de entrenamiento
X_train_scaled_array = scaler.fit_transform(df_train[columnas_predictivas])

# TRANSFORM en Test: Aplica las reglas aprendidas del train a las parcelas ciegas
X_test_scaled_array = scaler.transform(df_test[columnas_predictivas])

# 6. Reconstruir los DataFrames escalados para validación
df_train_scaled = pd.DataFrame(X_train_scaled_array, columns=columnas_predictivas)
df_train_scaled['ID_POLIGONO'] = df_train['ID_POLIGONO'].values
df_train_scaled['RENDIMIENTO_T_HA'] = df_train['RENDIMIENTO_T_HA'].values
df_train_scaled['CONJUNTO'] = 'ENTRENAMIENTO'

df_test_scaled = pd.DataFrame(X_test_scaled_array, columns=columnas_predictivas)
df_test_scaled['ID_POLIGONO'] = df_test['ID_POLIGONO'].values
df_test_scaled['RENDIMIENTO_T_HA'] = df_test['RENDIMIENTO_T_HA'].values
df_test_scaled['CONJUNTO'] = 'PREDICCION'

# Combinar todo de nuevo en un solo CSV listo para modelar
df_final_scaled = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)

# 7. Guardar resultados
print("4. Guardando datos escalados...")
df_final_scaled.to_csv('Tabla_Maestra_Escalada_MinMax.csv', index=False)

# Es una excelente práctica guardar el objeto 'scaler' para cuando quieras 
# predecir nuevas parcelas en el futuro sin tener que re-entrenar
joblib.dump(scaler, 'minmax_scaler_agrocebada.pkl')

print("¡Escalamiento MinMax completado con éxito!")