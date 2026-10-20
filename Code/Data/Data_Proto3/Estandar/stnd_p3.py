import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib 

print("1. Cargando la Tabla Maestra...")
df_maestra = pd.read_csv('Tabla_Maestra_AgroCebada_Mensual_Completa.csv')

df_train = df_maestra[df_maestra['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
df_test = df_maestra[df_maestra['CONJUNTO'] == 'PREDICCION'].copy()

columnas_excluidas = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA']
columnas_predictivas = [col for col in df_train.columns if col not in columnas_excluidas]

df_train[columnas_predictivas] = df_train[columnas_predictivas].apply(pd.to_numeric, errors='coerce')
df_test[columnas_predictivas] = df_test[columnas_predictivas].apply(pd.to_numeric, errors='coerce')

print("2. Tratando valores nulos...")
cols_nulas_completas = df_train[columnas_predictivas].columns[df_train[columnas_predictivas].isna().all()].tolist()
if cols_nulas_completas:
    print(f"   -> ADVERTENCIA: Eliminando {len(cols_nulas_completas)} columnas 100% nulas: {cols_nulas_completas}")
    columnas_predictivas = [c for c in columnas_predictivas if c not in cols_nulas_completas]

mediana_train = df_train[columnas_predictivas].median()
df_train[columnas_predictivas] = df_train[columnas_predictivas].fillna(mediana_train)
df_test[columnas_predictivas] = df_test[columnas_predictivas].fillna(mediana_train)

print("3. Aplicando MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled_array = scaler.fit_transform(df_train[columnas_predictivas])
X_test_scaled_array = scaler.transform(df_test[columnas_predictivas])

df_train_scaled = pd.DataFrame(X_train_scaled_array, columns=columnas_predictivas)
df_test_scaled = pd.DataFrame(X_test_scaled_array, columns=columnas_predictivas)

for col in columnas_excluidas:
    df_train_scaled[col] = df_train[col].values
    df_test_scaled[col] = df_test[col].values

# Reordenar las columnas para dejar el ID y la variable objetivo al principio
cols_finales = columnas_excluidas + columnas_predictivas
df_final_scaled = pd.concat([df_train_scaled[cols_finales], df_test_scaled[cols_finales]], ignore_index=True)

print("4. Guardando datos escalados...")
df_final_scaled.to_csv('Tabla_Maestra_Mensual_Escalada_MinMax.csv', index=False)
joblib.dump(scaler, 'minmax_scaler_agrocebada.pkl')

print("¡Escalamiento MinMax completado con éxito!")