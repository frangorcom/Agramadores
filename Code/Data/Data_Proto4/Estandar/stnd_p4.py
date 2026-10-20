import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib 

print("1. Recuperando identificadores de CONJUNTO...")
try:
    df_coords = pd.read_csv('Coordenadas_Parcelas_Dashboard.csv')
    df_conjunto = df_coords[['ID_POLIGONO', 'CONJUNTO']].copy()
    df_conjunto['ID_POLIGONO'] = df_conjunto['ID_POLIGONO'].astype(str).str.strip()
except FileNotFoundError:
    print("ERROR: No se encontró 'Coordenadas_Parcelas_Dashboard.csv'. Ejecuta primero la extracción de coordenadas.")
    exit()

# Definir los archivos generados en la limpieza
archivos_a_escalar = {
    'Lineal': 'Tabla_Maestra_GDD_FusionLineal.csv',
    'PCA': 'Tabla_Maestra_GDD_FusionPCA.csv'
}

for nombre_version, archivo_csv in archivos_a_escalar.items():
    print(f"\n{'='*50}")
    print(f"=== PROCESANDO VERSIÓN: {nombre_version.upper()} ===")
    print(f"{'='*50}")
    
    print(f"2. Cargando {archivo_csv}...")
    df_maestra = pd.read_csv(archivo_csv)
    df_maestra['ID_POLIGONO'] = df_maestra['ID_POLIGONO'].astype(str).str.strip()
    
    # Inyectar la columna CONJUNTO recuperada de las coordenadas (por si se perdió en la limpieza)
    if 'CONJUNTO' not in df_maestra.columns:
        df_maestra = df_maestra.merge(df_conjunto, on='ID_POLIGONO', how='left')
    
    df_train = df_maestra[df_maestra['CONJUNTO'] == 'ENTRENAMIENTO'].copy()
    df_test = df_maestra[df_maestra['CONJUNTO'] == 'PREDICCION'].copy()
    
    columnas_excluidas = ['ID_POLIGONO', 'CONJUNTO', 'RENDIMIENTO_T_HA', 'AREA_HA']
    columnas_predictivas = [col for col in df_train.columns if col not in columnas_excluidas]
    
    df_train[columnas_predictivas] = df_train[columnas_predictivas].apply(pd.to_numeric, errors='coerce')
    df_test[columnas_predictivas] = df_test[columnas_predictivas].apply(pd.to_numeric, errors='coerce')
    
    print("3. Tratando valores nulos residuales...")
    cols_nulas_completas = df_train[columnas_predictivas].columns[df_train[columnas_predictivas].isna().all()].tolist()
    if cols_nulas_completas:
        print(f"   -> ADVERTENCIA: Eliminando {len(cols_nulas_completas)} columnas 100% vacías.")
        columnas_predictivas = [c for c in columnas_predictivas if c not in cols_nulas_completas]
    
    # Imputación con mediana del conjunto de entrenamiento
    mediana_train = df_train[columnas_predictivas].median()
    df_train[columnas_predictivas] = df_train[columnas_predictivas].fillna(mediana_train)
    df_test[columnas_predictivas] = df_test[columnas_predictivas].fillna(mediana_train)
    
    print("4. Aplicando MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled_array = scaler.fit_transform(df_train[columnas_predictivas])
    X_test_scaled_array = scaler.transform(df_test[columnas_predictivas])
    
    df_train_scaled = pd.DataFrame(X_train_scaled_array, columns=columnas_predictivas)
    df_test_scaled = pd.DataFrame(X_test_scaled_array, columns=columnas_predictivas)
    
    # Reintegrar variables excluidas
    for col in columnas_excluidas:
        df_train_scaled[col] = df_train[col].values
        df_test_scaled[col] = df_test[col].values
    
    cols_finales = columnas_excluidas + columnas_predictivas
    df_final_scaled = pd.concat([df_train_scaled[cols_finales], df_test_scaled[cols_finales]], ignore_index=True)
    
    archivo_salida = f'Tabla_Maestra_Mensual_Escalada_{nombre_version}_MinMax.csv'
    scaler_salida = f'minmax_scaler_{nombre_version.lower()}.pkl'
    
    print(f"5. Guardando en '{archivo_salida}'...")
    df_final_scaled.to_csv(archivo_salida, index=False)
    joblib.dump(scaler, scaler_salida)
    print(f"¡Versión {nombre_version} completada!")

print("\n¡PROCESO TOTAL DE ESTANDARIZACIÓN FINALIZADO!")