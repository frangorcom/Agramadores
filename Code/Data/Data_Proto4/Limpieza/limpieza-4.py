import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("1. Cargando datos crudos y Shapefile de parcelas...")
df_basico = pd.read_csv('dataset_basico.csv')
df_pro = pd.read_csv('dataset_pro.csv')
df_ext = pd.read_csv('dataset_ext.csv', encoding='latin-1')
df_rendimiento = pd.read_csv('parcelas_rendimiento.csv')

# CORRECCIÓN: Renombrar columna truncada en el clima antes de limpiarla
if 'ID_POLIGON' in df_ext.columns:
    df_ext = df_ext.rename(columns={'ID_POLIGON': 'ID_POLIGONO'})

# Extraer Latitudes del Shapefile y crear df_lat
gdf_parcelas = gpd.read_file('Parcelas_Reto_AGC_CONJUNTO.shp')
if 'ID_POLIGON' in gdf_parcelas.columns:
    gdf_parcelas = gdf_parcelas.rename(columns={'ID_POLIGON': 'ID_POLIGONO'})

centroides = gdf_parcelas.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
gdf_parcelas['Latitud'] = centroides.y
df_lat = gdf_parcelas[['ID_POLIGONO', 'Latitud']].copy()

print("2. Procesando índices satelitales mensuales (Marzo - Octubre)...")
# Limpieza de llaves con validación de seguridad
for df in [df_rendimiento, df_basico, df_pro, df_ext, df_lat]:
    if 'ID_POLIGONO' in df.columns:
        df['ID_POLIGONO'] = df['ID_POLIGONO'].astype(str).str.strip()

# Ventana 2025 y Nubes
df_basico['fecha_captura'] = pd.to_datetime(df_basico['fecha_captura'], format='%d/%m/%Y', errors='coerce')
mask_basico = (df_basico['fecha_captura'] >= '2025-03-01') & (df_basico['fecha_captura'] <= '2025-10-31')
basico_2025 = df_basico[mask_basico].copy()
basico_2025['porcentaje_nubosidad'] = pd.to_numeric(basico_2025['porcentaje_nubosidad'], errors='coerce')
basico_2025 = basico_2025[basico_2025['porcentaje_nubosidad'] <= 30]
basico_2025['Mes'] = basico_2025['fecha_captura'].dt.strftime('%m')

# Forzar a numérico
ignorar_cols = ['ID_POLIGONO', 'fecha_captura', 'Mes', 'porcentaje_nubosidad']
vars_basico = [col for col in basico_2025.columns if col not in ignorar_cols]
for col in vars_basico:
    basico_2025[col] = pd.to_numeric(basico_2025[col], errors='coerce')

# Pivoteo Mensual (Promedio y Máximo)
df_features_basico = basico_2025.pivot_table(index='ID_POLIGONO', columns='Mes', values=vars_basico, aggfunc=['mean', 'max'])
df_features_basico.columns = [f"S2_{c[1]}_{c[2]}_{c[0]}" for c in df_features_basico.columns]
df_features_basico = df_features_basico.reset_index()

# ELIMINAR VARIANTES DEL MSI (Conservar solo el mean)
cols_msi_eliminar = [c for c in df_features_basico.columns if 'msi' in c.lower() and 'max' in c.lower()]
df_features_basico.drop(columns=cols_msi_eliminar, inplace=True)

print("3. Calculando Grados Día Desarrollo (GDD) mediante Splines Cúbicos...")
meses_num = ['03', '04', '05', '06', '07', '08', '09', '10']
dias_julianos_centro = np.array([74, 105, 135, 166, 196, 227, 258, 288]) # Centro aproximado de cada mes
dias_por_mes = {'03': 31, '04': 30, '05': 31, '06': 30, '07': 31, '08': 31, '09': 30, '10': 31}

df_clima = df_ext.rename(columns={'ID_POLIGON': 'ID_POLIGONO'})
df_clima = df_clima.merge(df_lat, on='ID_POLIGONO', how='left')

# Calcular GDD parcela por parcela
gdd_results = []
for index, row in df_clima.iterrows():
    tmax_mensual = [row.get(f'Tx25{m}_mea', np.nan) for m in meses_num]
    tmin_mensual = [row.get(f'Tm25{m}_mea', np.nan) for m in meses_num]
    
    parcela_gdd = {'ID_POLIGONO': row['ID_POLIGONO']}
    
    if not np.isnan(tmax_mensual).all() and not np.isnan(tmin_mensual).all():
        # Crear splines para simular el comportamiento diario a lo largo de los 8 meses
        cs_tmax = CubicSpline(dias_julianos_centro, tmax_mensual, bc_type='natural')
        cs_tmin = CubicSpline(dias_julianos_centro, tmin_mensual, bc_type='natural')
        
        for m, jd_centro in zip(meses_num, dias_julianos_centro):
            dias_mes = dias_por_mes[m]
            jd_inicio = jd_centro - (dias_mes // 2)
            jd_rango = np.arange(jd_inicio, jd_inicio + dias_mes)
            
            tmax_diario = cs_tmax(jd_rango)
            tmin_diario = cs_tmin(jd_rango)
            
            # GDD: ((Tmax + Tmin) / 2) - Tbase(0)
            gdd_diario = ((tmax_diario + tmin_diario) / 2)
            gdd_diario[gdd_diario < 0] = 0 # No hay GDD negativo
            
            parcela_gdd[f'GDD_25{m}'] = np.sum(gdd_diario)
    gdd_results.append(parcela_gdd)

df_gdd = pd.DataFrame(gdd_results)
df_clima = df_clima.merge(df_gdd, on='ID_POLIGONO', how='left')

print("4. Calculando Evapotranspiración (ET0) y Déficit Hídrico...")
def calc_ra_fao56(lat_deg, j_day):
    lat_rad = np.radians(lat_deg)
    dec = 0.409 * np.sin(2 * np.pi * j_day / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(dec), -1, 1))
    dr = 1 + 0.033 * np.cos(2 * np.pi * j_day / 365)
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    return Ra

for m, jd in zip(meses_num, dias_julianos_centro):
    tmax = df_clima[f'Tx25{m}_mea']
    tmin = df_clima[f'Tm25{m}_mea']
    tmed = (tmax + tmin) / 2
    precip = df_clima[f'Pr25{m}_mea']
    
    # Radiación Extraterrestre y ET0 (Hargreaves-Samani)
    Ra = calc_ra_fao56(df_clima['Latitud'], jd)
    ET0_diario = 0.0023 * Ra * (tmed + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None))
    ET0_mensual = ET0_diario * dias_por_mes[m]
    
    df_clima[f'DeficitHidrico_25{m}'] = precip - ET0_mensual

print("5. Ensamblando y aplicando Fusiones (Lineal y PCA)...")
maestra = df_rendimiento[['ID_POLIGONO', 'AREA_HA', 'RENDIMIENTO_T_HA', 'CONJUNTO']].copy()
maestra = maestra.merge(df_clima, on='ID_POLIGONO', how='left')
maestra = maestra.merge(df_features_basico, on='ID_POLIGONO', how='left')

# Preparar las dos versiones
maestra_lineal = maestra.copy()
maestra_pca = maestra.copy()

pca_model = PCA(n_components=1)
imputer = SimpleImputer(strategy='median') # El PCA requiere imputar nulos satelitales

for m in meses_num:
    deficit_col = f'DeficitHidrico_25{m}'
    msi_col = f'S2_msi_promedio_{m}_mean'
    
    if msi_col in maestra.columns and deficit_col in maestra.columns:
        # Fusión 1: Interacción Lineal
        maestra_lineal[f'Stress_Fusion_Lin_{m}'] = maestra_lineal[deficit_col] * maestra_lineal[msi_col]
        
        # Fusión 2: PCA
        # Imputamos temporalmente nulos solo para extraer el componente principal
        data_pca = imputer.fit_transform(maestra_pca[[deficit_col, msi_col]])
        maestra_pca[f'Stress_Fusion_PCA_{m}'] = pca_model.fit_transform(data_pca)
        
        # Eliminar el MSI original ya que su varianza está en la variable fusionada
        maestra_lineal.drop(columns=[msi_col], inplace=True)
        maestra_pca.drop(columns=[msi_col], inplace=True)

print("6. Eliminando variables crudas (Histórico completo)...")
# Detectar TODAS las columnas que empiecen con Tx (Tmax), Tm (Tmin) o Pr (Precipitación) de cualquier año
cols_clima_crudas = [c for c in df_clima.columns if c.startswith(('Tx', 'Tm', 'Pr'))]

# Agregamos Latitud a la lista de eliminación
cols_a_eliminar = cols_clima_crudas + ['Latitud']

# Eliminamos todo el ruido climático crudo de ambas tablas
maestra_lineal.drop(columns=cols_a_eliminar, inplace=True, errors='ignore')
maestra_pca.drop(columns=cols_a_eliminar, inplace=True, errors='ignore')

print(f"   -> Se eliminaron {len(cols_a_eliminar)} variables crudas y de geolocalización.")
print("7. Exportando resultados...")
maestra_lineal.to_csv('Tabla_Maestra_GDD_FusionLineal.csv', index=False)
maestra_pca.to_csv('Tabla_Maestra_GDD_FusionPCA.csv', index=False)

print("¡Proceso completado! Se han generado dos DataFrames limpios y sin variables crudas.")