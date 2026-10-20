import pandas as pd
import numpy as np
from scipy.integrate import simpson

# 1. LECTURA DE DATOS
# ---------------------------------------------------------
print("1. Cargando datos...")
df_basico = pd.read_csv('dataset_basico.csv')
df_pro = pd.read_csv('dataset_pro.csv')
df_ext = pd.read_csv('dataset_ext.csv', encoding='latin-1')
df_rendimiento = pd.read_csv('parcelas_rendimiento.csv')

# 2. HOMOLOGACIÓN Y LIMPIEZA EXTREMA DE LLAVES Y NUBOSIDAD
# ---------------------------------------------------------
print("2. Limpiando llaves y forzando tipos numéricos...")
df_ext = df_ext.rename(columns={'ID_POLIGON': 'ID_POLIGONO'})

# Forzar IDs a texto puro y quitar espacios ocultos
for df in [df_rendimiento, df_basico, df_pro, df_ext]:
    if 'ID_POLIGONO' in df.columns:
        df['ID_POLIGONO'] = df['ID_POLIGONO'].astype(str).str.strip()

# Forzar nubosidad a numérico (si hay texto como "20%", se volverá NaN, y eso es seguro)
df_basico['porcentaje_nubosidad'] = pd.to_numeric(df_basico['porcentaje_nubosidad'], errors='coerce')
df_pro['porcentaje_nubosidad'] = pd.to_numeric(df_pro['porcentaje_nubosidad'], errors='coerce')

# 3. CONVERSIÓN DE FECHAS
# ---------------------------------------------------------
print("3. Parseando fechas...")
df_basico['fecha_captura'] = pd.to_datetime(df_basico['fecha_captura'], format='%d/%m/%Y', errors='coerce')
df_pro['fecha_captura'] = pd.to_datetime(df_pro['fecha_captura'], format='%d/%m/%Y', errors='coerce')

# 4. AISLAMIENTO 2025 Y FILTRO DE NUBOSIDAD
# ---------------------------------------------------------
print("4. Filtrando ventana 2025 y nubes...")
mask_basico_25 = (df_basico['fecha_captura'] >= '2025-04-01') & (df_basico['fecha_captura'] <= '2025-10-31')
mask_pro_25 = (df_pro['fecha_captura'] >= '2025-04-01') & (df_pro['fecha_captura'] <= '2025-10-31')

basico_2025 = df_basico[mask_basico_25].copy()
pro_2025 = df_pro[mask_pro_25].copy()

# Aumentamos ligeramente el umbral de nubes a 30% para asegurar que sobrevivan parcelas
basico_2025 = basico_2025[basico_2025['porcentaje_nubosidad'] <= 30].sort_values(['ID_POLIGONO', 'fecha_captura'])
pro_2025 = pro_2025[pro_2025['porcentaje_nubosidad'] <= 30].sort_values(['ID_POLIGONO', 'fecha_captura'])

print(f"   -> Filas en BASICO 2025 útiles: {basico_2025.shape[0]}")
print(f"   -> Filas en PRO 2025 útiles: {pro_2025.shape[0]}")

# 5. SELECCIÓN DE COLUMNAS PRIORITARIAS
# ---------------------------------------------------------
cols_basico = ['ID_POLIGONO', 'fecha_captura', 'ndvi_promedio', 'evi_promedio', 'lai_promedio', 'fapar_promedio', 'ndwi_promedio', 'msi_promedio']
cols_pro = ['ID_POLIGONO', 'fecha_captura', 'ndvi_promedio', 'evi_promedio', 'lai_promedio', 'msavi_promedio']

# Evitar KeyError si alguna columna no existe en el CSV real
cols_basico_reales = [c for c in cols_basico if c in basico_2025.columns]
cols_pro_reales = [c for c in cols_pro if c in pro_2025.columns]

basico_2025 = basico_2025[cols_basico_reales]
pro_2025 = pro_2025[cols_pro_reales]

# 6. LÍNEA BASE HISTÓRICA (2022-2024)
# ---------------------------------------------------------
print("5. Calculando histórico...")
mask_historico = df_basico['fecha_captura'] < '2025-01-01'
basico_historico = df_basico[mask_historico].copy()
basico_historico = basico_historico[basico_historico['porcentaje_nubosidad'] <= 30]

# Solo agrupamos si hay datos históricos, evitamos error de groupby vacío
if not basico_historico.empty:
    historico_features = basico_historico.groupby('ID_POLIGONO').agg(
        S2_ndvi_historico_mean=('ndvi_promedio', 'mean'),
        S2_ndvi_historico_std=('ndvi_promedio', 'std'),
        S2_evi_historico_mean=('evi_promedio', 'mean')
    ).reset_index()
else:
    historico_features = pd.DataFrame(columns=['ID_POLIGONO', 'S2_ndvi_historico_mean', 'S2_ndvi_historico_std', 'S2_evi_historico_mean'])

# 7. FUNCIÓN DE FEATURE ENGINEERING SEGURO
# ---------------------------------------------------------
def calcular_features_seguro(grupo, sensor_prefix):
    features = {}
    # Eliminar nulos de las fechas para que no rompa el cálculo de días
    grupo = grupo.dropna(subset=['fecha_captura'])
    if grupo.empty:
        return pd.Series()

    tiempo = (grupo['fecha_captura'] - grupo['fecha_captura'].min()).dt.days.values
    
    mask_senescencia = grupo['fecha_captura'].dt.month.isin([8, 9, 10])
    grupo_sen = grupo[mask_senescencia]
    tiempo_sen = (grupo_sen['fecha_captura'] - grupo_sen['fecha_captura'].min()).dt.days.values if not grupo_sen.empty else np.array([])
    
    indices = [col for col in grupo.columns if 'promedio' in col]
    
    for idx in indices:
        # Asegurar que el vector es numérico y quitar NaNs
        vals = pd.to_numeric(grupo[idx], errors='coerce').values
        
        # Máximo
        features[f'{sensor_prefix}_{idx}_max'] = np.nanmax(vals) if np.count_nonzero(~np.isnan(vals)) > 0 else np.nan
        
        # Integral (requiere al menos 2 puntos limpios de nulos para no romper simpson)
        mask_validos = ~np.isnan(vals)
        vals_validos = vals[mask_validos]
        tiempo_validos = tiempo[mask_validos]
        
        if len(tiempo_validos) > 1:
            try:
                features[f'{sensor_prefix}_{idx}_auc'] = simpson(y=vals_validos, x=tiempo_validos)
            except Exception:
                features[f'{sensor_prefix}_{idx}_auc'] = np.nan
        else:
            features[f'{sensor_prefix}_{idx}_auc'] = np.nan

    # Pendiente de Senescencia
    if 'ndvi_promedio' in grupo.columns:
        ndvi_sen_vals = pd.to_numeric(grupo_sen['ndvi_promedio'], errors='coerce').values
        mask_validos_sen = ~np.isnan(ndvi_sen_vals)
        
        if len(tiempo_sen[mask_validos_sen]) > 1:
            try:
                slope, _ = np.polyfit(tiempo_sen[mask_validos_sen], ndvi_sen_vals[mask_validos_sen], 1)
                features[f'{sensor_prefix}_ndvi_pendiente_senescencia'] = slope
            except Exception:
                features[f'{sensor_prefix}_ndvi_pendiente_senescencia'] = np.nan
        else:
            features[f'{sensor_prefix}_ndvi_pendiente_senescencia'] = np.nan

        # Media de Julio
        mask_julio = grupo['fecha_captura'].dt.month == 7
        features[f'{sensor_prefix}_ndvi_julio'] = pd.to_numeric(grupo.loc[mask_julio, 'ndvi_promedio'], errors='coerce').mean()

    return pd.Series(features)

# 8. EJECUTAR INGENIERÍA DE CARACTERÍSTICAS
# ---------------------------------------------------------
print("6. Extrayendo curvas fenológicas (Feature Engineering)...")
# Usamos group_keys=False para mantener el comportamiento limpio en Pandas >= 1.5
df_features_basico = basico_2025.groupby('ID_POLIGONO', group_keys=False).apply(
    lambda x: calcular_features_seguro(x, 'S2')
).reset_index()

df_features_pro = pro_2025.groupby('ID_POLIGONO', group_keys=False).apply(
    lambda x: calcular_features_seguro(x, 'Planet')
).reset_index()

# 9. INTEGRACIÓN DE CLIMA (EXT) Y ENSAMBLAJE FINAL
# ---------------------------------------------------------
print("7. Ensamblando Tabla Maestra...")
meses_2025 = ['04', '05', '06', '07', '08', '09', '10']
cols_clima_2025 = ['ID_POLIGONO', 'TpEl_mean', 'TpPe_mean']
for var in ['Tx25', 'Tm25', 'Pr25']:
    cols_clima_2025.extend([f'{var}{mes}_mea' for mes in meses_2025])

# Evitar KeyError si alguna columna climática falta en EXT
cols_clima_reales = [c for c in cols_clima_2025 if c in df_ext.columns]
ext_2025 = df_ext[cols_clima_reales].copy()

# Precipitacion acumulada (sumar solo las columnas de lluvia que existan)
cols_pr = [c for c in ext_2025.columns if 'Pr25' in c]
if cols_pr:
    ext_2025['Pr_acumulada_Abr_Oct'] = ext_2025[cols_pr].sum(axis=1)

# El merge final paso a paso
maestra = df_rendimiento[['ID_POLIGONO', 'AREA_HA', 'RENDIMIENTO_T_HA', 'CONJUNTO']].copy()
maestra = maestra.merge(ext_2025, on='ID_POLIGONO', how='left')
maestra = maestra.merge(df_features_basico, on='ID_POLIGONO', how='left')
maestra = maestra.merge(df_features_pro, on='ID_POLIGONO', how='left')
maestra = maestra.merge(historico_features, on='ID_POLIGONO', how='left')

# Calcular Anomalías (evitando error si la columna no se calculó)
if 'S2_ndvi_julio' in maestra.columns and 'S2_ndvi_historico_mean' in maestra.columns:
    maestra['Anomalia_NDVI_Julio'] = maestra['S2_ndvi_julio'] - maestra['S2_ndvi_historico_mean']

if 'S2_evi_promedio_max' in maestra.columns and 'S2_evi_historico_mean' in maestra.columns:
    maestra['Anomalia_EVI_Max'] = maestra['S2_evi_promedio_max'] - maestra['S2_evi_historico_mean']

# 10. EXPORTACIÓN
# ---------------------------------------------------------
maestra.to_csv('Tabla_Maestra_AgroCebada_Completa.csv', index=False)
print("¡PROCESO FINALIZADO!")
print(f"Dimensiones de la Tabla Maestra: {maestra.shape}")