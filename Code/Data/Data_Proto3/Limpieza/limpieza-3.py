import pandas as pd
import numpy as np

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

for df in [df_rendimiento, df_basico, df_pro, df_ext]:
    if 'ID_POLIGONO' in df.columns:
        df['ID_POLIGONO'] = df['ID_POLIGONO'].astype(str).str.strip()

df_basico['porcentaje_nubosidad'] = pd.to_numeric(df_basico['porcentaje_nubosidad'], errors='coerce')
df_pro['porcentaje_nubosidad'] = pd.to_numeric(df_pro['porcentaje_nubosidad'], errors='coerce')

# 3. CONVERSIÓN DE FECHAS
# ---------------------------------------------------------
print("3. Parseando fechas...")
df_basico['fecha_captura'] = pd.to_datetime(df_basico['fecha_captura'], format='%d/%m/%Y', errors='coerce')
df_pro['fecha_captura'] = pd.to_datetime(df_pro['fecha_captura'], format='%d/%m/%Y', errors='coerce')

# 4. AISLAMIENTO 2025 (MARZO A OCTUBRE) Y FILTRO DE NUBOSIDAD
# ---------------------------------------------------------
print("4. Filtrando ventana 2025 (Marzo-Octubre) y nubes...")
# Ampliamos la ventana para incluir marzo, que es crítico como línea base del suelo
mask_basico_25 = (df_basico['fecha_captura'] >= '2025-03-01') & (df_basico['fecha_captura'] <= '2025-10-31')
mask_pro_25 = (df_pro['fecha_captura'] >= '2025-03-01') & (df_pro['fecha_captura'] <= '2025-10-31')

basico_2025 = df_basico[mask_basico_25].copy()
pro_2025 = df_pro[mask_pro_25].copy()

# Filtro del 30% de nubosidad y ordenamiento para evitar duplicados / preparar series temporales
basico_2025 = basico_2025[basico_2025['porcentaje_nubosidad'] <= 30].sort_values(['ID_POLIGONO', 'fecha_captura'])
pro_2025 = pro_2025[pro_2025['porcentaje_nubosidad'] <= 30].sort_values(['ID_POLIGONO', 'fecha_captura'])

# Extraer el mes como cadena (ej. '03', '04', ..., '10') para alinear con el clima
basico_2025['Mes'] = basico_2025['fecha_captura'].dt.strftime('%m')
pro_2025['Mes'] = pro_2025['fecha_captura'].dt.strftime('%m')

# 5. EXTRACCIÓN DINÁMICA DE TODAS LAS VARIABLES SATELITALES
# ---------------------------------------------------------
print("5. Agregando características mensuales (Fusión diaria -> mensual)...")
# Detectamos automáticamente todas las variables (excluyendo metadatos)
ignorar_cols = ['ID_POLIGONO', 'fecha_captura', 'Mes', 'porcentaje_nubosidad']
vars_basico = [col for col in basico_2025.columns if col not in ignorar_cols]
vars_pro = [col for col in pro_2025.columns if col not in ignorar_cols]

for col in vars_basico:
    basico_2025[col] = pd.to_numeric(basico_2025[col], errors='coerce')

for col in vars_pro:
    pro_2025[col] = pd.to_numeric(pro_2025[col], errors='coerce')

# Función auxiliar para Pivotear y aplanar columnas
def agregar_mensual(df, variables, prefijo):
    # Agrupamos por Polígono y Mes, sacando el Promedio y el Máximo
    tabla_pivot = df.pivot_table(
        index='ID_POLIGONO', 
        columns='Mes', 
        values=variables, 
        aggfunc=['mean', 'max'] # Aquí aplicamos el MVC y el Promedio
    )
    # Aplanar el multi-índice de columnas de Pandas a un formato limpio
    # Resultado ej: S2_ndvi_promedio_03_max
    nuevas_columnas = [f"{prefijo}_{col[1]}_{col[2]}_{col[0]}" for col in tabla_pivot.columns]
    tabla_pivot.columns = nuevas_columnas
    return tabla_pivot.reset_index()

df_features_basico = agregar_mensual(basico_2025, vars_basico, 'S2_L8')
df_features_pro = agregar_mensual(pro_2025, vars_pro, 'Planet')

# 6. INTEGRACIÓN DE CLIMA (EXT) Y TOPOGRAFÍA
# ---------------------------------------------------------
print("6. Ensamblando variables climáticas (Marzo a Octubre)...")
# Ahora incluimos el mes '03' para el clima
meses_2025 = ['03', '04', '05', '06', '07', '08', '09', '10']
cols_clima_estaticas = ['ID_POLIGONO', 'TpEl_mean', 'TpPe_mean'] # Topografía constante

# Generar nombres de columnas de clima esperadas (ej. Tx2503_mea, Pr2504_mea)
cols_clima_dinamicas = []
for var in ['Tx25', 'Tm25', 'Pr25']:
    cols_clima_dinamicas.extend([f'{var}{mes}_mea' for mes in meses_2025])

cols_clima_objetivo = cols_clima_estaticas + cols_clima_dinamicas
# Evitar KeyError si alguna columna climática falta en EXT
cols_clima_reales = [c for c in cols_clima_objetivo if c in df_ext.columns]
ext_2025 = df_ext[cols_clima_reales].copy()

# Precipitacion acumulada global (Abril - Octubre, ciclo productivo)
cols_pr_ciclo = [f'Pr25{mes}_mea' for mes in ['03','04','05','06','07','08','09','10'] if f'Pr25{mes}_mea' in ext_2025.columns]
if cols_pr_ciclo:
    ext_2025['Pr_acumulada_Ciclo'] = ext_2025[cols_pr_ciclo].sum(axis=1)

# 7. ENSAMBLAJE FINAL
# ---------------------------------------------------------
print("7. Construyendo Tabla Maestra...")
maestra = df_rendimiento[['ID_POLIGONO', 'AREA_HA', 'RENDIMIENTO_T_HA', 'CONJUNTO']].copy()

# Fusionar todo utilizando left join para preservar todos los polígonos
maestra = maestra.merge(ext_2025, on='ID_POLIGONO', how='left')
maestra = maestra.merge(df_features_basico, on='ID_POLIGONO', how='left')
maestra = maestra.merge(df_features_pro, on='ID_POLIGONO', how='left')

# 8. EXPORTACIÓN
# ---------------------------------------------------------
maestra.to_csv('Tabla_Maestra_AgroCebada_Mensual_Completa.csv', index=False)
print("¡PROCESO FINALIZADO!")
print(f"Dimensiones de la Tabla Maestra: {maestra.shape}")