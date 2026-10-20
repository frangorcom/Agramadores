import geopandas as gpd
import pandas as pd
import warnings

def extraer_coordenadas(shapefile_path, output_csv):
    print("1. Cargando capa vectorial de parcelas...")
    gdf_parcelas = gpd.read_file(shapefile_path)
    
    # -------------------------------------------------------------
    # CORRECCIÓN 1: Renombrar la columna truncada por el límite del Shapefile
    # -------------------------------------------------------------
    if 'ID_POLIGON' in gdf_parcelas.columns:
        gdf_parcelas = gdf_parcelas.rename(columns={'ID_POLIGON': 'ID_POLIGONO'})
    elif 'ID_POLIGO' in gdf_parcelas.columns: # Por si se truncó aún más
        gdf_parcelas = gdf_parcelas.rename(columns={'ID_POLIGO': 'ID_POLIGONO'})
        
    print("2. Calculando centroides con precisión espacial...")
    # -------------------------------------------------------------
    # CORRECCIÓN 2: Proyectar a métrico (Web Mercator 3857) para 
    # evitar el warning de CRS y obtener el centro físico real.
    # -------------------------------------------------------------
    # Reproyectamos a metros, calculamos el centro, y volvemos a grados (4326)
    gdf_proyectado = gdf_parcelas.to_crs(epsg=3857)
    centroides = gdf_proyectado.geometry.centroid.to_crs(epsg=4326)
    
    print("3. Extrayendo Latitud y Longitud...")
    gdf_parcelas['Latitud'] = centroides.y
    gdf_parcelas['Longitud'] = centroides.x
    
    # Verificación de seguridad de columnas disponibles
    cols_disponibles = gdf_parcelas.columns.tolist()
    cols_finales = ['ID_POLIGONO', 'Latitud', 'Longitud']
    
    # Asegurar que CONJUNTO existe antes de extraerlo
    if 'CONJUNTO' in cols_disponibles:
        cols_finales.append('CONJUNTO')
        
    df_coordenadas = gdf_parcelas[cols_finales].copy()
    
    print(f"4. Guardando {len(df_coordenadas)} coordenadas en {output_csv}...")
    df_coordenadas.to_csv(output_csv, index=False)
    print("¡Proceso completado exitosamente!")

extraer_coordenadas('Parcelas_Reto_AGC_CONJUNTO.shp', 'Coordenadas_Parcelas_Dashboard.csv')