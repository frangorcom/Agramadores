import os
from qgis.core import QgsRasterLayer
from qgis.analysis import QgsZonalStatistics

# 1. Definimos tu carpeta
carpeta_tifs = "C:/Users/inaoe/Desktop/AgroCebada/Respaldo/Reto_AgroCebada_Topografia_INEGI_CEM4"

# 2. Tomamos la capa seleccionada en QGIS
capa_parcelas = iface.activeLayer()

print("Iniciando inyección de datos (Modo API interno)...")

# 3. Iteramos sobre los archivos
for archivo in os.listdir(carpeta_tifs):
    if archivo.endswith(".tif"):
        ruta_raster = os.path.join(carpeta_tifs, archivo)
        
        # 4. Cargamos el TIF temporalmente en la memoria de QGIS
        raster_tmp = QgsRasterLayer(ruta_raster, archivo)
        
        if not raster_tmp.isValid():
            print(f"Error: No se pudo leer {archivo}")
            continue
            
        # 5. Generamos el prefijo (ej. Tx2201_)
        prefijo = "Tp" + archivo[0:2] + "_" 
        print(f"Procesando: {archivo} -> Nueva columna: {prefijo}mean")
        
        # 6. Llamamos al motor interno de QGIS para Estadísticas de Zona
        # QgsZonalStatistics.Mean le dice que solo calcule el promedio
        zs = QgsZonalStatistics(capa_parcelas, raster_tmp, prefijo, 1, QgsZonalStatistics.Mean)
        
        # 7. Ejecutamos: esto inyecta la columna directamente al .dbf de tu shapefile
        zs.calculateStatistics(None)

print("¡Proceso de este directorio terminado exitosamente!")