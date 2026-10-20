# AgroCebada 2026 — Dashboard de Predicción de Rendimiento

Dashboard interactivo para visualizar las predicciones del modelo de rendimiento
de cebada (t/ha) sobre parcelas del Estado de Hidalgo, México.

---

## Estructura del proyecto

```
dashboard_agro/
├── app.py                  ← Dashboard Streamlit (ejecutar este)
├── pipeline.py             ← Procesamiento de datos + entrenamiento del modelo
├── requirements.txt        ← Dependencias Python
├── .streamlit/
│   └── config.toml         ← Tema visual
└── data/
    ├── Conjunto_datos_BASICO_AgroCebada2026.csv     ← COPIAR AQUÍ
    ├── Conjunto_datos_PRO_AgroCebada.csv            ← COPIAR AQUÍ
    ├── ID_area_rendimiento_70_30_Reto_AgroCebada.csv← COPIAR AQUÍ
    ├── resultados_modelo.csv                        ← Generado por pipeline.py
    └── importancia_features.csv                     ← Generado por pipeline.py
```

---

## Instalación paso a paso

### 1. Instalar Python 3.10 o superior
Descarga desde https://www.python.org/downloads/
Activa "Add Python to PATH" al instalar en Windows.

### 2. Crear y activar entorno virtual

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Copiar los archivos de datos
Crea la carpeta `data/` y copia los tres CSV del dataset original.

### 5. Ejecutar el pipeline (genera los datos del dashboard)
```bash
python pipeline.py
```
Esto tarda ~2–3 minutos. Verás en consola las métricas del modelo.

### 6. Lanzar el dashboard
```bash
streamlit run app.py
```
El navegador se abre automáticamente en `http://localhost:8501`

---

## Despliegue en Streamlit Cloud (URL pública)

1. Sube el proyecto a un repositorio de GitHub (incluye `data/resultados_modelo.csv`)
2. Ve a https://share.streamlit.io
3. "New app" → selecciona tu repo → branch `main` → archivo `app.py`
4. Deploy → obtienes una URL pública permanente y gratuita
