"""
AgroCebada 2026 — Dashboard multipágina
Ejecutar:  streamlit run Inicio.py
"""

import streamlit as st

st.set_page_config(
    page_title="AgroCebada 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global compartido ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F2318; }
[data-testid="stSidebar"] * { color: #C8DCC8 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3E2A !important; }
footer { visibility: hidden; }

.page-hero {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #388E3C 100%);
    border-radius: 14px;
    padding: 40px 48px 36px;
    margin-bottom: 28px;
    color: white;
}
.page-hero h1 { font-size: 2.2em; font-weight: 800; margin: 0 0 8px; color: #fff; }
.page-hero p  { font-size: 1.0em; opacity: .88; margin: 0; color: #A5D6A7; }

.nav-card {
    border: 1.5px solid #C8E6C9;
    border-radius: 12px;
    padding: 24px 22px;
    transition: box-shadow .2s;
    background: #F9FBF9;
}
.nav-card:hover { box-shadow: 0 4px 18px rgba(27,94,32,.15); }
.nav-card h3 { margin: 8px 0 6px; font-size: 1.08em; color: #1B5E20; }
.nav-card p  { font-size: 0.82em; color: #555; margin: 0; line-height: 1.45; }
/* Contraste global y accesibilidad */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    color: #17251B !important;
}
[data-testid="stSidebar"] { background-color: #10291C !important; }
[data-testid="stSidebar"] * { color: #F0F7F1 !important; }
[data-testid="stSidebar"] hr { border-color: #42634D !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color: #F0F7F1 !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th, label, small {
    color: #17251B;
}
[data-testid="stMetricLabel"] { color: #34483A !important; }
[data-testid="stMetricValue"] { color: #10291C !important; }
[data-testid="stDataFrame"] { border: 1px solid #B6C5BA; border-radius: 8px; }
[data-testid="stTabs"] button { color: #263D2D !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #14532D !important; border-bottom-color: #167A3E !important;
}
button[kind="secondary"], button[kind="primary"] { font-weight: 600; }
footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  AgroCebada 2026")
    st.divider()
    st.markdown("**Navegación**")
    st.page_link("Inicio.py",            label="Inicio",            )
    st.page_link("pages/1_Datos.py",     label="Datos y procesamiento")
    st.page_link("pages/2_Modelo.py",    label="Modelo predictivo" )
    st.page_link("pages/3_Mapa.py",      label="Mapa de parcelas"  )
    st.divider()
    st.markdown("<p style='font-size:.72em;opacity:.5'>v1.0 · Reto AgroCebada 2026</p>",
                unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1> Predicción de Rendimiento de Cebada</h1>
  <p>Reto AgroCebada 2026 · Estado de Hidalgo, México ·
     Fuentes: CHIRPS (temperatura, precipitación), INEGI (topografía)</p>
</div>
""", unsafe_allow_html=True)

# ── Tarjetas de navegación ────────────────────────────────────────────────────
st.markdown("### Selecciona una sección")
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("""
    <div class="nav-card">
      <h3>Datos y procesamiento</h3>
      <p>Exploración del dataset EXT, variables climáticas y topográficas,
         estadísticos descriptivos y visualización del procesamiento.</p>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/1_Datos.py", label="Ir a Datos →", use_container_width=True)

with c2:
    st.markdown("""
    <div class="nav-card">
      <h3>Modelo predictivo</h3>
      <p>Arquitectura del modelo, métricas de desempeño, importancia de variables
         y comparación de resultados de predicción.</p>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/2_Modelo.py", label="Ir a Modelo →", use_container_width=True)

with c3:
    st.markdown("""
    <div class="nav-card">
      <h3>Mapa de parcelas</h3>
      <p>Visualización geoespacial de las 197 parcelas con sus predicciones
         de rendimiento. Haz clic en cada parcela para ver su detalle.</p>
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/3_Mapa.py", label="Ir a Mapa →", use_container_width=True)

st.divider()

# ── Métricas globales de contexto ─────────────────────────────────────────────
import pandas as pd
try:
    df = pd.read_csv("data/Conjunto_datos_EXT_AgroCebada.csv", encoding="latin1")
    df["Municipio"] = df["Municipio"].str.strip()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total parcelas",   len(df))
    m2.metric("Entrenamiento",    (df["CONJUNTO"]=="ENTRENAMIENTO").sum())
    m3.metric("Predicción",       (df["CONJUNTO"]=="PREDICCION").sum())
    m4.metric("Municipios",       df["Municipio"].nunique())
    m5.metric("Variables",        len(df.columns) - 6)
except Exception:
    st.info("Coloca los CSV en la carpeta data/ para ver las métricas.")
