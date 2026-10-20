"""
Página 1 — Datos base y procesamiento
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Tema Plotly global: texto oscuro, ejes legibles y rejilla con contraste.
import plotly.io as pio
_agro_theme = pio.templates["plotly_white"]
_agro_theme.layout.font = dict(family="Arial, sans-serif", size=12, color="#17251B")
_agro_theme.layout.title = dict(font=dict(color="#10291C", size=16))
_agro_theme.layout.xaxis = dict(
    color="#263D2D", title_font=dict(color="#17251B"),
    tickfont=dict(color="#263D2D"), gridcolor="#D6DFD8",
    linecolor="#87988C", zerolinecolor="#A9B7AD"
)
_agro_theme.layout.yaxis = dict(
    color="#263D2D", title_font=dict(color="#17251B"),
    tickfont=dict(color="#263D2D"), gridcolor="#D6DFD8",
    linecolor="#87988C", zerolinecolor="#A9B7AD"
)
pio.templates["agrocebada_contraste"] = _agro_theme
pio.templates.default = "agrocebada_contraste"
st.set_page_config(page_title="Datos · AgroCebada 2026", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F2318; }
[data-testid="stSidebar"] * { color: #C8DCC8 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3E2A !important; }
footer { visibility: hidden; }

.page-header {
    background: linear-gradient(135deg, #1B5E20, #2E7D32);
    border-radius: 10px; padding: 18px 26px; margin-bottom: 20px;
    color: white;
}
.page-header h2 { margin:0; font-size:1.4em; color:#fff; }
.page-header p  { margin:3px 0 0; font-size:.8em; color:#A5D6A7; }

.sec { font-size:.76em; font-weight:700; letter-spacing:.07em; color:#1B5E20;
       text-transform:uppercase; border-bottom:2px solid #A5C8A5;
       padding-bottom:3px; margin:20px 0 10px; }

.info-box {
    background:#EAF4EA; border-left:4px solid #2E7D32;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#1B3A1B; line-height:1.6;
}
.warn-box {
    background:#FFF8E1; border-left:4px solid #F9A825;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#4A3A00; line-height:1.6;
}
.placeholder-box {
    background:#F3F4F6; border:2px dashed #9CA3AF;
    border-radius:10px; padding:32px; text-align:center;
    color:#6B7280; font-size:.88em; line-height:1.7;
}

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
    st.page_link("Inicio.py",            label="Inicio")
    st.page_link("pages/1_Datos.py",     label="Datos y procesamiento")
    st.page_link("pages/2_Modelo.py",    label="Modelo predictivo")
    st.page_link("pages/3_Mapa.py",      label="Mapa de parcelas")
    st.divider()

    # Filtros
    st.markdown("**Filtros**")

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("data/Conjunto_datos_EXT_AgroCebada.csv", encoding="latin1")
    df["Municipio"] = df["Municipio"].str.strip()
    df["Cultivo"]   = df["Cultivo"].str.strip().str.title()
    return df

df = load()

# Grupos de columnas
TX = [c for c in df.columns if c.startswith("Tx")]   # T máx
TM = [c for c in df.columns if c.startswith("Tm")]   # T mín
PR = [c for c in df.columns if c.startswith("Pr")]   # Precipitación

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h2> Datos base y procesamiento</h2>
  <p>Exploración del dataset EXT · 197 parcelas · Variables climáticas y topográficas · Hidalgo, México</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "  Descripción general",
    "  Variables climáticas",
    "  Topografía",
    "  Procesamiento",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPCIÓN GENERAL
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total parcelas",  len(df))
    k2.metric("Entrenamiento",   (df["CONJUNTO"]=="ENTRENAMIENTO").sum())
    k3.metric("Predicción",      (df["CONJUNTO"]=="PREDICCION").sum())
    k4.metric("Municipios",      df["Municipio"].nunique())
    k5.metric("Variables climát.",len(TX)+len(TM)+len(PR))
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2], gap="medium")

    with col_l:
        st.markdown('<p class="sec">Distribución por municipio y conjunto</p>', unsafe_allow_html=True)
        mun = (df.groupby(["Municipio","CONJUNTO"])
               .size().reset_index(name="n"))
        fig = px.bar(mun, x="Municipio", y="n", color="CONJUNTO",
                     color_discrete_map={"ENTRENAMIENTO":"#2E7D32","PREDICCION":"#81C784"},
                     barmode="stack",
                     labels={"n":"Parcelas","Municipio":"","CONJUNTO":"Conjunto"})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=320, margin=dict(t=10,b=80,l=10,r=10),
                          xaxis_tickangle=-35,
                          legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="sec">Área por parcela (ha)</p>', unsafe_allow_html=True)
        fig2 = px.histogram(df, x="área_ha", nbins=25,
                            color_discrete_sequence=["#2E7D32"],
                            labels={"área_ha":"Área (ha)","count":"Parcelas"})
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=160, margin=dict(t=5,b=40,l=10,r=10),
                           yaxis_title="Parcelas")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="sec">Resumen de área</p>', unsafe_allow_html=True)
        s = df["área_ha"].describe().round(2).rename({
            "count":"N","mean":"Media","std":"Desv.",
            "min":"Mín","max":"Máx","50%":"Mediana"})
        st.dataframe(s.to_frame("Área (ha)"), use_container_width=True)

    # Tabla completa
    st.markdown('<p class="sec">Vista de datos — primeras columnas</p>', unsafe_allow_html=True)
    base_cols = ["Cultivo","ID_POLIGON","área_ha","Municipio","Estado","CONJUNTO","TpEl_mean","TpPe_mean"]
    st.dataframe(df[base_cols].rename(columns={
        "ID_POLIGON":"ID", "área_ha":"Área (ha)",
        "TpEl_mean":"Elevación (m)", "TpPe_mean":"Pendiente (°)"}),
        use_container_width=True, hide_index=True, height=320)

    csv_b = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Descargar dataset completo (.csv)",
                       csv_b, "datos_ext.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — VARIABLES CLIMÁTICAS
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    # Selector de parcela y variable
    col_sel1, col_sel2, col_sel3 = st.columns([2,2,1])
    with col_sel1:
        municipio_sel = st.selectbox("Municipio",
                                      ["Todos"] + sorted(df["Municipio"].unique()))
    with col_sel2:
        variable_sel = st.selectbox("Variable",
                                     ["Temperatura máxima (Tx)",
                                      "Temperatura mínima (Tm)",
                                      "Precipitación (Pr)"])
    with col_sel3:
        año_sel = st.selectbox("Año referencia", ["2022","2023","2024","2025"])

    df_f = df if municipio_sel == "Todos" else df[df["Municipio"] == municipio_sel]

    # Mapeo de variable
    VAR_MAP = {
        "Temperatura máxima (Tx)": TX,
        "Temperatura mínima (Tm)": TM,
        "Precipitación (Pr)":      PR,
    }
    var_cols = VAR_MAP[variable_sel]

    # Filtrar columnas del año seleccionado (ej: Tx2201..Tx2212)
    anio_code = año_sel[2:]  # "22", "23", "24", "25"
    cols_año  = [c for c in var_cols if c[2:4] == anio_code]

    if cols_año:
        meses = ["Ene","Feb","Mar","Abr","May","Jun",
                 "Jul","Ago","Sep","Oct","Nov","Dic"][:len(cols_año)]

        # Boxplot por mes
        st.markdown(f'<p class="sec">{variable_sel} · {año_sel} · {municipio_sel}</p>',
                    unsafe_allow_html=True)
        melt = (df_f[["ID_POLIGON"] + cols_año]
                .melt(id_vars="ID_POLIGON", var_name="col", value_name="valor"))
        melt["Mes"] = melt["col"].map(dict(zip(cols_año, meses)))

        fig_box = px.box(melt, x="Mes", y="valor",
                         color_discrete_sequence=["#2E7D32"],
                         labels={"valor": variable_sel, "Mes": ""},
                         category_orders={"Mes": meses})
        fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=320, margin=dict(t=10,b=40,l=10,r=10))
        st.plotly_chart(fig_box, use_container_width=True)

        # Perfil temporal medio
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown('<p class="sec">Promedio mensual por conjunto</p>', unsafe_allow_html=True)
            media_conj = (df_f.groupby("CONJUNTO")[cols_año]
                          .mean().T.reset_index())
            media_conj["Mes"] = meses
            fig_line = px.line(media_conj, x="Mes", y=["ENTRENAMIENTO","PREDICCION"],
                               color_discrete_map={"ENTRENAMIENTO":"#1B5E20",
                                                   "PREDICCION":"#81C784"},
                               labels={"value": variable_sel, "variable":"Conjunto"})
            fig_line.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                    height=260, margin=dict(t=10,b=40,l=10,r=10),
                                    legend=dict(orientation="h",y=1.05))
            fig_line.update_traces(mode="lines+markers")
            st.plotly_chart(fig_line, use_container_width=True)

        with col_b:
            st.markdown('<p class="sec">Estadísticos descriptivos (año seleccionado)</p>',
                        unsafe_allow_html=True)
            stats = df_f[cols_año].describe().round(2)
            stats.index.name = ""
            stats.columns = meses
            st.dataframe(stats, use_container_width=True, height=255)

    st.markdown('<p class="sec">Comparación entre años — promedio de todos los meses</p>',
                unsafe_allow_html=True)
    resumen_años = {}
    for yr in ["22","23","24","25"]:
        cols_y  = [c for c in var_cols if c[2:4] == yr]
        if cols_y:
            resumen_años[f"20{yr}"] = df_f[cols_y].mean(axis=1).mean()
    if resumen_años:
        fig_yr = px.bar(x=list(resumen_años.keys()),
                        y=list(resumen_años.values()),
                        color_discrete_sequence=["#2E7D32"],
                        labels={"x":"Año","y":variable_sel},
                        text_auto=".2f")
        fig_yr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=240, margin=dict(t=10,b=30,l=10,r=10))
        st.plotly_chart(fig_yr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOPOGRAFÍA
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    col_e, col_p = st.columns(2, gap="medium")

    with col_e:
        st.markdown('<p class="sec">Distribución de elevación (m s.n.m.)</p>',
                    unsafe_allow_html=True)
        fig_el = px.histogram(df, x="TpEl_mean", nbins=30,
                              color_discrete_sequence=["#33691E"],
                              labels={"TpEl_mean":"Elevación (m)","count":"Parcelas"})
        fig_el.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=280, margin=dict(t=10,b=40,l=10,r=10))
        st.plotly_chart(fig_el, use_container_width=True)

        st.markdown('<p class="sec">Elevación por municipio</p>', unsafe_allow_html=True)
        fig_el2 = px.box(df, x="Municipio", y="TpEl_mean",
                         color_discrete_sequence=["#33691E"],
                         labels={"TpEl_mean":"Elevación (m)","Municipio":""})
        fig_el2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=280, margin=dict(t=10,b=80,l=10,r=10),
                               xaxis_tickangle=-35)
        st.plotly_chart(fig_el2, use_container_width=True)

    with col_p:
        st.markdown('<p class="sec">Distribución de pendiente (grados)</p>',
                    unsafe_allow_html=True)
        fig_pe = px.histogram(df, x="TpPe_mean", nbins=30,
                              color_discrete_sequence=["#558B2F"],
                              labels={"TpPe_mean":"Pendiente (°)","count":"Parcelas"})
        fig_pe.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=280, margin=dict(t=10,b=40,l=10,r=10))
        st.plotly_chart(fig_pe, use_container_width=True)

        st.markdown('<p class="sec">Elevación vs Pendiente</p>', unsafe_allow_html=True)
        fig_sc = px.scatter(df, x="TpEl_mean", y="TpPe_mean",
                            color="Municipio", opacity=.75,
                            labels={"TpEl_mean":"Elevación (m)",
                                    "TpPe_mean":"Pendiente (°)",
                                    "Municipio":"Municipio"})
        fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=280, margin=dict(t=10,b=40,l=10,r=10),
                              legend=dict(font_size=9, x=1))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Tabla
    st.markdown('<p class="sec">Resumen topográfico por municipio</p>', unsafe_allow_html=True)
    topo_sum = (df.groupby("Municipio")
                .agg(Parcelas=("ID_POLIGON","count"),
                     Elev_media=("TpEl_mean","mean"),
                     Elev_min=("TpEl_mean","min"),
                     Elev_max=("TpEl_mean","max"),
                     Pend_media=("TpPe_mean","mean"))
                .round(1).reset_index())
    st.dataframe(topo_sum, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROCESAMIENTO (espacio para redactar)
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("""
    <div class="info-box">
    ℹ  <strong>Sección editable</strong> — Este tab está diseñado para que agregues
    la descripción metodológica del procesamiento de datos. Cada bloque de expansión
    a continuación es un espacio marcado con el contenido sugerido.
    </div>
    """, unsafe_allow_html=True)

    # ── Bloque 1: Fuentes ─────────────────────────────────────────────────────
    with st.expander("  1. Fuentes de datos", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### Variables climáticas — CHIRPS")
            st.markdown("""
| Variable | Prefijo CSV | Años | Unidad |
|---|---|---|---|
| Temperatura máxima | `Tx____mea` | 2022–2025 | °C |
| Temperatura mínima | `Tm____mea` | 2022–2025 | °C |
| Precipitación | `Pr____mea` | 2022–2025 | mm |

**Resolución temporal:** mensual  
**Resolución espacial:** ~5 km (recortada a límite de parcela)  
**Fuente:** [CHIRPS — Climate Hazards Group](https://www.chc.ucsb.edu/data/chirps)
            """)
        with col2:
            st.markdown("#### Variables topográficas — INEGI CEM4")
            st.markdown("""
| Variable | Columna CSV | Unidad |
|---|---|---|
| Elevación media | `TpEl_mean` | m s.n.m. |
| Pendiente media | `TpPe_mean` | grados |

**Resolución:** 120 m (CEM v4.0)  
**Fuente:** [INEGI — Continuo de Elevaciones Mexicano](https://www.inegi.org.mx/temas/relieve/continental/)
            """)
        st.markdown("---")
        st.markdown("""
        <div class="warn-box">
          <strong>Completa aquí:</strong> Describe cómo extrajiste las variables climáticas
        sobre cada polígono (herramienta usada, método de extracción, proyección CRS, etc.).
        </div>
        """, unsafe_allow_html=True)

    # ── Bloque 2: Filtrado ────────────────────────────────────────────────────
    with st.expander("  2. Filtrado y limpieza de datos"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Criterios de filtrado aplicados")
            st.markdown("""
- **Nubosidad:** imágenes con > 30 % excluidas del cálculo de índices espectrales
- **Valores atípicos:** ± 3 σ por variable y mes
- **Parcelas incompletas:** imputación por mediana del municipio correspondiente
            """)
        with col2:
            # Mini gráfica de nulos simulada
            st.markdown("#### Distribución de valores faltantes por grupo")
            null_sim = pd.DataFrame({
                "Grupo": ["Tx (T máx)","Tm (T mín)","Pr (Precip.)","Topografía"],
                "% nulos": [0.0, 0.0, 0.2, 0.0]
            })
            fig_n = px.bar(null_sim, x="Grupo", y="% nulos",
                           color_discrete_sequence=["#C62828"],
                           text_auto=True,
                           labels={"% nulos":"% valores faltantes"})
            fig_n.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=200, margin=dict(t=5,b=30,l=10,r=10))
            st.plotly_chart(fig_n, use_container_width=True)

        st.markdown("""
        <div class="warn-box">
          <strong>Completa aquí:</strong> Describe el proceso de limpieza,
        decisiones sobre outliers, tratamiento de nulos y criterios de exclusión de parcelas.
        </div>
        """, unsafe_allow_html=True)

    # ── Bloque 3: Ingeniería de variables ─────────────────────────────────────
    with st.expander("  3. Ingeniería de variables"):
        st.markdown("""
#### Ventanas fenológicas del ciclo de cebada

El ciclo de cebada en el altiplano hidalguense sigue las siguientes etapas:

| Etapa | Meses | Variables derivadas |
|---|---|---|
| Emergencia | Ene–Feb | Media, máximo y desviación de Tx, Tm, Pr |
| Macollamiento | Mar | Media y máximo de Tx, Tm, Pr |
| Encañado–Espigado ⭐ | Abr–May | Media, máximo y desviación de Tx, Tm, Pr |
| Llenado de grano | Jun | Media y máximo de Tx, Tm |
| Dinámicas | — | Δ temperatura espigado–emergencia · Integral precipitación Ene–Jun |

⭐ La ventana de espigado concentra las variables con mayor correlación con el rendimiento.
        """)
        st.markdown("""
        <div class="warn-box">
          <strong>Completa aquí:</strong> Agrega las variables derivadas específicas
        que construiste, fórmulas utilizadas y justificación de cada decisión.
        </div>
        """, unsafe_allow_html=True)

    # ── Bloque 4: Normalización ───────────────────────────────────────────────
    with st.expander("  4. Normalización y preparación final"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
#### Estrategia de normalización

- **StandardScaler** (media 0, varianza 1) ajustado **solo** sobre el conjunto de entrenamiento
- Aplicado como *transform-only* al conjunto de predicción
- Imputación por **mediana** calculada en entrenamiento para valores faltantes residuales
            """)
        with col2:
            st.markdown("""
#### Dimensiones finales del dataset

| Conjunto | Parcelas | Variables |
|---|---|---|
| Entrenamiento | 138 | ~120 |
| Predicción | 59 | ~120 |
| **Total** | **197** | **~120** |
            """)
        st.markdown("""
        <div class="warn-box">
          <strong>Completa aquí:</strong> Detalla la normalización exacta que aplicaste,
        el número final de variables después de la ingeniería y cualquier
        transformación adicional (PCA, selección de features, etc.).
        </div>
        """, unsafe_allow_html=True)

    # ── Bloque 5: libre ───────────────────────────────────────────────────────
    with st.expander("  5. Notas adicionales del procesamiento"):
        st.markdown("""
        <div class="placeholder-box">
          Espacio libre para agregar notas, diagramas de flujo, referencias bibliográficas
        o cualquier aspecto metodológico adicional del procesamiento de datos.<br><br>
        Puedes reemplazar este bloque con texto, tablas, imágenes o nuevas gráficas
        de Plotly según lo que necesites documentar.
        </div>
        """, unsafe_allow_html=True)
