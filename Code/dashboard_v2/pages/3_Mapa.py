"""
Página 3 — Mapa interactivo de parcelas tipo DENUE/INEGI
Boceto con datos de ejemplo. Reemplaza los datos simulados con resultados
reales del pipeline cuando estén disponibles.
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
st.set_page_config(page_title="Mapa · AgroCebada 2026", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F2318; }
[data-testid="stSidebar"] * { color: #C8DCC8 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3E2A !important; }
footer { visibility: hidden; }

.page-header {
    background: linear-gradient(135deg, #4A148C, #6A1B9A);
    border-radius: 10px; padding: 18px 26px; margin-bottom: 20px;
    color: white;
}
.page-header h2 { margin:0; font-size:1.4em; color:#fff; }
.page-header p  { margin:3px 0 0; font-size:.8em; color:#CE93D8; }

.sec { font-size:.76em; font-weight:700; letter-spacing:.07em; color:#4A148C;
       text-transform:uppercase; border-bottom:2px solid #CE93D8;
       padding-bottom:3px; margin:20px 0 10px; }

.demo-badge {
    display:inline-block; background:#FF6F00; color:white;
    padding:3px 10px; border-radius:12px; font-size:.72em;
    font-weight:700; letter-spacing:.05em; margin-bottom:8px;
}
.info-box {
    background:#F3E5F5; border-left:4px solid #6A1B9A;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#1A0033; line-height:1.6;
}

/* Panel de detalle de parcela */
.detail-card {
    background: #FAFAFA;
    border: 1.5px solid #CE93D8;
    border-radius: 12px;
    padding: 18px 20px;
}
.detail-card h4 { margin: 0 0 10px; font-size:1.0em; color:#4A148C; }
.cond-pill {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:.85em; font-weight:700; margin-bottom:12px;
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
    st.markdown("**Controles del mapa**")

# ── Carga de coordenadas reales + datos simulados ─────────────────────────────
@st.cache_data
def load_coords():
    df = pd.read_csv("data/Coordenadas_Parcelas_Dashboard.csv", encoding="latin1")
    return df

@st.cache_data
def load_ext():
    df = pd.read_csv("data/Conjunto_datos_EXT_AgroCebada.csv", encoding="latin1")
    df["Municipio"] = df["Municipio"].str.strip()
    df = df.rename(columns={"ID_POLIGON":"ID_POLIGONO"})
    return df[["ID_POLIGONO","área_ha","Municipio","CONJUNTO","TpEl_mean","TpPe_mean"]]

@st.cache_data
def build_demo_data():
    """
    Genera datos de predicción simulados sobre las coordenadas reales.
    REEMPLAZAR con resultados de pipeline.py cuando estén disponibles.
    """
    coords = load_coords()
    ext    = load_ext()
    df     = coords.merge(ext, on="ID_POLIGONO", how="left")

    # Semilla fija para reproducibilidad del boceto
    rng = np.random.default_rng(2026)
    n   = len(df)

    # Rendimiento predicho simulado (distribución cercana a la real: μ≈3.96, σ≈0.85)
    df["rendimiento_predicho"] = np.clip(
        rng.normal(3.96, 0.80, n), 1.5, 5.9
    ).round(3)

    # Rendimiento real solo para entrenamiento
    df["rendimiento_real"] = np.where(
        df["CONJUNTO"] == "ENTRENAMIENTO",
        np.clip(df["rendimiento_predicho"] + rng.normal(0, 0.35, n), 1.5, 5.9),
        np.nan
    ).round(3)

    df["error_absoluto"] = np.where(
        df["CONJUNTO"] == "ENTRENAMIENTO",
        np.abs(df["rendimiento_real"] - df["rendimiento_predicho"]),
        np.nan
    ).round(3)

    # Condición semáforo
    def cond(v):
        if v >= 4.3: return "Bueno"
        if v >= 3.5: return "Alerta 1"
        if v >= 2.8: return "Alerta 2"
        return "Emergencia"

    df["condicion"] = df["rendimiento_predicho"].apply(cond)

    # Texto del hover completo
    df["hover"] = df.apply(lambda r: (
        f"<b>{r['ID_POLIGONO']}</b><br>"
        f"Municipio: {r.get('Municipio','—')}<br>"
        f"Conjunto: {r['CONJUNTO']}<br>"
        f"Área: {r.get('área_ha','—')} ha<br>"
        f"Elevación: {r.get('TpEl_mean','—'):.0f} m<br>"
        f"Pendiente: {r.get('TpPe_mean','—'):.2f}°<br>"
        f"──────────────<br>"
        f"<b>Rendimiento predicho: {r['rendimiento_predicho']:.3f} t/ha</b><br>"
        + (f"Rendimiento real: {r['rendimiento_real']:.3f} t/ha<br>"
           f"Error absoluto: {r['error_absoluto']:.3f} t/ha"
           if not np.isnan(r['rendimiento_real']) else
           "Rendimiento real: no disponible")
    ), axis=1)

    return df

# ── Intento de cargar resultados reales ───────────────────────────────────────
@st.cache_data
def try_real_results():
    try:
        coords = load_coords()
        res    = pd.read_csv("data/resultados_modelo.csv")
        res    = res.rename(columns={"ID_POLIGONO":"ID_POLIGONO_RES"})
        ext    = load_ext()
        coords = coords.merge(ext, on="ID_POLIGONO", how="left")
        df = coords.merge(res, left_on="ID_POLIGONO",
                          right_on="ID_POLIGONO_RES", how="left")
        return df, True
    except Exception:
        return build_demo_data(), False

df_map, es_real = try_real_results()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h2> Mapa de parcelas</h2>
  <p>Visualización geoespacial de las 197 parcelas · Estado de Hidalgo, México</p>
</div>
""", unsafe_allow_html=True)

# Banner de estado de datos
if not es_real:
    st.markdown("""
    <div class="info-box">
    <span class="demo-badge"> BOCETO</span><br>
    El mapa usa <strong>coordenadas reales</strong> del archivo
    <code>Coordenadas_Parcelas_Dashboard.csv</code>, pero los
    <strong>valores de rendimiento son simulados</strong> para mostrar
    el comportamiento interactivo. Ejecuta <code>python pipeline.py</code> y
    aparecerán los resultados reales automáticamente.
    </div>
    """, unsafe_allow_html=True)
else:
    st.success("  Mapa usando resultados reales de `data/resultados_modelo.csv`.")

# ── Sidebar — controles ───────────────────────────────────────────────────────
with st.sidebar:
    # Filtro: conjunto
    conj_opts = ["Todos"] + sorted(df_map["CONJUNTO"].unique().tolist())
    conj_sel  = st.selectbox("Conjunto", conj_opts)

    # Filtro: condición
    cond_opts = sorted(df_map["condicion"].unique().tolist())
    cond_sel  = st.multiselect("Condición", cond_opts, default=cond_opts)

    # Variable de color
    color_var = st.radio("Colorear por",
                         ["Condición (semáforo)", "Rendimiento predicho (gradiente)"])

    # Tamaño del marcador
    marker_sz = st.slider("Tamaño de marcadores", 6, 20, 10)

    st.divider()
    st.markdown("**Mapa base**")
    mapa_base = st.selectbox("Estilo",
                              ["carto-positron",
                               "open-street-map",
                               "stamen-terrain",
                               "carto-darkmatter"])

# ── Filtrado ──────────────────────────────────────────────────────────────────
df_f = df_map.copy()
if conj_sel != "Todos":
    df_f = df_f[df_f["CONJUNTO"] == conj_sel]
df_f = df_f[df_f["condicion"].isin(cond_sel)]

# ── Layout: mapa + panel lateral ──────────────────────────────────────────────
col_map, col_det = st.columns([4, 1], gap="medium")

COLORES = {
    "Bueno":      "#27AE60",
    "Alerta 1":   "#F1C40F",
    "Alerta 2":   "#E67E22",
    "Emergencia": "#C0392B",
}

with col_map:
    st.markdown('<p class="sec">Haz clic en una parcela para ver el detalle</p>',
                unsafe_allow_html=True)

    if color_var == "Condición (semáforo)":
        fig = px.scatter_mapbox(
            df_f,
            lat="Latitud", lon="Longitud",
            color="condicion",
            color_discrete_map=COLORES,
            hover_name="ID_POLIGONO",
            custom_data=["ID_POLIGONO","CONJUNTO","rendimiento_predicho",
                         "rendimiento_real","error_absoluto","condicion",
                         "área_ha","Municipio","TpEl_mean","TpPe_mean"],
            size_max=marker_sz,
            zoom=10,
            center={"lat": df_f["Latitud"].mean(), "lon": df_f["Longitud"].mean()},
            mapbox_style=mapa_base,
            height=560,
        )
    else:
        fig = px.scatter_mapbox(
            df_f,
            lat="Latitud", lon="Longitud",
            color="rendimiento_predicho",
            color_continuous_scale=[
                (0.00, "#C0392B"),
                (0.35, "#E67E22"),
                (0.55, "#F1C40F"),
                (1.00, "#27AE60"),
            ],
            range_color=[df_f["rendimiento_predicho"].min(),
                         df_f["rendimiento_predicho"].max()],
            hover_name="ID_POLIGONO",
            custom_data=["ID_POLIGONO","CONJUNTO","rendimiento_predicho",
                         "rendimiento_real","error_absoluto","condicion",
                         "área_ha","Municipio","TpEl_mean","TpPe_mean"],
            zoom=10,
            center={"lat": df_f["Latitud"].mean(), "lon": df_f["Longitud"].mean()},
            mapbox_style=mapa_base,
            height=560,
        )

    # Personalizar hover
    fig.update_traces(
        marker=dict(size=marker_sz, opacity=0.88),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Municipio: %{customdata[7]}<br>"
            "Conjunto: %{customdata[1]}<br>"
            "Área: %{customdata[6]:.2f} ha<br>"
            "──────────────<br>"
            "<b>Rendimiento predicho: %{customdata[2]:.3f} t/ha</b><br>"
            "Rendimiento real: %{customdata[3]:.3f} t/ha<br>"
            "Error absoluto: %{customdata[4]:.3f} t/ha<br>"
            "Condición: %{customdata[5]}<br>"
            "──────────────<br>"
            "Elevación: %{customdata[8]:.0f} m<br>"
            "Pendiente: %{customdata[9]:.2f}°"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(
            title="Condición",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            font_size=12, color="#17251B",
            x=0.01, y=0.99,
        ),
        coloraxis_colorbar=dict(title="t/ha", len=0.6, y=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Selector de parcela para el panel de detalle
    st.markdown('<p class="sec">Seleccionar parcela manualmente</p>', unsafe_allow_html=True)
    parcela_sel = st.selectbox(
        "Parcela",
        options=df_f["ID_POLIGONO"].tolist(),
        help="También puedes pasar el cursor sobre el mapa para ver el detalle en el tooltip"
    )

with col_det:
    st.markdown('<p class="sec">Detalle</p>', unsafe_allow_html=True)

    if parcela_sel:
        row = df_f[df_f["ID_POLIGONO"] == parcela_sel].iloc[0]

        # Color de la condición
        cond_colors = {"Bueno":"#27AE60","Alerta 1":"#F1C40F",
                       "Alerta 2":"#E67E22","Emergencia":"#C0392B"}
        bg  = cond_colors.get(row["condicion"], "#95A5A6")
        txt = "#000" if row["condicion"] == "Alerta 1" else "#fff"

        st.markdown(f"""
        <div class="detail-card">
          <h4> {row['ID_POLIGONO']}</h4>
          <span class="cond-pill" style="background:{bg};color:{txt}">
            {row['condicion']}
          </span><br>
          <small><b>Municipio:</b> {row.get('Municipio','—')}</small><br>
          <small><b>Conjunto:</b> {row['CONJUNTO']}</small><br>
          <small><b>Área:</b> {row.get('área_ha','—'):.2f} ha</small>
          <hr style="margin:10px 0">
          <small><b>Lat:</b> {row['Latitud']:.6f}</small><br>
          <small><b>Lon:</b> {row['Longitud']:.6f}</small>
          <hr style="margin:10px 0">
          <b style="font-size:.95em"> Rendimiento</b><br>
          <big style="color:{bg};font-weight:800">{row['rendimiento_predicho']:.3f} t/ha</big>
          <p style="font-size:.75em;margin:2px 0">predicho</p>
        """, unsafe_allow_html=True)

        if not np.isnan(row["rendimiento_real"]):
            st.markdown(f"""
          <small><b>Real:</b> {row['rendimiento_real']:.3f} t/ha</small><br>
          <small><b>Error abs.:</b> {row['error_absoluto']:.3f} t/ha</small>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
          <small style="color:#888">Rendimiento real: no disponible</small>
            """, unsafe_allow_html=True)

        st.markdown("""
          <hr style="margin:10px 0">
          <b style="font-size:.85em"> Topografía</b>
        """, unsafe_allow_html=True)
        st.markdown(f"""
          <small><b>Elevación:</b> {row.get('TpEl_mean',0):.0f} m</small><br>
          <small><b>Pendiente:</b> {row.get('TpPe_mean',0):.2f}°</small>
        </div>
        """, unsafe_allow_html=True)

        # Mini indicador de rendimiento
        st.markdown("<br>", unsafe_allow_html=True)
        rend_v = row["rendimiento_predicho"]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rend_v,
            number={"suffix":" t/ha","font":{"size":16,"color":"#17251B"}},
            gauge={
                "axis":{"range":[1.5, 6.0],"tickwidth":1,"tickfont_size":8},
                "bar":{"color":bg},
                "steps":[
                    {"range":[1.5, 2.8], "color":"#FFEBEE"},
                    {"range":[2.8, 3.5], "color":"#FFF3E0"},
                    {"range":[3.5, 4.3], "color":"#FFFDE7"},
                    {"range":[4.3, 6.0], "color":"#E8F5E9"},
                ],
                "threshold":{"line":{"color":"#1B5E20","width":2},
                             "thickness":.75,"value":3.96},
            },
            domain={"x":[0,1],"y":[0,1]}
        ))
        fig_g.update_layout(height=160, margin=dict(t=10,b=10,l=10,r=10),
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

# ── Estadísticos del mapa ─────────────────────────────────────────────────────
st.divider()
col_s1, col_s2, col_s3 = st.columns(3, gap="medium")

with col_s1:
    st.markdown('<p class="sec">Distribución por condición (vista filtrada)</p>',
                unsafe_allow_html=True)
    cnt = df_f["condicion"].value_counts().reset_index()
    cnt.columns = ["Condición","Parcelas"]
    cnt["%"] = (cnt["Parcelas"]/cnt["Parcelas"].sum()*100).round(1)
    st.dataframe(cnt, hide_index=True, use_container_width=True,
                 column_config={"%": st.column_config.NumberColumn(format="%.1f %%")})

with col_s2:
    st.markdown('<p class="sec">Rendimiento predicho por municipio</p>',
                unsafe_allow_html=True)
    if "Municipio" in df_f.columns:
        mun_g = (df_f.groupby("Municipio")["rendimiento_predicho"]
                 .mean().sort_values(ascending=False).reset_index())
        fig_mun = px.bar(mun_g, x="rendimiento_predicho", y="Municipio",
                         orientation="h",
                         color="rendimiento_predicho",
                         color_continuous_scale=["#C0392B","#F1C40F","#27AE60"],
                         range_color=[2.5,5.0],
                         text_auto=".2f",
                         labels={"rendimiento_predicho":"Prom. predicho (t/ha)",
                                 "Municipio":""})
        fig_mun.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=260, margin=dict(t=5,b=10,l=10,r=10),
                               coloraxis_showscale=False,
                               yaxis=dict(tickfont_size=9))
        st.plotly_chart(fig_mun, use_container_width=True)

with col_s3:
    st.markdown('<p class="sec">Histograma de rendimiento predicho</p>',
                unsafe_allow_html=True)
    fig_hist = px.histogram(df_f, x="rendimiento_predicho", nbins=20,
                             color="condicion", color_discrete_map=COLORES,
                             opacity=.82, barmode="overlay",
                             labels={"rendimiento_predicho":"Rendimiento predicho (t/ha)",
                                     "condicion":"Condición"})
    fig_hist.add_vline(x=df_f["rendimiento_predicho"].mean(),
                       line_dash="dash", line_color="#37474F", line_width=1.5,
                       annotation_text=f"Media {df_f['rendimiento_predicho'].mean():.2f}",
                       annotation_font_size=9)
    fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            height=260, margin=dict(t=10,b=40,l=10,r=10),
                            showlegend=False,
                            yaxis_title="Parcelas")
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tabla completa ────────────────────────────────────────────────────────────
with st.expander("  Tabla de todas las parcelas visibles"):
    cols_tab = ["ID_POLIGONO","Municipio","CONJUNTO","área_ha",
                "Latitud","Longitud","rendimiento_predicho",
                "rendimiento_real","condicion"]
    cols_tab = [c for c in cols_tab if c in df_f.columns]
    st.dataframe(df_f[cols_tab].round(3),
                 hide_index=True, use_container_width=True, height=350,
                 column_config={
                     "rendimiento_predicho": st.column_config.NumberColumn(
                         "Predicho (t/ha)", format="%.3f"),
                     "rendimiento_real": st.column_config.NumberColumn(
                         "Real (t/ha)", format="%.3f"),
                     "área_ha": st.column_config.NumberColumn(
                         "Área (ha)", format="%.2f"),
                 })
    csv_b = df_f[cols_tab].to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Descargar datos del mapa (.csv)",
                       csv_b, "mapa_parcelas.csv", "text/csv")
