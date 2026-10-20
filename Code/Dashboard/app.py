"""
app.py — Dashboard interactivo de predicción de rendimiento de cebada.
Reto AgroCebada 2026 | Estado de Hidalgo, México.

Uso:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroCebada 2026",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores ──────────────────────────────────────────────────────────
C = {
    "Bueno":      "#27AE60",
    "Alerta 1":   "#F1C40F",
    "Alerta 2":   "#E67E22",
    "Emergencia": "#C0392B",
    "Sin datos":  "#95A5A6",
    "primary":    "#1B5E20",   # verde campo oscuro
    "bg_card":    "#F0F4F0",
    "text_dark":  "#1A2B1A",
    "rule":       "#A5C8A5",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: #132313;
    border-right: 1px solid #2D4A2D;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: #C8DCC8 !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background-color: #1E361E !important;
    border-color: #2D4A2D !important;
    color: #E0EEE0 !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: #2D4A2D !important;
}}

/* ── Encabezado ───────────────────────────────────────────────────────── */
.app-header {{
    background: linear-gradient(135deg, {C['primary']} 0%, #2E7D32 60%, #388E3C 100%);
    padding: 20px 28px 16px;
    border-radius: 12px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 16px;
}}
.app-header-icon {{ font-size: 2.6em; line-height: 1; }}
.app-header-text h1 {{
    margin: 0;
    color: #FFFFFF;
    font-size: 1.55em;
    font-weight: 700;
    letter-spacing: -0.3px;
}}
.app-header-text p {{
    margin: 3px 0 0;
    color: #A5D6A7;
    font-size: 0.82em;
}}

/* ── KPI cards ────────────────────────────────────────────────────────── */
.kpi {{
    border-radius: 10px;
    padding: 16px 12px 14px;
    text-align: center;
    color: #fff;
    box-shadow: 0 2px 10px rgba(0,0,0,.18);
    min-height: 108px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
}}
.kpi-value {{
    font-size: 2.05em;
    font-weight: 800;
    line-height: 1;
    margin: 0;
    letter-spacing: -1px;
}}
.kpi-label {{
    font-size: 0.70em;
    opacity: 0.92;
    line-height: 1.35;
    margin: 0;
}}

/* ── Rótulos de sección ───────────────────────────────────────────────── */
.sec-label {{
    font-size: 0.78em;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: {C['primary']};
    text-transform: uppercase;
    border-bottom: 2px solid {C['rule']};
    padding-bottom: 4px;
    margin: 18px 0 10px;
}}

/* ── Nota al pie ──────────────────────────────────────────────────────── */
.app-footer {{
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid #D4E4D4;
    font-size: 0.72em;
    color: #7A9A7A;
    text-align: center;
}}

/* ── Ocultar "Made with Streamlit" ───────────────────────────────────── */
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    p = Path("data/resultados_modelo.csv")
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_importance():
    p = Path("data/importancia_features.csv")
    return pd.read_csv(p) if p.exists() else None


df_all = load_results()
imp_df = load_importance()

if df_all is None:
    st.error("### ⚠️  Archivo de resultados no encontrado")
    st.markdown("""
    El dashboard necesita el archivo `data/resultados_modelo.csv`.  
    Genera los datos ejecutando el pipeline primero:
    ```bash
    python pipeline.py
    ```
    Luego recarga esta página.
    """)
    st.stop()


# ── Sidebar — Filtros ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgroCebada 2026")
    st.markdown("*Panel de control*")
    st.divider()

    # Conjunto de datos
    conjuntos   = ["Todos"] + sorted(df_all["CONJUNTO"].unique().tolist())
    conjunto_sel = st.selectbox("Conjunto", conjuntos,
                                 help="Entrenamiento: rendimiento real conocido. "
                                      "Predicción: solo estimación del modelo.")

    # Condición semáforo
    cond_opts = sorted(df_all["condicion"].unique().tolist())
    cond_sel  = st.multiselect("Condición", cond_opts, default=cond_opts)

    # Rango de rendimiento predicho
    rmin = float(df_all["rendimiento_predicho"].min())
    rmax = float(df_all["rendimiento_predicho"].max())
    rango = st.slider("Rendimiento predicho (t/ha)",
                       min_value=round(rmin, 1),
                       max_value=round(rmax, 1),
                       value=(round(rmin, 1), round(rmax, 1)),
                       step=0.1)

    st.divider()

    # Contador de parcelas visibles
    _df_tmp = df_all.copy()
    if conjunto_sel != "Todos":
        _df_tmp = _df_tmp[_df_tmp["CONJUNTO"] == conjunto_sel]
    _df_tmp = _df_tmp[
        _df_tmp["condicion"].isin(cond_sel) &
        (_df_tmp["rendimiento_predicho"] >= rango[0]) &
        (_df_tmp["rendimiento_predicho"] <= rango[1])
    ]
    st.metric("Parcelas visibles", f"{len(_df_tmp)} / {len(df_all)}")

    if st.button("↺ Restablecer filtros", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("<p style='font-size:0.72em;opacity:.6'>v1.0 · Reto AgroCebada 2026</p>",
                unsafe_allow_html=True)


# ── Filtrado reactivo ──────────────────────────────────────────────────────────
df = df_all.copy()
if conjunto_sel != "Todos":
    df = df[df["CONJUNTO"] == conjunto_sel]
df = df[
    df["condicion"].isin(cond_sel) &
    (df["rendimiento_predicho"] >= rango[0]) &
    (df["rendimiento_predicho"] <= rango[1])
]


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-header-icon">🌾</div>
  <div class="app-header-text">
    <h1>Predicción de Rendimiento de Cebada</h1>
    <p>Reto AgroCebada 2026 · Estado de Hidalgo, México · 
       Fuentes: Sentinel-2, Planet PRO, CHIRPS, INEGI</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_dash, tab_modelo, tab_datos = st.tabs(
    ["📊  Dashboard", "🤖  Modelo predictivo", "📋  Datos completos"]
)


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # ── KPI cards ──────────────────────────────────────────────────────────────
    n_vis       = len(df)
    n_b         = (df["condicion"] == "Bueno").sum()
    n_a1        = (df["condicion"] == "Alerta 1").sum()
    n_a2        = (df["condicion"] == "Alerta 2").sum()
    n_em        = (df["condicion"] == "Emergencia").sum()
    rend_mean   = df["rendimiento_predicho"].mean()
    pct_vis     = 100 * n_vis / len(df_all)

    def kpi(col, val, lbl, color, fmt="{:.0f}"):
        col.markdown(
            f'<div class="kpi" style="background:{color}">'
            f'<p class="kpi-value">{fmt.format(val)}</p>'
            f'<p class="kpi-label">{lbl}</p></div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, pct_vis,   "% parcelas<br>visualizadas",       "#1B5E20", "{:.1f}%")
    kpi(c2, n_b,       "Parcelas<br>Buenas ≥4.3 t/ha",     C["Bueno"])
    kpi(c3, n_a1,      "Parcelas<br>Alerta 1",              C["Alerta 1"])
    kpi(c4, n_a2,      "Parcelas<br>Alerta 2",              C["Alerta 2"])
    kpi(c5, n_em,      "Parcelas<br>Emergencia",            C["Emergencia"])
    kpi(c6, rend_mean, "Rendimiento prom.<br>predicho t/ha","#0D47A1", "{:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Histograma + Donut ──────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:
        st.markdown('<p class="sec-label">Distribución de rendimiento predicho</p>',
                    unsafe_allow_html=True)

        fig_hist = px.histogram(
            df, x="rendimiento_predicho", color="condicion",
            nbins=28, color_discrete_map=C, opacity=0.82,
            barmode="overlay",
            labels={"rendimiento_predicho": "Rendimiento predicho (t/ha)",
                    "condicion": "Condición"},
        )
        for umbral, etiq, col_ in [
            (4.3, "Bueno", C["Bueno"]),
            (3.5, "Alerta 1", C["Alerta 1"]),
            (2.8, "Alerta 2", C["Alerta 2"]),
        ]:
            fig_hist.add_vline(x=umbral, line_dash="dot", line_color=col_,
                               line_width=1.5,
                               annotation_text=etiq,
                               annotation_font_size=9,
                               annotation_font_color=col_,
                               annotation_position="top")
        fig_hist.add_vline(
            x=rend_mean, line_dash="dash", line_color="#37474F", line_width=2,
            annotation_text=f"Media {rend_mean:.2f}",
            annotation_font_size=10, annotation_position="top right",
        )
        fig_hist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=330, margin=dict(t=30, b=40, l=10, r=10),
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right",
                        font_size=10),
            xaxis=dict(showgrid=True, gridcolor="#E8F5E9"),
            yaxis=dict(showgrid=True, gridcolor="#E8F5E9", title="Nº parcelas"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        st.markdown('<p class="sec-label">Condición de parcelas</p>',
                    unsafe_allow_html=True)

        cnt = df["condicion"].value_counts().reset_index()
        cnt.columns = ["condicion", "n"]

        fig_pie = px.pie(
            cnt, names="condicion", values="n",
            color="condicion", color_discrete_map=C, hole=0.44,
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont_size=11,
            marker=dict(line=dict(color="white", width=2)),
        )
        fig_pie.update_layout(
            showlegend=False, paper_bgcolor="white",
            height=330, margin=dict(t=20, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Mini tabla de condición
        tbl = (df.groupby("condicion")
               .agg(Parcelas=("ID_POLIGONO", "count"),
                    Prom=("rendimiento_predicho", "mean"))
               .round({"Prom": 2})
               .reset_index())
        tbl["%"] = (tbl["Parcelas"] / tbl["Parcelas"].sum() * 100).round(1)
        tbl.columns = ["Condición", "Parcelas", "Prom (t/ha)", "%"]
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=175,
                     column_config={
                         "%": st.column_config.NumberColumn(format="%.1f %%")
                     })

    # ── Scatter real vs predicho + Barras por parcela ─────────────────────────
    col_sc, col_bar = st.columns([2, 3], gap="medium")

    with col_sc:
        df_tr = df[df["CONJUNTO"] == "ENTRENAMIENTO"]
        if len(df_tr) > 0:
            st.markdown('<p class="sec-label">Real vs. predicho — entrenamiento</p>',
                        unsafe_allow_html=True)
            lim_lo = min(df_tr["rendimiento_real"].min(),
                         df_tr["rendimiento_predicho"].min()) - 0.2
            lim_hi = max(df_tr["rendimiento_real"].max(),
                         df_tr["rendimiento_predicho"].max()) + 0.2

            fig_sc = px.scatter(
                df_tr,
                x="rendimiento_real", y="rendimiento_predicho",
                color="condicion", color_discrete_map=C,
                hover_data=["ID_POLIGONO", "error_absoluto"],
                labels={"rendimiento_real": "Real (t/ha)",
                        "rendimiento_predicho": "Predicho (t/ha)",
                        "condicion": "Condición"},
                opacity=0.82,
            )
            fig_sc.add_shape(
                type="line",
                x0=lim_lo, y0=lim_lo, x1=lim_hi, y1=lim_hi,
                line=dict(dash="dash", color="#90A4AE", width=1.5),
            )
            fig_sc.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=295, margin=dict(t=10, b=40, l=10, r=10),
                xaxis=dict(showgrid=True, gridcolor="#E8F5E9", range=[lim_lo, lim_hi]),
                yaxis=dict(showgrid=True, gridcolor="#E8F5E9", range=[lim_lo, lim_hi]),
                legend=dict(orientation="h", y=1.05, font_size=9),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Selecciona el conjunto de entrenamiento para ver esta gráfica.")

    with col_bar:
        st.markdown('<p class="sec-label">Rendimiento predicho por parcela</p>',
                    unsafe_allow_html=True)

        df_ord = df.sort_values("rendimiento_predicho", ascending=False).reset_index(drop=True)
        fig_bp = px.bar(
            df_ord, x=df_ord.index, y="rendimiento_predicho",
            color="condicion", color_discrete_map=C,
            hover_data=["ID_POLIGONO", "CONJUNTO", "AREA_HA"],
            labels={"rendimiento_predicho": "Rendimiento predicho (t/ha)",
                    "index": "Parcelas (ordenadas)"},
        )
        for umbral, col_, etiq in [
            (4.3, C["Bueno"],    "4.3"),
            (3.5, C["Alerta 1"], "3.5"),
            (2.8, C["Alerta 2"], "2.8"),
        ]:
            fig_bp.add_hline(
                y=umbral, line_dash="dot", line_color=col_, line_width=1.2,
                annotation_text=f"{etiq} t/ha",
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color=col_,
            )
        fig_bp.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=295, margin=dict(t=10, b=40, l=10, r=60),
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="#E8F5E9"),
            showlegend=False,
        )
        st.plotly_chart(fig_bp, use_container_width=True)

    # ── Botón de descarga ───────────────────────────────────────────────────────
    cols_exp = ["ID_POLIGONO", "CONJUNTO", "AREA_HA",
                "rendimiento_real", "rendimiento_predicho",
                "error_absoluto", "condicion"]
    cols_exp = [c for c in cols_exp if c in df.columns]
    csv_bytes = df[cols_exp].to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Descargar datos filtrados (.csv)",
                       csv_bytes, "resultados_filtrados.csv", "text/csv")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
      Reto AgroCebada 2026 &nbsp;·&nbsp; Modelo: Ridge + Random Forest Ensemble
      &nbsp;·&nbsp; Sentinel-2 · Planet PRO · CHIRPS · INEGI
      &nbsp;·&nbsp; Estado de Hidalgo, México
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODELO PREDICTIVO
# ════════════════════════════════════════════════════════════════════════════════
with tab_modelo:
    st.markdown("### Explicación del modelo predictivo")

    # Métricas CV
    st.markdown("#### Desempeño en validación cruzada (k = 5 folds)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE",            "0.627 t/ha",
              help="Raíz del error cuadrático medio sobre folds de validación")
    m2.metric("MAE",             "0.481 t/ha",
              help="Error absoluto medio sobre folds de validación")
    m3.metric("R²",              "0.391",
              help="Coeficiente de determinación")
    m4.metric("Error relativo",  "~12–14 %",
              help="Respecto a la media de rendimiento (3.96 t/ha)")

    st.divider()

    col_i, col_p = st.columns([2, 2], gap="large")

    with col_i:
        st.markdown("#### Variables más importantes (Random Forest)")
        if imp_df is not None:
            top = imp_df.head(14).sort_values("importancia")
            fig_imp = px.bar(
                top, x="importancia", y="feature", orientation="h",
                color="importancia", color_continuous_scale="Greens",
                labels={"importancia": "Importancia", "feature": "Variable"},
            )
            fig_imp.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=400, margin=dict(t=10, b=20, l=10, r=10),
                coloraxis_showscale=False,
                yaxis=dict(tickfont_size=10),
                xaxis=dict(showgrid=True, gridcolor="#E8F5E9"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Ejecuta `python pipeline.py` para ver la importancia de variables.")

    with col_p:
        st.markdown("#### Arquitectura del modelo")
        st.markdown("""
| Componente | Peso | Descripción |
|---|---|---|
| **Ridge Regression** | 40 % | Regularización L2; robusto ante multicolinealidad entre índices espectrales |
| **Random Forest** | 60 % | 500 árboles; captura interacciones no lineales entre variables |

**Preprocesamiento** (ajustado solo en entrenamiento):
- Imputación por mediana para datos faltantes por nubosidad
- Estandarización `StandardScaler`

**Validación:** K-Fold estratificado k = 5 por quintiles de rendimiento
        """)

        st.markdown("#### Ingeniería de variables por etapa fenológica")
        st.markdown("""
| Ventana | Meses | Variables |
|---|---|---|
| Emergencia | Ene–Feb | NDVI, EVI, SAVI, LAI, FAPAR, CRC (media, máx, std) |
| Macollamiento | Mar | NDVI, EVI, SAVI, LAI, FAPAR, CRC (media, máx) |
| **Espigado ⭐** | **Abr–May** | **NDVI, EVI, SAVI, LAI, FAPAR, CRC (media, máx, std)** |
| Llenado | Jun | NDVI, EVI, LAI (media, máx) |
| Dinámicas | — | Δ NDVI espigado–emergencia · Integral NDVI (Ene–Jun) |
| Planet PRO | Ene–Jun | NDVI, EVI, LAI, MSAVI mensual |

⭐ Espigado (abr–may) contiene las variables de mayor correlación con el rendimiento (r ≈ 0.53–0.62).
        """)

    st.divider()

    # Correlaciones NDVI por mes
    st.markdown("#### Correlación NDVI mensual vs. rendimiento real")
    corr_df = pd.DataFrame({
        "Mes":     ["Ene", "Feb", "Mar", "Abr", "May", "Jun"],
        "r":       [0.337, 0.339, 0.443, 0.469, 0.527, 0.531],
        "Ventana": ["Emergencia", "Emergencia", "Macollamiento",
                    "Espigado", "Espigado", "Llenado"],
    })
    col_map = {"Emergencia": "#AED6F1", "Macollamiento": "#82E0AA",
               "Espigado": "#27AE60", "Llenado": "#1B5E20"}

    fig_corr = px.bar(
        corr_df, x="Mes", y="r", color="Ventana",
        color_discrete_map=col_map,
        text="r", range_y=[0, 0.65],
        labels={"r": "Correlación de Pearson (r)", "Mes": ""},
    )
    fig_corr.add_hline(y=0.5, line_dash="dash", line_color="#C0392B",
                       annotation_text="r = 0.50",
                       annotation_font_color="#C0392B", annotation_font_size=10)
    fig_corr.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                           textfont_size=10)
    fig_corr.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=300, margin=dict(t=20, b=30, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="#E8F5E9"),
        legend=dict(orientation="h", y=1.05, font_size=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DATOS COMPLETOS
# ════════════════════════════════════════════════════════════════════════════════
with tab_datos:
    st.markdown("### Tabla completa de resultados")

    cols_show = ["ID_POLIGONO", "CONJUNTO", "AREA_HA",
                 "rendimiento_real", "rendimiento_predicho",
                 "error_absoluto", "error_relativo_pct", "condicion"]
    cols_show = [c for c in cols_show if c in df.columns]
    df_show   = df[cols_show].copy()

    for c in ("rendimiento_real", "rendimiento_predicho", "error_absoluto"):
        if c in df_show:
            df_show[c] = df_show[c].round(3)
    if "error_relativo_pct" in df_show:
        df_show["error_relativo_pct"] = df_show["error_relativo_pct"].round(1)

    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        height=460,
        column_config={
            "ID_POLIGONO":          st.column_config.TextColumn("Polígono"),
            "CONJUNTO":             st.column_config.TextColumn("Conjunto", width="small"),
            "AREA_HA":              st.column_config.NumberColumn("Área (ha)", format="%.2f"),
            "rendimiento_real":     st.column_config.NumberColumn("Real (t/ha)",    format="%.3f"),
            "rendimiento_predicho": st.column_config.NumberColumn("Predicho (t/ha)",format="%.3f"),
            "error_absoluto":       st.column_config.NumberColumn("Error abs.",      format="%.3f"),
            "error_relativo_pct":   st.column_config.NumberColumn("Error %",         format="%.1f %%"),
            "condicion":            st.column_config.TextColumn("Condición"),
        },
    )

    st.divider()
    st.markdown("#### Estadísticos descriptivos")
    num_cols = [c for c in ("rendimiento_real", "rendimiento_predicho",
                            "error_absoluto", "AREA_HA") if c in df.columns]
    st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

    csv_all = df[cols_show].to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Descargar tabla completa (.csv)",
                       csv_all, "predicciones_completas.csv", "text/csv")
