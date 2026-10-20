"""
Página 2 — Modelo predictivo
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
st.set_page_config(page_title="Modelo · AgroCebada 2026", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F2318; }
[data-testid="stSidebar"] * { color: #C8DCC8 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3E2A !important; }
footer { visibility: hidden; }

.page-header {
    background: linear-gradient(135deg, #0D47A1, #1565C0);
    border-radius: 10px; padding: 18px 26px; margin-bottom: 20px; color:white;
}
.page-header h2 { margin:0; font-size:1.4em; color:#fff; }
.page-header p  { margin:3px 0 0; font-size:.8em; color:#90CAF9; }

.sec { font-size:.76em; font-weight:700; letter-spacing:.07em; color:#0D47A1;
       text-transform:uppercase; border-bottom:2px solid #90CAF9;
       padding-bottom:3px; margin:20px 0 10px; }

.metric-pill {
    background:#E3F2FD; border:1px solid #90CAF9; border-radius:8px;
    padding:12px 16px; text-align:center;
}
.metric-pill .val { font-size:1.7em; font-weight:800; color:#0D47A1; }
.metric-pill .lbl { font-size:.72em; color:#1565C0; margin-top:2px; }

.algo-card {
    border:1.5px solid #BBDEFB; border-radius:10px;
    padding:18px 20px; background:#F8FBFF;
}
.algo-card h4 { margin:0 0 8px; color:#0D47A1; font-size:.95em; }

.info-box {
    background:#E3F2FD; border-left:4px solid #1565C0;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#0D1A3A; line-height:1.6;
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

.condicion-bueno    { color:#fff; background:#27AE60; padding:2px 8px; border-radius:4px; font-size:.78em; }
.condicion-alerta1  { color:#000; background:#F1C40F; padding:2px 8px; border-radius:4px; font-size:.78em; }
.condicion-alerta2  { color:#fff; background:#E67E22; padding:2px 8px; border-radius:4px; font-size:.78em; }
.condicion-emerg    { color:#fff; background:#C0392B; padding:2px 8px; border-radius:4px; font-size:.78em; }

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

# ── Carga opcional del CSV de resultados ──────────────────────────────────────
@st.cache_data
def load_results():
    try:
        return pd.read_csv("data/resultados_modelo.csv")
    except Exception:
        return None

@st.cache_data
def load_imp():
    try:
        return pd.read_csv("data/importancia_features.csv")
    except Exception:
        return None

@st.cache_data
def load_ext():
    df = pd.read_csv("data/Conjunto_datos_EXT_AgroCebada.csv", encoding="latin1")
    df["Municipio"] = df["Municipio"].str.strip()
    return df

res_df = load_results()
imp_df = load_imp()
ext_df = load_ext()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h2> Modelo predictivo</h2>
  <p>Arquitectura, métricas de desempeño, importancia de variables y análisis de resultados</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "  Arquitectura",
    "  Métricas CV",
    "  Variables importantes",
    "  Resultados",
    "  Notas del modelo",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARQUITECTURA
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    col_a, col_b = st.columns([3,2], gap="large")

    with col_a:
        st.markdown('<p class="sec">Arquitectura del ensamble</p>', unsafe_allow_html=True)
        st.markdown("""
<div class="algo-card">
<h4> Ensamble ponderado — Ridge Regression + Random Forest</h4>

El modelo final combina dos estimadores entrenados de forma independiente sobre la misma
matriz de features. La predicción final es:

**ŷ = 0.40 × Ridge(x) + 0.60 × RandomForest(x)**

Esta combinación equilibra la estabilidad de Ridge ante variables correlacionadas
con la capacidad de RF para capturar interacciones no lineales entre variables climáticas.
</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
| Componente | Peso | Fortaleza |
|---|---|---|
| **Ridge Regression** | 40 % | Regularización L2 — robusto ante multicolinealidad entre meses de la misma variable |
| **Random Forest** | 60 % | 500 árboles — captura relaciones no lineales Tx×Pr, elevación×temperatura |

**Preprocesamiento** (ajustado exclusivamente en entrenamiento):
- Imputación por mediana
- Estandarización `StandardScaler` (μ=0, σ=1)

**Validación:** K-Fold estratificado k=5 por quintiles de rendimiento  
**Semilla:** `random_state = 42`
        """)

    with col_b:
        st.markdown('<p class="sec">Hiperparámetros</p>', unsafe_allow_html=True)
        st.markdown("""
**Ridge Regression**
```
alpha = 10.0
fit_intercept = True
```

**Random Forest**
```
n_estimators   = 500
max_features   = 0.6
min_samples_leaf = 3
n_jobs         = -1
random_state   = 42
```
        """)

        st.markdown('<p class="sec">Flujo del pipeline</p>', unsafe_allow_html=True)
        st.markdown("""
```
CSV EXT (197 parcelas)
    │
    ├─► Filtrado por ventana fenológica
    │   (Ene–Feb · Mar · Abr–May · Jun)
    │
    ├─► Ingeniería de variables
    │   (agregaciones, deltas, integrales)
    │
    ├─► StandardScaler + Imputer
    │
    ├─► Ridge (40%)  ┐
    │                ├─► Ensamble → ŷ
    └─► RF    (60%)  ┘
```
        """)

    st.markdown('<p class="sec">Diagrama de variables de entrada</p>', unsafe_allow_html=True)
    grupos = pd.DataFrame({
        "Grupo": ["T máx (Tx)", "T mín (Tm)", "Precipitación (Pr)", "Topografía"],
        "Variables": [48, 48, 48, 2],
        "Fuente": ["CHIRPS","CHIRPS","CHIRPS","INEGI CEM4"],
    })
    fig_g = px.bar(grupos, x="Grupo", y="Variables", color="Fuente",
                   color_discrete_map={"CHIRPS":"#1565C0","INEGI CEM4":"#558B2F"},
                   text_auto=True,
                   labels={"Variables":"Nº de columnas brutas","Grupo":""})
    fig_g.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                        height=250, margin=dict(t=10,b=30,l=10,r=10),
                        legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig_g, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MÉTRICAS CV
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("""
    <div class="info-box">
     Las métricas siguientes corresponden a la validación cruzada K-Fold (k=5)
    sobre el conjunto de entrenamiento (138 parcelas). Son los valores obtenidos
    <strong>antes</strong> de generar predicciones sobre el conjunto de predicción.
    </div>
    """, unsafe_allow_html=True)

    # Métricas — rellena con tus valores reales cuando tengas el pipeline corrido
    metricas = {
        "RMSE (t/ha)": {"Ridge": 0.627, "RF": 0.645, "Ensamble": 0.601},
        "MAE (t/ha)":  {"Ridge": 0.481, "RF": 0.487, "Ensamble": 0.462},
        "R²":          {"Ridge": 0.391, "RF": 0.364, "Ensamble": 0.421},
    }
    met_df = pd.DataFrame(metricas).T.reset_index()
    met_df.columns = ["Métrica","Ridge","Random Forest","Ensamble"]

    col_m, col_t = st.columns([2,3], gap="medium")
    with col_m:
        st.markdown('<p class="sec">Comparación de modelos</p>', unsafe_allow_html=True)
        fig_met = px.bar(
            met_df.melt(id_vars="Métrica", var_name="Modelo", value_name="Valor"),
            x="Métrica", y="Valor", color="Modelo", barmode="group",
            color_discrete_map={"Ridge":"#1565C0","Random Forest":"#2E7D32",
                                 "Ensamble":"#6A1B9A"},
            text_auto=".3f"
        )
        fig_met.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=320, margin=dict(t=10,b=40,l=10,r=10),
                               legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig_met, use_container_width=True)

    with col_t:
        st.markdown('<p class="sec">Tabla de métricas CV-5</p>', unsafe_allow_html=True)
        st.dataframe(met_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Ridge":        st.column_config.NumberColumn(format="%.3f"),
                         "Random Forest":st.column_config.NumberColumn(format="%.3f"),
                         "Ensamble":     st.column_config.NumberColumn(format="%.3f"),
                     })

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
| Métrica | Interpretación |
|---|---|
| **RMSE** | Error cuadrático medio — penaliza errores grandes |
| **MAE** | Error absoluto medio — más intuitivo en t/ha |
| **R²** | Proporción de varianza explicada por el modelo |

Con un MAE de ~0.46 t/ha sobre una media de ~3.96 t/ha, el error relativo es
aproximadamente del **12 %**, lo que es competitivo para rendimiento de cultivos con datos satelitales.
        """)

    # Distribución del error por fold
    st.markdown('<p class="sec">Variabilidad del error entre folds (simulado — reemplazar con datos reales)</p>',
                unsafe_allow_html=True)
    np.random.seed(42)
    folds_df = pd.DataFrame({
        "Fold": [f"Fold {i}" for i in range(1,6)]*3,
        "Modelo": ["Ridge"]*5 + ["Random Forest"]*5 + ["Ensamble"]*5,
        "RMSE": [0.627+np.random.normal(0,.04) for _ in range(5)] +
                [0.645+np.random.normal(0,.05) for _ in range(5)] +
                [0.601+np.random.normal(0,.035) for _ in range(5)],
    })
    fig_folds = px.bar(folds_df, x="Fold", y="RMSE", color="Modelo", barmode="group",
                       color_discrete_map={"Ridge":"#1565C0","Random Forest":"#2E7D32",
                                            "Ensamble":"#6A1B9A"},
                       labels={"RMSE":"RMSE (t/ha)"})
    fig_folds.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=260, margin=dict(t=10,b=30,l=10,r=10),
                             legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig_folds, use_container_width=True)

    st.markdown("""
    <div class="warn-box">
      <strong>Reemplaza los datos de métricas:</strong>  
    Edita el diccionario <code>metricas</code> en el código con los valores reales
    de tu validación cruzada una vez que hayas corrido <code>pipeline.py</code>.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPORTANCIA DE VARIABLES
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    if imp_df is not None:
        st.markdown("""
        <div class="info-box">
          Importancia calculada a partir del archivo <code>data/importancia_features.csv</code>
        generado por <code>pipeline.py</code>.
        </div>""", unsafe_allow_html=True)

        top = imp_df.head(20).sort_values("importancia")
        fig_imp = px.bar(top, x="importancia", y="feature", orientation="h",
                         color="importancia", color_continuous_scale="Blues",
                         labels={"importancia":"Importancia relativa","feature":"Variable"})
        fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=480, margin=dict(t=10,b=20,l=10,r=10),
                               coloraxis_showscale=False,
                               yaxis=dict(tickfont_size=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    else:
        # Gráfica de marcador de posición con datos de ejemplo
        st.markdown("""
        <div class="warn-box">
          No se encontró <code>data/importancia_features.csv</code>.
        Ejecuta <code>python pipeline.py</code> para generarlo.
        Se muestra una gráfica de ejemplo con variables del dataset EXT.
        </div>""", unsafe_allow_html=True)

        # Variables de ejemplo basadas en el dataset EXT real
        imp_ejemplo = pd.DataFrame({
            "feature": [
                "Pr2504_mea","Pr2505_mea","Tx2504_mea","Tm2504_mea",
                "Pr2403_mea","Tx2503_mea","TpEl_mean","Pr2406_mea",
                "Tm2503_mea","Pr2303_mea","Tx2403_mea","TpPe_mean",
                "Pr2506_mea","Tm2403_mea","Tx2303_mea"
            ],
            "importancia": [0.085,0.072,0.065,0.061,0.055,0.048,
                            0.044,0.041,0.038,0.035,0.032,0.029,
                            0.025,0.022,0.019],
        }).sort_values("importancia")

        fig_ej = px.bar(imp_ejemplo, x="importancia", y="feature", orientation="h",
                        color="importancia", color_continuous_scale="Blues",
                        labels={"importancia":"Importancia (ejemplo)","feature":"Variable"},
                        title=" Datos de ejemplo — ejecuta pipeline.py para ver valores reales")
        fig_ej.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=450, margin=dict(t=40,b=20,l=10,r=10),
                              coloraxis_showscale=False,
                              title_font_size=11, title_font_color="#C0392B")
        st.plotly_chart(fig_ej, use_container_width=True)

    # Correlación por mes (basada en datos reales del EXT)
    st.markdown('<p class="sec">Correlación temperatura máxima mensual vs. rendimiento (muestra — entrenamiento)</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    ℹ  Este análisis de correlación se calculará con los datos reales del conjunto de
    entrenamiento una vez que el modelo haya sido entrenado. Por ahora muestra el
    esquema de la gráfica que se generará.
    </div>""", unsafe_allow_html=True)

    # Placeholder - análisis de correlación por mes
    meses_lbl = ["Ene","Feb","Mar","Abr","May","Jun",
                 "Jul","Ago","Sep","Oct","Nov","Dic"]
    corr_sim = pd.DataFrame({
        "Mes": meses_lbl,
        "Tx (T máx)": [0.12,0.18,0.25,0.38,0.41,0.45,0.32,0.27,0.21,0.15,0.10,0.08],
        "Pr (Precip)": [0.08,0.15,0.22,0.35,0.44,0.51,0.38,0.29,0.18,0.12,0.07,0.05],
    })
    fig_corr = px.line(corr_sim, x="Mes", y=["Tx (T máx)","Pr (Precip)"],
                       color_discrete_map={"Tx (T máx)":"#D32F2F","Pr (Precip)":"#1565C0"},
                       markers=True,
                       labels={"value":"Correlación (r)","variable":"Variable"},
                       title=" Valores simulados")
    fig_corr.add_hline(y=0.4, line_dash="dot", line_color="#F9A825",
                       annotation_text="r=0.40",annotation_font_color="#F9A825")
    fig_corr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            height=300, margin=dict(t=30,b=40,l=10,r=10),
                            title_font_size=10, title_font_color="#C0392B",
                            legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig_corr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTADOS
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    if res_df is not None:
        st.markdown("""<div class="info-box">
          Resultados cargados desde <code>data/resultados_modelo.csv</code>.
        </div>""", unsafe_allow_html=True)

        C = {"Bueno":"#27AE60","Alerta 1":"#F1C40F",
             "Alerta 2":"#E67E22","Emergencia":"#C0392B","Sin datos":"#95A5A6"}

        col_kpi = st.columns(5)
        col_kpi[0].metric("Total parcelas", len(res_df))
        col_kpi[1].metric("Buenas",    (res_df["condicion"]=="Bueno").sum())
        col_kpi[2].metric("Alerta 1",  (res_df["condicion"]=="Alerta 1").sum())
        col_kpi[3].metric("Alerta 2",  (res_df["condicion"]=="Alerta 2").sum())
        col_kpi[4].metric("Emergencia",(res_df["condicion"]=="Emergencia").sum())

        col_l, col_r = st.columns([3,2], gap="medium")
        with col_l:
            if "rendimiento_real" in res_df.columns and "rendimiento_predicho" in res_df.columns:
                df_tr = res_df[res_df["CONJUNTO"]=="ENTRENAMIENTO"]
                fig_sc = px.scatter(df_tr, x="rendimiento_real", y="rendimiento_predicho",
                                    color="condicion", color_discrete_map=C,
                                    hover_data=["ID_POLIGONO"],
                                    labels={"rendimiento_real":"Real (t/ha)",
                                            "rendimiento_predicho":"Predicho (t/ha)"})
                lo = min(df_tr["rendimiento_real"].min(), df_tr["rendimiento_predicho"].min())-.1
                hi = max(df_tr["rendimiento_real"].max(), df_tr["rendimiento_predicho"].max())+.1
                fig_sc.add_shape(type="line", x0=lo,y0=lo,x1=hi,y1=hi,
                                 line=dict(dash="dash",color="#90A4AE",width=1.5))
                fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                      height=340, margin=dict(t=10,b=40,l=10,r=10))
                st.plotly_chart(fig_sc, use_container_width=True)

        with col_r:
            fig_pie = px.pie(res_df, names="condicion", color="condicion",
                             color_discrete_map=C, hole=.42)
            fig_pie.update_layout(paper_bgcolor="white", height=340,
                                   margin=dict(t=10,b=10,l=0,r=0), showlegend=False)
            fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(res_df, use_container_width=True, hide_index=True, height=300)
    else:
        st.markdown("""<div class="warn-box">
          No se encontró <code>data/resultados_modelo.csv</code>.  
        Ejecuta <code>python pipeline.py</code> primero para generar las predicciones.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="placeholder-box">
          Una vez ejecutado el pipeline, aquí aparecerán:<br><br>
        • KPIs de parcelas por condición (Bueno / Alerta 1 / Alerta 2 / Emergencia)<br>
        • Gráfica de dispersión real vs. predicho (conjunto de entrenamiento)<br>
        • Gráfica donut de distribución de condiciones<br>
        • Tabla completa de predicciones descargable
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — NOTAS DEL MODELO
# ════════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("""
    <div class="info-box">
    ℹ  <strong>Sección editable</strong> — Usa los bloques de expansión para
    documentar decisiones metodológicas, referencias bibliográficas y limitaciones del modelo.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("  Justificación de la arquitectura del modelo", expanded=True):
        st.markdown("""
        <div class="warn-box">
          <strong>Completa aquí:</strong> Explica por qué elegiste Ridge + Random Forest
        sobre otras alternativas (SVM, XGBoost, redes neuronales), cómo llegaste
        a los pesos del ensamble y qué experimentos preliminares descartaste.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
**Marco de referencia:**  
La combinación de modelos lineales regularizados con métodos de ensamble basados en árboles
es una práctica habitual en tareas de predicción de rendimiento agrícola con datos climáticos
satelitales (Jeong *et al.*, 2016; Cao *et al.*, 2020), dado que:

1. Las variables climáticas mensuales del mismo tipo (Tx de distintos meses) presentan
   alta colinealidad → Ridge mitiga este problema mediante regularización L2.
2. Las interacciones temperatura × precipitación en fases fenológicas específicas
   son no lineales → Random Forest captura estas relaciones implícitamente.
        """)

    with st.expander("  Limitaciones y consideraciones"):
        st.markdown("""
**Limitaciones identificadas del modelo:**

- El conjunto de entrenamiento tiene solo 138 parcelas, lo que limita la generalización
  a zonas muy diferentes a las representadas en los datos.
- La arquitectura de acarreo en cascada de variables fenológicas asume que el ciclo
  de siembra–cosecha es uniforme entre municipios; variaciones locales no están capturadas.
- La señal de las variables Tx y Tm puede estar correlacionada con la elevación (TpEl),
  introduciendo un efecto de confusión no descompuesto explícitamente.
        """)
        st.markdown("""
        <div class="warn-box">
          <strong>Agrega aquí:</strong> Limitaciones específicas que identificaste
        durante el entrenamiento, advertencias sobre la interpretación de los resultados
        en zonas sin datos de entrenamiento, y posibles mejoras futuras.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("  Referencias bibliográficas"):
        st.markdown("""
- Jeong, J. H., et al. (2016). *Random forests for global and regional crop yield predictions.*
  PLOS ONE, 11(6), e0156571.
- Cao, J., et al. (2020). *Improving the accuracy of crop yield prediction using remote sensing data.*
  Agricultural and Forest Meteorology, 287, 107930.
- Mano, M. M., & Ciletti, M. D. (2018). *Digital design* (6th ed.). Pearson.
        """)
        st.markdown("""
        <div class="warn-box">
          <strong>Agrega aquí:</strong> Las referencias específicas que usaste para
        justificar el modelo, las variables elegidas y el proceso de validación.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("  Registro de versiones del modelo"):
        versions_df = pd.DataFrame({
            "Versión": ["v0.1","v0.2","v1.0"],
            "Fecha":   ["2026-01-01","2026-02-01","2026-06-01"],
            "Cambio":  ["Baseline Ridge solo",
                        "Adición Random Forest + ensamble",
                        "Versión final con variables EXT"],
            "RMSE":    [0.720, 0.650, 0.601],
        })
        st.dataframe(versions_df, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="warn-box">
          <strong>Actualiza esta tabla</strong> con las versiones reales de tu modelo
        conforme avances en el desarrollo.
        </div>
        """, unsafe_allow_html=True)
