"""
pipeline.py — Procesamiento de datos y entrenamiento del modelo predictivo.
Genera data/resultados_modelo.csv y data/importancia_features.csv.

Uso:
    python pipeline.py

Requiere en data/:
    - Conjunto_datos_BASICO_AgroCebada2026.csv
    - Conjunto_datos_PRO_AgroCebada.csv
    - ID_area_rendimiento_70_30_Reto_AgroCebada.csv
"""

import pandas as pd
import numpy as np
from functools import reduce
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── Configuración ──────────────────────────────────────────────────────────────
DATA_DIR    = "data"
OUT_RESULTS = f"{DATA_DIR}/resultados_modelo.csv"
OUT_IMP     = f"{DATA_DIR}/importancia_features.csv"

UMBRAL_BUENO    = 4.30   # t/ha  →  ≥ Bueno
UMBRAL_ALERTA1  = 3.50   # t/ha  →  ≥ Alerta 1
UMBRAL_ALERTA2  = 2.80   # t/ha  →  ≥ Alerta 2
                          #         <  Emergencia

# Ventanas fenológicas de cebada, altiplano de Hidalgo (meses del año)
VENTANAS = {
    "emerg":  [1, 2],
    "macoll": [3],
    "espig":  [4, 5],
    "llen":   [6],
}

INDICES_S2 = [
    "ndvi_promedio", "evi_promedio", "savi_promedio", "lai_promedio",
    "fapar_promedio", "crc_promedio", "ndwi_promedio", "msi_promedio",
]

INDICES_PRO = [
    "ndvi_promedio", "evi_promedio", "lai_promedio", "msavi_promedio",
]


# ── Utilidades ─────────────────────────────────────────────────────────────────
def _fix_col(df):
    """Corrige el BOM que aparece en el nombre de la columna ID."""
    return df.rename(columns={c: "ID_POLIGONO"
                               for c in df.columns
                               if "ID_POLIGONO" in c and c != "ID_POLIGONO"})


def _load():
    print("Cargando archivos CSV...")
    b = _fix_col(pd.read_csv(f"{DATA_DIR}/Conjunto_datos_BASICO_AgroCebada2026.csv",
                              encoding="latin1"))
    p = _fix_col(pd.read_csv(f"{DATA_DIR}/Conjunto_datos_PRO_AgroCebada.csv",
                              encoding="latin1"))
    t = _fix_col(pd.read_csv(f"{DATA_DIR}/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
                              encoding="latin1"))
    for df in [b, p]:
        df["fecha_captura"] = pd.to_datetime(df["fecha_captura"], dayfirst=True, errors="coerce")
        df["mes"]  = df["fecha_captura"].dt.month
        df["anio"] = df["fecha_captura"].dt.year
    print(f"  BÁSICO : {len(b):>7,} filas · {b['ID_POLIGONO'].nunique()} polígonos")
    print(f"  PRO    : {len(p):>7,} filas · {p['ID_POLIGONO'].nunique()} polígonos")
    train_n = (t["CONJUNTO"] == "ENTRENAMIENTO").sum()
    pred_n  = (t["CONJUNTO"] == "PREDICCION").sum()
    print(f"  TARGET :   {len(t)} polígonos  ({train_n} entrena / {pred_n} predicción)")
    return b, p, t


def _feats_s2(df_b, ids):
    """Agregación por ventana fenológica sobre Sentinel-2."""
    s2 = df_b[
        (df_b["sensor"] == "Sentinel-2") &
        (df_b["porcentaje_nubosidad"] < 30) &
        (df_b["ID_POLIGONO"].isin(ids))
    ].copy()

    frames = []
    for sfx, meses in VENTANAS.items():
        sub  = s2[s2["mes"].isin(meses)]
        avail = [c for c in INDICES_S2 if c in sub.columns]
        if not avail:
            continue
        agg = {}
        for c in avail:
            nm = c.replace("_promedio", "")
            agg[f"{nm}_mean_{sfx}"] = (c, "mean")
            agg[f"{nm}_max_{sfx}"]  = (c, "max")
            if sfx in ("espig", "emerg"):
                agg[f"{nm}_std_{sfx}"] = (c, "std")
        frames.append(sub.groupby("ID_POLIGONO").agg(**agg).reset_index())

    if not frames:
        return pd.DataFrame({"ID_POLIGONO": list(ids)})

    result = reduce(lambda a, b: a.merge(b, on="ID_POLIGONO", how="outer"), frames)

    # Features dinámicas
    for idx in ("ndvi", "savi", "evi"):
        c_esp = f"{idx}_mean_espig"
        c_em  = f"{idx}_mean_emerg"
        if c_esp in result.columns and c_em in result.columns:
            result[f"{idx}_delta_espig_emerg"] = result[c_esp] - result[c_em]

    ndvi_cols = [f"ndvi_mean_{s}" for s in VENTANAS if f"ndvi_mean_{s}" in result.columns]
    if ndvi_cols:
        result["ndvi_integral_1_6"] = result[ndvi_cols].sum(axis=1)

    return result


def _feats_pro(df_p, ids):
    """Pivot mensual sobre Planet PRO."""
    pro   = df_p[(df_p["porcentaje_nubosidad"] < 30) & (df_p["ID_POLIGONO"].isin(ids))].copy()
    avail = [c for c in INDICES_PRO if c in pro.columns]
    if not avail:
        return pd.DataFrame({"ID_POLIGONO": list(ids)})

    monthly = pro.groupby(["ID_POLIGONO", "mes"])[avail].mean().reset_index()
    pivots  = []
    for c in avail:
        nm  = c.replace("_promedio", "")
        pv  = monthly.pivot(index="ID_POLIGONO", columns="mes", values=c)
        pv.columns = [f"{nm}_pro_m{int(col)}" for col in pv.columns]
        pivots.append(pv)

    return pd.concat(pivots, axis=1).reset_index()


def _condicion(val):
    if pd.isna(val):        return "Sin datos"
    if val >= UMBRAL_BUENO: return "Bueno"
    if val >= UMBRAL_ALERTA1: return "Alerta 1"
    if val >= UMBRAL_ALERTA2: return "Alerta 2"
    return "Emergencia"


def _train_predict(X_tr, y_tr, X_all):
    print("\nEntrenando modelos...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def make_pipe(mdl):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("m",   mdl),
        ])

    pipe_r = make_pipe(Ridge(alpha=10.0))
    pipe_f = make_pipe(RandomForestRegressor(
        n_estimators=500, max_features=0.6,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    ))

    for name, pipe in [("Ridge", pipe_r), ("Random Forest", pipe_f)]:
        cv = cross_validate(pipe, X_tr, y_tr, cv=kf,
                            scoring=["neg_root_mean_squared_error",
                                     "r2",
                                     "neg_mean_absolute_error"])
        rmse = -cv["test_neg_root_mean_squared_error"].mean()
        mae  = -cv["test_neg_mean_absolute_error"].mean()
        r2   =  cv["test_r2"].mean()
        print(f"  {name:<15}: RMSE={rmse:.3f} t/ha | MAE={mae:.3f} t/ha | R²={r2:.3f}")

    # Entrenamiento final
    pipe_r.fit(X_tr, y_tr)
    pipe_f.fit(X_tr, y_tr)

    # Ensamble ponderado 40 / 60
    pred = 0.4 * pipe_r.predict(X_all) + 0.6 * pipe_f.predict(X_all)

    # Importancia de variables
    feat_names = X_tr.columns.tolist()
    imp = pd.DataFrame({
        "feature":    feat_names,
        "importancia": pipe_f.named_steps["m"].feature_importances_
    }).sort_values("importancia", ascending=False)

    print("\nTop 5 variables:")
    for _, row in imp.head(5).iterrows():
        print(f"  {row['feature']:<42} {row['importancia']:.4f}")

    return pred, imp


# ── Pipeline principal ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PIPELINE — Predicción de Rendimiento de Cebada 2026")
    print("=" * 60)

    df_b, df_p, df_t = _load()

    all_ids   = df_t["ID_POLIGONO"].tolist()
    tr_mask   = df_t["CONJUNTO"] == "ENTRENAMIENTO"

    print("\nConstruyendo features...")
    fs2  = _feats_s2(df_b, all_ids)
    fpro = _feats_pro(df_p, all_ids)
    print(f"  Sentinel-2 : {fs2.shape[1]-1} variables")
    print(f"  Planet PRO : {fpro.shape[1]-1} variables")

    full = (
        df_t[["ID_POLIGONO", "RENDIMIENTO_T_HA", "AREA_HA", "CONJUNTO"]]
        .merge(fs2,  on="ID_POLIGONO", how="left")
        .merge(fpro, on="ID_POLIGONO", how="left")
    )

    feat_cols = [c for c in full.columns
                 if c not in ("ID_POLIGONO", "RENDIMIENTO_T_HA", "AREA_HA", "CONJUNTO")]
    print(f"  Total      : {len(feat_cols)} variables")

    X_all = full[feat_cols]
    X_tr  = full.loc[tr_mask, feat_cols]
    y_tr  = full.loc[tr_mask, "RENDIMIENTO_T_HA"]

    pred_all, imp_df = _train_predict(X_tr, y_tr, X_all)

    # ── Construir tabla de resultados ──────────────────────────────────────────
    res = df_t[["ID_POLIGONO", "RENDIMIENTO_T_HA", "AREA_HA", "CONJUNTO"]].copy()
    res.rename(columns={"RENDIMIENTO_T_HA": "rendimiento_real"}, inplace=True)

    res["rendimiento_predicho"] = np.clip(pred_all, 1.0, 7.0)

    is_train = res["CONJUNTO"] == "ENTRENAMIENTO"
    res["error_absoluto"]     = np.where(is_train,
        np.abs(res["rendimiento_real"] - res["rendimiento_predicho"]), np.nan)
    res["error_relativo_pct"] = np.where(is_train,
        100 * res["error_absoluto"] / res["rendimiento_real"], np.nan)

    res["condicion"] = res["rendimiento_predicho"].apply(_condicion)

    # Añadir NDVI espigado como referencia visual en el dashboard
    if "ndvi_mean_espig" in fs2.columns:
        res = res.merge(fs2[["ID_POLIGONO", "ndvi_mean_espig"]], on="ID_POLIGONO", how="left")

    # ── Guardar ────────────────────────────────────────────────────────────────
    res.to_csv(OUT_RESULTS, index=False)
    imp_df.to_csv(OUT_IMP, index=False)

    print(f"\nDistribución de condiciones (197 parcelas):")
    print(res["condicion"].value_counts().to_frame("parcelas").to_string())

    print(f"\n✓  {OUT_RESULTS}")
    print(f"✓  {OUT_IMP}")
    print("\nPipeline completado.")


if __name__ == "__main__":
    main()
