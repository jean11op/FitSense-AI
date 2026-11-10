"""Preparación transparente de variables para clustering."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df: pd.DataFrame, features: list[str], missing: str = "median", outliers: str = "keep"):
    if not features:
        raise ValueError("Seleccione al menos una variable numérica.")
    absent = set(features) - set(df.columns)
    if absent:
        raise ValueError(f"Variables no disponibles: {sorted(absent)}")
    metadata_cols = [c for c in ["Id", "Date"] if c in df.columns]
    meta = df[metadata_cols].copy()
    x = df[features].apply(pd.to_numeric, errors="coerce").copy()
    if x.isna().all().any():
        raise ValueError("Hay variables completamente vacías o no numéricas.")
    if missing == "drop":
        keep = x.notna().all(axis=1); x = x.loc[keep]; meta = meta.loc[keep]
    elif missing in {"median", "mean"}:
        x = x.fillna(x.median() if missing == "median" else x.mean())
    elif missing == "zero":
        x = x.fillna(0)
    else:
        raise ValueError("Imputación debe ser drop, median, mean o zero.")
    if outliers not in {"keep", "iqr", "winsorize"}:
        raise ValueError("Tratamiento extremo debe ser keep, iqr o winsorize.")
    bounds = {}
    if outliers in {"iqr", "winsorize"}:
        for col in features:
            q1, q3 = x[col].quantile([.25,.75]); spread=q3-q1
            bounds[col] = (float(q1-1.5*spread), float(q3+1.5*spread))
            x[col] = x[col].clip(*bounds[col])
    scaler = StandardScaler(); scaled = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(scaled, columns=features, index=x.index)
    return {"raw": x.reset_index(drop=True), "scaled": scaled_df.reset_index(drop=True), "metadata": meta.reset_index(drop=True), "scaler": scaler, "features": features, "bounds": bounds, "missing_strategy": missing, "outlier_strategy": outliers}

