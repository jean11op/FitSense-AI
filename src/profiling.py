"""Perfiles descriptivos en unidades originales; no realiza inferencias clínicas."""
import numpy as np
import pandas as pd

def profile_clusters(raw_features, labels):
    frame=raw_features.copy(); frame["cluster"]=np.asarray(labels)
    grouped=frame.groupby("cluster").agg(["count","mean","median","std"])
    return grouped

def outlier_days(metadata, labels, distances=None):
    labels=np.asarray(labels)
    if len(metadata)!=len(labels): raise ValueError("Metadatos y etiquetas deben tener la misma longitud.")
    out=metadata.copy(); out["cluster"]=labels; out["potencial_atipico"]=labels==-1
    if distances is not None:
        distances=np.asarray(distances,dtype=float)
        if len(distances)!=len(labels): raise ValueError("Las distancias deben corresponder a las etiquetas.")
        out["distancia_centroide"]=distances
    return out.loc[out["potencial_atipico"]].drop(columns=["Id"],errors="ignore")


def kmeans_outlier_candidates(X_scaled, labels, metadata, percentile=95):
    """Marca distancias altas a la media del cluster en el espacio original escalado."""
    X=np.asarray(X_scaled,dtype=float); labels=np.asarray(labels)
    if X.ndim!=2 or len(X)!=len(labels) or len(metadata)!=len(labels):
        raise ValueError("X, etiquetas y metadatos deben estar alineados por observación.")
    if not 0 < float(percentile) <= 100:
        raise ValueError("El percentil debe estar en el intervalo (0, 100].")
    if not np.isfinite(X).all():
        raise ValueError("X contiene valores no finitos.")
    assigned=labels!=-1
    distances=np.full(len(labels),np.nan)
    for cluster in np.unique(labels[assigned]):
        mask=labels==cluster
        center=X[mask].mean(axis=0)
        distances[mask]=np.linalg.norm(X[mask]-center,axis=1)
    candidates=distances[assigned]
    threshold=float(np.percentile(candidates,percentile)) if len(candidates) else float("nan")
    result=metadata.copy(); result["cluster"]=labels; result["distancia_centroide"]=distances
    result["umbral_percentil"]=threshold; result["potencial_atipico"]=(labels!=-1)&(distances>threshold)
    return result.loc[result["potencial_atipico"]].reset_index(drop=True),threshold

