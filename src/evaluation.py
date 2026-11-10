"""Métricas internas con validaciones explícitas para ruido y muestras pequeñas."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def clustering_metrics(X, labels, noise_label=-1):
    X=np.asarray(X); labels=np.asarray(labels)
    result={"valid":False,"reason":None,"n_clusters":0,"n_evaluated":0,"n_noise":0,"coverage":0.0,"silhouette":np.nan,"davies_bouldin":np.nan,"calinski_harabasz":np.nan}
    if X.ndim != 2:
        result["reason"]="X debe ser una matriz bidimensional."; return result
    if labels.ndim != 1 or len(labels) != len(X):
        result["reason"]="Las etiquetas deben ser un vector alineado con las observaciones."; return result
    try:
        finite = np.isfinite(X.astype(float)).all()
    except (TypeError, ValueError):
        finite = False
    if not finite:
        result["reason"]="X contiene valores no numéricos o no finitos."; return result
    missing = pd.isna(labels)
    if np.asarray(missing).any():
        result["reason"]="Las etiquetas contienen valores faltantes."; return result
    valid = np.ones(len(labels), dtype=bool) if noise_label is None else labels != noise_label
    xv, yv = X[valid], labels[valid]
    clusters=np.unique(yv); n=len(yv)
    result.update(n_clusters=int(len(clusters)),n_evaluated=int(n),n_noise=int((~valid).sum()),coverage=float(valid.mean()) if len(labels) else 0.0)
    if len(clusters) < 2:
        result["reason"]="Se requieren al menos dos clusters válidos después de excluir ruido."; return result
    if n <= len(clusters):
        result["reason"]="Las observaciones evaluadas deben superar el número de clusters."; return result
    try:
        result.update(valid=True,silhouette=float(silhouette_score(xv,yv)),davies_bouldin=float(davies_bouldin_score(xv,yv)),calinski_harabasz=float(calinski_harabasz_score(xv,yv)))
    except (ValueError, TypeError) as exc:
        result.update(reason=f"No se pudieron calcular las métricas: {exc}")
    return result

def compare_labels(X, labels_a, labels_b, name_a="A", name_b="B"):
    a,b=np.asarray(labels_a),np.asarray(labels_b)
    X=np.asarray(X)
    if a.ndim != 1 or b.ndim != 1 or len(a)!=len(b) or len(a)!=len(X):
        return {"common_n":0,"valid":False,"reason":"Las matrices y vectores de etiquetas deben alinear por observación.",name_a:clustering_metrics(np.empty((0,0)),[]),name_b:clustering_metrics(np.empty((0,0)),[])}
    if pd.isna(a).any() or pd.isna(b).any():
        return {"common_n":0,"valid":False,"reason":"Las etiquetas contienen valores faltantes."}
    common=(a!=-1)&(b!=-1)
    left=clustering_metrics(X[common],a[common],noise_label=None)
    right=clustering_metrics(X[common],b[common],noise_label=None)
    valid=left["valid"] and right["valid"]
    reasons=[f"{name}: {metrics['reason']}" for name,metrics in [(name_a,left),(name_b,right)] if not metrics["valid"]]
    return {"common_n":int(common.sum()),"valid":bool(valid),"reason":"; ".join(reasons) if reasons else None,name_a:left,name_b:right}

