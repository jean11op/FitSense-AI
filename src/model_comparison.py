"""Comparación en el espacio original escalado con conteo evaluado."""
import numpy as np
import pandas as pd
from .evaluation import clustering_metrics

def compare_models(X, label_map):
    rows=[]
    for name,labels in label_map.items():
        m=clustering_metrics(X,labels)
        rows.append({"modelo":name,**m,"noise_pct":100*m["n_noise"]/len(labels) if len(labels) else float("nan"),"coverage_pct":100*m["coverage"]})
    return pd.DataFrame(rows)


def common_assigned_comparison(X, label_map, required_models=("K-Means", "DBSCAN", "PCA + K-Means", "PCA + DBSCAN")):
    """Evalúa las cuatro etiquetas sobre la intersección asignada por ambos DBSCAN."""
    missing=[name for name in required_models if name not in label_map]
    if missing:
        return {"valid":False,"n_common":0,"reason":f"Faltan resultados vigentes: {', '.join(missing)}.","table":pd.DataFrame()}
    X=np.asarray(X)
    if X.ndim != 2:
        return {"valid":False,"n_common":0,"reason":"La representación original escalada debe ser bidimensional.","table":pd.DataFrame()}
    labels={name:np.asarray(label_map[name]) for name in required_models}
    if any(y.ndim != 1 or len(y) != len(X) for y in labels.values()):
        return {"valid":False,"n_common":0,"reason":"Las etiquetas no están alineadas con las observaciones de X.","table":pd.DataFrame()}
    if any(pd.isna(y).any() for y in labels.values()):
        return {"valid":False,"n_common":0,"reason":"Hay etiquetas faltantes en uno o más modelos.","table":pd.DataFrame()}
    common=(labels["DBSCAN"]!=-1)&(labels["PCA + DBSCAN"]!=-1)
    n_common=int(common.sum())
    rows=[]; reasons=[]
    for name in required_models:
        metrics=clustering_metrics(X[common],labels[name][common],noise_label=None)
        rows.append({"modelo":name,**metrics,"n_common":n_common})
        if not metrics["valid"]: reasons.append(f"{name}: {metrics['reason']}")
    return {"valid":not reasons,"n_common":n_common,"reason":"; ".join(reasons) if reasons else None,"table":pd.DataFrame(rows)}

