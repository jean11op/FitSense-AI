"""Métricas internas con validaciones explícitas para ruido y muestras pequeñas."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def clustering_metrics(X, labels, noise_label=-1):
    X=np.asarray(X); labels=np.asarray(labels)
    valid=labels != noise_label
    xv, yv = X[valid], labels[valid]
    clusters=np.unique(yv); n=len(yv)
    result={"n_clusters":int(len(clusters)),"n_evaluated":int(n),"n_noise":int((~valid).sum()),"coverage":float(valid.mean()) if len(labels) else 0.0,"silhouette":np.nan,"davies_bouldin":np.nan,"calinski_harabasz":np.nan}
    if len(clusters)>=2 and n>len(clusters):
        result.update(silhouette=float(silhouette_score(xv,yv)),davies_bouldin=float(davies_bouldin_score(xv,yv)),calinski_harabasz=float(calinski_harabasz_score(xv,yv)))
    return result

def compare_labels(X, labels_a, labels_b, name_a="A", name_b="B"):
    a,b=np.asarray(labels_a),np.asarray(labels_b)
    if len(a)!=len(b) or len(a)!=len(X): raise ValueError("Matrices y etiquetas deben alinear por observación.")
    common=(a!=-1)&(b!=-1)
    if common.sum()<3 or len(np.unique(a[common]))<2 or len(np.unique(b[common]))<2:
        return {"common_n":int(common.sum()),"valid":False,"reason":"Subconjunto común insuficiente o con menos de dos grupos."}
    return {"common_n":int(common.sum()),"valid":True,name_a:clustering_metrics(np.asarray(X)[common],a[common],noise_label=999),name_b:clustering_metrics(np.asarray(X)[common],b[common],noise_label=999)}

