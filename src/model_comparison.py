"""Comparación en el espacio original escalado con conteo evaluado."""
import pandas as pd
from .evaluation import clustering_metrics

def compare_models(X, label_map):
    rows=[]
    for name,labels in label_map.items():
        m=clustering_metrics(X,labels)
        rows.append({"modelo":name,**m,"noise_pct":100*m["n_noise"]/len(labels),"coverage_pct":100*m["coverage"]})
    return pd.DataFrame(rows)

