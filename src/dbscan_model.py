"""DBSCAN, curva k-distances y exploración de parámetros."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from .evaluation import clustering_metrics

def k_distances(X, min_samples=5):
    X=np.asarray(X); k=max(2,min(int(min_samples),len(X)))
    distances=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:,-1]
    return np.sort(distances)

def fit_dbscan(X, eps=.8, min_samples=5):
    X=np.asarray(X); model=DBSCAN(eps=float(eps),min_samples=int(min_samples)).fit(X); labels=model.labels_.copy()
    metrics=clustering_metrics(X,labels)
    core=np.zeros(len(labels),dtype=bool); core[model.core_sample_indices_]=True
    kinds=np.where(labels==-1,"ruido",np.where(core,"núcleo","borde"))
    return model, labels, metrics, kinds

def explore_dbscan(X, eps_values, min_samples_values):
    rows=[]
    for ms in min_samples_values:
        for eps in eps_values:
            _, labels, metrics, _=fit_dbscan(X,eps,ms)
            rows.append({"eps":float(eps),"min_samples":int(ms),"noise_pct":100*metrics["n_noise"]/len(labels),"coverage_pct":100*metrics["coverage"],**metrics})
    return pd.DataFrame(rows)

