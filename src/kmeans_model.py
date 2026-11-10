"""K-Means y búsqueda de K reproducibles."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .evaluation import clustering_metrics

def search_k(X, k_values=range(2,11), n_init=20, random_state=42):
    X=np.asarray(X); rows=[]
    for k in k_values:
        if 2 <= int(k) < len(X):
            model=KMeans(n_clusters=int(k),init="k-means++",n_init=max(10,int(n_init)),random_state=random_state).fit(X)
            m=clustering_metrics(X,model.labels_,noise_label=None)
            sizes=pd.Series(model.labels_).value_counts().sort_index()
            rows.append({"k":int(k),"inertia":float(model.inertia_),
                "tamaños_grupo":"; ".join(f"{cluster}: {count}" for cluster,count in sizes.items()),
                "grupo_menor_n":int(sizes.min()),"grupo_mayor_n":int(sizes.max()),
                "grupo_menor_pct":100*float(sizes.min())/len(X),**m})
    return pd.DataFrame(rows)

def fit_kmeans(X, k=3, n_init=20, random_state=42):
    model=KMeans(n_clusters=int(k),init="k-means++",n_init=max(10,int(n_init)),random_state=int(random_state)).fit(np.asarray(X))
    return model, model.labels_.copy()

def centroid_table(model, feature_names, scaler=None):
    centers=model.cluster_centers_
    if scaler is not None: centers=scaler.inverse_transform(centers)
    return pd.DataFrame(centers,columns=feature_names).rename_axis("cluster").reset_index()

