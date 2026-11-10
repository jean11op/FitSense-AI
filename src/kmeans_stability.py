"""Estabilidad de K-Means mediante semillas y ARI."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

def assess_kmeans_stability(X,k=3,seeds=(11,23,42,77,101),n_init=10):
    X=np.asarray(X); labels=[]; rows=[]
    for seed in seeds:
        model=KMeans(n_clusters=k,init="k-means++",n_init=max(10,n_init),random_state=seed).fit(X); labels.append(model.labels_)
        rows.append({"seed":seed,"silhouette":silhouette_score(X,model.labels_),"davies_bouldin":davies_bouldin_score(X,model.labels_),"calinski_harabasz":calinski_harabasz_score(X,model.labels_)})
    pair=[adjusted_rand_score(labels[i],labels[j]) for i in range(len(labels)) for j in range(i+1,len(labels))]
    return pd.DataFrame(rows), {"ari_mean":float(np.mean(pair)) if pair else float("nan"),"ari_std":float(np.std(pair)) if pair else float("nan")}

