"""Estabilidad DBSCAN: ARI incluye explícitamente el ruido como etiqueta propia."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from .dbscan_model import fit_dbscan

def assess_dbscan_stability(X, eps_values, min_samples_values):
    configs=[]; labs=[]; X=np.asarray(X)
    for ms in min_samples_values:
        for eps in eps_values:
            _,y,m,_=fit_dbscan(X,eps,ms); configs.append({"eps":eps,"min_samples":ms,"clusters":m["n_clusters"],"noise_pct":100*m["n_noise"]/len(y),"coverage_pct":100*m["coverage"]}); labs.append(y)
    rows=[]
    for i in range(len(labs)):
        for j in range(i+1,len(labs)):
            both=(labs[i]!=-1)&(labs[j]!=-1)
            rows.append({"config_a":i,"config_b":j,"ari_including_noise":adjusted_rand_score(labs[i],labs[j]),"both_clustered_n":int(both.sum()),"both_clustered_fraction":float(both.mean()),"ari_common_clustered":adjusted_rand_score(labs[i][both],labs[j][both]) if both.sum()>=2 else np.nan})
    return pd.DataFrame(configs),pd.DataFrame(rows)

