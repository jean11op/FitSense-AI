"""Análisis PCA sobre las variables ya estandarizadas."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def fit_pca(X, n_components=None, variance_threshold=.90):
    X=np.asarray(X,dtype=float)
    if n_components is None:
        probe=PCA().fit(X); n_components=int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_),variance_threshold)+1)
    model=PCA(n_components=int(n_components),svd_solver="full").fit(X)
    scores=model.transform(X); reconstructed=model.inverse_transform(scores)
    rmse=float(np.sqrt(mean_squared_error(X,reconstructed)))
    return model,scores,{"explained_variance_ratio":model.explained_variance_ratio_,"cumulative_variance":np.cumsum(model.explained_variance_ratio_),"loadings":pd.DataFrame(model.components_.T,columns=[f"PC{i+1}" for i in range(model.n_components_) ]),"n_components":int(model.n_components_),"variance_retained":float(model.explained_variance_ratio_.sum()),"reconstruction_rmse_scaled":rmse}

