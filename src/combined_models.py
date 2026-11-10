"""PCA compartido seguido de K-Means o DBSCAN."""
import numpy as np
from .kmeans_model import fit_kmeans
from .dbscan_model import fit_dbscan

def fit_pca_kmeans(scores, k=3, n_init=20, random_state=42):
    model,labels=fit_kmeans(scores,k,n_init,random_state)
    return model,labels

def fit_pca_dbscan(scores,eps=.8,min_samples=5):
    return fit_dbscan(scores,eps,min_samples)

def reconstruct_centroids(kmeans_model, pca_model, scaler):
    return scaler.inverse_transform(pca_model.inverse_transform(kmeans_model.cluster_centers_))

