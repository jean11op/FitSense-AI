"""Perfiles descriptivos en unidades originales; no realiza inferencias clínicas."""
import numpy as np
import pandas as pd

def profile_clusters(raw_features, labels):
    frame=raw_features.copy(); frame["cluster"]=np.asarray(labels)
    grouped=frame.groupby("cluster").agg(["count","mean","median","std"])
    return grouped

def outlier_days(metadata, labels, distances=None):
    out=metadata.copy(); out["cluster"]=np.asarray(labels); out["potencial_atipico"]=np.asarray(labels)==-1
    if distances is not None: out["distancia_centroide"]=np.asarray(distances)
    if "Id" in out: out["Participante"]=out["Id"].astype("string").str[-4:].radd("Participante …")
    return out.loc[out["potencial_atipico"] | (out.get("distancia_centroide",pd.Series(False,index=out.index))>out.get("distancia_centroide",pd.Series([np.inf]*len(out))).quantile(.95))].drop(columns=["Id"],errors="ignore")

