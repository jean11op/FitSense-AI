"""Perfiles descriptivos en unidades originales; no realiza inferencias clínicas."""
import numpy as np
import pandas as pd

def profile_clusters(raw_features, labels):
    frame=raw_features.copy(); frame["cluster"]=np.asarray(labels)
    grouped=frame.groupby("cluster").agg(["count","mean","median","std"])
    return grouped


PROFILE_AXES = {
    "TotalSteps": ("Más pasos", "Menos pasos"),
    "actividad_intensa_moderada": ("Actividad intensa/moderada elevada", "Actividad intensa/moderada baja"),
    "LightlyActiveMinutes": ("Actividad ligera elevada", "Actividad ligera baja"),
    "SedentaryMinutes": ("Sedentarismo elevado", "Sedentarismo reducido"),
    "TotalMinutesAsleep": ("Mayor duración del sueño", "Menor duración del sueño"),
}


def profile_details(raw_features, scaled_features, labels, threshold=0.35):
    """Resume grupos y propone nombres a partir de las medias z observadas."""
    raw=raw_features.reset_index(drop=True).copy()
    scaled=scaled_features.reset_index(drop=True).copy()
    labels=np.asarray(labels)
    if len(raw)!=len(scaled) or len(raw)!=len(labels):
        raise ValueError("Características y etiquetas deben estar alineadas.")
    if list(raw.columns)!=list(scaled.columns):
        raise ValueError("Las matrices original y estandarizada deben compartir variables y orden.")
    if float(threshold)<=0:
        raise ValueError("El umbral descriptivo debe ser positivo.")

    raw["cluster"]=labels
    scaled["cluster"]=labels
    total=len(labels); summaries=[]; statistics=[]; zmeans=[]
    for cluster,group in raw.groupby("cluster",sort=True):
        mask=labels==cluster
        standardized=scaled.loc[mask,scaled_features.columns].mean()
        axes={}
        for feature in ("TotalSteps","LightlyActiveMinutes","SedentaryMinutes","TotalMinutesAsleep"):
            if feature in standardized.index:
                axes[feature]=float(standardized[feature])
        active=[c for c in ("VeryActiveMinutes","FairlyActiveMinutes") if c in standardized.index]
        if active:
            axes["actividad_intensa_moderada"]=float(standardized[active].mean())

        if cluster == -1:
            profile="Días no asignados (ruido DBSCAN)"
            interpretation=("DBSCAN no asignó estos días a un grupo de densidad; se muestran como ruido, "
                            "no como un perfil de cluster.")
        else:
            relevant=sorted(((axis,value) for axis,value in axes.items() if abs(value)>=threshold),
                            key=lambda item:abs(item[1]),reverse=True)
            names=[]
            for axis,value in relevant[:2]:
                high,low=PROFILE_AXES[axis]
                names.append(high if value>0 else low)
            profile=" · ".join(names) if names else "Perfil cercano a la media observada"
            if relevant:
                phrases=[]
                for axis,value in relevant:
                    high,low=PROFILE_AXES[axis]
                    phrases.append(f"{high.lower() if value>0 else low.lower()} ({value:+.2f} DE)")
                interpretation="Respecto de la media del conjunto: "+"; ".join(phrases)+"."
            else:
                interpretation=f"Ninguna media se separa al menos {threshold:.2f} desviaciones estándar de la media global."

        summaries.append({"cluster":cluster,"observaciones":int(len(group)),
            "porcentaje":100*len(group)/total if total else 0.0,
            "perfil_sugerido":profile,"interpretacion":interpretation})
        for feature in raw_features.columns:
            values=group[feature]
            statistics.append({"cluster":cluster,"variable":feature,"media":values.mean(),
                "mediana":values.median(),"desviacion_estandar":values.std(ddof=1)})
        zmeans.append({"cluster":cluster,**standardized.to_dict()})

    summary=pd.DataFrame(summaries)
    details=pd.DataFrame(statistics)
    heatmap=pd.DataFrame(zmeans).set_index("cluster") if zmeans else pd.DataFrame(columns=scaled_features.columns)
    return summary,details,heatmap

def outlier_days(metadata, labels, distances=None):
    labels=np.asarray(labels)
    if len(metadata)!=len(labels): raise ValueError("Metadatos y etiquetas deben tener la misma longitud.")
    out=metadata.copy(); out["cluster"]=labels; out["potencial_atipico"]=labels==-1
    if distances is not None:
        distances=np.asarray(distances,dtype=float)
        if len(distances)!=len(labels): raise ValueError("Las distancias deben corresponder a las etiquetas.")
        out["distancia_centroide"]=distances
    return out.loc[out["potencial_atipico"]].drop(columns=["Id"],errors="ignore")


def kmeans_outlier_candidates(X_scaled, labels, metadata, percentile=95):
    """Marca distancias altas a la media del cluster en el espacio original escalado."""
    X=np.asarray(X_scaled,dtype=float); labels=np.asarray(labels)
    if X.ndim!=2 or len(X)!=len(labels) or len(metadata)!=len(labels):
        raise ValueError("X, etiquetas y metadatos deben estar alineados por observación.")
    if not 0 < float(percentile) <= 100:
        raise ValueError("El percentil debe estar en el intervalo (0, 100].")
    if not np.isfinite(X).all():
        raise ValueError("X contiene valores no finitos.")
    assigned=labels!=-1
    distances=np.full(len(labels),np.nan)
    for cluster in np.unique(labels[assigned]):
        mask=labels==cluster
        center=X[mask].mean(axis=0)
        distances[mask]=np.linalg.norm(X[mask]-center,axis=1)
    candidates=distances[assigned]
    threshold=float(np.percentile(candidates,percentile)) if len(candidates) else float("nan")
    result=metadata.copy(); result["cluster"]=labels; result["distancia_centroide"]=distances
    result["umbral_percentil"]=threshold; result["potencial_atipico"]=(labels!=-1)&(distances>threshold)
    return result.loc[result["potencial_atipico"]].reset_index(drop=True),threshold

