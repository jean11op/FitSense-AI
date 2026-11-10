"""Matriz de evidencia; no elige automáticamente un algoritmo."""
import pandas as pd

MODEL_TRADEOFFS={
    "K-Means":("Asigna todas las observaciones; centroides y perfiles directos.","Hay que elegir K; sensible a extremos y a geometrías no esféricas."),
    "DBSCAN":("Puede identificar ruido y grupos de distintas formas sin fijar K.","Cobertura y cantidad de grupos dependen de eps/min_samples."),
    "PCA + K-Means":("Combina reducción dimensional con asignación completa.","Pierde parte de la varianza y los grupos dependen del espacio PCA."),
    "PCA + DBSCAN":("Reduce dimensiones y puede marcar ruido en ese espacio.","La cobertura depende de PCA y de eps/min_samples; cuesta más interpretar."),
}

def evidence_matrix(comparison: pd.DataFrame, stability: dict | None = None, pca_variance: float | None = None):
    stability=stability or {}
    rows=[]
    indexed=comparison.set_index("modelo") if not comparison.empty and "modelo" in comparison else pd.DataFrame()
    for model,(pros,cons) in MODEL_TRADEOFFS.items():
        if model in indexed.index:
            row=indexed.loc[model].to_dict()
        else:
            row={"valid":False,"reason":"Modelo no ejecutado o resultado obsoleto.","n_clusters":None,"n_evaluated":None,"n_noise":None,"coverage_pct":None,"silhouette":None,"davies_bouldin":None,"calinski_harabasz":None}
        row.update({"modelo":model,"estabilidad":stability.get(model,"No evaluada"),"varianza_pca":pca_variance if model.startswith("PCA +") else None,"ventajas":pros,"limitaciones":cons})
        rows.append(row)
    return pd.DataFrame(rows)

def decision_guidance(evidence: pd.DataFrame, selected_model="Sin definir", arguments="", limitations=""):
    return {
        "evidence":evidence,
        "selected_model":selected_model,
        "arguments":arguments,
        "limitations":limitations,
        "note":"La decisión académica final requiere interpretación humana conjunta de métricas, cobertura, estabilidad, perfiles, simplicidad y pérdida de información. Silhouette por sí sola no elige el modelo.",
    }
