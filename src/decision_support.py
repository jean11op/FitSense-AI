"""Apoyo a decisión sin selector automático basado en una sola métrica."""
def decision_guidance(comparison, pca_variance=None):
    return {"comparison":comparison,"pca_variance_retained":pca_variance,"note":"La selección requiere valorar conjuntamente métricas, cobertura, estabilidad, interpretabilidad, simplicidad y utilidad académica. No existe recomendación automática."}

