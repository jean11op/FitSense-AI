# Distribución del trabajo y contratos

## Integrante 1: datos y K-Means

Responsable de `data_loader`, `integration`, `data_validation`, `data_dictionary`, `eda`, `preprocessing`, `kmeans_model`, `kmeans_stability` y las secciones de datos/EDA/preprocesamiento/K-Means de Streamlit.

## Integrante 2: PCA, DBSCAN y evaluación

Responsable de `dbscan_model`, `pca_model`, `combined_models`, `evaluation`, `dbscan_stability`, `model_comparison` y sus secciones de interfaz.

## Trabajo conjunto

Ambos acuerdan variables, perfiles, decisión y conclusiones. Contratos: las funciones de modelos reciben matrices NumPy bidimensionales y devuelven estimador/etiquetas; la integración produce una fila única por Id-Date; preprocesamiento devuelve matriz escalada y metadatos alineados; las etiquetas conservan el orden de filas de la matriz.

Los módulos de datos y modelos se pueden trabajar por separado mientras se respeten esos contratos. Los originales no son editables por la canalización.
