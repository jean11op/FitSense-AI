# Metodología reproducible

1. Definir problema y unidad persona-día.
2. Cargar fuentes, revisar estructura, faltantes, duplicados, distribuciones y correlaciones.
3. Elegir variables justificadas, separar metadatos, imputar faltantes según opción declarada, elegir tratamiento extremo y aplicar StandardScaler.
4. Explorar K-Means con K=2..10 sujeto al tamaño, k-means++, n_init ≥ 10, semilla y métricas internas.
5. Explorar DBSCAN sobre eps/min_samples, curva k-distances, ruido y cobertura. Las métricas excluyen ruido.
6. Ajustar PCA sobre la matriz estandarizada y comunicar varianza retenida, cargas y RMSE de reconstrucción.
7. Aplicar ambos algoritmos a las mismas puntuaciones PCA seleccionadas.
8. Comparar etiquetas en el espacio original escalado; las métricas PCA se presentan solo como diagnóstico y se informa el subconjunto evaluado.
9. Medir estabilidad con semillas para K-Means y variaciones de parámetros para DBSCAN. El ARI DBSCAN que incluye ruido trata -1 como categoría; también se reporta coincidencia de observaciones agrupadas.
10. Interpretar medias, medianas y desviaciones en unidades originales, y tomar decisión humana justificada.

Los ceros se conservan. Ninguna opción de faltantes o extremos se aplica en silencio. El clustering no aporta causalidad ni interpretación médica.
