# Etapa 1. Comprensión del problema

## Problema y objetivo

FitSense AI estudia rutinas diarias de actividad física y descanso mediante observaciones persona-día anonimizadas de dispositivos vestibles. El objetivo es comparar K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN para encontrar perfiles descriptivos y marcar días potencialmente atípicos.

## Pregunta de investigación

¿Qué rutinas de actividad física y descanso se pueden diferenciar mediante técnicas de aprendizaje no supervisado y qué días presentan comportamientos atípicos?

## Unidad, utilidad y límites

La unidad de análisis es una persona durante una fecha. La utilidad esperada es resumir patrones de comportamiento para exploración académica. Los resultados no son diagnósticos ni predicciones clínicas.

Un agrupamiento será útil si forma grupos con separación razonable, tamaño y cobertura comunicados, estabilidad entre ejecuciones, perfiles entendibles en unidades originales y utilidad para la pregunta. DBSCAN puede dejar observaciones sin asignar; estas se reportan como ruido y no se ocultan.

El aprendizaje no supervisado busca estructura en datos sin una etiqueta objetivo preexistente. La predicción supervisada aprende una relación entre predictores y etiquetas conocidas para estimar etiquetas en casos nuevos. Este proyecto realiza agrupamiento, no predicción supervisada.

## Criterio de decisión

Se valoran conjuntamente métricas, cobertura, estabilidad, interpretabilidad, simplicidad y varianza perdida por PCA. Ninguna métrica por sí sola determina el modelo recomendado.
