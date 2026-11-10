# Revisión final y mejora de presentación — FitSense AI

**Fecha:** 10 de noviembre de 2026  
**Alcance:** correcciones de presentación, trazabilidad de resultados, comparación de modelos e interpretación descriptiva con los archivos y datos disponibles en el proyecto. Los CSV fuente no se modificaron.

## 1. Resultado de la revisión

La aplicación ahora presenta de forma coherente las cuatro alternativas solicitadas, separa las métricas individuales de la comparación sobre soporte común y permite interpretar los grupos usando estadísticas de los datos. Los resultados de cada modelo quedan asociados a la configuración que los generó; si cambia una configuración, la alternativa afectada queda pendiente hasta volver a entrenarla. Las configuraciones de K-Means original y PCA + K-Means se mantienen independientes.

La interfaz conserva el tema oscuro y su navegación por diez secciones. La recomendación de modelo sigue siendo una decisión académica del equipo: ni el valor máximo de Silhouette ni las demás métricas seleccionan automáticamente una alternativa.

## 2. Archivos modificados

- `.streamlit/config.toml`: establece el tema oscuro, color principal y colores de fondo/texto.
- `src/kmeans_model.py`: agrega a la búsqueda de K tamaños de grupo y tamaño mínimo porcentual para apoyar la interpretación de soluciones fragmentadas.
- `src/model_comparison.py`: agrega una tabla con las cuatro alternativas y su estado, configuración, métricas, cobertura, ruido y estabilidad; mantiene la evaluación de soporte común.
- `src/profiling.py`: calcula conteos, proporciones, medias, medianas, desviaciones estándar, diferencias estandarizadas y descripciones de perfil basadas en las variables observadas.
- `src/streamlit_app.py`: reorganiza las vistas de K-Means, DBSCAN, modelos combinados, comparación y perfiles; añade explicaciones y descargas, y vincula los resultados con sus configuraciones activas.
- `tests/test_core.py`: incorpora comprobaciones para perfiles descriptivos, filas pendientes de comparación y validez de resultados según configuración.
- `docs/guia_archivos_proyecto.md`: documenta el propósito de archivos de código, configuración, datos, documentación y salidas.
- `outputs/reports/REVISION_FINAL_FITSENSE_AI.md`: este registro de cambios y resultados reproducidos.

## 3. Correcciones de presentación y método

### Comparación de cuatro alternativas

La vista comparativa incluye siempre K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN. Si un modelo no se ejecutó con la configuración vigente, aparece como pendiente, sin presentar cifras antiguas como actuales. La tabla individual indica el número de grupos, observaciones evaluadas, ruido, cobertura, Silhouette, Davies–Bouldin, Calinski–Harabasz y estado del análisis de estabilidad.

Las métricas de DBSCAN excluyen la etiqueta de ruido (`-1`); por ello sus soportes y cobertura se muestran explícitamente. La segunda tabla restringe las cuatro particiones a las mismas observaciones asignadas por ambos DBSCAN, y calcula las métricas en la matriz original estandarizada. Si la intersección o alguna partición no satisface los requisitos matemáticos, la interfaz informa que la comparación no es válida.

### Selección y explicación de K-Means

La búsqueda presenta métricas internas y tamaños de grupo, identifica visualmente el K activo y muestra la distribución de sus integrantes. También permite documentar la razón humana para mantener un valor de K.

En la matriz estandarizada usada para esta revisión, K=7 obtuvo el mayor Silhouette de K=2…10 (0.2719), ligeramente por encima de K=6 (0.2709). Sin embargo, K=7 produce un grupo de 15 observaciones (3.66 % del total) y siete segmentos; K=3 produce tres grupos de 160, 180 y 70 observaciones (39.02 %, 43.90 % y 17.07 %). K=3 se conserva como configuración base para esta ejecución porque forma grupos más amplios y facilita una descripción acotada, no porque maximice Silhouette. Es una decisión ilustrativa que el equipo debe respaldar y registrar en la interfaz.

### Selección y explicación de DBSCAN

La vista identifica los valores de `eps` y `min_samples` del modelo entrenado, su cobertura y porcentaje de ruido; además distingue puntos núcleo, borde y ruido. La curva k-distances y la grilla de sensibilidad se ejecutan mediante acciones explícitas. El equipo puede escribir el razonamiento para elegir parámetros.

Se mantiene `eps=0.8`, `min_samples=5` como punto de referencia reproducible, no como óptimo automático. Con `min_samples=5`, variar `eps` de 0.6 a 1.2 cambia la cobertura de 36.59 % a 90.24 %, y el ruido de 63.41 % a 9.76 %. A `eps=0.8`, el DBSCAN original identifica un grupo de 244 días y cuatro grupos pequeños de 5, 5, 5 y 8 días; 143 días quedan como ruido. Silhouette por sí sola no resuelve el intercambio entre compactación, cantidad/tamaño de grupos y cobertura.

### Modelos combinados y PCA

PCA se ajusta una vez a las características estandarizadas y comparte puntuaciones entre las dos alternativas combinadas. PCA + K-Means usa su propio K, `n_init` y semilla, distintos a los de K-Means original. Para reproducir esta revisión se usó K=4 en PCA + K-Means y umbral PCA de 90 %. Se retuvieron cuatro componentes, con 92.56 % de varianza explicada y RMSE de reconstrucción escalado 0.2727. PCA + DBSCAN tiene sus propios `eps` y `min_samples`; para esta revisión se usó 0.8 y 5.

Los gráficos de PCA y de clusters usan tema oscuro consistente. Las vistas combinadas reportan expresamente el espacio donde se calcularon sus métricas y despliegan detalles de configuración.

### Perfiles, días inusuales y decisión

Los perfiles se forman con medias estandarizadas observadas, con las medias, medianas y desviaciones estándar disponibles en detalle en unidades originales. Las etiquetas sugeridas describen diferencias frente al promedio del conjunto y no asignan categorías clínicas. DBSCAN conserva “ruido” como categoría aparte. Para K-Means se calculan candidatos inusuales por distancia al centroide asignado, usando un percentil explícito; se muestran junto a metadatos seudonimizados.

La vista de conclusiones recoge evidencia, ventajas y límites de las alternativas y deja campos para que el equipo elija y argumente su recomendación. La decisión guardada queda ligada a las configuraciones vigentes. Si cambian, se debe volver a documentar la elección.

## 4. Configuraciones y métricas reproducidas

Se integraron los CSV en **410 observaciones persona-día × 6 variables**. Las seis variables usadas fueron `TotalSteps`, `VeryActiveMinutes`, `FairlyActiveMinutes`, `LightlyActiveMinutes`, `SedentaryMinutes` y `TotalMinutesAsleep`. Las métricas individuales se calculan en el espacio original estandarizado; DBSCAN excluye ruido de las métricas internas.

| Alternativa | Configuración | Grupos | Evaluadas | Ruido | Cobertura | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K-Means | K=3, `n_init=20`, semilla 42 | 3 | 410 | 0 | 100.00 % | 0.2286 | 1.3780 | 116.9864 |
| DBSCAN | `eps=0.8`, `min_samples=5` | 5 | 267 | 143 | 65.12 % | 0.1215 | 0.7499 | 19.3912 |
| PCA + K-Means | PCA 90 % (4 componentes), K=4, `n_init=20`, semilla 42 | 4 | 410 | 0 | 100.00 % | 0.2191 | 1.3094 | 114.9803 |
| PCA + DBSCAN | PCA 90 % (4 componentes), `eps=0.8`, `min_samples=5` | 3 | 338 | 72 | 82.44 % | 0.1100 | 0.7655 | 14.2805 |

Las métricas individuales **no deben ordenarse como si evaluaran los mismos días**. Ambos modelos DBSCAN asignaron 267 observaciones en común. Sobre esas mismas 267 filas y en el espacio original estandarizado, se obtuvo:

| Alternativa | Silhouette común | Davies–Bouldin común | Calinski–Harabasz común |
|---|---:|---:|---:|
| K-Means | 0.2706 | 1.2230 | 106.7728 |
| DBSCAN | 0.1215 | 0.7499 | 19.3912 |
| PCA + K-Means | 0.2699 | 1.2003 | 108.8544 |
| PCA + DBSCAN | 0.1638 | 0.7143 | 16.3364 |

La comparación común mejora la equivalencia de soporte, pero tampoco resume por sí sola calidad, cobertura o utilidad interpretativa. El proyecto muestra ambas perspectivas y advierte sobre sus diferencias.

## 5. Perfiles observados en esta ejecución

Los nombres siguientes son resúmenes calculados de las medias estandarizadas; las proporciones se calculan sobre 410 días persona-día.

| Modelo | Grupo | Días | Proporción | Perfil descriptivo sugerido |
|---|---:|---:|---:|---|
| K-Means K=3 | 0 | 160 | 39.02 % | Más pasos; actividad intensa/moderada elevada |
| K-Means K=3 | 1 | 180 | 43.90 % | Mayor duración del sueño; actividad intensa/moderada baja |
| K-Means K=3 | 2 | 70 | 17.07 % | Sedentarismo elevado; menos pasos |
| PCA + K-Means K=4 | 0 | 132 | 32.20 % | Actividad intensa/moderada elevada; más pasos |
| PCA + K-Means K=4 | 1 | 54 | 13.17 % | Sedentarismo elevado; menor duración del sueño |
| PCA + K-Means K=4 | 2 | 123 | 30.00 % | Actividad ligera elevada; sedentarismo reducido |
| PCA + K-Means K=4 | 3 | 101 | 24.63 % | Menos pasos; mayor duración del sueño |

DBSCAN original marcó 143 días como ruido (34.88 %); PCA + DBSCAN marcó 72 (17.56 %). Los restantes grupos pequeños de ambos modelos pueden representar estructuras locales o ser sensibles a los parámetros. Se recomienda describirlos con sus tamaños y distribución, sin asignarles significado poblacional fuerte de forma automática.

## 6. Estado de hallazgos de auditoría atendidos

- **Resultados obsoletos:** cada ajuste conserva su preprocesamiento, parámetros, huella y, cuando aplica, huella PCA. La comparación y los perfiles solo incluyen modelos vigentes. Los parámetros activos persisten al navegar.
- **Exposición de identificadores:** las vistas públicas y tablas de observaciones usan pseudónimos estables y fecha; el `Id` original no se presenta en esas tablas.
- **Soporte distinto:** se separan métricas individuales y evaluación común para las cuatro alternativas.
- **Validez de comparación:** las métricas comunes conservan sus validaciones y la UI comunica condiciones no satisfechas.
- **Reruns y costo:** búsqueda/grillas/estabilidad se invocan mediante botones y los cálculos repetibles usan caché por parámetros.
- **Atípicos K-Means:** se añade distancia al centroide con percentil configurable, expuesta como indicador descriptivo.
- **Matriz escalada:** la página de preprocesamiento distingue/exporta datos originales seleccionados y datos estandarizados utilizados por los modelos.
- **Interpretación:** se presentan perfiles basados en datos, matriz de calor, limitaciones y decisión humana documentable.
- **Fecha fuente en integración:** la carga mantiene texto de fecha original para la auditoría de duplicados exactos.

## 7. Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| `.venv\Scripts\python.exe -m pytest -q` | 16 pruebas aprobadas |
| `python -m compileall -q src app.py tests` | Sin errores de compilación |
| `python -m pip check` | Sin dependencias rotas |
| Importación y ejecución de cuatro modelos con datos reales | 410 filas; cuatro resultados reproducidos |
| Comparación común de cuatro modelos | 267 observaciones; validación correcta |
| Streamlit en modo headless y petición HTTP local | HTTP 200; proceso detenido al terminar la comprobación |
| Notebook reproducible en copia temporal | `nbconvert` terminó con código 0; 6 celdas de código y 0 salidas de error |

## 8. Límites pendientes para la entrega académica

Los nombres de perfil son etiquetas descriptivas basadas en umbrales de desviación estandarizada, no una validación externa de tipos de usuario. DBSCAN continúa siendo sensible a sus parámetros y los grupos pequeños requieren cautela. La cohorte disponible tiene 33 participantes y aproximadamente un mes de mediciones; la generalización a otras poblaciones no está establecida. El sueño no cubre todos los días con actividad. Las mediciones provienen de dispositivos y no son evaluaciones clínicas.

El equipo debe completar en la app los argumentos y limitaciones específicas de su decisión final y revisarlos con criterio académico. Los resultados estadísticos informan esa decisión, pero no la sustituyen.

La ejecución aislada del notebook emitió una advertencia de compatibilidad del event loop/ZMQ de Windows; la advertencia no impidió ejecutar sus celdas.

| Etapa | Estado después de la revisión | Pendiente o evidencia |
|---|---|---|
| 1. Comprensión del problema | Cumplida | Alcance descriptivo, unidad persona-día y límites documentados. |
| 2. Carga y EDA | Cumplida técnicamente | La discusión sustantiva debe integrarse al informe del equipo. |
| 3. Preprocesamiento | Cumplida | Se exponen opciones y matriz estandarizada. |
| 4. K-Means | Cumplida técnicamente | Justificar el K elegido para la pregunta del proyecto. |
| 5. DBSCAN | Cumplida técnicamente | Explicar el intercambio entre cobertura, ruido y estructura pequeña. |
| 6. PCA | Cumplida técnicamente | Reportar varianza retenida y reconstrucción al presentar resultados. |
| 7. Modelos combinados | Cumplida técnicamente | Parámetros independientes y PCA compartido trazables. |
| 8. Comparación | Cumplida técnicamente | Interpretar métricas junto con cobertura y soporte. |
| 9. Estabilidad | Cumplida técnicamente | Revisar y discutir la evidencia de estabilidad de la configuración elegida. |
| 10. Interpretación y decisión | Parcialmente cumplida | Requiere elección, argumentos y limitaciones escritos por el equipo. |

**Conclusión:** la interfaz y el flujo de análisis ofrecen ahora comparaciones trazables, perfiles basados en los datos y explicaciones para explorar K y DBSCAN. Para cerrar la entrega académica falta que el equipo registre y defienda su decisión, con las limitaciones correspondientes.
