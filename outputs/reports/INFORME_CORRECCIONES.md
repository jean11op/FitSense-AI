# Informe de correcciones — FitSense AI

**Fecha:** 10 de noviembre de 2026  
**Alcance:** correcciones técnicas de los hallazgos de la auditoría, pruebas de regresión y verificación reproducible con los CSV existentes. Se conservaron los archivos fuente originales y el entorno `.venv`. La ejecución del pipeline y del notebook que escribe artefactos se hizo en directorios temporales.

## Resultado

Se resolvieron H-01 y H-02, los ocho hallazgos medios M-01 a M-08 y cuatro hallazgos bajos B-02 a B-05. B-01 queda como decisión estructural pendiente: las diez vistas siguen dentro de `src/streamlit_app.py` usando navegación por radio. La elección académica final permanece en manos del equipo.

## Cambios realizados

| Área | Cambio | Archivos principales |
|---|---|---|
| Trazabilidad de modelos | Huellas de datos y orden, variables, imputación, extremos, PCA e hiperparámetros. Los artefactos guardan estimador, etiquetas copiadas, métricas, configuración y hora. Resultados incompatibles o desalineados se excluyen de comparación/perfiles. | `src/result_tracking.py`, `src/streamlit_app.py` |
| Privacidad | Pseudónimos deterministas para tablas, carga alternativa, días inusuales y descargas. `Id` no se incluye en las matrices públicas. | `src/result_tracking.py`, `src/streamlit_app.py` |
| Comparación | Cobertura y ruido explícitos para cada modelo; tabla sobre filas comunes no ruidosas por ambos DBSCAN, con las cuatro etiquetas evaluadas sobre idénticas observaciones en el espacio original escalado. | `src/model_comparison.py`, `src/evaluation.py`, `src/streamlit_app.py` |
| Validez de métricas | Guards para etiquetas alineadas y no faltantes, datos numéricos finitos, al menos dos grupos y más observaciones que grupos. | `src/evaluation.py` |
| Ejecución Streamlit | Búsqueda K, grilla DBSCAN, ajustes y estabilidad son acciones explícitas; los cálculos puros se cachean. Se guardan y se muestran parámetros de exploración. Se eliminó `dict.get(..., fit_pca(...))` con efecto de evaluación anticipada. | `src/streamlit_app.py` |
| Atípicos | Distancia a la media del cluster asignado en el espacio original escalado, con percentil seleccionable y tabla de pseudónimo/fecha/grupo/distancia. Ruido DBSCAN conserva pseudo-ID y fecha. | `src/profiling.py`, `src/streamlit_app.py` |
| Preprocesamiento | Dos vistas y dos exportaciones: características antes de escalar y matriz StandardScaler utilizada por modelos, alineadas con metadatos pseudonimizados. IQR de Tukey y winsorización 5–95 % ahora difieren. | `src/preprocessing.py`, `src/result_tracking.py`, `src/streamlit_app.py` |
| Interpretación | Matriz de evidencia con métricas, soporte, estabilidad, varianza PCA y pros/contras; limitaciones estructuradas y campos para que el equipo documente elección y argumentos sin selección automática. | `src/decision_support.py`, `src/streamlit_app.py` |
| Fechas fuente e integración | Se preservan cadenas originales de fecha. Deduplicación exacta considera los valores fuente; las claves duplicadas tras normalización detienen el proceso y guardan filas conflictivas y duplicados exactos descartados en auditoría. | `src/data_loader.py`, `src/integration.py` |
| Compatibilidad/entregables | `width="stretch"`, reglas de Git ignore para datos con IDs y excepción para informes Markdown, siete IDs de celda únicos. No se creó repositorio Git. | `src/streamlit_app.py`, `.gitignore`, `notebooks/analisis_reproducible.ipynb` |

## Estado de hallazgos

| Hallazgo | Estado |
|---|---|
| H-01 Resultados obsoletos | Corregido y vinculado a huella de configuración. |
| H-02 Exposición de `Id` | Corregido en elementos públicos de la interfaz y exportaciones. |
| M-01 Soportes distintos | Corregido con soporte común para las cuatro alternativas. |
| M-02 Validez límite | Corregido y cubierto con pruebas de etiquetas/soportes pequeños. |
| M-03 Recálculos en reruns | Corregido mediante botones y caché por entradas. |
| M-04 Atípicos K-Means | Corregido con distancia y percentil configurable. |
| M-05 Matriz escalada | Corregido con vista y descarga. |
| M-06 Interpretación final | Corregido como soporte de decisión humana; el juicio sustantivo lo completa el equipo. |
| M-07 Grilla DBSCAN | Corregido con botón, parámetros guardados y rótulos; k-distances se cachea. |
| M-08 Hora fuente | Corregido; colisiones normalizadas conservan filas para auditoría. |
| B-01 Navegación monolítica | Pendiente aceptado; no se dividió la app en directorio `pages/`. |
| B-02 IQR/winsorize iguales | Corregido con límites distintos y prueba. |
| B-03 `use_container_width` | Corregido a `width="stretch"`. |
| B-04 Entregables ignorados | Corregido para Markdown de informes; CSV con datos/IDs permanecen ignorados. |
| B-05 IDs notebook | Corregido y validado con nbformat. |

## Verificación

- `.venv\Scripts\python.exe -m pytest -q`: **14 passed**. Incluye métricas y alineación, soporte común, trazabilidad, pseudónimos, exportación estandarizada, distancias a centroides, límites de extremos y navegación/acciones principales de Streamlit.
- `compileall` para `src`, `app.py` y `tests`: salida limpia.
- `pip check`: no encontró dependencias rotas.
- Pipeline de integración con CSV reales en carpeta temporal: **410 filas**, auditoría creada; originales sin cambios.
- Modelos en memoria: K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN ejecutados. Matriz de entrada: 410 × 6; PCA retuvo 4 componentes y 92.56 %; los cuatro DBSCAN/alternativas se compararon sobre 267 observaciones comunes asignadas.
- `AppTest`: las diez vistas y los botones de K-Means, DBSCAN, PCA y modelos combinados terminaron sin excepciones; descarga de matriz estandarizada disponible.
- Streamlit respondió **HTTP 200**.
- Notebook: siete IDs únicos, `nbformat.validate` correcto y seis celdas de código ejecutadas en copia temporal sin errores. Se emitieron advertencias propias del event loop/ZMQ de Windows, sin impedir la ejecución.

`AppTest` y HTTP 200 confirman ejecución de los componentes, pero no equivalen a una revisión visual manual en navegador. No se inicializó Git ni se modificaron los CSV originales.
