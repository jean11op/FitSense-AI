# Guía de archivos del proyecto FitSense AI

Esta guía describe el propósito de los archivos de código, documentación y datos que componen el proyecto. La aplicación estudia patrones de actividad y sueño a nivel persona-día con K-Means, DBSCAN, PCA y sus combinaciones. Los resultados son descriptivos y no tienen uso clínico.

## 1. Entrada y configuración

| Archivo | Funcionamiento |
|---|---|
| `app.py` | Punto de entrada de Streamlit. Importa `run` desde `src.streamlit_app` y lo ejecuta al iniciar con `streamlit run app.py`. |
| `src/__init__.py` | Marca `src` como paquete Python y presenta el proyecto como la lógica reutilizable de FitSense AI. |
| `src/config.py` | Define la raíz del proyecto, las rutas a datos y salidas, la semilla `42` y las seis variables de clustering predeterminadas. Resuelve las rutas a partir de la ubicación del archivo. |
| `requirements.txt` | Fija las versiones de Python packages que usa la app, análisis, gráficos, Excel, pruebas y notebook. |
| `.gitignore` | Evita incluir entorno virtual, cachés, secretos y archivos de datos originales/procesados. Permite conservar los informes Markdown de `outputs/reports/`. |
| `.streamlit/config.toml` | Define el tema oscuro de Streamlit y sus colores de acento, fondo y texto. |
| `README.md` | Resume el objetivo, arquitectura, instrucciones para instalar dependencias, regenerar datos y arrancar la aplicación. |

## 2. Carga, integración y diccionario de datos

| Archivo | Funcionamiento |
|---|---|
| `src/data_loader.py` | Lee los CSV de actividad y sueño, exige sus columnas principales, mantiene `Id` como texto, conserva la fecha fuente y crea fechas normalizadas. Falla si encuentra fechas ilegibles. |
| `src/integration.py` | Integra actividad y sueño por `Id` y fecha con una relación uno a uno. Cuenta duplicados exactos de sueño, detecta claves persona-fecha conflictivas y mantiene las filas involucradas para auditoría. `run_integration` guarda los CSV/Excel procesados y el libro de auditoría. |
| `src/data_validation.py` | Comprueba que exista la clave `Id`-`Date` e informa unicidad y valores faltantes en esas columnas. Es una validación reutilizable para revisar una tabla integrada. |
| `src/data_dictionary.py` | Mantiene descripciones de las variables del dataset y genera la tabla del diccionario que presenta Streamlit. |
| `docs/diccionario_datos.xlsx` | Libro Excel con el diccionario de variables entregado como material de consulta. |

## 3. EDA y preprocesamiento

| Archivo | Funcionamiento |
|---|---|
| `src/eda.py` | Calcula faltantes, estadísticas descriptivas, correlaciones y hallazgos de resumen sobre un DataFrame. Los hallazgos son descriptivos y aclaran que correlación no implica causalidad. |
| `src/preprocessing.py` | Selecciona variables numéricas y conserva `Id`/fecha como metadatos alineados. Permite imputar faltantes mediante mediana, media o cero, o excluir filas incompletas. Trata extremos con conservación, límites de Tukey (IQR) o winsorización a percentiles 5 y 95; finalmente aplica `StandardScaler`. |
| `src/result_tracking.py` | Genera pseudónimos deterministas, prepara matrices públicas para exportación y calcula huellas de datos/configuración. Registra modelo, etiquetas, métricas, parámetros, fecha de ejecución y verifica si un resultado sigue correspondiendo a la configuración activa. |

## 4. Modelos y análisis estadístico

| Archivo | Funcionamiento |
|---|---|
| `src/kmeans_model.py` | Ajusta K-Means reproducible con inicialización k-means++ y al menos diez inicializaciones; busca varios valores de K con métricas y tamaños de grupo, y construye tablas de centroides. |
| `src/dbscan_model.py` | Calcula la curva k-distances, ajusta DBSCAN y clasifica puntos como ruido, núcleo o borde. También explora combinaciones de `eps` y `min_samples`. |
| `src/pca_model.py` | Ajusta PCA sobre variables estandarizadas, elige el número de componentes según un umbral de varianza y calcula puntuaciones, cargas, varianza retenida y error de reconstrucción. |
| `src/combined_models.py` | Aplica K-Means o DBSCAN sobre puntuaciones PCA y permite reconstruir los centroides de PCA + K-Means en unidades originales. |
| `src/evaluation.py` | Calcula Silhouette, Davies–Bouldin y Calinski–Harabasz con validaciones de dimensión, finitud, alineación y cantidad de observaciones/grupos. `compare_labels` evalúa dos particiones sobre filas comunes asignadas. |
| `src/model_comparison.py` | Compara modelos, conserva las cuatro alternativas aunque alguna esté pendiente, e informa grupos, puntos de ruido, soporte, cobertura y estabilidad. Incluye evaluación común de las cuatro etiquetas sobre los días asignados por ambos DBSCAN. |
| `src/kmeans_stability.py` | Ejecuta K-Means con varias semillas y resume métricas por corrida junto con ARI entre particiones. |
| `src/dbscan_stability.py` | Evalúa DBSCAN con varias combinaciones de parámetros; resume grupos, ruido, cobertura y ARI entre configuraciones, tanto incluyendo ruido como en observaciones agrupadas por ambas. |
| `src/profiling.py` | Calcula perfiles con medias, medianas, desviaciones estándar y medias estandarizadas; sugiere descripciones basadas en los datos. También lista ruido DBSCAN y marca candidatos inusuales por distancia al centroide K-Means y percentil configurable. |
| `src/decision_support.py` | Construye una matriz de evidencia con ventajas, limitaciones, métricas, estabilidad y varianza PCA. Guarda una elección argumentada por el equipo; no elige un algoritmo automáticamente. |

## 5. Interfaz, visualización y exportación

| Archivo | Funcionamiento |
|---|---|
| `src/streamlit_app.py` | Implementa diez secciones en una aplicación Streamlit: inicio, calidad, EDA, preprocesamiento, K-Means, DBSCAN, PCA, modelos combinados, comparación/estabilidad y perfiles/conclusiones. Controla el estado de sesión, descarta resultados incompatibles, permite ejecutar búsquedas y estabilidad por acción, cachea cálculos repetibles y presenta observaciones públicas con pseudónimos. |
| `src/visualization.py` | Ofrece una función reutilizable para generar un gráfico Plotly de las dos primeras puntuaciones y colorearlas por grupo. |
| `src/export_results.py` | Exporta un DataFrame como CSV y crea la carpeta de destino si hace falta. Es una utilidad genérica de exportación. |
| `src/run_pipeline.py` | Ejecuta la integración desde los CSV originales, prepara las características predeterminadas, guarda la matriz estandarizada y registra su configuración. Se ejecuta con `python -m src.run_pipeline`. |
| `notebooks/analisis_reproducible.ipynb` | Presenta un recorrido ejecutable de integración, EDA, preprocesamiento, cuatro modelos y comparación. La integración del notebook escribe artefactos procesados y auditoría en la raíz usada por el notebook. |

## 6. Scripts iniciales de EDA

Estos cuatro archivos son ejercicios breves e independientes que preceden a la app modular. Leen los CSV originales y muestran resultados en consola; no sustituyen a `src/eda.py` ni generan la interfaz Streamlit.

| Archivo | Funcionamiento |
|---|---|
| `eda/01_carga_Datos.py` | Carga ambos CSV y muestra una vista inicial de cada fuente. |
| `eda/02_dimensiones_estructura.py` | Muestra dimensiones, nombres de columnas, primeras y últimas filas de ambas fuentes. |
| `eda/03_informacion_estructural.py` | Imprime el resumen estructural de los DataFrames con `info()`. |
| `eda/04_estadisticas_descriptivas.py` | Muestra estadísticas descriptivas de las columnas, excluyendo `Id`. |

## 7. Pruebas

| Archivo | Funcionamiento |
|---|---|
| `tests/test_core.py` | Reúne pruebas de integración y duplicados, normalización de fechas, preprocesamiento, métricas, modelos, comparación común, estabilidad, pseudónimos, exportación y candidatos atípicos. También recorre las diez vistas principales de Streamlit y ejecuta los botones principales de modelos con `AppTest`. |

Para ejecutar las pruebas desde la raíz: `.venv\\Scripts\\python.exe -m pytest -q` en Windows.

## 8. Documentación del proyecto

| Archivo | Funcionamiento |
|---|---|
| `docs/procedencia_dataset.md` | Documenta archivos fuente, unión, duplicados y auditoría del dataset. |
| `docs/problema_y_objetivos.md` | Define problema, pregunta de investigación, unidad persona-día, objetivo y límites de interpretación. |
| `docs/metodologia.md` | Describe las etapas de EDA, preprocesamiento, modelos, comparación, estabilidad e interpretación. |
| `docs/presentacion_estructura.md` | Propone el orden y contenido de una presentación académica de doce diapositivas. |
| `docs/informe_resultados_template.md` | Plantilla para redactar el informe de resultados después de ejecutar y revisar los modelos. |
| `docs/distribucion_trabajo.md` | Resume el reparto de tareas y los contratos de datos entre módulos e integrantes. |
| `outputs/reports/AUDITORIA_TECNICA.md` | Registra el diagnóstico técnico inicial y añade el estado verificado después de las correcciones. |
| `outputs/reports/INFORME_CORRECCIONES.md` | Resume los cambios realizados, el estado de los hallazgos y las comprobaciones ejecutadas. |
| `outputs/reports/REVISION_FINAL_FITSENSE_AI.md` | Documenta las mejoras de presentación, configuraciones y métricas reproducidas, perfiles observados, verificaciones y asuntos académicos pendientes. |
| `outputs/.gitkeep` | Mantiene el directorio `outputs/` en un control de versiones aunque no contenga otros archivos versionables. |

## 9. Archivos de datos y salidas generadas

| Archivo | Funcionamiento |
|---|---|
| `data/original/dailyActivity_merged.csv` | Fuente local con medidas diarias de actividad. La canalización la lee sin modificarla. |
| `data/original/sleepDay_merged.csv` | Fuente local con medidas diarias de sueño. La canalización la lee sin modificarla. |
| `data/integrated/FitSense_AI_Dataset_Integrado.xlsx` | Excel de referencia usado para contrastar la integración reproducida. No reemplaza los CSV originales como fuente. |
| `data/processed/person_day_integrated.csv` | Resultado tabular de la unión persona-día, generado por el pipeline. |
| `data/processed/person_day_integrated.xlsx` | Copia Excel de la tabla integrada. |
| `data/processed/activity_without_sleep.csv` | Filas de actividad sin observación de sueño coincidente. |
| `data/processed/sleep_without_activity.csv` | Filas de sueño sin actividad coincidente; puede contener solo encabezados si no hay filas. |
| `data/processed/preprocessed_features.csv` | Metadatos más características estandarizadas generadas por `src.run_pipeline`. |
| `data/processed/preprocessing_config.json` | Variables, imputación, tratamiento de extremos y escalador usados en esa preparación. |
| `outputs/integration_audit.xlsx` | Libro con conteos de integración, discrepancias con el Excel de referencia, conflictos y filas exactas descartadas. |

**Privacidad:** los CSV y Excel de `data/` son artefactos internos y pueden incluir el identificador original `Id`. `.gitignore` evita versionar estas carpetas. No se deben compartir directamente los archivos procesados; para presentar resultados, usar la interfaz o exportaciones que sustituyen `Id` por pseudónimos.

## 10. Recorrido general

```text
data/original/*.csv
        │
        ├── src/data_loader.py ── src/integration.py ── data/processed/ + outputs/integration_audit.xlsx
        │                                           │
        │                                           └── src/eda.py
        │
        └── src/preprocessing.py ── StandardScaler ── K-Means / DBSCAN / PCA / modelos combinados
                                                        │
                                                        ├── evaluation, comparación y estabilidad
                                                        └── perfiles y vistas en src/streamlit_app.py
```
