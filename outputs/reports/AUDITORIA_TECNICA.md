# Auditoría técnica y metodológica — FitSense AI

**Fecha:** 10 de noviembre de 2026  
**Alcance:** revisión de solo lectura del proyecto, ejecución de pruebas y modelos con los datos existentes. No se corrigieron fuentes ni configuraciones. La ejecución del notebook se hizo en una copia temporal para no regenerar los artefactos procesados del workspace.

> **Actualización posterior:** este documento conserva debajo el diagnóstico original como fotografía previa a los cambios. La sección 19 y [INFORME_CORRECCIONES.md](INFORME_CORRECCIONES.md) reflejan el estado verificado después de aplicar las correcciones autorizadas el 10 de noviembre de 2026. Los datos originales y el entorno virtual se conservaron.

## 1. Resumen ejecutivo

La revisión inicial que sigue encontró 410 filas persona-día, y 5 pruebas unitarias pasaban entonces. Después de las correcciones, **14 pruebas pasan**; el compilador no reporta errores, `pip check` no encuentra conflictos, las diez vistas y cuatro acciones de modelos funcionan con `AppTest`, y Streamlit responde HTTP 200. El notebook validado ejecutó seis celdas de código en una copia temporal.

Se corrigieron los dos hallazgos altos: cada resultado queda asociado a la configuración y al número/orden de observaciones; las tablas y descargas de la interfaz usan pseudónimos. También se añadieron comparación sobre soporte común, días potencialmente atípicos de K-Means, matrices pre y post escalado, registro de duplicados fuente y una vista de evidencia para decisión humana. La interfaz no se inspeccionó visualmente en un navegador; `AppTest` y HTTP 200 verifican ejecución, no legibilidad visual.

### Resultados calculados para una configuración de auditoría

Con las seis variables por defecto, K-Means `K=3`, DBSCAN `eps=0.8`, `min_samples=5` y PCA con umbral de varianza 90 %:

- La matriz compartida tiene **410 observaciones × 6 características**. `Id` y `Date` no entran al escalador ni al clustering.
- K-Means: 3 grupos; Silhouette 0.2286, Davies–Bouldin 1.3780 y Calinski–Harabasz 116.9864. Entre cinco semillas, ARI medio 0.9927.
- DBSCAN original: 5 grupos, 267 observaciones evaluadas, 143 puntos de ruido y cobertura 65.12 %. Las métricas excluyen el ruido.
- PCA retuvo 4 componentes y 92.56 % de varianza; RMSE de reconstrucción escalado 0.2727.
- PCA + K-Means: evaluado en el espacio original escalado, Silhouette 0.2308.
- PCA + DBSCAN: 3 grupos, 338 observaciones evaluadas y cobertura 82.44 %; métricas principales calculadas en el espacio original escalado.
- El subconjunto común asignado por los dos DBSCAN contiene 267 observaciones. La estabilidad DBSCAN cambia sustancialmente con los parámetros auditados: ruido entre 14.88 % y 63.41 %, y entre 4 y 11 grupos en la grilla usada.

Estas cifras describen configuraciones de comprobación, **no son una recomendación automática**. Las métricas DBSCAN de la tabla general usan muestras asignadas de distinto tamaño y no deben ordenarse como si fueran comparaciones sobre idénticas observaciones.

## 2. Arquitectura examinada

La separación entre carga, integración, preprocesamiento, modelos, evaluación y Streamlit está implementada en `src/`. `app.py` funciona como punto de entrada y resuelve las rutas de datos desde `src/config.py` con `Path(__file__).resolve()`, no desde el directorio actual. Los cuatro scripts de `eda/` se conservan como ejercicios iniciales independientes.

La aplicación presenta diez vistas mediante un único `st.radio` en `src/streamlit_app.py`; no existe una carpeta `pages/` ni páginas Streamlit independientes. La navegación por las diez vistas sí se ejecutó con `AppTest`. Esto funciona, aunque difiere de la arquitectura multipágina propuesta y concentra mucho comportamiento en un módulo de unas 174 líneas.

No existe directorio `.git` en el workspace, por lo que no se pudo revisar historial, rama ni seguimiento de archivos. `.gitignore` excluye `data/processed/*` y `outputs/*`; la excepción actual solo conserva `outputs/.gitkeep`. Al inicializar Git, los CSV procesados, la auditoría y este informe no se incluirían sin reglas adicionales. `requirements.txt` fija las dependencias directas, incluyendo Jupyter; `pip check` no encontró conflictos. `seaborn` está instalado/declarado pero no aparece usado en la lógica del proyecto.

## 3. Auditoría de datos

| Fuente o artefacto | Filas × columnas | Faltantes | Duplicados exactos | Observación |
|---|---:|---:|---:|---|
| `dailyActivity_merged.csv` | 940 × 15 | 0 | 0 | 33 participantes; 2016-04-12 a 2016-05-12 |
| `sleepDay_merged.csv` | 413 × 5 | 0 | 3 | 24 participantes; mismo rango de fechas |
| Integrado reproducido | 410 × 18 | 0 | 0 | 24 participantes coincidentes; clave Id-fecha única |
| Actividad sin sueño | 530 × 16 | 0 | — | Ningún registro de sueño quedó sin actividad |
| Sueño sin actividad | 0 × 6 | 0 | — | Archivo vacío, con encabezados |
| Variables procesadas | 410 × 8 | 0 | 0 | Id, fecha y seis características escaladas |

El CSV integrado y el XLSX procesado están alineados y reproducen los valores del Excel de referencia (410 filas; cero diferencias tras ordenar y normalizar tipos). El libro de auditoría informa 940 actividad, 413 sueño, 3 duplicados exactos de sueño, 410 coincidencias y unicidad persona-día.

Los ceros se conservan. En los 410 días integrados, `VeryActiveMinutes` tiene 35.9 % de ceros y `FairlyActiveMinutes` 32.7 %; no son faltantes. La correlación más alta en las seis variables candidatas es `SedentaryMinutes`–`TotalMinutesAsleep` (r≈−0.60). No se incluyeron en las características por defecto los pares de distancia redundantes (`TotalDistance`/`TrackerDistance`) ni `TotalTimeInBed` junto con `TotalMinutesAsleep`. La selección por defecto es razonable como candidata, pero el proyecto no registra una justificación empírica formal de selección más allá de permitir modificarla.

## 4. Auditoría de integración

`load_activity` y `load_sleep` leen `Id` como texto y validan las columnas requeridas. Las fechas se convierten y normalizan. `integrate` hace una unión interna `one_to_one` con `validate="one_to_one"`, quita duplicados exactos solo de sueño y detiene la integración ante claves Id-fecha repetidas después de quitar esos exactos. En los datos reales no existen duplicados de clave de actividad; las tres filas duplicadas de sueño son exactas y, tras deduplicarlas, no queda ninguna clave conflictiva.

Se reprodujo `integrate` directamente en memoria desde ambos CSV reales, sin escribir sobre `data/processed/`; el resultado coincide con el CSV procesado y con el Excel de referencia. `Date` se conserva como metadato, e `Id` sigue siendo texto. Los originales no se escribieron durante esta auditoría.

**Límite verificado:** la función de bajo nivel `integrate` cuenta duplicados después de crear una fecha normalizada, pero conserva las columnas de fecha originales que recibe. En el flujo normal, `load_sleep` normaliza `SleepDay` antes de invocarla; por tanto, dos filas con distinta hora original y mismas mediciones podrían convertirse en duplicados exactos antes del conteo. En el CSV actual todas las horas son medianoche, de modo que no cambia el resultado observado. Recomendación: contar y quitar duplicados exactos sobre las filas fuente antes de sobrescribir la fecha original o guardar explícitamente la fecha bruta.

## 5. Auditoría de EDA

`src/eda.py` implementa resumen descriptivo, faltantes, correlaciones y cuatro hallazgos automáticos. La interfaz ofrece dimensiones, tipos, duplicados, faltantes, estadísticas, histogramas con caja marginal y matriz de correlación. Los textos aclaran que la correlación no implica causalidad. El análisis real confirma cero faltantes en ambas fuentes.

Parcialidad: los cuatro hallazgos son genéricos y no explican por sí mismos la distribución de ceros ni los pares redundantes detectados; la interpretación académica sustantiva todavía debe redactarse. Los cuatro scripts históricos de `eda/` solo cargan/imprimen datos y no constituyen una segunda implementación completa del EDA.

## 6. Auditoría del preprocesamiento

`prepare_features` excluye implícitamente metadatos al seleccionar solo columnas indicadas, convierte las candidatas a numéricas, admite `drop`, mediana, media o cero, ofrece conservar o limitar extremos y aplica `StandardScaler`. Los metadatos se reinician en el mismo orden que las filas procesadas. En los datos reales el resultado escalado tiene media cero y desviación estándar poblacional uno por característica. No se detectaron faltantes en la integración, por lo que la imputación mediana por defecto no alteró valores; la estrategia queda en `preprocessing_config.json`.

La selección por defecto de seis variables se comparte en la ejecución normal de los cuatro modelos. Sin embargo, la UI descarga y muestra `prep["raw"]`, no la matriz estandarizada que alimenta el clustering; el archivo `preprocessed_features.csv` creado por el pipeline sí contiene columnas escaladas. La UI debería hacer explícita esta diferencia para que el usuario vea y descargue la representación realmente usada.

Las opciones `iqr` y `winsorize` ejecutan exactamente el mismo recorte a los límites Q1−1.5×IQR y Q3+1.5×IQR. La duplicación de opciones no cambia los resultados, pero sugiere tratamientos diferentes cuando hoy son equivalentes.

## 7. Auditoría de K-Means

`fit_kmeans` y `search_k` usan k-means++, `n_init` nunca menor que 10 y semilla configurable. La búsqueda real de K=2..10 funciona, informa inercia, Silhouette, Davies–Bouldin y Calinski–Harabasz. `centroid_table` y la UI revierten centroides a unidades originales. Para K=3 y cinco semillas, el ARI medio fue 0.9927 y las métricas variaron poco.

La UI no guarda el K y la identidad de la matriz junto con el resultado: si el usuario mueve el slider después de entrenar, la tabla de métricas/centroides puede seguir mostrando el modelo previo mientras la estabilidad se calcula con el K recién seleccionado. Este comportamiento forma parte del hallazgo alto de estado obsoleto (sección 17).

## 8. Auditoría de DBSCAN

`fit_dbscan` usa `eps` y `min_samples`, mantiene ruido con etiqueta −1, reconoce puntos núcleo/borde/ruido y calcula métricas sobre los no-ruido. `clustering_metrics` deja métricas como NaN si no hay al menos dos grupos o si el número de filas no supera al de grupos. Para la configuración auditada, 207 fueron núcleo, 60 borde y 143 ruido. Cobertura y número evaluado se informan.

`k_distances` calcula la curva del k-ésimo vecino. La interfaz explora una grilla fija de seis `eps` y tres valores de `min_samples` alrededor del control. La grilla y el resumen de estabilidad se vuelven a calcular en cada rerun de la aplicación, aunque no se haya pedido un nuevo análisis.

## 9. Auditoría de PCA

`fit_pca` ajusta PCA a la matriz recibida, informa varianza por componente, acumulada, cargas y RMSE de reconstrucción en escala estandarizada. Para la matriz por defecto, el umbral 90 % requiere 4 componentes y retiene 92.56 %; el RMSE fue 0.2727. El ajuste se alimenta de datos escalados en el flujo de Streamlit y notebook.

Parcialidad: la vista PCA muestra scree, acumulada y cargas, pero no presenta por sí sola la gráfica PC1–PC2; esta aparece en la vista combinada. La función `fit_pca` documenta que recibe datos estandarizados, pero no valida ese contrato ni el rango del umbral cuando se invoca desde otro módulo.

## 10. Auditoría de modelos combinados

Los wrappers combinados aplican K-Means o DBSCAN a los mismos scores PCA que reciben. En la UI, ambos se ajustan sobre el mismo objeto PCA y el mismo número de componentes seleccionados; K-Means reconstruye sus centroides con `inverse_transform` y el escalador original. Los cuatro vectores de etiquetas mantienen las 410 posiciones en la ejecución por defecto.

La garantía depende del estado de sesión: si se cambia el preprocesamiento después de ajustar modelos, no se invalidan los scores ni las etiquetas previas. No hay registro persistente de un ID de configuración que pruebe qué variables, scaler, componentes e hiperparámetros produjeron cada etiqueta.

## 11. Auditoría de métricas y comparación

`compare_models` calcula la tabla sobre el espacio original escalado para todas las etiquetas disponibles. DBSCAN excluye ruido e informa `n_evaluated`, `n_noise` y cobertura. La UI añade una evaluación DBSCAN vs. PCA+DBSCAN en el subconjunto asignado por ambos; para la configuración comprobada fueron 267 días.

**Precaución metodológica:** la tabla principal coloca lado a lado métricas que usan 410 observaciones en K-Means, 267 en DBSCAN original, 410 en PCA+K-Means y 338 en PCA+DBSCAN. Aunque la UI documenta exclusión del ruido y la comparación común adicional, las cifras de Silhouette/Davies–Bouldin/Calinski–Harabasz de esa tabla no son equivalentes muestra a muestra. Deben rotularse como resultados por cobertura y acompañarse por una tabla común de soporte para todas las alternativas donde sea posible.

`compare_labels` comprueba que haya al menos 3 filas comunes y al menos 2 etiquetas por modelo, pero no comprueba `n_observations > n_clusters` antes de devolver `valid=True`. En un caso límite, `clustering_metrics` retornaría NaN, mientras la envoltura marcaría la comparación como válida. No ocurrió con los datos actuales; se recomienda alinear la condición de validez con la de las métricas internas.

La inercia queda solo en la búsqueda de K dentro de cada espacio; no se usa para comparar espacio original con PCA. No se produce una selección automática por Silhouette.

## 12. Auditoría de estabilidad

K-Means ejecuta cinco semillas por defecto y devuelve las tres métricas por semilla más ARI medio y desviación. Los datos reales muestran ARI medio 0.9927. DBSCAN cruza `eps` y `min_samples`, reporta grupos, ruido, cobertura, ARI incluyendo −1 como etiqueta y coincidencia/ARI de los elementos agrupados en ambos modelos.

La evidencia confirma que DBSCAN es sensible a sus hiperparámetros en la grilla auditada. No se deben presentar sus resultados como estables sin fijar un criterio de perturbación y revisar la grilla completa. En la UI, la grilla es fija y sus valores no se exportan junto al resultado escogido ni quedan asociados de forma persistente a este.

## 13. Auditoría de interpretación

`profile_clusters` calcula cantidad, media, mediana y desviación estándar por variable y grupo. La UI expresa diferencias de medias estandarizadas respecto de la media general, sin asignar nombres automáticos basados en el número de cluster. Se muestra una advertencia de que los resultados no son clínicos.

Parcialidad: la detección de atípicos en la interfaz se limita a etiquetas −1 de DBSCAN. La función `outlier_days` puede marcar el 5 % más distante si se le dan distancias, pero no se invoca y ningún flujo calcula distancias K-Means a centroides para marcar días complementarios. La tabla de días ruidosos elimina `Id` por completo; no presenta el identificador seudónimo y fecha que solicita el planteamiento. La vista final tampoco incorpora una sección efectiva de limitaciones y recomendación argumentada: el módulo `decision_support` devuelve solo una nota genérica y no se usa desde la UI.

## 14. Auditoría de Streamlit

`streamlit run app.py` inició sin error y una petición HTTP a la raíz devolvió 200. `AppTest` recorrió las diez opciones sin excepciones; también ejecutó los botones de K-Means, DBSCAN y los modelos combinados, y abrió comparación/perfiles sin excepción. Esto verifica ejecución de script y callbacks, **no inspección visual de los gráficos en un navegador**.

`st.session_state` conserva etiquetas/modelos, pero no conserva su configuración completa ni invalida resultados obsoletos al cambiar variables, imputación, tratamiento de extremos o hiperparámetros. K-Means recalcula la búsqueda completa al renderizar; DBSCAN recalcula la grilla al renderizar; la comparación vuelve a correr estabilidad. Además, `dict.get("pca", fit_pca(...))` evalúa el argumento por defecto antes de llamar `get`, así que ajusta PCA aunque ya exista en sesión. Estas acciones pueden repetirse en cada interacción sin pulsar el botón anunciado.

Streamlit 1.67.0 emitió advertencias de que `use_container_width` será eliminado después del 31-12-2025. Las tablas integradas y alternativas muestran la columna `Id` completa. La UI tampoco ofrece una visualización/descarga explícita de la matriz escalada, solo de la tabla previa al escalado.

## 15. Pruebas ejecutadas

| Comando/comprobación | Resultado |
|---|---|
| `.venv\Scripts\python.exe --version` | Python 3.14.4 |
| `.venv\Scripts\python.exe -m pytest -q` | 5 passed |
| `.venv\Scripts\python.exe -m pip check` | No broken requirements found |
| `.venv\Scripts\python.exe -m compileall -q src app.py tests` | Exit 0 |
| Import de módulos de `src` | Todos los módulos importaron |
| `integrate` con los dos CSV originales (en memoria) | 410 filas; unicidad Id-Date; coincide con procesado y Excel de referencia |
| Búsqueda K=2..10 + cuatro modelos + comparación + estabilidad (datos reales) | Ejecutados; métricas y cifras registradas en secciones 1, 7, 8, 11 y 12 |
| `AppTest` | Diez vistas y botones K-Means/DBSCAN/combinados sin excepciones |
| `streamlit run app.py` + petición HTTP | Servidor inició; HTTP 200 |
| `jupyter nbconvert --to notebook --execute` en copia temporal | Exit 0; 6 celdas de código ejecutadas; 0 salidas tipo error |

El notebook emitió advertencias de `nbformat` porque las celdas no tienen `id`, y una advertencia de compatibilidad del event loop/ZMQ de Windows; no impidieron ejecutar las celdas. No se comprobaron visualmente descargas ni legibilidad en navegador.

Las cinco pruebas unitarias usan fixtures sintéticos controlados. No hay pruebas automatizadas para lectura con `data_loader`, exportación, `data_validation`, tratamiento extremo, carga alternativa por la UI, límites de validez de `compare_labels` o la ejecución completa de `run_integration` ante conflicto. Los archivos reales sí se usaron en la reproducción in-memory, el notebook temporal y los modelos.

## 16. Entregables encontrados

- CSV originales: presentes en `data/original/`.
- Excel integrado de referencia: presente en `data/integrated/`.
- CSV/Excel integrados, archivos sin coincidencia y CSV escalado: presentes en `data/processed/`.
- Diccionario Excel: `docs/diccionario_datos.xlsx`.
- Auditoría de integración: `outputs/integration_audit.xlsx`.
- Código modular, `app.py`, notebook, `requirements.txt`, README y documentación metodológica: presentes.
- Plantilla del informe y estructura de presentación: presentes.
- Pruebas unitarias: presentes, con cobertura funcional limitada.
- No se encontró `pages/` ni un repositorio `.git`.
- El informe de resultados académicos sigue siendo plantilla; no existe una recomendación final respaldada y redactada.

## 17. Problemas identificados

### H-01 — Resultados de modelos quedan obsoletos al cambiar configuración (ALTO)

- **Archivo/sección:** `src/streamlit_app.py`, preprocesamiento 76–87; K-Means 88–99; DBSCAN 100–109; combinados 113–125; comparación 127–154; perfiles 155–168.
- **Descripción:** las etiquetas/modelos se guardan con claves fijas (`labels_kmeans`, etc.) sin guardar variables, tratamiento de faltantes/extremos, PCA, K, `eps` o `min_samples` usados. Modificar la configuración no elimina ni reentrena automáticamente las etiquetas guardadas.
- **Evidencia:** la página de preprocesamiento asigna siempre `st.session_state["prep"]`; los resultados permanecen en claves separadas. Comparación y perfiles aplican las etiquetas existentes al `prep` que esté activo.
- **Consecuencia:** una comparación puede puntuar etiquetas antiguas en una matriz nueva; perfiles pueden agrupar valores distintos de los que generaron el modelo. Si cambia el número de filas, se puede producir error de longitudes.
- **Corrección recomendada:** guardar un fingerprint/versionado de datos, features, estrategia, scaler, PCA y parámetros junto a cada resultado; invalidar resultados dependientes al cambiar el fingerprint y bloquear comparaciones con configuraciones incompatibles.

### H-02 — Exposición del identificador de participante en la interfaz (ALTO)

- **Archivo/sección:** `src/streamlit_app.py:59–64`, vista de calidad y carga alternativa.
- **Descripción:** `df.head(50)` y la vista de integración alternativa presentan `Id` sin enmascarar.
- **Evidencia:** `Id` se conserva y se envía directamente a `st.dataframe`; los valores reales son identificadores numéricos de 10 dígitos.
- **Consecuencia:** una publicación pública expone identificadores que el requisito de interfaz pide proteger.
- **Corrección recomendada:** retirar/enmascarar `Id` en todas las tablas públicas y usar un pseudónimo estable solo cuando la interpretación necesite distinguir participantes.

### M-01 — Métricas de la tabla comparativa usan soportes distintos (MEDIO)

- **Archivo/sección:** `src/model_comparison.py:5–10`; `src/streamlit_app.py:127–154`.
- **Descripción:** las métricas están calculadas en el mismo espacio, pero DBSCAN descarta el ruido y por tanto usa menos observaciones que K-Means. La comparación DBSCAN común solo cubre los dos DBSCAN, no todas las alternativas.
- **Evidencia:** con la configuración auditada, `n_evaluated` varía 267, 338 y 410 según modelo.
- **Consecuencia:** ordenar los cuatro Silhouette o Davies–Bouldin de la tabla puede inducir una conclusión inválida por diferencias de cobertura y composición de muestra.
- **Corrección recomendada:** titular claramente métricas “sobre observaciones asignadas”; añadir una tabla secundaria con soporte comparable común para las cuatro alternativas, reportando cuando no sea válido.

### M-02 — Límite incompleto de validez en comparación común (MEDIO)

- **Archivo/sección:** `src/evaluation.py:16–22`.
- **Descripción:** `compare_labels` marca `valid=True` con solo verificar ≥3 observaciones y ≥2 etiquetas. No garantiza `n > n_clusters` para cada etiqueta.
- **Evidencia:** `clustering_metrics` exige `n > len(clusters)` en líneas 11–13, pero la envoltura no hereda esta verificación antes de fijar `valid=True`.
- **Consecuencia:** puede anunciarse una comparación válida con métricas NaN en un caso límite.
- **Corrección recomendada:** comprobar las condiciones matemáticas de cada métrica, exponer por separado las métricas disponibles y dejar `valid=False` cuando falte alguna condición.

### M-03 — Recalcula exploraciones y estabilidad en reruns (MEDIO)

- **Archivo/sección:** `src/streamlit_app.py:91`, `99`, `109`, `114`, `133–145`.
- **Descripción:** búsqueda K, grilla DBSCAN y estabilidad se ejecutan durante la renderización. El valor por defecto de `dict.get` ajusta PCA aun cuando `pca` exista.
- **Evidencia:** llamadas a `search_k`, `explore_dbscan`, `assess_*_stability` fuera de botones; llamada `st.session_state.get("pca", fit_pca(...))` evalúa `fit_pca` primero.
- **Consecuencia:** cada cambio de widget puede repetir ajuste y renderizar tablas más lentas; escala peor al ampliar datos o grillas.
- **Corrección recomendada:** usar botones/formularios y resultados cacheados por fingerprint de configuración; reemplazar el `get` por una condición explícita.

### M-04 — No se muestran días K-Means alejados del centroide (MEDIO)

- **Archivo/sección:** `src/profiling.py:10–14`; `src/streamlit_app.py:155–168`.
- **Descripción:** la UI solo lista etiquetas DBSCAN −1; nunca calcula distancia a centroides para K-Means ni invoca `outlier_days`.
- **Evidencia:** la vista de perfiles filtra únicamente `labels == -1`; no hay componente que calcule dichas distancias.
- **Consecuencia:** no cumple la detección complementaria de observaciones potencialmente atípicas solicitada.
- **Corrección recomendada:** calcular distancia al centroide asignado en el espacio apropiado, definir y declarar el umbral, asociarlo a Id-fecha y mostrarlo como indicador descriptivo, no error.

### M-05 — Vista de preprocesamiento no muestra la matriz escalada usada (MEDIO)

- **Archivo/sección:** `src/streamlit_app.py:76–87`.
- **Descripción:** la pantalla/descarga usa `prep["raw"]`, aunque los modelos consumen `prep["scaled"]`.
- **Evidencia:** la matriz estandarizada se crea en `prepare_features`, pero esta vista nunca la muestra ni la incorpora al botón de descarga.
- **Consecuencia:** el usuario no puede inspeccionar ni descargar desde la UI la representación que produjo los clusters.
- **Corrección recomendada:** presentar ambas matrices con títulos y exportación inequívocos, incluyendo metadatos alineados y configuración aplicada.

### M-06 — Interpretación final y recomendación siguen incompletas (MEDIO)

- **Archivo/sección:** `src/decision_support.py:1–4`; página Perfiles en `src/streamlit_app.py:155–168`.
- **Descripción:** el módulo solo devuelve una nota fija y no se utiliza desde la UI. La última página muestra estadísticas, diferencias estandarizadas y una advertencia, pero no limitaciones estructuradas ni una alternativa recomendada con argumentos.
- **Evidencia:** la función `decision_guidance` no evalúa métricas, estabilidad, cobertura o interpretabilidad; no se importa en `streamlit_app.py`.
- **Consecuencia:** la Etapa 10 y las conclusiones académicas no están listas para sustentación.
- **Corrección recomendada:** mostrar explícitamente limitaciones y una plantilla de decisión revisada por el equipo, enlazando métricas, cobertura, estabilidad, perfiles y varianza PCA sin elegir automáticamente.

### M-07 — K-distances usa una sola grilla de exploración no confirmada por acción (MEDIO)

- **Archivo/sección:** `src/streamlit_app.py:100–109`.
- **Descripción:** cada rerun recalcula curva y 18 combinaciones alrededor del control; la grilla se ejecuta aunque no se pulse “Ejecutar DBSCAN”.
- **Consecuencia:** resultados de la búsqueda y modelo ajustado pueden corresponder a momentos/valores diferentes de la pantalla; desperdicia ejecuciones.
- **Corrección recomendada:** botón independiente para explorar, persistir los parámetros de grilla y rotular cuál fila corresponde al modelo entrenado.

### M-08 — Auditoría automática no conserva la hora fuente al deduplicar (MEDIO)

- **Archivo/sección:** `src/data_loader.py:18–24`; `src/integration.py:12–18`.
- **Descripción:** `SleepDay` se sobrescribe con fecha normalizada en la carga antes de que `integrate` cuente duplicados exactos.
- **Evidencia:** con datos que difieran solo en la hora, la carga puede convertirlos en filas exactas y eliminar una.
- **Consecuencia:** un dataset alternativo podría perder una fila que no era duplicado exacto según el archivo fuente.
- **Corrección recomendada:** hacer conteo/deduplicación exacta antes de normalizar o conservar `SleepDay_raw` para auditar la igualdad fuente.

### B-01 — Interfaz monolítica en lugar de módulos de página (BAJO)

- **Archivo/sección:** `src/streamlit_app.py:24–174`; no existe `pages/`.
- **Descripción:** las diez vistas funcionan mediante radio en un único script, no como páginas independientes.
- **Consecuencia:** reparto de trabajo y pruebas por página menos aislados que lo previsto.
- **Corrección recomendada:** separar páginas o mantener la radio pero documentar oficialmente esta arquitectura y extraer funciones de render por módulo.

### B-02 — Tratamientos `iqr` y `winsorize` son el mismo recorte (BAJO)

- **Archivo/sección:** `src/preprocessing.py:24–31`.
- **Descripción:** ambos valores ejecutan el mismo clip de 1.5×IQR.
- **Consecuencia:** la opción da una elección aparente sin cambiar el procedimiento.
- **Corrección recomendada:** eliminar la opción duplicada o implementar y describir estrategias con semántica distinta.

### B-03 — `use_container_width` está obsoleto en Streamlit instalado (BAJO)

- **Archivo/sección:** varios `st.dataframe`/`st.plotly_chart` en `src/streamlit_app.py`.
- **Descripción:** Streamlit 1.67.0 emite advertencia de migración a `width="stretch"`/`"content"`.
- **Consecuencia:** compatibilidad futura y consola ruidosa; no impidió ejecutar.
- **Corrección recomendada:** migrar antes de actualizar Streamlit.

### B-04 — La regla de ignorado excluye entregables que se desea versionar (BAJO)

- **Archivo/sección:** `.gitignore`, regla `outputs/*`.
- **Descripción:** ignora `outputs/reports/AUDITORIA_TECNICA.md`, la auditoría de integración y cualquier salida dentro de subcarpetas. Tampoco se encontró `.git`.
- **Consecuencia:** al inicializar Git, los informes no aparecerán en `git status` ni se publicarán a menos que se ajusten excepciones.
- **Corrección recomendada:** definir de forma explícita qué artefactos pequeños deben versionarse y qué datos generados se regeneran.

### B-05 — Notebook sin IDs de celda (BAJO)

- **Archivo/sección:** `notebooks/analisis_reproducible.ipynb`.
- **Descripción:** `nbformat` emitió `MissingIDFieldWarning` al ejecutarlo.
- **Consecuencia:** futura versión de nbformat puede convertir la advertencia en error de validación.
- **Corrección recomendada:** normalizar/agregar IDs únicos a todas las celdas.

## 18. Recomendaciones de corrección

1. Antes de divulgar la app, ocultar `Id` en todo el recorrido de tablas.
2. Vincular cada resultado a la configuración exacta y limpiar estados al cambiar datos, variables, imputación, extremos, PCA o parámetros.
3. Separar la tabla de métricas por soporte y crear una comparación común explícita; corregir el guard de `compare_labels`.
4. Mover búsqueda de K, grillas DBSCAN y estabilidad detrás de acciones explícitas o caché por configuración.
5. Mostrar y exportar la matriz escalada y sus metadatos en la página de preprocesamiento.
6. Agregar distancia a centroides para atípicos K-Means y completar limitaciones/decisión final con validación humana.
7. Añadir pruebas para loader, exportación, claves conflictivas desde el wrapper de escritura, estado obsoleto, comparaciones límite y descarga.
8. Preservar valores de fecha bruta para que duplicados exactos se determinen sobre registros fuente.
9. Resolver advertencias de Streamlit/nbformat antes de publicar, y ajustar `.gitignore` una vez se defina el repositorio.

## Matriz de cumplimiento de las diez etapas

| Etapa | Requisito | Estado | Evidencia y motivo |
|---|---|---|---|
| 1 | Comprensión del problema | CUMPLIDO | Objetivo, pregunta, persona-día, no supervisado vs. supervisado y límites documentados. |
| 2 | Carga y EDA | PARCIALMENTE CUMPLIDO | Estadísticas, calidad, gráficas y hallazgos funcionan; la interpretación sustantiva requiere desarrollo humano. |
| 3 | Preprocesamiento | PARCIALMENTE CUMPLIDO | Faltantes, ceros, extremos y escalado configurables; UI no muestra/descarga la matriz escalada y hay opciones duplicadas. |
| 4 | K-Means | PARCIALMENTE CUMPLIDO | Algoritmo, búsqueda, métricas, centroides y estabilidad operan; persistencia no vincula resultados con parámetros/configuración. |
| 5 | DBSCAN | PARCIALMENTE CUMPLIDO | Ruido, cobertura, tipos de punto, métricas y sensibilidad implementados; búsqueda corre en reruns y los resultados no quedan ligados a su configuración. |
| 6 | PCA | PARCIALMENTE CUMPLIDO | Varianza, cargas, selección y RMSE funcionan; falta PC1–PC2 en la vista PCA independiente y no se valida escalado en la API. |
| 7 | Modelos combinados | PARCIALMENTE CUMPLIDO | Ambos usan scores PCA compartidos en una sesión y centroides reconstruidos; falta trazabilidad/invalidez del estado al cambiar configuración. |
| 8 | Comparación justa | PARCIALMENTE CUMPLIDO | Métricas en espacio original, ruido/cobertura y subconjunto común DBSCAN; la tabla principal mezcla soportes y el guard de comparación tiene un caso límite. |
| 9 | Estabilidad | PARCIALMENTE CUMPLIDO | ARI/variación y sensibilidad se ejecutan; DBSCAN muestra sensibilidad alta y las grillas no se documentan con el resultado elegido. |
| 10 | Interpretación y decisión | PARCIALMENTE CUMPLIDO | Estadísticas descriptivas y advertencia no clínica; faltan atípicos K-Means, pseudónimo-fecha y recomendación/limitaciones desarrolladas. |

**Conclusión de la auditoría inicial:** la base computacional corría y reproducía los datos reales, pero requería resolver H-01/H-02 y completar interpretación/comparabilidad. El estado actualizado de esos puntos se detalla en la sección 19.

## 19. Estado verificado después de las correcciones

| Hallazgo | Estado actual | Evidencia de corrección |
|---|---|---|
| H-01 Estado obsoleto | Corregido | Fingerprints de fuente/orden, variables, imputación, extremos, scaler, PCA e hiperparámetros. Comparación y perfiles aceptan solo resultados vigentes y alineados. |
| H-02 IDs expuestos | Corregido | Tablas, vistas de días, matrices descargables y carga alternativa sustituyen `Id` por pseudónimo determinista; el ID real sigue solo en memoria para joins. |
| M-01 Soportes distintos | Corregido | Tabla de asignados declara cobertura y exclusión de ruido; tabla común puntúa las cuatro etiquetas en las mismas filas asignadas por ambos DBSCAN y en el espacio original escalado. |
| M-02 Validez matemática | Corregido | Métricas validan dimensión, finitud, alineación, faltantes, cantidad de grupos y `n > k`; la comparación común hereda la validez. |
| M-03 Cálculos en reruns | Corregido | Búsquedas/grillas/ajustes de estabilidad quedan detrás de acciones; los cálculos puros se cachean por entradas y PCA no se ajusta por un valor por defecto evaluado con `dict.get`. |
| M-04 Atípicos K-Means | Corregido | Distancia euclidiana al centroide del grupo en espacio escalado, umbral percentil configurable y salida con pseudónimo, fecha, grupo y distancia. |
| M-05 Matriz escalada | Corregido | La interfaz muestra y permite descargar características antes de escalar y matriz estandarizada, ambas con metadatos pseudonimizados alineados. |
| M-06 Decisión/limitaciones | Corregido | Evidencia reúne métricas, soporte, cobertura, estabilidad y varianza PCA; presenta pros/contras, limitaciones y campos de decisión argumentada humana. |
| M-07 Grilla DBSCAN | Corregido | Grilla de sensibilidad requiere botón y expone los valores de `eps` y `min_samples`; curvas k-distances se cachean por matriz y parámetro. |
| M-08 Fecha fuente | Corregido | Se retienen fechas fuente para definir duplicados exactos; colisiones de clave tras normalización detienen la unión y exponen las filas conflictivas. |
| B-02 IQR/winsorize | Corregido | Tukey limita a Q1−1.5×IQR/Q3+1.5×IQR; winsorize limita a percentiles 5 y 95. |
| B-03 Streamlit obsoleto | Corregido | Se migraron los componentes de ancho a `width="stretch"`. |
| B-04 Versionado de entregables | Corregido | `.gitignore` excluye CSV fuente/procesados con IDs y permite versionar informes Markdown; no se inicializó Git. |
| B-05 IDs notebook | Corregido | Siete IDs únicos; `nbformat.validate` y ejecución aislada sin errores. |
| B-01 Arquitectura de páginas | Pendiente aceptado | Se mantiene la navegación existente con diez opciones en un solo módulo; no afecta la ejecución funcional comprobada. |

### Matriz de cumplimiento actualizada

| Etapa | Estado actual | Evidencia |
|---|---|---|
| 1. Comprensión | Cumplido | Objetivo, pregunta y límites no clínicos documentados. |
| 2. Carga y EDA | Parcial | Calidad, estadísticas y gráficos implementados; la interpretación académica sustantiva aún requiere revisión del equipo. |
| 3. Preprocesamiento | Cumplido | Opciones diferenciadas, matriz escalada visible/descargable y resultados enlazados a su configuración. |
| 4. K-Means | Cumplido | Ajuste, métricas, centroides, estabilidad y candidatos atípicos trazables. |
| 5. DBSCAN | Cumplido | Ruido, cobertura, grilla explícita y estabilidad trazable. |
| 6. PCA | Cumplido | Componentes, varianza, cargas, reconstrucción y proyección PC1–PC2. |
| 7. Modelos combinados | Cumplido | PCA compartido, etiquetas/configuración ligadas y métricas diagnósticas en ambos espacios. |
| 8. Comparación | Cumplido | Soporte asignado explícito y evaluación común de cuatro modelos. |
| 9. Estabilidad | Cumplido | Acciones explícitas, resultados asociados a configuración y sensibilidad DBSCAN reportada. |
| 10. Interpretación/decisión | Parcial | Evidencia y límites estructurados; selección final y argumentos siguen siendo responsabilidad humana. |

### Comprobaciones posteriores

- `.venv\Scripts\python.exe -m pytest -q`: **14 passed**.
- `compileall` de `src`, `app.py` y `tests`: sin errores.
- `pip check`: sin dependencias rotas.
- Reproducción desde CSV en memoria: 410 filas × 6 características; tres duplicados exactos de sueño descartados; 267 observaciones en el soporte común.
- Cuatro modelos: 410/410/410/410 etiquetas; en métricas asignadas, DBSCAN evaluó 267 y PCA+DBSCAN 338; las cuatro métricas comunes usaron 267 filas.
- `AppTest`: diez vistas, cuatro botones principales de ejecución y descarga de matriz estandarizada sin excepciones.
- Inicio Streamlit y solicitud HTTP: **200**.
- Notebook: seis celdas ejecutadas en una copia temporal, cero salidas de error; advertencias del event loop/ZMQ de Windows no impidieron la ejecución.
- No se efectuó inspección visual del navegador ni se inicializó Git.
