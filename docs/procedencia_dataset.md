# Procedencia y auditoría del dataset

Fuentes locales esperadas: `data/original/dailyActivity_merged.csv` y `data/original/sleepDay_merged.csv`. Los originales se leen sin modificación; `Id` se conserva como texto. `ActivityDate` y `SleepDay` se normalizan a fechas.

La rutina `src.integration.run_integration` elimina únicamente duplicados exactos del CSV de sueño. Claves persona-fecha repetidas con valores distintos detienen la integración y se detallan en `outputs/integration_audit.xlsx`, evitando elegir una fila arbitrariamente. La unión es interna, uno a uno, por Id y fecha.

La integración, los archivos sin pareja y el informe se regeneran a partir de las fuentes. El Excel de referencia se contrasta fila por fila contra la unión reproducida, tras normalizar tipos y ordenar por persona-fecha; no sustituye a los CSV fuente. Consulte `Resumen` y `Verificación Excel` de la auditoría para cantidades y discrepancias calculadas.
