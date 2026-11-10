# FitSense AI

Aplicación académica en español para descubrir perfiles descriptivos de actividad y descanso con K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN.

## Inicio rápido en Windows

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.run_pipeline
.venv\Scripts\streamlit.exe run app.py
```

La aplicación también puede abrirse después de preparar los datos. Las fuentes deben estar en `data/original/`. La unión interna se hace por participante y fecha y no modifica fuentes.

## Arquitectura

- `src/`: carga, integración, EDA, preprocesamiento, modelos, métricas, estabilidad, perfiles y exportación.
- `app.py`: entrada Streamlit; la navegación presenta diez secciones.
- `data/original/`: archivos fuente inmutables.
- `data/processed/`: integración y características listas.
- `outputs/`: auditoría y salidas reproducibles.
- `docs/`: metodología y documentos académicos.
- `notebooks/`: recorrido reproducible fuera de la interfaz.

## Reglas de interpretación

Los ceros no se confunden con faltantes; Id y fecha son metadatos; los extremos son configurables. DBSCAN informa ruido y cobertura. La comparación principal utiliza el espacio original escalado y nunca recomienda un modelo con una sola métrica. Los perfiles no son clínicos.

## Reproducibilidad y validación

Semilla inicial 42. K-Means usa k-means++ y al menos diez inicializaciones. El pipeline registra auditoría en `outputs/integration_audit.xlsx`. Ejecute `pytest` para pruebas unitarias.
