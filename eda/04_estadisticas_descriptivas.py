
from pathlib import Path
import pandas as pd

# 1. Identificar la carpeta principal
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Definir rutas
ACTIVITY_PATH = PROJECT_DIR / "data" / "original" / "dailyActivity_merged.csv"
SLEEP_PATH = PROJECT_DIR / "data" / "original" / "sleepDay_merged.csv"

# 3. Cargar los DataFrames
df_activity = pd.read_csv(ACTIVITY_PATH)
df_sleep = pd.read_csv(SLEEP_PATH)


# Mostrar todas las columnas sin abreviaciones
print("\n========== ESTADÍSTICAS DE ACTIVIDAD ==========")

actividad_stats = df_activity.drop(
    columns=["Id"]
).describe().T.round(2)

print(actividad_stats.to_string())


print("\n========== ESTADÍSTICAS DE SUEÑO ==========")

sueno_stats = df_sleep.drop(
    columns=["Id"]
).describe().T.round(2)

print(sueno_stats.to_string())