
from pathlib import Path
import pandas as pd

# 1. Identificar la carpeta principal
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Definir las rutas de los archivos
ACTIVITY_PATH = PROJECT_DIR / "data" / "original" / "dailyActivity_merged.csv"
SLEEP_PATH = PROJECT_DIR / "data" / "original" / "sleepDay_merged.csv"

# 3. Cargar los DataFrames
df_activity = pd.read_csv(ACTIVITY_PATH)
df_sleep = pd.read_csv(SLEEP_PATH)


# 4. Analizar dimensiones
print("\n========== DIMENSIONES ==========")

print("Actividad física:", df_activity.shape)
print("Sueño:", df_sleep.shape)


# 5. Analizar nombres de columnas
print("\n========== COLUMNAS DE ACTIVIDAD ==========")
print(df_activity.columns.tolist())

print("\n========== COLUMNAS DE SUEÑO ==========")
print(df_sleep.columns.tolist())


# 6. Mostrar primeras observaciones
print("\n========== PRIMERAS 5 FILAS DE ACTIVIDAD ==========")
print(df_activity.head())

print("\n========== PRIMERAS 5 FILAS DE SUEÑO ==========")
print(df_sleep.head())


# 7. Mostrar últimas observaciones
print("\n========== ÚLTIMAS 5 FILAS DE ACTIVIDAD ==========")
print(df_activity.tail())

print("\n========== ÚLTIMAS 5 FILAS DE SUEÑO ==========")
print(df_sleep.tail())
