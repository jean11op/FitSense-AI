
from pathlib import Path
import pandas as pd

# 1. Identificar la carpeta principal del proyecto
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Definir las rutas de los archivos CSV
ACTIVITY_PATH = PROJECT_DIR / "data" / "original" / "dailyActivity_merged.csv"
SLEEP_PATH = PROJECT_DIR / "data" / "original" / "sleepDay_merged.csv"


# 3. Cargar los archivos CSV
df_activity = pd.read_csv(ACTIVITY_PATH)
df_sleep = pd.read_csv(SLEEP_PATH)


# 4. Mostrar las primeras cinco filas
print("========== ACTIVIDAD FÍSICA ==========")
print(df_activity.head())

print("\n========== DESCANSO ==========")
print(df_sleep.head())
