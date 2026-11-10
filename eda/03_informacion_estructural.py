
from pathlib import Path
import pandas as pd

# 1. Identificar la carpeta principal
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Definir las rutas
ACTIVITY_PATH = PROJECT_DIR / "data" / "original" / "dailyActivity_merged.csv"
SLEEP_PATH = PROJECT_DIR / "data" / "original" / "sleepDay_merged.csv"

# 3. Cargar los archivos
df_activity = pd.read_csv(ACTIVITY_PATH)
df_sleep = pd.read_csv(SLEEP_PATH)


# 4. Información estructural del DataFrame de actividad
print("\n========== INFORMACIÓN DE ACTIVIDAD ==========")
df_activity.info()


# 5. Información estructural del DataFrame de sueño
print("\n========== INFORMACIÓN DE SUEÑO ==========")
df_sleep.info()
