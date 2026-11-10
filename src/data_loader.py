"""Carga y validación básica de fuentes originales."""
from pathlib import Path
import pandas as pd

ACTIVITY_COLUMNS = ["Id", "ActivityDate", "TotalSteps", "TotalDistance", "TrackerDistance", "LoggedActivitiesDistance", "VeryActiveDistance", "ModeratelyActiveDistance", "LightActiveDistance", "SedentaryActiveDistance", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes", "Calories"]
SLEEP_COLUMNS = ["Id", "SleepDay", "TotalSleepRecords", "TotalMinutesAsleep", "TotalTimeInBed"]

def load_activity(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Id": "string"})
    missing = set(ACTIVITY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de actividad: {sorted(missing)}")
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], errors="coerce", format="mixed").dt.normalize()
    if df["ActivityDate"].isna().any():
        raise ValueError("Hay fechas de actividad que no se pudieron interpretar.")
    return df

def load_sleep(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Id": "string"})
    missing = set(SLEEP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas de sueño: {sorted(missing)}")
    df["SleepDay"] = pd.to_datetime(df["SleepDay"], errors="coerce", format="mixed").dt.normalize()
    if df["SleepDay"].isna().any():
        raise ValueError("Hay fechas de sueño que no se pudieron interpretar.")
    return df

