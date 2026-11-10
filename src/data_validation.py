"""Validaciones de contrato de datos."""
import pandas as pd

def validate_person_day(df):
    required={"Id","Date"}
    if not required.issubset(df.columns): raise ValueError(f"Se requieren columnas {sorted(required)}")
    return {"unique_person_day":bool(not df.duplicated(["Id","Date"]).any()),"missing_id":int(df.Id.isna().sum()),"missing_date":int(df.Date.isna().sum())}

