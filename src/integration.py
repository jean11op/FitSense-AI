"""Integración auditable de actividad y sueño a nivel persona-día."""
from pathlib import Path
import pandas as pd
from .data_loader import load_activity, load_sleep

class DuplicateKeyError(ValueError):
    """Las fuentes tienen mediciones diferentes para una misma clave."""

def integrate(activity: pd.DataFrame, sleep: pd.DataFrame):
    a, s = activity.copy(), sleep.copy()
    a["Id"] = a["Id"].astype("string"); s["Id"] = s["Id"].astype("string")
    a["Date"] = pd.to_datetime(a["ActivityDate"], format="mixed").dt.normalize()
    s["Date"] = pd.to_datetime(s["SleepDay"], format="mixed").dt.normalize()
    exact_sleep = int(s.duplicated().sum())
    exact_activity = int(a.duplicated().sum())
    s = s.drop_duplicates().copy()  # solo duplicados exactos de sueño
    activity_keys = a.duplicated(["Id", "Date"], keep=False)
    sleep_keys = s.duplicated(["Id", "Date"], keep=False)
    conflicts_a = a.loc[activity_keys].sort_values(["Id", "Date"])
    conflicts_s = s.loc[sleep_keys].sort_values(["Id", "Date"])
    if activity_keys.any() or sleep_keys.any():
        raise DuplicateKeyError("Claves Id-fecha repetidas con potenciales mediciones distintas; revisar informe de claves antes de integrar.")
    merged = a.merge(s, on=["Id", "Date"], how="inner", suffixes=("_activity", "_sleep"), validate="one_to_one")
    merged = merged.drop(columns=["ActivityDate", "SleepDay"])
    activity_without = a.loc[~a.set_index(["Id", "Date"]).index.isin(s.set_index(["Id", "Date"]).index)].copy()
    sleep_without = s.loc[~s.set_index(["Id", "Date"]).index.isin(a.set_index(["Id", "Date"]).index)].copy()
    participants_a, participants_s = set(a.Id.dropna()), set(s.Id.dropna())
    audit = pd.DataFrame([
        ("Registros originales de actividad", len(a)), ("Registros originales de sueño", len(s)+exact_sleep),
        ("Duplicados exactos actividad", exact_activity), ("Duplicados exactos sueño", exact_sleep),
        ("Claves duplicadas actividad", int(activity_keys.sum())), ("Claves duplicadas sueño tras quitar exactos", int(sleep_keys.sum())),
        ("Registros coincidentes persona-día", len(merged)), ("Actividad sin sueño", len(activity_without)),
        ("Sueño sin actividad", len(sleep_without)), ("Participantes actividad", len(participants_a)),
        ("Participantes sueño", len(participants_s)), ("Participantes coincidentes", len(participants_a & participants_s)),
        ("Cobertura integración sobre actividad (%)", round(100*len(merged)/max(len(a),1), 2)),
        ("Cobertura integración sobre sueño deduplicado (%)", round(100*len(merged)/max(len(s),1), 2)),
        ("Unicidad persona-día", bool(~merged.duplicated(["Id","Date"]).any())),
    ], columns=["Indicador", "Valor"])
    return merged, activity_without, sleep_without, audit, conflicts_a, conflicts_s

def run_integration(activity_path: str | Path, sleep_path: str | Path, root: str | Path):
    root = Path(root); processed = root / "data" / "processed"; outputs = root / "outputs"
    processed.mkdir(parents=True, exist_ok=True); outputs.mkdir(parents=True, exist_ok=True)
    a, s = load_activity(activity_path), load_sleep(sleep_path)
    # Persist key-conflict details even if integration cannot continue.
    try:
        merged, a0, s0, audit, ca, cs = integrate(a, s)
    except DuplicateKeyError:
        ca = a.loc[a.duplicated(["Id","ActivityDate"], keep=False)]
        sd = s.drop_duplicates().copy(); sd["_date"] = sd.SleepDay
        cs = sd.loc[sd.duplicated(["Id","_date"], keep=False)]
        outputs.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(outputs / "integration_audit.xlsx", engine="openpyxl") as writer:
            pd.DataFrame({"Indicador":["Claves duplicadas conflictivas"],"Valor":["Integración detenida para evitar descartar mediciones"]}).to_excel(writer, sheet_name="Resumen", index=False)
            ca.to_excel(writer, sheet_name="Claves actividad", index=False); cs.to_excel(writer, sheet_name="Claves sueño", index=False)
        raise
    merged.to_csv(processed / "person_day_integrated.csv", index=False)
    merged.to_excel(processed / "person_day_integrated.xlsx", index=False)
    a0.to_csv(processed / "activity_without_sleep.csv", index=False)
    s0.to_csv(processed / "sleep_without_activity.csv", index=False)
    reference = root / "data" / "integrated" / "FitSense_AI_Dataset_Integrado.xlsx"
    reference_audit = pd.DataFrame([{"Indicador":"Excel de referencia","Valor":"No encontrado"}])
    if reference.exists():
        try:
            ref = pd.read_excel(reference, sheet_name="Datos_combinados", header=3)
            ref = ref.rename(columns={"Fecha":"Date"})
            if "Date" in ref:
                ref["Date"] = pd.to_datetime(ref["Date"], errors="coerce").dt.normalize()
            if "Id" in ref: ref["Id"] = ref["Id"].astype("string")
            common = [c for c in merged.columns if c in ref.columns]
            left = merged[common].sort_values(["Id","Date"]).reset_index(drop=True)
            right = ref[common].sort_values(["Id","Date"]).reset_index(drop=True)
            same_rows = len(left)==len(right) and len(common)==len(merged.columns) and left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>"))
            reference_audit = pd.DataFrame([
                {"Indicador":"Filas Excel de referencia","Valor":len(ref)},
                {"Indicador":"Columnas esperadas presentes en Excel","Valor":len(common)==len(merged.columns)},
                {"Indicador":"Filas idénticas tras normalizar tipos y ordenar","Valor":bool(same_rows)},
                {"Indicador":"Diferencia de filas (reproducido - Excel)","Valor":len(merged)-len(ref)},
            ])
        except Exception as exc:
            reference_audit = pd.DataFrame([{"Indicador":"Error al verificar Excel de referencia","Valor":str(exc)}])
    audit = pd.concat([audit, reference_audit], ignore_index=True)
    with pd.ExcelWriter(outputs / "integration_audit.xlsx", engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="Resumen", index=False)
        reference_audit.to_excel(writer, sheet_name="Verificación Excel", index=False)
        ca.to_excel(writer, sheet_name="Claves actividad", index=False); cs.to_excel(writer, sheet_name="Claves sueño", index=False)
        a0.to_excel(writer, sheet_name="Actividad sin sueño", index=False); s0.to_excel(writer, sheet_name="Sueño sin actividad", index=False)
    return merged, audit

