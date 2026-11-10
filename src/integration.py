"""Integración auditable de actividad y sueño a nivel persona-día."""
from pathlib import Path
import pandas as pd
from .data_loader import load_activity, load_sleep

class DuplicateKeyError(ValueError):
    """Las fuentes tienen mediciones diferentes para una misma clave."""

    def __init__(self, message, activity_conflicts=None, sleep_conflicts=None, sleep_duplicates=None):
        super().__init__(message)
        self.activity_conflicts = activity_conflicts if activity_conflicts is not None else pd.DataFrame()
        self.sleep_conflicts = sleep_conflicts if sleep_conflicts is not None else pd.DataFrame()
        self.sleep_duplicates = sleep_duplicates if sleep_duplicates is not None else pd.DataFrame()

def integrate(activity: pd.DataFrame, sleep: pd.DataFrame):
    a, s = activity.copy(), sleep.copy()
    a["Id"] = a["Id"].astype("string"); s["Id"] = s["Id"].astype("string")
    if "ActivityDate_source" not in a:
        a["ActivityDate_source"] = a["ActivityDate"].astype("string")
    if "SleepDay_source" not in s:
        s["SleepDay_source"] = s["SleepDay"].astype("string")
    a["Date"] = pd.to_datetime(a["ActivityDate"], format="mixed").dt.normalize()
    s["Date"] = pd.to_datetime(s["SleepDay"], format="mixed").dt.normalize()
    activity_exact_columns = [c for c in a.columns if c not in {"ActivityDate", "Date"}]
    sleep_exact_columns = [c for c in s.columns if c not in {"SleepDay", "Date"}]
    exact_sleep_mask = s.duplicated(subset=sleep_exact_columns, keep="first")
    exact_sleep_discarded = s.loc[exact_sleep_mask].copy()
    exact_sleep = int(exact_sleep_mask.sum())
    exact_activity = int(a.duplicated(subset=activity_exact_columns, keep="first").sum())
    s = s.loc[~exact_sleep_mask].copy()  # solo filas exactamente repetidas en la fuente
    activity_keys = a.duplicated(["Id", "Date"], keep=False)
    sleep_keys = s.duplicated(["Id", "Date"], keep=False)
    conflicts_a = a.loc[activity_keys].sort_values(["Id", "Date"])
    conflicts_s = s.loc[sleep_keys].sort_values(["Id", "Date"])
    if activity_keys.any() or sleep_keys.any():
        raise DuplicateKeyError(
            "Claves Id-fecha repetidas tras normalizar. Se conservaron las filas conflictivas para auditoría.",
            conflicts_a, conflicts_s, exact_sleep_discarded,
        )
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
    return merged, activity_without, sleep_without, audit, conflicts_a, conflicts_s, exact_sleep_discarded

def run_integration(activity_path: str | Path, sleep_path: str | Path, root: str | Path):
    root = Path(root); processed = root / "data" / "processed"; outputs = root / "outputs"
    processed.mkdir(parents=True, exist_ok=True); outputs.mkdir(parents=True, exist_ok=True)
    a, s = load_activity(activity_path), load_sleep(sleep_path)
    # Persist key-conflict details even if integration cannot continue.
    try:
        merged, a0, s0, audit, ca, cs, exact_sleep_discarded = integrate(a, s)
    except DuplicateKeyError as exc:
        outputs.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(outputs / "integration_audit.xlsx", engine="openpyxl") as writer:
            pd.DataFrame({"Indicador":["Claves duplicadas conflictivas"],"Valor":["Integración detenida para evitar descartar mediciones"]}).to_excel(writer, sheet_name="Resumen", index=False)
            exc.activity_conflicts.to_excel(writer, sheet_name="Claves actividad", index=False)
            exc.sleep_conflicts.to_excel(writer, sheet_name="Claves sueño", index=False)
            exc.sleep_duplicates.to_excel(writer, sheet_name="Exactos sueño descartados", index=False)
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
            source_date_columns = {"ActivityDate_source", "SleepDay_source"}
            canonical = [c for c in merged.columns if c not in source_date_columns]
            common = [c for c in canonical if c in ref.columns]
            left = merged[common].sort_values(["Id","Date"]).reset_index(drop=True)
            right = ref[common].sort_values(["Id","Date"]).reset_index(drop=True)
            same_rows = len(left)==len(right) and len(common)==len(canonical) and left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>"))
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
        exact_sleep_discarded.to_excel(writer, sheet_name="Exactos sueño descartados", index=False)
        a0.to_excel(writer, sheet_name="Actividad sin sueño", index=False); s0.to_excel(writer, sheet_name="Sueño sin actividad", index=False)
    return merged, audit

