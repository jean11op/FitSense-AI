"""Pseudónimos públicos y trazabilidad de resultados de análisis."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def participant_mapping(ids) -> dict[str, str]:
    """Asignación determinista y estable para el conjunto de IDs recibido."""
    values = sorted({str(value) for value in ids if not pd.isna(value)})
    return {participant_id: f"Participante_{index:03d}" for index, participant_id in enumerate(values, start=1)}


def pseudonymize_frame(frame: pd.DataFrame, mapping: dict[str, str], drop_id: bool = True) -> pd.DataFrame:
    """Devuelve una copia presentable; nunca reemplaza el Id interno."""
    result = frame.copy()
    if "Id" in result.columns:
        result["Participante"] = result["Id"].astype("string").map(mapping).fillna("Participante_sin_asignar")
        if drop_id:
            result = result.drop(columns="Id")
    return result


def feature_export_frames(metadata: pd.DataFrame, raw: pd.DataFrame, scaled: pd.DataFrame, mapping: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Crea descargas alineadas con metadatos seudonimizados y features sin Id."""
    if len(metadata)!=len(raw) or len(raw)!=len(scaled):
        raise ValueError("Metadatos y matrices de características deben estar alineados.")
    exports={}
    for name,features in (("antes_de_escalar",raw),("estandarizada",scaled)):
        frame=metadata.reset_index(drop=True).copy().join(features.reset_index(drop=True))
        frame=pseudonymize_frame(frame,mapping)
        ordered=[column for column in ("Participante","Date") if column in frame]
        ordered.extend(list(features.columns))
        exports[name]=frame[ordered]
    return exports


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash sensible al orden de filas, columnas y valores."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in frame.columns], ensure_ascii=False).encode("utf-8"))
    hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype=np.uint64)
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def fingerprint(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def preprocessing_fingerprint(source: pd.DataFrame, features, missing: str, outliers: str, scaler_params=None) -> str:
    return fingerprint({
        "observations_order_hash": dataframe_fingerprint(source),
        "features": list(features),
        "missing": missing,
        "outliers": outliers,
        "scaler": scaler_params or {"class": "StandardScaler", "with_mean": True, "with_std": True},
    })


def pca_fingerprint(preprocessing_id: str, variance_threshold: float) -> str:
    return fingerprint({"preprocessing_id": preprocessing_id, "transform": "PCA", "variance_threshold": float(variance_threshold)})


def result_fingerprint(preprocessing_id: str, model_name: str, model_config: dict, pca_id: str | None = None) -> str:
    return fingerprint({"preprocessing_id": preprocessing_id, "model": model_name, "model_config": model_config, "pca_id": pca_id})


def register_result(model, labels, metrics, configuration: dict, configuration_id: str, **extra) -> dict:
    return {
        "model": model,
        "labels": np.asarray(labels).copy(),
        "metrics": dict(metrics),
        "configuration": dict(configuration),
        "configuration_id": configuration_id,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def result_matches(result: dict | None, expected_configuration_id: str, row_count: int | None = None) -> bool:
    if not result or result.get("configuration_id") != expected_configuration_id:
        return False
    labels = result.get("labels")
    return labels is not None and (row_count is None or len(labels) == row_count)
