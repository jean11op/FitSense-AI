"""Exportación tabular de resultados."""
from pathlib import Path
import pandas as pd

def export_csv(frame: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)

