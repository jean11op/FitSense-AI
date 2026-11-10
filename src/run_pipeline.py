"""Genera integración y matriz escalada reproducible desde CSV originales."""
from .config import ROOT, DATA_ORIGINAL, DATA_PROCESSED, DEFAULT_FEATURES
from .integration import run_integration
from .preprocessing import prepare_features
from .data_loader import load_activity, load_sleep
import json

def main():
    merged,audit=run_integration(DATA_ORIGINAL/"dailyActivity_merged.csv",DATA_ORIGINAL/"sleepDay_merged.csv",ROOT)
    prep=prepare_features(merged,[c for c in DEFAULT_FEATURES if c in merged],missing="median",outliers="keep")
    exported=prep["metadata"].copy()
    exported=exported.join(prep["scaled"])
    DATA_PROCESSED.mkdir(parents=True,exist_ok=True)
    exported.to_csv(DATA_PROCESSED/"preprocessed_features.csv",index=False)
    (DATA_PROCESSED/"preprocessing_config.json").write_text(json.dumps({"features":prep["features"],"missing_strategy":prep["missing_strategy"],"outlier_strategy":prep["outlier_strategy"],"scaler":"StandardScaler"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(audit.to_string(index=False)); print(f"\nMatriz preparada: {exported.shape}; variables={prep['features']}")

if __name__ == "__main__": main()
