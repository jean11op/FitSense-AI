from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ORIGINAL = ROOT / "data" / "original"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
SEED = 42
DEFAULT_FEATURES = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]

