from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
