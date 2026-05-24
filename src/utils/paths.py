from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
OUTPUT_DIR = DATA_DIR / "output"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SQL_DIR = PROJECT_ROOT / "sql"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_runtime_directories() -> None:
    for path in [RAW_DIR, BRONZE_DIR, OUTPUT_DIR, ARTIFACTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
