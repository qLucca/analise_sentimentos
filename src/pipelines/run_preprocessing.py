from __future__ import annotations

import pandas as pd

from src.preprocessing.text_preprocessing import preprocess_reviews
from src.utils.io import write_dataframe
from src.utils.logger import setup_logging
from src.utils.paths import (
    BRONZE_UNIFIED_DATASET_PATH,
    GOLD_DASHBOARD_UNIFIED_DATASET_PATH,
    RAW_DIR,
    SILVER_UNIFIED_DATASET_PATH,
    ensure_runtime_directories,
)


def build_unified_bronze() -> pd.DataFrame:
    datasets = []
    mapping = {
        "google_play": RAW_DIR / "google_play" / "google_play_reviews_processed.csv",
        "youtube": RAW_DIR / "youtube" / "youtube_processed.csv",
        "consumidor_gov": RAW_DIR / "consumidor_gov" / "consumidor_gov_processed.csv",
    }
    expected_columns = [
        "id_registro",
        "fonte",
        "data_publicacao",
        "texto_original",
        "nota",
        "status_reclamacao",
        "categoria_problema",
        "uf",
        "versao_app",
        "sentimento_real",
    ]
    source_aliases = {
        "status_reclamacao": ["status_reclamacao", "status"],
        "categoria_problema": ["categoria_problema", "categoria"],
        "uf": ["uf", "UF"],
        "versao_app": ["versao_app"],
    }

    for source_name, path in mapping.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for target_column, aliases in source_aliases.items():
            for alias in aliases:
                if alias in df.columns:
                    df[target_column] = df[alias]
                    break
        for column in expected_columns:
            if column not in df.columns:
                df[column] = None
        df["fonte"] = df["fonte"].fillna(source_name)
        datasets.append(df[expected_columns])

    if not datasets:
        return pd.DataFrame(columns=expected_columns)

    return pd.concat(datasets, ignore_index=True)


def run() -> pd.DataFrame:
    setup_logging()
    ensure_runtime_directories()
    bronze = build_unified_bronze()
    write_dataframe(bronze, BRONZE_UNIFIED_DATASET_PATH)
    silver = preprocess_reviews(bronze)
    write_dataframe(silver, SILVER_UNIFIED_DATASET_PATH)
    # The dashboard consumes the cleaned unified base as its default dataset.
    write_dataframe(silver, GOLD_DASHBOARD_UNIFIED_DATASET_PATH)
    return silver


if __name__ == "__main__":
    run()
