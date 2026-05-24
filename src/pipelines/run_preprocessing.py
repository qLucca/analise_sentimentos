from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.text_preprocessing import preprocess_reviews
from src.utils.io import write_dataframe
from src.utils.logger import setup_logging
from src.utils.paths import BRONZE_DIR, RAW_DIR, ensure_runtime_directories


def build_unified_bronze() -> pd.DataFrame:
    datasets = []
    mapping = {
        "google_play": RAW_DIR / "google_play" / "google_play_reviews_raw.csv",
        "youtube": RAW_DIR / "youtube" / "reclame_aqui_raw.csv",
        "consumidor_gov": RAW_DIR / "consumidor_gov" / "consumidor_gov_raw.csv",
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

    for source_name, path in mapping.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
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
    write_dataframe(bronze, BRONZE_DIR / "dados_unificados_bronze.csv")
    silver = preprocess_reviews(bronze)
    write_dataframe(silver, BRONZE_DIR / "dados_unificados_silver_local.csv")
    return silver


if __name__ == "__main__":
    run()
