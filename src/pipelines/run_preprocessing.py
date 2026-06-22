from __future__ import annotations

import pandas as pd

from processing.text_preprocessing import preprocess_reviews
from src.utils.io import write_dataframe
from src.utils.logger import setup_logging
from src.utils.paths import (
    BRONZE_UNIFIED_DATASET_PATH,
    GOLD_DASHBOARD_UNIFIED_DATASET_PATH,
    PROJECT_ROOT,
    RAW_DIR,
    SILVER_UNIFIED_DATASET_PATH,
    ensure_runtime_directories,
)


def _load_legacy_unified_dataset() -> pd.DataFrame:
    legacy_path = PROJECT_ROOT / "data" / "processed" / "unified_dataset.csv"
    if not legacy_path.exists():
        return pd.DataFrame()
    legacy = pd.read_csv(legacy_path)
    legacy = legacy.rename(
        columns={
            "categoria": "categoria_problema",
            "status": "status_reclamacao",
            "UF": "uf",
        }
    )
    return legacy


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
    loaded_sources: set[str] = set()

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
        loaded_sources.add(source_name)

    legacy = _load_legacy_unified_dataset()
    missing_sources = set(mapping) - loaded_sources
    if not legacy.empty and missing_sources:
        for source_name in sorted(missing_sources):
            legacy_subset = legacy.loc[legacy["fonte"] == source_name].copy()
            if legacy_subset.empty:
                continue
            for column in expected_columns:
                if column not in legacy_subset.columns:
                    legacy_subset[column] = None
            datasets.append(legacy_subset[expected_columns])

    if not datasets:
        if legacy.empty:
            return pd.DataFrame(columns=expected_columns)
        for column in expected_columns:
            if column not in legacy.columns:
                legacy[column] = None
        return legacy[expected_columns]

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
