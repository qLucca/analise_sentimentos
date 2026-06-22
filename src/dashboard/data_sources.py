from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.database.queries import read_table
from src.storage.s3 import S3ArtifactStore


class DashboardDataLoader:
    def __init__(
        self,
        table_loader=read_table,
        artifact_store: S3ArtifactStore | None = None,
        processed_data_dir: Path | None = None,
    ) -> None:
        self.table_loader = table_loader
        self.artifact_store = artifact_store
        if self.artifact_store is None and os.getenv("S3_BUCKET"):
            self.artifact_store = S3ArtifactStore()
        self.processed_data_dir = processed_data_dir or Path("data/processed")

    def _read_model_summary(self) -> pd.DataFrame:
        if self.artifact_store is not None:
            try:
                return self.artifact_store.read_csv("processed/model_comparison_summary.csv")
            except Exception:
                pass

        local_path = self.processed_data_dir / "model_comparison_summary.csv"
        return pd.read_csv(local_path)

    def load(self) -> dict[str, pd.DataFrame]:
        return {
            "Base unificada": self.table_loader("reviews_cleaned", "silver"),
            "Resumo de modelos": self._read_model_summary(),
            "YouTube + BERTimbau": self.table_loader(
                "sentiment_analysis",
                "gold",
                where_clause="fonte = 'youtube'",
            ),
            "Consumidor.gov": self.table_loader(
                "reviews_cleaned",
                "silver",
                where_clause="fonte = 'consumidor_gov'",
            ),
        }
