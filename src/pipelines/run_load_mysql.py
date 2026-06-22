from __future__ import annotations

import pandas as pd

from src.database.ingestion import load_gold_dataframe, load_silver_reviews
from src.utils.logger import setup_logging
from src.utils.paths import (
    GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH,
    SILVER_UNIFIED_DATASET_PATH,
)


def run() -> None:
    setup_logging()

    silver = pd.read_csv(SILVER_UNIFIED_DATASET_PATH)
    gold = pd.read_csv(GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH)

    load_silver_reviews(silver, if_exists="replace")
    load_gold_dataframe(gold, "sentiment_analysis", if_exists="replace")


if __name__ == "__main__":
    run()
