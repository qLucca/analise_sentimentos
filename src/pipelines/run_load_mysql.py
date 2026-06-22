from __future__ import annotations

import pandas as pd

from src.database.ingestion import load_gold_dataframe, load_silver_reviews
from src.utils.logger import setup_logging
from src.utils.paths import PROCESSED_DIR


def run() -> None:
    setup_logging()

    silver = pd.read_csv(PROCESSED_DIR / "textual_dataset_preprocessed.csv")
    gold = pd.read_csv(PROCESSED_DIR / "youtube_with_predicted_sentiment_bertimbau.csv")

    load_silver_reviews(silver, if_exists="replace")
    load_gold_dataframe(gold, "sentiment_analysis", if_exists="replace")


if __name__ == "__main__":
    run()
