from __future__ import annotations

import pandas as pd

from src.database.load_gold_to_sqlserver import load_gold_dataframe
from src.database.load_silver_to_sqlserver import load_silver_reviews
from src.utils.logger import setup_logging
from src.utils.paths import (
    GOLD_ANALYTICS_SENTIMENT_PATH,
    GOLD_TOPICS_ANALYSIS_PATH,
    SILVER_UNIFIED_DATASET_PATH,
)


def run() -> None:
    setup_logging()
    silver = pd.read_csv(SILVER_UNIFIED_DATASET_PATH)
    load_silver_reviews(silver)

    sentiment = pd.read_csv(GOLD_ANALYTICS_SENTIMENT_PATH)
    topics = pd.read_csv(GOLD_TOPICS_ANALYSIS_PATH)
    load_gold_dataframe(sentiment, "sentiment_analysis")
    load_gold_dataframe(topics, "topic_analysis")


if __name__ == "__main__":
    run()
