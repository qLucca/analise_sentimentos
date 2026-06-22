from __future__ import annotations

import pandas as pd

from src.database.ingestion import load_gold_dataframe, load_silver_reviews
from src.utils.logger import setup_logging
from src.utils.paths import (
    GOLD_ANALYTICS_SENTIMENT_PATH,
    GOLD_DASHBOARD_MODEL_COMPARISON_PATH,
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
    if GOLD_DASHBOARD_MODEL_COMPARISON_PATH.exists():
        metrics = pd.read_csv(GOLD_DASHBOARD_MODEL_COMPARISON_PATH)
        load_gold_dataframe(metrics, "model_metrics")


if __name__ == "__main__":
    run()
