from __future__ import annotations

import pandas as pd

from src.database.load_gold_to_sqlserver import load_gold_dataframe
from src.database.load_silver_to_sqlserver import load_silver_reviews
from src.utils.logger import setup_logging
from src.utils.paths import BRONZE_DIR


def run() -> None:
    setup_logging()
    silver = pd.read_csv(BRONZE_DIR / "dados_unificados_silver_local.csv")
    load_silver_reviews(silver)

    sentiment = pd.read_csv(BRONZE_DIR / "gold_sentiment_analysis_local.csv")
    topics = pd.read_csv(BRONZE_DIR / "gold_topic_analysis_local.csv")
    load_gold_dataframe(sentiment, "sentiment_analysis")
    load_gold_dataframe(topics, "topic_analysis")


if __name__ == "__main__":
    run()
