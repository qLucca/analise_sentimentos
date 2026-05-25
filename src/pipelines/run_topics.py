from __future__ import annotations

import pandas as pd

from src.topics.topic_modeling import extract_topics
from src.utils.logger import setup_logging
from src.utils.paths import (
    GOLD_ANALYTICS_SENTIMENT_PATH,
    GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH,
    GOLD_TOPICS_ANALYSIS_PATH,
)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_logging()
    predicted = pd.read_csv(GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH)
    sentiment_df, topic_df = extract_topics(predicted)
    sentiment_df.to_csv(GOLD_ANALYTICS_SENTIMENT_PATH, index=False)
    topic_df.to_csv(GOLD_TOPICS_ANALYSIS_PATH, index=False)
    return sentiment_df, topic_df


if __name__ == "__main__":
    run()
