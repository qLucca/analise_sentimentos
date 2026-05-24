from __future__ import annotations

import pandas as pd

from src.topics.topic_modeling import extract_topics
from src.utils.logger import setup_logging
from src.utils.paths import BRONZE_DIR


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_logging()
    predicted = pd.read_csv(BRONZE_DIR / "dados_com_sentimento_previsto.csv")
    sentiment_df, topic_df = extract_topics(predicted)
    sentiment_df.to_csv(BRONZE_DIR / "gold_sentiment_analysis_local.csv", index=False)
    topic_df.to_csv(BRONZE_DIR / "gold_topic_analysis_local.csv", index=False)
    return sentiment_df, topic_df


if __name__ == "__main__":
    run()
