from __future__ import annotations

import pandas as pd

from src.models.predict_sentiment import predict_sentiment
from src.models.train_sentiment_model import train_sentiment_model
from src.utils.logger import setup_logging
from src.utils.paths import (
    GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH,
    SILVER_UNIFIED_DATASET_PATH,
)


def run() -> tuple[pd.DataFrame, dict]:
    setup_logging()
    df = pd.read_csv(SILVER_UNIFIED_DATASET_PATH)
    model, metrics, vectorizer = train_sentiment_model(df)
    predicted = predict_sentiment(df, model, vectorizer)
    predicted.to_csv(GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH, index=False)
    return predicted, metrics


if __name__ == "__main__":
    run()
