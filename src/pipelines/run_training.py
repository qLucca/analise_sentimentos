from __future__ import annotations

import pandas as pd

from src.models.predict_sentiment import predict_sentiment
from src.models.train_sentiment_model import train_sentiment_model
from src.utils.logger import setup_logging
from src.utils.paths import BRONZE_DIR


def run() -> tuple[pd.DataFrame, dict]:
    setup_logging()
    silver_path = BRONZE_DIR / "dados_unificados_silver_local.csv"
    df = pd.read_csv(silver_path)
    model, metrics, vectorizer = train_sentiment_model(df)
    predicted = predict_sentiment(df, model, vectorizer)
    predicted.to_csv(BRONZE_DIR / "dados_com_sentimento_previsto.csv", index=False)
    return predicted, metrics


if __name__ == "__main__":
    run()
