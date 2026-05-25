from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd

from src.database.load_gold_to_sqlserver import load_gold_dataframe
from src.database.load_silver_to_sqlserver import load_silver_reviews
from src.pipelines.run_ingestion import run as run_ingestion
from src.pipelines.run_preprocessing import run as run_preprocessing
from src.pipelines.run_topics import run as run_topics
from src.pipelines.run_training import run as run_training
from src.utils.logger import setup_logging


def build_model_metrics_dataframe(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id_execucao": str(uuid4()),
                "modelo": metrics.get("modelo", "logistic_regression"),
                "vetorizador": metrics.get("vetorizador", "tfidf"),
                "accuracy": metrics.get("accuracy"),
                "precision_macro": metrics.get("precision_macro"),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "data_treinamento": metrics.get("data_treinamento", datetime.now()),
                "observacoes": metrics.get("observacoes"),
            }
        ]
    )


def run() -> None:
    setup_logging()
    run_ingestion()
    silver = run_preprocessing()
    load_silver_reviews(silver)
    predicted, metrics = run_training()
    sentiment_df, topic_df = run_topics()
    load_gold_dataframe(sentiment_df, "sentiment_analysis")
    load_gold_dataframe(topic_df, "topic_analysis")
    load_gold_dataframe(build_model_metrics_dataframe(metrics), "model_metrics")


if __name__ == "__main__":
    run()
