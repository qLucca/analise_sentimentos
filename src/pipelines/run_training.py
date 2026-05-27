from __future__ import annotations

from uuid import uuid4

import pandas as pd

from src.models.bertimbau_runner import run_bertimbau_evaluation
from src.models.model_summary import build_model_summary_row, merge_model_summaries
from src.models.predict_sentiment import predict_sentiment
from src.models.train_sentiment_model import train_sentiment_model
from src.utils.logger import setup_logging
from src.utils.paths import (
    GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH,
    GOLD_DASHBOARD_MODEL_COMPARISON_PATH,
    GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH,
    SILVER_UNIFIED_DATASET_PATH,
)


def run() -> tuple[pd.DataFrame, dict]:
    setup_logging()
    df = pd.read_csv(SILVER_UNIFIED_DATASET_PATH)
    model, metrics, vectorizer = train_sentiment_model(df)
    metrics.setdefault("id_execucao", str(uuid4()))
    predicted = predict_sentiment(df, model, vectorizer)
    predicted.to_csv(GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH, index=False)
    classic_model_name = metrics.get("model_name") or model.__class__.__name__
    classic_candidate_rows = metrics.get("classic_candidate_rows") or [metrics]
    classic_summary_rows = []
    for candidate_metrics in classic_candidate_rows:
        missing_classic_summary_metrics = [
            key
            for key in [
                "precision_macro",
                "recall_macro",
                "roc_auc_macro",
                "roc_auc_weighted",
                "f1_negativo",
                "f1_neutro",
                "f1_positivo",
            ]
            if candidate_metrics.get(key) is None or pd.isna(candidate_metrics.get(key))
        ]
        candidate_name = candidate_metrics.get("modelo") or classic_model_name
        if candidate_name == classic_model_name:
            insight_summary = (
                "Modelo classico selecionado via validacao cruzada estratificada para uso operacional."
            )
        else:
            insight_summary = (
                "Baseline classico concorrente avaliado no mesmo criterio de robustez para comparacao."
            )
        if missing_classic_summary_metrics:
            insight_summary += (
                f" Campos ausentes neste resumo: {', '.join(missing_classic_summary_metrics)}."
            )
        classic_summary_rows.append(
            build_model_summary_row(
                model_name=candidate_name,
                metrics=candidate_metrics,
                is_primary=False,
                insight_summary=insight_summary,
                provenance="classic_runtime",
            )
        )
    bert_summary_row, bert_info = run_bertimbau_evaluation(df)
    merged_summary = merge_model_summaries(classic_summary_rows, bert_summary_row)
    pd.DataFrame(merged_summary).to_csv(GOLD_DASHBOARD_MODEL_COMPARISON_PATH, index=False)
    predicted.to_csv(GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH, index=False)
    metrics["bertimbau"] = bert_info
    return predicted, metrics


if __name__ == "__main__":
    run()
