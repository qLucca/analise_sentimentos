from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine


GOLD_SENTIMENT_COLUMNS = [
    "id_registro",
    "fonte",
    "data_publicacao",
    "texto_original",
    "texto_limpo",
    "nota",
    "status_reclamacao",
    "categoria_problema",
    "uf",
    "versao_app",
    "sentimento_real",
    "sentimento_previsto",
    "topico",
    "data_processamento",
]
GOLD_MODEL_METRICS_COLUMNS = [
    "id_execucao",
    "modelo",
    "vetorizador",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "data_treinamento",
    "observacoes",
]


def prepare_gold_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy().rename(
        columns={
            "status": "status_reclamacao",
            "categoria": "categoria_problema",
            "UF": "uf",
        }
    )
    for column in GOLD_SENTIMENT_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[GOLD_SENTIMENT_COLUMNS]


def prepare_gold_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "id_execucao" not in prepared.columns:
        prepared["id_execucao"] = [str(uuid4()) for _ in range(len(prepared))]
    if "vetorizador" not in prepared.columns:
        prepared["vetorizador"] = None
    if "proveniencia" in prepared.columns:
        inferred_vectorizer = prepared["proveniencia"].fillna("").map(
            lambda value: "bertimbau"
            if str(value).startswith("bert") or str(value) == "benchmark"
            else "tfidf"
        )
    else:
        inferred_vectorizer = "tfidf"
    prepared["vetorizador"] = prepared["vetorizador"].fillna(inferred_vectorizer)
    prepared["data_treinamento"] = prepared.get("data_treinamento", datetime.now())
    if "observacoes" not in prepared.columns:
        prepared["observacoes"] = ""
    observacoes_base = prepared["observacoes"]
    if "insight_resumo" in prepared.columns or "proveniencia" in prepared.columns:
        prepared["observacoes"] = (
            observacoes_base.fillna("").astype(str).str.strip()
            + " "
            + (
                prepared["insight_resumo"].fillna("").astype(str).str.strip()
                if "insight_resumo" in prepared.columns
                else ""
            )
            + " "
            + (
                prepared["proveniencia"].fillna("").astype(str).str.strip()
                if "proveniencia" in prepared.columns
                else ""
            )
        ).str.strip()
    for column in GOLD_MODEL_METRICS_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[GOLD_MODEL_METRICS_COLUMNS]


def load_gold_dataframe(df: pd.DataFrame, table_name: str, schema: str = "gold", if_exists: str = "append") -> None:
    engine = get_engine()
    dataframe = df
    if table_name == "sentiment_analysis":
        dataframe = prepare_gold_sentiment_analysis(df)
    elif table_name == "model_metrics":
        dataframe = prepare_gold_model_metrics(df)
    if if_exists == "append":
        with engine.begin() as connection:
            connection.execute(text(f"DELETE FROM {schema}.{table_name}"))
    dataframe.to_sql(table_name, con=engine, schema=schema, if_exists=if_exists, index=False)
