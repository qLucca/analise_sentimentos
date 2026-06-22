from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine


GENERAL_METRICS_QUERY = "SELECT * FROM gold.vw_general_metrics"
SENTIMENT_BY_SOURCE_QUERY = "SELECT * FROM gold.vw_sentiment_by_source"
SENTIMENT_BY_MONTH_QUERY = "SELECT * FROM gold.vw_sentiment_by_month"
TOPICS_BY_SENTIMENT_QUERY = "SELECT * FROM gold.vw_topics_by_sentiment"
NEGATIVE_TOPICS_QUERY = "SELECT * FROM gold.vw_negative_topics"
SOURCE_SUMMARY_QUERY = "SELECT * FROM gold.vw_source_summary"


def run_query(query: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(query, con=engine)


def read_table(table_name: str, schema: str, where_clause: str | None = None) -> pd.DataFrame:
    qualified_table = f"{schema}.{table_name}"
    query = f"SELECT * FROM {qualified_table}"
    if where_clause:
        query = f"{query} WHERE {where_clause}"

    engine = get_engine()
    with engine.connect() as connection:
        return pd.read_sql(text(query), con=connection)
