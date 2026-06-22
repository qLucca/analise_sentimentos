"""Database integration modules."""

from src.database.connection import build_connection_string, get_engine
from src.database.ingestion import (
    load_dataframe,
    load_gold_dataframe,
    load_silver_reviews,
)
from src.database.queries import (
    GENERAL_METRICS_QUERY,
    NEGATIVE_TOPICS_QUERY,
    SENTIMENT_BY_MONTH_QUERY,
    SENTIMENT_BY_SOURCE_QUERY,
    SOURCE_SUMMARY_QUERY,
    TOPICS_BY_SENTIMENT_QUERY,
    run_query,
)

__all__ = [
    "build_connection_string",
    "get_engine",
    "load_dataframe",
    "load_gold_dataframe",
    "load_silver_reviews",
    "GENERAL_METRICS_QUERY",
    "NEGATIVE_TOPICS_QUERY",
    "SENTIMENT_BY_MONTH_QUERY",
    "SENTIMENT_BY_SOURCE_QUERY",
    "SOURCE_SUMMARY_QUERY",
    "TOPICS_BY_SENTIMENT_QUERY",
    "run_query",
]
