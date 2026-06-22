from __future__ import annotations

import pandas as pd

from src.database.connection import get_engine


def load_dataframe(
    df: pd.DataFrame,
    table_name: str,
    schema: str,
    if_exists: str = "append",
) -> None:
    engine = get_engine()
    df.to_sql(table_name, con=engine, schema=schema, if_exists=if_exists, index=False)


def load_silver_reviews(
    df: pd.DataFrame,
    table_name: str = "reviews_cleaned",
    schema: str = "silver",
    if_exists: str = "append",
) -> None:
    load_dataframe(df, table_name=table_name, schema=schema, if_exists=if_exists)


def load_gold_dataframe(
    df: pd.DataFrame,
    table_name: str,
    schema: str = "gold",
    if_exists: str = "append",
) -> None:
    load_dataframe(df, table_name=table_name, schema=schema, if_exists=if_exists)
