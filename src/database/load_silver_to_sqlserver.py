from __future__ import annotations

import pandas as pd

from src.database.connection import get_engine


def load_silver_reviews(df: pd.DataFrame, table_name: str = "reviews_cleaned", schema: str = "silver") -> None:
    engine = get_engine()
    df.to_sql(table_name, con=engine, schema=schema, if_exists="append", index=False)
