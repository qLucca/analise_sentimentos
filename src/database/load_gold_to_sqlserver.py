from __future__ import annotations

import pandas as pd

from src.database.connection import get_engine


def load_gold_dataframe(df: pd.DataFrame, table_name: str, schema: str = "gold", if_exists: str = "append") -> None:
    engine = get_engine()
    df.to_sql(table_name, con=engine, schema=schema, if_exists=if_exists, index=False)
