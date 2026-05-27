from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine


SILVER_COLUMNS = [
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
    "data_processamento",
]


def prepare_silver_reviews(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy().rename(
        columns={
            "status": "status_reclamacao",
            "categoria": "categoria_problema",
            "UF": "uf",
        }
    )
    for column in SILVER_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[SILVER_COLUMNS]


def load_silver_reviews(df: pd.DataFrame, table_name: str = "reviews_cleaned", schema: str = "silver") -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text(f"DELETE FROM {schema}.{table_name}"))
    prepare_silver_reviews(df).to_sql(
        table_name,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
    )
