from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine


def build_connection_string() -> str:
    load_dotenv()
    host = os.getenv("SQLSERVER_HOST", "localhost")
    database = os.getenv("SQLSERVER_DATABASE", "NubankSentimentAnalysis")
    user = os.getenv("SQLSERVER_USER", "")
    driver = os.getenv("SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server")
    trusted = os.getenv("SQLSERVER_TRUSTED_CONNECTION", "yes").lower()

    if trusted == "yes":
        params = quote_plus(
            f"DRIVER={{{driver}}};SERVER={host};DATABASE={database};Trusted_Connection=yes;"
        )
    else:
        params = quote_plus(
            f"DRIVER={{{driver}}};SERVER={host};DATABASE={database};UID={user};PWD={password};"
        )
    return f"mssql+pyodbc:///?odbc_connect={params}"

print (build_connection_string())


def get_engine():
    return create_engine(build_connection_string())
