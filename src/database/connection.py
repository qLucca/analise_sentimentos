from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def build_connection_string() -> str:
    load_dotenv()
    backend = os.getenv("DB_BACKEND", "").lower()

    if backend == "mysql":
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        database = os.getenv("DB_NAME", "analytics")
        user = os.getenv("DB_USER", "")
        password = os.getenv("DB_PASSWORD", "")
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

    host = os.getenv("SQLSERVER_HOST", "localhost")
    database = os.getenv("SQLSERVER_DATABASE", "NubankSentimentAnalysis")
    user = os.getenv("SQLSERVER_USER", "")
    password = os.getenv("SQLSERVER_PASSWORD", "")
    driver = os.getenv("SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server")
    trusted = os.getenv("SQLSERVER_TRUSTED_CONNECTION", "yes").lower()
    trust_server_certificate = os.getenv("SQLSERVER_TRUST_SERVER_CERTIFICATE", "yes").lower()

    common_params = (
        f"DRIVER={{{driver}}};"
        f"SERVER={host};"
        f"DATABASE={database};"
        f"TrustServerCertificate={trust_server_certificate};"
    )

    if trusted == "yes":
        params = quote_plus(
            f"{common_params}Trusted_Connection=yes;"
        )
    else:
        params = quote_plus(
            f"{common_params}UID={user};PWD={password};"
        )
    return f"mssql+pyodbc:///?odbc_connect={params}"


def get_engine() -> Engine:
    return create_engine(build_connection_string())
