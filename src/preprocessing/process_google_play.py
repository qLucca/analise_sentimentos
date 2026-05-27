from __future__ import annotations

import pandas as pd

from src.utils.paths import RAW_DIR


def main() -> None:
    input_path = RAW_DIR / "google_play" / "google_play_reviews_raw.csv"
    output_path = RAW_DIR / "google_play" / "google_play_reviews_processed.csv"

    df = pd.read_csv(input_path, encoding="utf-8")
    print(df.columns.tolist())
    print(df.head())

    df["data_publicacao"] = pd.to_datetime(df["data_publicacao"], errors="coerce")
    df["titulo"] = None
    df["usuario"] = None
    df["categoria"] = None
    df["status"] = None

    df = df[
        [
            "id_registro",
            "fonte",
            "data_publicacao",
            "titulo",
            "texto_original",
            "nota",
            "usuario",
            "categoria",
            "status",
            "sentimento_real",
            "versao_app",
        ]
    ]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)


if __name__ == "__main__":
    main()
