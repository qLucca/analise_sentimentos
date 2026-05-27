from __future__ import annotations

import pandas as pd

from src.utils.paths import (
    NOTEBOOK_DATA_DIR,
    RAW_DIR,
)


COMMON_COLUMNS = [
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
]


def main() -> None:
    google_play_path = RAW_DIR / "google_play" / "google_play_reviews_processed.csv"
    consumidor_gov_path = RAW_DIR / "consumidor_gov" / "consumidor_gov_processed.csv"
    youtube_path = RAW_DIR / "youtube" / "youtube_processed.csv"

    unified_output_path = NOTEBOOK_DATA_DIR / "unified_dataset.csv"
    textual_output_path = NOTEBOOK_DATA_DIR / "textual_dataset.csv"
    structured_output_path = NOTEBOOK_DATA_DIR / "structured_dataset.csv"

    df_google_play = pd.read_csv(google_play_path, encoding="utf-8")
    df_consumidor_gov = pd.read_csv(consumidor_gov_path, encoding="utf-8")
    df_youtube = pd.read_csv(youtube_path, encoding="utf-8")

    print("Google Play:", df_google_play.shape)
    print("Consumidor.gov:", df_consumidor_gov.shape)
    print("YouTube:", df_youtube.shape)

    df_google_play = df_google_play[COMMON_COLUMNS].copy()
    df_consumidor_gov = df_consumidor_gov[COMMON_COLUMNS].copy()
    df_youtube = df_youtube[COMMON_COLUMNS].copy()

    for dataframe in [df_google_play, df_consumidor_gov, df_youtube]:
        dataframe["data_publicacao"] = pd.to_datetime(
            dataframe["data_publicacao"],
            errors="coerce",
        )

    df_unified = pd.concat(
        [df_google_play, df_consumidor_gov, df_youtube],
        ignore_index=True,
    )
    df_unified = df_unified.drop_duplicates(
        subset=["id_registro", "fonte"],
        keep="first",
    )
    df_unified = df_unified.sort_values(
        "data_publicacao",
        ascending=False,
    ).reset_index(drop=True)

    print("UNIFIED")
    print(df_unified.shape)
    print(df_unified.columns.tolist())
    print(df_unified.head())
    print(df_unified.isna().sum())

    df_textual = df_unified[
        df_unified["texto_original"].notna()
        & (df_unified["texto_original"].astype(str).str.strip() != "")
    ].copy()

    print("TEXTUAL")
    print(df_textual.shape)
    print(df_textual.head())
    print(df_textual.isna().sum())

    df_structured = df_unified[
        df_unified["categoria"].notna()
        | df_unified["status"].notna()
        | df_unified["nota"].notna()
    ].copy()

    print("STRUCTURED")
    print(df_structured.shape)
    print(df_structured.head())
    print(df_structured.isna().sum())

    NOTEBOOK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_unified.to_csv(unified_output_path, index=False, encoding="utf-8-sig")
    df_textual.to_csv(textual_output_path, index=False, encoding="utf-8-sig")
    df_structured.to_csv(structured_output_path, index=False, encoding="utf-8-sig")

    print(unified_output_path)
    print(textual_output_path)
    print(structured_output_path)


if __name__ == "__main__":
    main()
