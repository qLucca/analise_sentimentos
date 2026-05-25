from __future__ import annotations

import pandas as pd

from src.utils.paths import RAW_DIR


def main() -> None:
    input_path = RAW_DIR / "youtube" / "nubank_youtube_comments.csv"
    output_path = RAW_DIR / "youtube" / "youtube_processed.csv"

    df_youtube = pd.read_csv(input_path, encoding="utf-8")

    print(df_youtube.shape)
    print(df_youtube.columns.tolist())
    print(df_youtube.head())

    df_youtube["review_date"] = pd.to_datetime(
        df_youtube["review_date"],
        errors="coerce",
        utc=True,
    )
    df_youtube["comment_updated_at"] = pd.to_datetime(
        df_youtube["comment_updated_at"],
        errors="coerce",
        utc=True,
    )

    df_youtube["id_registro"] = df_youtube["comment_id"]
    df_youtube["fonte"] = "youtube"
    df_youtube["data_publicacao"] = (
        df_youtube["review_date"].dt.tz_convert("America/Sao_Paulo").dt.date
    )
    df_youtube["titulo"] = df_youtube["video_title"]
    df_youtube["texto_original"] = df_youtube["content"]
    df_youtube["nota"] = None
    df_youtube["usuario"] = df_youtube["user_name"]
    df_youtube["categoria"] = df_youtube["query"]
    df_youtube["status"] = None
    df_youtube["sentimento_real"] = df_youtube["sentiment"]

    df_processed = df_youtube[
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
            "video_id",
            "channel_title",
            "like_count",
            "reply_count",
            "comment_updated_at",
        ]
    ].copy()

    print(df_processed.shape)
    print(df_processed.columns.tolist())
    print(df_processed.head())
    print(df_processed.isna().sum())

    df_processed.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)


if __name__ == "__main__":
    main()
