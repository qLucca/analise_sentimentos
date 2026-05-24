from __future__ import annotations
import pandas as pd
from google_play_scraper import Sort, reviews
from src.utils.logger import get_logger, setup_logging
from src.utils.paths import RAW_DIR
logger = get_logger(__name__)

def map_sentiment_from_rating(rating: float | int | None) -> str | None:
    if rating in (1, 2):
        return "Negativo"
    if rating == 3:
        return "Neutro"
    if rating in (4, 5):
        return "Positivo"
    return None

def collect_google_play_reviews(
    app_id: str = "com.nu.production",
    output_path: str | None = None,
    limit =  75000,
    start_date: str | None = "2025-10-01",
    end_date: str | None = "2026-03-31",
    batch_size: int = 200,
    max_reviews_to_scan = 75000
) -> pd.DataFrame:
    """
    Coleta avaliacoes publicas da Google Play Store para o app do Nubank
    usando a biblioteca google-play-scraper.
    """
    logger.info("Preparando coleta da Google Play para o app_id=%s", app_id)

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_reviews_to_scan is not None and max_reviews_to_scan <= 0:
        raise ValueError("max_reviews_to_scan must be greater than zero")

    start_timestamp = pd.to_datetime(start_date).normalize() if start_date else None
    end_timestamp = pd.to_datetime(end_date).normalize() if end_date else None

    if start_timestamp is not None and end_timestamp is not None and start_timestamp > end_timestamp:
        raise ValueError("start_date cannot be later than end_date")

    rows = []
    scanned = 0
    continuation_token = None

    while len(rows) < limit and (max_reviews_to_scan is None or scanned < max_reviews_to_scan):
        count = batch_size
        if max_reviews_to_scan is not None:
            count = min(batch_size, max_reviews_to_scan - scanned)
            if count <= 0:
                break

        resultados, continuation_token = reviews(
            app_id,
            lang="pt",
            country="br",
            sort=Sort.NEWEST,
            count=count,
            continuation_token=continuation_token,
        )

        if not resultados:
            logger.info("Nenhum review adicional retornado pela API.")
            break

        scanned += len(resultados)
        reached_older_than_start = False

        for review in resultados:
            published_at = pd.to_datetime(review.get("at"), errors="coerce")
            if pd.isna(published_at):
                continue

            published_date = published_at.date()

            if end_timestamp is not None and published_date > end_timestamp.date():
                continue

            if start_timestamp is not None and published_date < start_timestamp.date():
                reached_older_than_start = True
                break

            rows.append(
                {
                    "id_registro": review.get("reviewId"),
                    "fonte": "google_play",
                    "data_publicacao": published_date,
                    "texto_original": review.get("content"),
                    "nota": review.get("score"),
                    "versao_app": review.get("reviewCreatedVersion") or review.get("appVersion"),
                    "sentimento_real": map_sentiment_from_rating(review.get("score")),
                }
            )

            if len(rows) >= limit:
                break

        if reached_older_than_start:
            logger.info("A coleta atingiu registros anteriores ao inicio do periodo.")
            break

        if continuation_token is None:
            logger.info("Nao ha mais paginas para continuar a coleta.")
            break

    if (
        not rows
        and start_timestamp is not None
        and max_reviews_to_scan is not None
        and scanned >= max_reviews_to_scan
    ):
        logger.warning(
            "Nenhum review encontrado no periodo antes de atingir o limite de varredura (%s reviews).",
            max_reviews_to_scan,
        )

    logger.info("Quantidade coletada no periodo: %s", len(rows))
    logger.info("Quantidade analisada pela API: %s", scanned)

    df = pd.DataFrame(
        rows,
        columns=[
            "id_registro",
            "fonte",
            "data_publicacao",
            "texto_original",
            "nota",
            "versao_app",
            "sentimento_real",
        ],
    )

    destination = output_path or str(RAW_DIR / "google_play" / "google_play_reviews_raw.csv")
    pd.io.common.check_parent_directory(destination)
    df.to_csv(destination, index=False)

    logger.info("Arquivo bruto salvo em %s", destination)

    if not df.empty:
        logger.info(
            "Intervalo coletado: %s ate %s",
            df["data_publicacao"].min(),
            df["data_publicacao"].max(),
        )

    return df


if __name__ == "__main__":
    setup_logging()
    collect_google_play_reviews()