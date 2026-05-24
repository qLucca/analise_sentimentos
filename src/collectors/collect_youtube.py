from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger, setup_logging
from src.utils.paths import RAW_DIR

logger = get_logger(__name__)
load_dotenv()


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
DEFAULT_OUTPUT_PATH = RAW_DIR / "youtube" / "nubank_youtube_comments.csv"
DEFAULT_VIDEOS_OUTPUT_PATH = RAW_DIR / "youtube" / "nubank_youtube_videos.csv"
DEFAULT_QUERIES = [
    "Nubank",
    "Nubank app",
    "Nubank cartão",
    "Nubank conta",
    "Nubank pix",
    "Nubank empréstimo",
    "Nubank reclamação",
    "Nubank problema",
    "Nubank conta bloqueada",
    "Nubank atendimento",
]
EMPTY_COMMENTS_COLUMNS = [
    "source",
    "video_id",
    "video_title",
    "video_description",
    "channel_id",
    "channel_title",
    "video_published_at",
    "query",
    "comment_id",
    "review_date",
    "content",
    "user_name",
    "author_channel_id",
    "like_count",
    "reply_count",
    "comment_updated_at",
    "sentiment",
]


def map_sentiment_from_text_placeholder(text: str | None) -> str | None:
    """
    Placeholder para manter padrão com as outras coletas.

    Aqui não temos rating numérico como Google Play/App Store.
    Depois você pode aplicar modelo de sentimento/NLP sobre a coluna content.
    """
    return None


def request_youtube_api(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    sleep: float = 0.2,
) -> dict[str, Any] | None:
    try:
        response = session.get(url, params=params, timeout=30)

        if response.status_code == 403:
            logger.error("Erro 403 na API do YouTube: %s", response.text[:500])
            return None

        if response.status_code == 400:
            logger.error("Erro 400 na API do YouTube: %s", response.text[:500])
            return None

        response.raise_for_status()

        time.sleep(sleep)

        return response.json()

    except requests.exceptions.RequestException as error:
        logger.error("Erro de requisição na API do YouTube: %s", error)
        return None


def search_youtube_videos(
    session: requests.Session,
    api_key: str,
    query: str,
    start_date: str,
    end_date: str,
    max_videos: int = 100,
    region_code: str = "BR",
    relevance_language: str = "pt",
    sleep: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Busca vídeos públicos no YouTube dentro do intervalo informado.

    O endpoint search.list permite usar publishedAfter e publishedBefore
    em formato RFC3339. Cada chamada search.list custa 100 unidades de quota.
    """

    logger.info("Buscando vídeos para query: %s", query)

    videos: list[dict[str, Any]] = []
    next_page_token: str | None = None

    published_after = f"{start_date}T00:00:00Z"
    published_before = f"{end_date}T23:59:59Z"

    while len(videos) < max_videos:
        params = {
            "key": api_key,
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "relevance",
            "maxResults": 50,
            "regionCode": region_code,
            "relevanceLanguage": relevance_language,
            "publishedAfter": published_after,
            "publishedBefore": published_before,
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        data = request_youtube_api(
            session=session,
            url=YOUTUBE_SEARCH_URL,
            params=params,
            sleep=sleep,
        )

        if not data:
            break

        items = data.get("items", [])

        for item in items:
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})

            if not video_id:
                continue

            videos.append(
                {
                    "video_id": video_id,
                    "video_title": snippet.get("title"),
                    "video_description": snippet.get("description"),
                    "channel_id": snippet.get("channelId"),
                    "channel_title": snippet.get("channelTitle"),
                    "video_published_at": snippet.get("publishedAt"),
                    "query": query,
                }
            )

            if len(videos) >= max_videos:
                break

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    logger.info("Query '%s' retornou %s vídeos", query, len(videos))

    return videos


def collect_comments_from_video(
    session: requests.Session,
    api_key: str,
    video: dict[str, Any],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    max_comments_per_video: int = 500,
    sleep: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Coleta comentários públicos de um vídeo.

    O endpoint commentThreads.list retorna comentários em páginas.
    maxResults aceita valores de 1 a 100.
    """

    video_id = video["video_id"]

    logger.info("Coletando comentários do vídeo: %s", video_id)

    comments: list[dict[str, Any]] = []
    next_page_token: str | None = None

    while len(comments) < max_comments_per_video:
        params = {
            "key": api_key,
            "part": "snippet",
            "videoId": video_id,
            "order": "time",
            "textFormat": "plainText",
            "maxResults": 100,
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        data = request_youtube_api(
            session=session,
            url=YOUTUBE_COMMENTS_URL,
            params=params,
            sleep=sleep,
        )

        if not data:
            break

        items = data.get("items", [])

        if not items:
            break

        stop_video_due_old_comments = False

        for item in items:
            snippet = item.get("snippet", {})
            top_comment = snippet.get("topLevelComment", {})
            top_comment_snippet = top_comment.get("snippet", {})

            comment_id = top_comment.get("id")
            comment_text = top_comment_snippet.get("textOriginal")
            author_name = top_comment_snippet.get("authorDisplayName")
            author_channel_id = (
                top_comment_snippet
                .get("authorChannelId", {})
                .get("value")
            )
            comment_published_at = top_comment_snippet.get("publishedAt")
            comment_updated_at = top_comment_snippet.get("updatedAt")
            like_count = top_comment_snippet.get("likeCount")
            reply_count = snippet.get("totalReplyCount")

            comment_dt = pd.to_datetime(
                comment_published_at,
                errors="coerce",
                utc=True,
            )

            if pd.isna(comment_dt):
                continue

            # Como order=time vem dos mais recentes para os mais antigos,
            # se já ficou antes do início, dá para parar esse vídeo.
            if comment_dt < start_dt:
                stop_video_due_old_comments = True
                continue

            if comment_dt >= end_dt:
                continue

            comments.append(
                {
                    "source": "youtube_comments",
                    "video_id": video_id,
                    "video_title": video.get("video_title"),
                    "video_description": video.get("video_description"),
                    "channel_id": video.get("channel_id"),
                    "channel_title": video.get("channel_title"),
                    "video_published_at": video.get("video_published_at"),
                    "query": video.get("query"),
                    "comment_id": comment_id,
                    "review_date": comment_published_at,
                    "content": comment_text,
                    "user_name": author_name,
                    "author_channel_id": author_channel_id,
                    "like_count": like_count,
                    "reply_count": reply_count,
                    "comment_updated_at": comment_updated_at,
                    "sentiment": map_sentiment_from_text_placeholder(comment_text),
                }
            )

            if len(comments) >= max_comments_per_video:
                break

        if len(comments) >= max_comments_per_video:
            break

        if stop_video_due_old_comments:
            break

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    logger.info(
        "Vídeo %s retornou %s comentários dentro do intervalo",
        video_id,
        len(comments),
    )

    return comments


def collect_youtube_comments_about_nubank(
    output_path: str | None = None,
    videos_output_path: str | None = None,
    start_date: str = "2025-10-01",
    end_date: str = "2026-03-31",
    queries: list[str] | None = None,
    max_videos_per_query: int = 50,
    max_comments_per_video: int = 300,
    sleep: float = 0.2,
) -> pd.DataFrame:
    """
    Coleta comentários públicos do YouTube sobre Nubank.

    Fonte complementar para análise de sentimento/NLP:
    - Google Play: reviews de app
    - Consumidor.gov: reclamações formais
    - YouTube: comentários públicos em vídeos relacionados ao Nubank
    """

    api_key = os.getenv("YOUTUBE_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_KEY não encontrada. "
            "Crie um arquivo .env na raiz do projeto com YOUTUBE_KEY=SUA_KEY."
        )

    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
    videos_output_path = (
        Path(videos_output_path) if videos_output_path else DEFAULT_VIDEOS_OUTPUT_PATH
    )
    if queries is None:
        queries = DEFAULT_QUERIES

    logger.info("Iniciando coleta de comentários do YouTube")
    logger.info("Intervalo: %s até %s", start_date, end_date)
    logger.info("Total de queries: %s", len(queries))

    session = requests.Session()
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)
    all_videos: list[dict[str, Any]] = []

    for query in queries:
        videos = search_youtube_videos(
            session=session,
            api_key=api_key,
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_videos=max_videos_per_query,
            sleep=sleep,
        )

        all_videos.extend(videos)

    if not all_videos:
        logger.warning("Nenhum vídeo encontrado.")
        return pd.DataFrame()

    df_videos = pd.DataFrame(all_videos)

    df_videos = df_videos.drop_duplicates(subset=["video_id"], keep="first")

    logger.info("Total de vídeos únicos encontrados: %s", len(df_videos))

    videos_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_videos.to_csv(videos_output_path, index=False, encoding="utf-8-sig")

    logger.info("Arquivo de vídeos salvo em: %s", videos_output_path)

    all_comments: list[dict[str, Any]] = []
    videos_records = df_videos.to_dict(orient="records")

    for index, video in enumerate(videos_records, start=1):
        logger.info(
            "Processando vídeo %s/%s",
            index,
            len(videos_records),
        )

        comments = collect_comments_from_video(
            session=session,
            api_key=api_key,
            video=video,
            start_dt=start_dt,
            end_dt=end_dt,
            max_comments_per_video=max_comments_per_video,
            sleep=sleep,
        )

        all_comments.extend(comments)

    if not all_comments:
        logger.warning("Nenhum comentário encontrado dentro do intervalo.")

        df_empty = pd.DataFrame(columns=EMPTY_COMMENTS_COLUMNS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_empty.to_csv(output_path, index=False, encoding="utf-8-sig")

        return df_empty

    df_comments = pd.DataFrame(all_comments)

    df_comments["review_date"] = pd.to_datetime(
        df_comments["review_date"],
        errors="coerce",
        utc=True,
    )

    df_comments = df_comments.dropna(subset=["review_date"])

    df_comments = df_comments[
        (df_comments["review_date"] >= start_dt)
        & (df_comments["review_date"] < end_dt)
    ].copy()

    df_comments = df_comments.drop_duplicates(
        subset=["comment_id", "content"],
        keep="first",
    )

    df_comments = df_comments.sort_values("review_date", ascending=False)

    df_comments["review_date"] = (
        df_comments["review_date"]
        .dt.tz_convert("America/Sao_Paulo")
        .dt.tz_localize(None)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_comments.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("Coleta finalizada.")
    logger.info("Total de comentários salvos: %s", len(df_comments))
    logger.info("Arquivo salvo em: %s", output_path)

    return df_comments


if __name__ == "__main__":
    setup_logging()

    collect_youtube_comments_about_nubank(
        output_path=RAW_DIR / "youtube" / "nubank_youtube_comments.csv",
        videos_output_path=RAW_DIR / "youtube" / "nubank_youtube_videos.csv",
        start_date="2025-10-01",
        end_date="2026-03-31",
        max_videos_per_query=50,
        max_comments_per_video=300,
        sleep=0.2,
    )
