from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger, setup_logging
from src.utils.paths import RAW_DIR
import requests
from bs4 import BeautifulSoup

logger = get_logger(__name__)


def collect_consumidor_gov(
        dataset_url: str = "https://dados.mj.gov.br/dataset/reclamacoes-do-consumidor-gov-br",
        target_months: set[str] | None = None,
):
    logger.info("Collecting data from Consumidor.gov.br")
    logger.info(f"Dataset URL: {dataset_url}")

    if target_months is None:
        target_months = {
            "Outubro_2025",
            "Novembro_2025",
            "Dezembro_2025",
            "Janeiro_2026",
            "Fevereiro_2026",
            "Março_2026",
        }
    logger.info("Target months: %s", sorted(target_months))

    output_dir = RAW_DIR / "consumidor_gov"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    logger.info("Downloading dataset page...")
    response = requests.get(dataset_url, timeout=30)
    response.raise_for_status()
    logger.info("Page loaded successfully. Status code: %s", response.status_code)

    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    logger.info("HTML parsed successfully")

    resource_items = soup.find_all("li", class_="resource-item")
    logger.info("Quantidade de recursos encontrados: %s", len(resource_items))

    for item in resource_items:
        heading = item.find("a", class_="heading")
        download_link = None
        for link in item.find_all("a", href=True):
            href = link.get("href", "")
            if "/download/" in href:
                download_link = link
                break

        if not heading or not download_link:
            continue

        title = heading.get("title", "").strip()
        download_url = download_link.get("href", "").strip()

        if any(month in title for month in target_months):
            logger.info("Baixando recurso: %s", title)
            logger.info("URL: %s", download_url)

            file_name = download_url.split("/")[-1]
            file_path = output_dir / file_name

            file_response = requests.get(download_url, timeout=60)
            file_response.raise_for_status()

            with open(file_path, "wb") as file:
                file.write(file_response.content)

            logger.info("Arquivo salvo em: %s", file_path)

    return soup



if __name__ == "__main__":
    setup_logging()
    collect_consumidor_gov()
