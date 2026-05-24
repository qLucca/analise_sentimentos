import logging
import logging.config
from pathlib import Path

import yaml

from src.utils.paths import CONFIG_DIR


def setup_logging(config_path: Path | None = None) -> None:
    path = config_path or CONFIG_DIR / "logging.yaml"
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
