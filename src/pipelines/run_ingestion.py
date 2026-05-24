from __future__ import annotations

import pandas as pd

from src.collectors.collect_consumidor_gov import collect_consumidor_gov
from src.collectors.collect_google_play import collect_google_play_reviews
from src.collectors.collect_youtube import collect_reclame_aqui
from src.utils.logger import setup_logging
from src.utils.paths import ensure_runtime_directories


def run() -> dict[str, pd.DataFrame]:
    setup_logging()
    ensure_runtime_directories()
    return {
        "google_play": collect_google_play_reviews(),
        "consumidor_gov": collect_consumidor_gov(),
        "youtube": collect_reclame_aqui(),
    }


if __name__ == "__main__":
    run()
