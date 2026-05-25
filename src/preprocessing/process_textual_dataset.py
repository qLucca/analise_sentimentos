from __future__ import annotations

import pandas as pd

from src.preprocessing.text_preprocessing import preprocess_reviews
from src.utils.paths import NOTEBOOK_DATA_DIR, SILVER_TEXTUAL_DATASET_PATH


def main() -> None:
    input_path = NOTEBOOK_DATA_DIR / "textual_dataset.csv"
    output_path = SILVER_TEXTUAL_DATASET_PATH

    df = pd.read_csv(input_path, encoding="utf-8")
    df_processed = preprocess_reviews(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(df.shape)
    print(df_processed.shape)
    print(output_path)


if __name__ == "__main__":
    main()
