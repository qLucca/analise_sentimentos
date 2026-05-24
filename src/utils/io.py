from pathlib import Path

import pandas as pd


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Formato não suportado: {path}")


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=index)
        return
    if suffix == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
        return
    raise ValueError(f"Formato não suportado: {path}")
