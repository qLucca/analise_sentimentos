import pandas as pd
from src.pipelines.run_preprocessing import build_unified_bronze


def test_build_unified_bronze_returns_dataframe():
    result = build_unified_bronze()
    assert isinstance(result, pd.DataFrame)
