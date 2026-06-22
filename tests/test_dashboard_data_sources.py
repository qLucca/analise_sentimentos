from pathlib import Path

import pandas as pd

from src.dashboard.data_sources import DashboardDataLoader


def test_dashboard_loader_uses_database_and_s3_sources():
    silver_df = pd.DataFrame([{"fonte": "youtube", "texto_original": "bom"}])
    gold_df = pd.DataFrame(
        [{"fonte": "youtube", "texto_original": "otimo", "sentimento_previsto_bert": "Positivo"}]
    )
    model_df = pd.DataFrame([{"modelo": "BERTimbau", "accuracy": 0.85}])

    queried_tables = []

    def fake_table_loader(table_name, schema, where_clause=None):
        queried_tables.append((schema, table_name, where_clause))
        if schema == "silver":
            return silver_df
        return gold_df

    class FakeStore:
        def read_csv(self, key):
            assert key == "processed/model_comparison_summary.csv"
            return model_df

    loader = DashboardDataLoader(
        table_loader=fake_table_loader,
        artifact_store=FakeStore(),
        processed_data_dir=Path("data/processed"),
    )

    datasets = loader.load()

    assert datasets["Base unificada"].equals(silver_df)
    assert datasets["YouTube + BERTimbau"].equals(gold_df)
    assert datasets["Consumidor.gov"].equals(silver_df)
    assert datasets["Resumo de modelos"].equals(model_df)
    assert queried_tables == [
        ("silver", "reviews_cleaned", None),
        ("gold", "sentiment_analysis", "fonte = 'youtube'"),
        ("silver", "reviews_cleaned", "fonte = 'consumidor_gov'"),
    ]
