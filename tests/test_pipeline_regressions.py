import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.database.connection import build_connection_string
from src.database.load_gold_to_sqlserver import (
    load_gold_dataframe,
    prepare_gold_model_metrics,
    prepare_gold_sentiment_analysis,
)
from src.database.load_silver_to_sqlserver import load_silver_reviews, prepare_silver_reviews
from src.pipelines.run_preprocessing import build_unified_bronze
from src.preprocessing.process_consumidor_gov import main as process_consumidor_gov


def test_build_connection_string_supports_sql_auth(monkeypatch):
    monkeypatch.setenv("SQLSERVER_HOST", "localhost")
    monkeypatch.setenv("SQLSERVER_DATABASE", "NubankSentimentAnalysis")
    monkeypatch.setenv("SQLSERVER_DRIVER", "ODBC Driver 18 for SQL Server")
    monkeypatch.setenv("SQLSERVER_TRUSTED_CONNECTION", "no")
    monkeypatch.setenv("SQLSERVER_USER", "sa")
    monkeypatch.setenv("SQLSERVER_PASSWORD", "secret")

    connection_string = build_connection_string()

    assert "UID%3Dsa" in connection_string
    assert "PWD%3Dsecret" in connection_string


def test_build_connection_string_defaults_to_trusting_server_certificate(monkeypatch):
    monkeypatch.setenv("SQLSERVER_HOST", "localhost")
    monkeypatch.setenv("SQLSERVER_DATABASE", "NubankSentimentAnalysis")
    monkeypatch.setenv("SQLSERVER_DRIVER", "ODBC Driver 18 for SQL Server")
    monkeypatch.setenv("SQLSERVER_TRUSTED_CONNECTION", "yes")
    monkeypatch.delenv("SQLSERVER_TRUST_SERVER_CERTIFICATE", raising=False)

    connection_string = build_connection_string()

    assert "TrustServerCertificate%3Dyes" in connection_string


def test_run_ingestion_module_imports():
    sys.modules.pop("src.pipelines.run_ingestion", None)
    module = importlib.import_module("src.pipelines.run_ingestion")
    assert hasattr(module, "run")


def test_run_training_module_imports():
    sys.modules.pop("src.pipelines.run_training", None)
    module = importlib.import_module("src.pipelines.run_training")
    assert hasattr(module, "run")


def test_build_unified_bronze_falls_back_to_legacy_processed_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    legacy_dataset = pd.DataFrame(
        [
            {
                "id_registro": "1",
                "fonte": "google_play",
                "data_publicacao": "2026-03-31",
                "titulo": None,
                "texto_original": "App muito bom",
                "nota": 5,
                "usuario": None,
                "categoria": None,
                "status": None,
                "sentimento_real": "Positivo",
            }
        ]
    )
    legacy_path = processed_dir / "unified_dataset.csv"
    legacy_dataset.to_csv(legacy_path, index=False)

    monkeypatch.setattr("src.pipelines.run_preprocessing.RAW_DIR", raw_dir)
    monkeypatch.setattr("src.pipelines.run_preprocessing.PROJECT_ROOT", tmp_path)

    result = build_unified_bronze()

    assert len(result) == 1
    assert str(result.loc[0, "id_registro"]) == "1"


def test_prepare_silver_reviews_normalizes_legacy_columns():
    df = pd.DataFrame(
        [
            {
                "id_registro": "1",
                "fonte": "google_play",
                "data_publicacao": "2026-03-31",
                "texto_original": "App muito bom",
                "texto_limpo": "app muito bom",
                "nota": 5,
                "categoria": "Atendimento",
                "status": "Fechado",
                "sentimento_real": "Positivo",
            }
        ]
    )

    prepared = prepare_silver_reviews(df)

    assert list(prepared.columns) == [
        "id_registro",
        "fonte",
        "data_publicacao",
        "texto_original",
        "texto_limpo",
        "nota",
        "status_reclamacao",
        "categoria_problema",
        "uf",
        "versao_app",
        "sentimento_real",
        "data_processamento",
    ]
    assert prepared.loc[0, "categoria_problema"] == "Atendimento"
    assert prepared.loc[0, "status_reclamacao"] == "Fechado"


def test_prepare_gold_sentiment_analysis_normalizes_legacy_columns():
    df = pd.DataFrame(
        [
            {
                "id_registro": "1",
                "fonte": "youtube",
                "data_publicacao": "2026-03-31",
                "texto_original": "Atendimento ruim",
                "texto_limpo": "atendimento ruim",
                "nota": None,
                "categoria": "Nubank atendimento",
                "status": None,
                "sentimento_real": None,
                "sentimento_previsto": "Negativo",
                "topico": "Topico 1",
            }
        ]
    )

    prepared = prepare_gold_sentiment_analysis(df)

    assert prepared.loc[0, "categoria_problema"] == "Nubank atendimento"
    assert "titulo" not in prepared.columns
    assert "categoria" not in prepared.columns


def test_prepare_gold_model_metrics_maps_dashboard_summary_to_sql_schema():
    df = pd.DataFrame(
        [
            {
                "modelo": "LogisticRegression",
                "accuracy": 0.85,
                "precision_macro": 0.70,
                "recall_macro": 0.66,
                "f1_macro": 0.63,
                "insight_resumo": "Modelo principal operacional.",
                "proveniencia": "classic_runtime",
            }
        ]
    )

    prepared = prepare_gold_model_metrics(df)

    assert list(prepared.columns) == [
        "id_execucao",
        "modelo",
        "vetorizador",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "data_treinamento",
        "observacoes",
    ]
    assert prepared.loc[0, "modelo"] == "LogisticRegression"
    assert prepared.loc[0, "vetorizador"] == "tfidf"
    assert "classic_runtime" in prepared.loc[0, "observacoes"]


def test_load_silver_reviews_refreshes_table_before_append(monkeypatch):
    executed = []
    to_sql_calls = []

    class FakeConnection:
        def execute(self, statement):
            executed.append(str(statement))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def begin(self):
            return FakeConnection()

    def fake_to_sql(self, table_name, con, schema, if_exists, index):
        to_sql_calls.append(
            {
                "table_name": table_name,
                "schema": schema,
                "if_exists": if_exists,
                "index": index,
            }
        )

    monkeypatch.setattr("src.database.load_silver_to_sqlserver.get_engine", lambda: FakeEngine())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=False)

    load_silver_reviews(
        pd.DataFrame(
            [
                {
                    "id_registro": "1",
                    "fonte": "google_play",
                    "data_publicacao": "2026-03-31",
                    "texto_original": "App muito bom",
                    "texto_limpo": "app muito bom",
                    "nota": 5,
                }
            ]
        )
    )

    assert any("DELETE FROM silver.reviews_cleaned" in statement for statement in executed)
    assert to_sql_calls[0]["if_exists"] == "append"


def test_load_gold_dataframe_refreshes_table_before_append(monkeypatch):
    executed = []
    to_sql_calls = []

    class FakeConnection:
        def execute(self, statement):
            executed.append(str(statement))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def begin(self):
            return FakeConnection()

    def fake_to_sql(self, table_name, con, schema, if_exists, index):
        to_sql_calls.append(
            {
                "table_name": table_name,
                "schema": schema,
                "if_exists": if_exists,
                "index": index,
            }
        )

    monkeypatch.setattr("src.database.load_gold_to_sqlserver.get_engine", lambda: FakeEngine())
    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=False)

    load_gold_dataframe(
        pd.DataFrame(
            [
                {
                    "id_registro": "1",
                    "fonte": "youtube",
                    "data_publicacao": "2026-03-31",
                    "texto_original": "Atendimento ruim",
                    "texto_limpo": "atendimento ruim",
                    "sentimento_previsto": "Negativo",
                }
            ]
        ),
        "sentiment_analysis",
    )

    assert any("DELETE FROM gold.sentiment_analysis" in statement for statement in executed)
    assert to_sql_calls[0]["if_exists"] == "append"


def test_build_model_summary_row_contains_required_fields():
    from src.models.model_summary import build_model_summary_row

    metrics = {
        "accuracy": 0.8,
        "precision_macro": 0.7,
        "recall_macro": 0.6,
        "f1_macro": 0.65,
        "roc_auc_macro": 0.81,
        "roc_auc_weighted": 0.9,
        "f1_negativo": 0.62,
        "f1_neutro": 0.31,
        "f1_positivo": 0.88,
    }
    row = build_model_summary_row(
        model_name="LogisticRegression",
        metrics=metrics,
        is_primary=False,
        insight_summary="baseline estavel",
        provenance="classic_runtime",
    )

    assert row["modelo"] == "LogisticRegression"
    assert row["accuracy"] == metrics["accuracy"]
    assert row["precision_macro"] == metrics["precision_macro"]
    assert row["recall_macro"] == metrics["recall_macro"]
    assert row["f1_macro"] == metrics["f1_macro"]
    assert row["roc_auc_macro"] == metrics["roc_auc_macro"]
    assert row["roc_auc_weighted"] == metrics["roc_auc_weighted"]
    assert row["f1_negativo"] == metrics["f1_negativo"]
    assert row["f1_neutro"] == metrics["f1_neutro"]
    assert row["f1_positivo"] == metrics["f1_positivo"]
    assert row["modelo_principal"] is False
    assert row["insight_resumo"] == "baseline estavel"
    assert row["proveniencia"] == "classic_runtime"
    assert set(row).issuperset(
        {
            "modelo",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc_macro",
            "roc_auc_weighted",
            "f1_negativo",
            "f1_neutro",
            "f1_positivo",
            "modelo_principal",
            "insight_resumo",
            "proveniencia",
        }
    )


def test_choose_primary_model_prefers_roc_auc_then_f1_then_accuracy():
    from src.models.model_summary import choose_primary_model

    rows = [
        {"modelo": "A", "roc_auc_macro": 0.91, "f1_macro": 0.60, "accuracy": 0.88},
        {"modelo": "B", "roc_auc_macro": 0.91, "f1_macro": 0.62, "accuracy": 0.87},
        {"modelo": "C", "roc_auc_macro": 0.90, "f1_macro": 0.80, "accuracy": 0.95},
    ]

    winner = choose_primary_model(rows)

    assert winner["modelo"] == "B"


def test_choose_primary_model_treats_none_metrics_as_lowest_value():
    from src.models.model_summary import choose_primary_model

    rows = [
        {"modelo": "A", "roc_auc_macro": None, "f1_macro": 0.95, "accuracy": 0.95},
        {"modelo": "B", "roc_auc_macro": 0.90, "f1_macro": None, "accuracy": 0.90},
        {"modelo": "C", "roc_auc_macro": 0.90, "f1_macro": 0.70, "accuracy": None},
        {"modelo": "D", "roc_auc_macro": 0.90, "f1_macro": 0.70, "accuracy": 0.80},
    ]

    winner = choose_primary_model(rows)

    assert winner["modelo"] == "D"


def test_choose_primary_model_treats_nan_metrics_as_lowest_value():
    from src.models.model_summary import choose_primary_model

    rows = [
        {"modelo": "A", "roc_auc_macro": float("nan"), "f1_macro": 0.95, "accuracy": 0.95},
        {"modelo": "B", "roc_auc_macro": 0.90, "f1_macro": float("nan"), "accuracy": 0.90},
        {"modelo": "C", "roc_auc_macro": 0.90, "f1_macro": 0.70, "accuracy": float("nan")},
        {"modelo": "D", "roc_auc_macro": 0.90, "f1_macro": 0.70, "accuracy": 0.80},
    ]

    winner = choose_primary_model(rows)

    assert winner["modelo"] == "D"


def test_choose_primary_model_treats_pd_na_metrics_as_lowest_value():
    from src.models.model_summary import choose_primary_model

    rows = [
        {"modelo": "A", "roc_auc_macro": pd.NA, "f1_macro": 0.95, "accuracy": 0.95},
        {"modelo": "B", "roc_auc_macro": 0.90, "f1_macro": pd.NA, "accuracy": 0.90},
        {"modelo": "C", "roc_auc_macro": 0.90, "f1_macro": 0.70, "accuracy": pd.NA},
        {"modelo": "D", "roc_auc_macro": 0.90, "f1_macro": 0.70, "accuracy": 0.80},
    ]

    winner = choose_primary_model(rows)

    assert winner["modelo"] == "D"


def test_merge_model_summaries_marks_only_the_selected_primary_model():
    from src.models.model_summary import merge_model_summaries

    classic_rows = [
        {
            "modelo": "LogisticRegression",
            "roc_auc_macro": 0.88,
            "f1_macro": 0.63,
            "accuracy": 0.85,
            "proveniencia": "classic_runtime",
        }
    ]
    bert_row = {
        "modelo": "BERTimbau",
        "roc_auc_macro": 0.91,
        "f1_macro": 0.61,
        "accuracy": 0.90,
        "proveniencia": "benchmark",
    }

    merged = merge_model_summaries(classic_rows, bert_row)

    winners = [row["modelo"] for row in merged if row["modelo_principal"]]

    assert winners == ["LogisticRegression"]


def test_merge_model_summaries_marks_only_one_primary_when_model_names_repeat():
    from src.models.model_summary import merge_model_summaries

    classic_rows = [
        {
            "modelo": "LinearSVC",
            "roc_auc_macro": 0.88,
            "f1_macro": 0.62,
            "accuracy": 0.84,
            "proveniencia": "classic_runtime",
        },
        {
            "modelo": "LinearSVC",
            "roc_auc_macro": 0.81,
            "f1_macro": 0.55,
            "accuracy": 0.80,
            "proveniencia": "classic_runtime",
        },
    ]

    merged = merge_model_summaries(classic_rows, None)

    assert sum(bool(row["modelo_principal"]) for row in merged) == 1
    assert bool(merged[0]["modelo_principal"]) is True
    assert bool(merged[1]["modelo_principal"]) is False


def test_run_bertimbau_evaluation_returns_benchmark_summary_row():
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    benchmark_metrics = {
        "modelo": "BERTimbau",
        "accuracy": 0.9,
        "f1_macro": 0.61,
        "roc_auc_macro": 0.91,
        "roc_auc_weighted": 0.95,
        "f1_negativo": 0.81,
        "f1_neutro": 0.08,
        "f1_positivo": 0.95,
    }
    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "src.models.bertimbau_runner._load_bertimbau_benchmark_metrics",
        lambda: (benchmark_metrics, ""),
    )

    try:
        row, info = run_bertimbau_evaluation(df)
    finally:
        monkeypatch.undo()

    assert row is not None
    assert row["modelo"] == "BERTimbau"
    assert row["proveniencia"] == "benchmark"
    assert pd.notna(row["roc_auc_macro"])
    assert "benchmark" in row["insight_resumo"].lower()
    assert info["available"] is True
    assert info["source"] == "benchmark"


def test_bertimbau_runtime_annotation_uses_full_base_export():
    from src.models.bertimbau_runner import (
        _annotate_bert_predictions,
        _build_prediction_frame,
        _prepare_runtime_text_columns,
    )

    df = pd.DataFrame(
        [
            {
                "fonte": "google_play",
                "texto_original": "App travando muito",
                "texto_limpo": "app travando muito",
            },
            {
                "fonte": "youtube",
                "texto_original": "Problema com pix",
                "texto_limpo": "problema com pix",
            },
        ]
    )
    prepared = _prepare_runtime_text_columns(df)
    prediction_df = _build_prediction_frame(prepared)

    class FakePredOutput:
        predictions = pd.DataFrame(
            [
                [0.1, 0.2, 0.7],
                [0.7, 0.2, 0.1],
            ]
        ).to_numpy()

    annotated = _annotate_bert_predictions(
        prediction_df,
        FakePredOutput(),
        {0: "Negativo", 1: "Neutro", 2: "Positivo"},
    )

    assert len(annotated) == 2
    assert "sentimento_previsto_bert" in annotated.columns
    assert "tema_negocio" in annotated.columns
    assert "texto_modelo" in annotated.columns
    assert annotated.loc[0, "texto_modelo"].startswith("fonte: Google Play")
    assert set(annotated["sentimento_previsto_bert"]) == {"Positivo", "Negativo"}
    assert set(annotated["tema_negocio"]) <= {"App e estabilidade", "Pix e transferências"}


def test_run_bertimbau_evaluation_degrades_when_benchmark_schema_is_incomplete(monkeypatch):
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    monkeypatch.setattr(
        "src.models.bertimbau_runner._load_bertimbau_benchmark_metrics",
        lambda: ({"modelo": "BERTimbau", "accuracy": 0.9, "f1_macro": 0.61}, ""),
    )

    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )

    row, info = run_bertimbau_evaluation(df)

    assert row is None
    assert info["available"] is False
    assert "schema" in info["reason"].lower()


def test_run_bertimbau_evaluation_uses_last_valid_comparison_rows_assignment(
    tmp_path, monkeypatch
):
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    benchmark_file = tmp_path / "bertimbau_experiment.py"
    benchmark_file.write_text(
        "comparison_rows = [\n"
        "    {'modelo': 'BERTimbau', 'accuracy': 0.1, 'f1_macro': 0.1, 'roc_auc_macro': 0.1}\n"
        "]\n"
        "comparison_rows = [\n"
        "    {'modelo': 'BERTimbau', 'accuracy': 0.9, 'f1_macro': 0.8, 'roc_auc_macro': 0.95}\n"
        "]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.models.bertimbau_runner.BERTIMBAU_EXPERIMENT_PATH",
        benchmark_file,
    )

    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )

    row, info = run_bertimbau_evaluation(df)

    assert row is not None
    assert row["accuracy"] == 0.9
    assert info["available"] is True


def test_run_bertimbau_evaluation_supports_annotated_comparison_rows_assignment(
    tmp_path, monkeypatch
):
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    benchmark_file = tmp_path / "bertimbau_experiment.py"
    benchmark_file.write_text(
        "comparison_rows: list[dict[str, float]] = [\n"
        "    {'modelo': 'BERTimbau', 'accuracy': 0.93, 'f1_macro': 0.79, 'roc_auc_macro': 0.96}\n"
        "]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.models.bertimbau_runner.BERTIMBAU_EXPERIMENT_PATH",
        benchmark_file,
    )

    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )

    row, info = run_bertimbau_evaluation(df)

    assert row is not None
    assert row["accuracy"] == 0.93
    assert info["available"] is True


def test_run_bertimbau_evaluation_degrades_safely_when_benchmark_file_is_corrupted(
    tmp_path, monkeypatch
):
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    corrupted_benchmark = tmp_path / "bertimbau_experiment.py"
    corrupted_benchmark.write_text("comparison_rows = build_rows(", encoding="utf-8")
    monkeypatch.setattr(
        "src.models.bertimbau_runner.BERTIMBAU_EXPERIMENT_PATH",
        corrupted_benchmark,
    )

    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )

    row, info = run_bertimbau_evaluation(df)

    assert row is None
    assert info["available"] is False
    assert "sintaxe" in info["reason"].lower()


def test_run_bertimbau_evaluation_degrades_safely_when_comparison_rows_is_programmatic(
    tmp_path, monkeypatch
):
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    programmatic_benchmark = tmp_path / "bertimbau_experiment.py"
    programmatic_benchmark.write_text(
        "def build_rows():\n    return []\n\ncomparison_rows = build_rows()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.models.bertimbau_runner.BERTIMBAU_EXPERIMENT_PATH",
        programmatic_benchmark,
    )

    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )

    row, info = run_bertimbau_evaluation(df)

    assert row is None
    assert info["available"] is False
    assert "nao-literal" in info["reason"].lower()


def test_run_bertimbau_evaluation_degrades_safely_when_comparison_rows_literal_is_malformed(
    tmp_path, monkeypatch
):
    from src.models.bertimbau_runner import run_bertimbau_evaluation

    malformed_benchmark = tmp_path / "bertimbau_experiment.py"
    malformed_benchmark.write_text(
        "comparison_rows = ['BERTimbau', {'modelo': 'LogisticRegression'}]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.models.bertimbau_runner.BERTIMBAU_EXPERIMENT_PATH",
        malformed_benchmark,
    )

    df = pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
        ]
    )

    row, info = run_bertimbau_evaluation(df)

    assert row is None
    assert info["available"] is False
    assert "malformad" in info["reason"].lower()


def test_run_training_writes_model_comparison_summary_with_classic_metrics(tmp_path, monkeypatch):
    from src.pipelines.run_training import run

    class DummyClassicModel:
        pass

    silver_path = tmp_path / "silver.csv"
    dashboard_summary = tmp_path / "model_comparison_summary.csv"
    dashboard_predicted = tmp_path / "predicted.csv"
    dashboard_primary = tmp_path / "primary.csv"
    pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
            {"texto_limpo": "otimo atendimento", "sentimento_real": "Positivo"},
            {"texto_limpo": "app travando", "sentimento_real": "Negativo"},
            {"texto_limpo": "servico ok", "sentimento_real": "Neutro"},
            {"texto_limpo": "gostei bastante", "sentimento_real": "Positivo"},
            {"texto_limpo": "horrivel usar", "sentimento_real": "Negativo"},
            {"texto_limpo": "nada demais", "sentimento_real": "Neutro"},
            {"texto_limpo": "excelente suporte", "sentimento_real": "Positivo"},
            {"texto_limpo": "péssimo retorno", "sentimento_real": "Negativo"},
            {"texto_limpo": "razoavel ate aqui", "sentimento_real": "Neutro"},
        ]
    ).to_csv(silver_path, index=False)

    monkeypatch.setattr("src.pipelines.run_training.SILVER_UNIFIED_DATASET_PATH", silver_path)
    monkeypatch.setattr("src.pipelines.run_training.GOLD_DASHBOARD_MODEL_COMPARISON_PATH", dashboard_summary)
    monkeypatch.setattr("src.pipelines.run_training.GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH", dashboard_predicted)
    monkeypatch.setattr("src.pipelines.run_training.GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH", dashboard_primary)
    monkeypatch.setattr(
        "src.pipelines.run_training.train_sentiment_model",
        lambda df: (
            DummyClassicModel(),
            {
                "accuracy": 0.85,
                "precision_macro": 0.7,
                "recall_macro": float("nan"),
                "f1_macro": 0.63,
                "roc_auc_macro": float("nan"),
                "model_name": "LinearSVC",
            },
            object(),
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.run_training.predict_sentiment",
        lambda df, model, vectorizer: df.assign(sentimento_previsto="Positivo"),
    )
    monkeypatch.setattr(
        "src.pipelines.run_training.run_bertimbau_evaluation",
        lambda df: (
            {
                "modelo": "BERTimbau",
                "accuracy": 0.9,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": 0.61,
                "roc_auc_macro": 0.91,
                "roc_auc_weighted": 0.95,
                "f1_negativo": 0.81,
                "f1_neutro": 0.08,
                "f1_positivo": 0.95,
                "modelo_principal": False,
                "insight_resumo": "Benchmark historico reaproveitado.",
                "proveniencia": "benchmark",
            },
            {"available": True, "source": "benchmark", "reason": "ok"},
        ),
    )

    run()

    summary_df = pd.read_csv(dashboard_summary)
    assert "modelo" in summary_df.columns
    assert "f1_macro" in summary_df.columns
    assert "modelo_principal" in summary_df.columns
    assert "insight_resumo" in summary_df.columns
    assert "proveniencia" in summary_df.columns
    assert len(summary_df) >= 1
    assert "BERTimbau" in summary_df["modelo"].tolist()
    bert_row = summary_df.loc[summary_df["modelo"] == "BERTimbau"].iloc[0]
    classic_row = summary_df.loc[summary_df["modelo"] == "LinearSVC"].iloc[0]
    assert "benchmark" in str(bert_row["insight_resumo"]).lower()
    assert bert_row["proveniencia"] == "benchmark"
    assert classic_row["proveniencia"] == "classic_runtime"
    assert bool(classic_row["modelo_principal"]) is True
    assert bool(bert_row["modelo_principal"]) is False
    assert "recall_macro" in str(classic_row["insight_resumo"])
    assert "roc_auc_macro" in str(classic_row["insight_resumo"])
    primary_df = pd.read_csv(dashboard_primary)
    predicted_df = pd.read_csv(dashboard_predicted)
    pd.testing.assert_frame_equal(primary_df, predicted_df)


def test_build_model_interpretation_prioritizes_runtime_primary_model():
    from dashboard.app import build_model_interpretation

    modelos_dataframe = pd.DataFrame(
        [
            {
                "modelo": "LinearSVC",
                "accuracy": 0.85,
                "precision_macro": 0.70,
                "recall_macro": 0.66,
                "f1_macro": 0.63,
                "roc_auc_macro": 0.88,
                "f1_neutro": 0.31,
                "modelo_principal": True,
                "proveniencia": "classic_runtime",
                "insight_resumo": "Modelo classico baseline.",
            },
            {
                "modelo": "BERTimbau",
                "accuracy": 0.90,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": 0.61,
                "roc_auc_macro": 0.91,
                "f1_neutro": 0.08,
                "modelo_principal": False,
                "proveniencia": "benchmark",
                "insight_resumo": "Benchmark historico.",
            },
        ]
    )

    interpretation = build_model_interpretation(modelos_dataframe)

    assert "LinearSVC" in interpretation
    assert "benchmark" in interpretation.lower()
    assert "BERTimbau" in interpretation


def test_dashboard_uses_business_first_section_labels():
    from dashboard.app import SECTION_OPTIONS

    assert SECTION_OPTIONS == (
        "Visão Geral",
        "O que está bom",
        "O que está ruim",
        "O que melhorar",
        "Tendência",
    )


def test_load_app_data_degrades_when_optional_youtube_dataset_is_missing(monkeypatch):
    from dashboard.app import load_app_data

    monkeypatch.setattr(
        "dashboard.app.load_unified_dataset",
        lambda file_mtime_ns: pd.DataFrame(
            [{"data_publicacao": "2026-03-31", "fonte": "google_play", "texto_original": "ok"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.load_model_comparison_summary",
        lambda file_mtime_ns: pd.DataFrame(
            [{"modelo": "LinearSVC", "modelo_principal": True, "proveniencia": "classic_runtime"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.load_primary_model_dataset",
        lambda file_mtime_ns: pd.DataFrame(
            [{"data_publicacao": "2026-03-31", "sentimento_previsto": "Positivo"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.load_consumidor_gov_dataset",
        lambda file_mtime_ns: pd.DataFrame(
            [{"data_publicacao": "2026-03-31", "status": "Resolvida"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.DASHBOARD_DATASET_PATHS",
        {
            "Base unificada": "base",
            "Resumo de modelos": "resumo",
            "Modelo principal": "principal",
            "YouTube + BERTimbau": "youtube",
            "Consumidor.gov": "consumidor",
        },
    )

    def fake_get_file_mtime_ns(path):
        if path == "youtube":
            raise FileNotFoundError("missing")
        return 1

    monkeypatch.setattr("dashboard.app.get_file_mtime_ns", fake_get_file_mtime_ns)

    datasets = load_app_data()

    assert "YouTube + BERTimbau" in datasets
    assert list(datasets["YouTube + BERTimbau"].columns) == [
        "data_publicacao",
        "titulo",
        "texto_original",
        "texto_limpo",
        "usuario",
        "sentimento_previsto_bert",
    ]
    assert datasets["YouTube + BERTimbau"].empty


def test_load_app_data_degrades_when_optional_consumidor_dataset_is_missing(monkeypatch):
    from dashboard.app import load_app_data

    monkeypatch.setattr(
        "dashboard.app.load_unified_dataset",
        lambda file_mtime_ns: pd.DataFrame(
            [{"data_publicacao": "2026-03-31", "fonte": "google_play", "texto_original": "ok"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.load_model_comparison_summary",
        lambda file_mtime_ns: pd.DataFrame(
            [{"modelo": "LinearSVC", "modelo_principal": True, "proveniencia": "classic_runtime"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.load_primary_model_dataset",
        lambda file_mtime_ns: pd.DataFrame(
            [{"data_publicacao": "2026-03-31", "sentimento_previsto": "Positivo"}]
        ),
    )
    monkeypatch.setattr(
        "dashboard.app.DASHBOARD_DATASET_PATHS",
        {
            "Base unificada": "base",
            "Resumo de modelos": "resumo",
            "Modelo principal": "principal",
            "YouTube + BERTimbau": "youtube",
            "Consumidor.gov": "consumidor",
        },
    )
    monkeypatch.setattr(
        "dashboard.app.load_csv",
        lambda path, file_mtime_ns: pd.DataFrame(
            [{"data_publicacao": "2026-03-31", "sentimento_previsto_bert": "Positivo"}]
        ),
    )

    def fake_get_file_mtime_ns(path):
        if path == "consumidor":
            raise FileNotFoundError("missing")
        return 1

    monkeypatch.setattr("dashboard.app.get_file_mtime_ns", fake_get_file_mtime_ns)

    datasets = load_app_data()

    assert "Consumidor.gov" in datasets
    assert list(datasets["Consumidor.gov"].columns) == [
        "data_publicacao",
        "categoria_problema",
        "texto_original",
        "status_reclamacao",
        "uf",
        "nota",
    ]
    assert datasets["Consumidor.gov"].empty


def test_dashboard_app_bootstraps_project_root_for_streamlit_import():
    app_path = "C:\\Users\\Lucca\\Documents\\analise_sentimentov2\\analise_sentimentos\\dashboard\\app.py"
    original_sys_path = list(sys.path)
    sys.modules.pop("dashboard_app_bootstrap_test", None)
    try:
        sys.path = [
            path
            for path in sys.path
            if "analise_sentimentov2\\analise_sentimentos" not in path.lower()
        ]
        spec = importlib.util.spec_from_file_location("dashboard_app_bootstrap_test", app_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path = original_sys_path
        sys.modules.pop("dashboard_app_bootstrap_test", None)

    assert hasattr(module, "get_dashboard_dataset_paths")


def test_train_sentiment_model_raises_clear_error_for_single_class_training_split():
    from src.models.train_sentiment_model import train_sentiment_model

    df = pd.DataFrame(
        [
            {"texto_limpo": "bom 1", "sentimento_real": "Positivo"},
            {"texto_limpo": "bom 2", "sentimento_real": "Positivo"},
            {"texto_limpo": "bom 3", "sentimento_real": "Positivo"},
            {"texto_limpo": "ruim 1", "sentimento_real": "Negativo"},
        ]
    )

    with pytest.raises(ValueError, match="Treino do modelo ficou com apenas uma classe apos o split."):
        train_sentiment_model(df)


def test_train_sentiment_model_returns_classic_candidate_comparison():
    from src.models.train_sentiment_model import train_sentiment_model

    rows = []
    for idx in range(15):
        rows.append({"texto_limpo": f"otimo atendimento {idx}", "sentimento_real": "Positivo"})
        rows.append({"texto_limpo": f"pessimo app {idx}", "sentimento_real": "Negativo"})
        rows.append({"texto_limpo": f"servico regular {idx}", "sentimento_real": "Neutro"})

    df = pd.DataFrame(rows)

    model, metrics, vectorizer = train_sentiment_model(df)

    assert model is not None
    assert metrics["model_name"] in {"LogisticRegression", "LinearSVC", "MultinomialNB"}
    assert isinstance(metrics["classic_candidate_rows"], list)
    assert len(metrics["classic_candidate_rows"]) == 3
    assert all("modelo" in row for row in metrics["classic_candidate_rows"])
    assert all("f1_macro" in row for row in metrics["classic_candidate_rows"])
    assert "neutral_noise_removed" in metrics
    assert vectorizer is not None or hasattr(model, "predict")


def test_train_sentiment_model_ignores_nao_rotulado_rows():
    from src.models.train_sentiment_model import train_sentiment_model

    rows = []
    for idx in range(8):
        rows.append({"texto_limpo": f"otimo atendimento {idx}", "sentimento_real": "Positivo"})
        rows.append({"texto_limpo": f"pessimo app {idx}", "sentimento_real": "Negativo"})
        rows.append({"texto_limpo": f"servico regular {idx}", "sentimento_real": "Neutro"})
    for idx in range(4):
        rows.append(
            {
                "texto_limpo": f"conta corrente cobranca indevida {idx}",
                "sentimento_real": "Não rotulado",
            }
        )

    _, metrics, _ = train_sentiment_model(pd.DataFrame(rows))

    assert metrics["model_name"] in {"LogisticRegression", "LinearSVC", "MultinomialNB"}


def test_predict_sentiment_supports_pipeline_models_without_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.models.predict_sentiment import predict_sentiment

    df = pd.DataFrame(
        [
            {"texto_limpo": "excelente app", "sentimento_real": "Positivo"},
            {"texto_limpo": "horrivel atendimento", "sentimento_real": "Negativo"},
            {"texto_limpo": "servico normal", "sentimento_real": None},
        ]
    )
    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(
        df.loc[df["sentimento_real"].notna(), "texto_limpo"],
        df.loc[df["sentimento_real"].notna(), "sentimento_real"],
    )

    predicted = predict_sentiment(df, pipeline, None)

    assert "sentimento_previsto" in predicted.columns
    assert "confianca_modelo" in predicted.columns
    assert "predicao_incerta" in predicted.columns
    assert predicted.loc[2, "sentimento_previsto"] in {"Positivo", "Negativo"}
    assert predicted.loc[0, "sentimento_previsto"] == "Positivo"


def test_predict_sentiment_adds_probability_columns_when_available():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.models.predict_sentiment import predict_sentiment

    df = pd.DataFrame(
        [
            {"texto_limpo": "excelente app", "sentimento_real": "Positivo"},
            {"texto_limpo": "horrivel atendimento", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
            {"texto_limpo": "servico normal", "sentimento_real": None},
        ]
    )
    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(
        df.loc[df["sentimento_real"].notna(), "texto_limpo"],
        df.loc[df["sentimento_real"].notna(), "sentimento_real"],
    )

    predicted = predict_sentiment(df, pipeline, None)

    assert {"score_negativo", "score_neutro", "score_positivo"}.issubset(predicted.columns)
    assert pd.notna(predicted.loc[3, "confianca_modelo"])
    assert "tema_negocio" in predicted.columns


def test_predict_sentiment_generates_prediction_for_nao_rotulado_rows():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.models.predict_sentiment import predict_sentiment

    df = pd.DataFrame(
        [
            {"texto_limpo": "excelente app", "sentimento_real": "Positivo"},
            {"texto_limpo": "horrivel atendimento", "sentimento_real": "Negativo"},
            {"texto_limpo": "cobranca indevida no cartao", "sentimento_real": "Não rotulado"},
        ]
    )
    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(
        df.loc[df["sentimento_real"].isin(["Positivo", "Negativo"]), "texto_limpo"],
        df.loc[df["sentimento_real"].isin(["Positivo", "Negativo"]), "sentimento_real"],
    )

    predicted = predict_sentiment(df, pipeline, None)

    assert predicted.loc[2, "sentimento_previsto"] in {"Positivo", "Negativo"}
    assert predicted.loc[2, "sentimento_previsto"] != "Não rotulado"
    assert predicted.loc[2, "tema_negocio"] in {"Cobrança e contestação", "Cartão e fatura", "Conta e cadastro", "Outros temas"}


def test_predict_sentiment_classifies_business_theme_from_text():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.models.predict_sentiment import predict_sentiment

    df = pd.DataFrame(
        [
            {"texto_limpo": "bloqueio de conta e acesso ao app", "sentimento_real": None},
            {"texto_limpo": "problema com pix e transferencia", "sentimento_real": None},
        ]
    )
    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(
        pd.Series(["excelente app", "horrivel atendimento"]),
        pd.Series(["Positivo", "Negativo"]),
    )

    predicted = predict_sentiment(df, pipeline, None)

    assert predicted.loc[0, "tema_negocio"] == "Conta bloqueada e acesso"
    assert predicted.loc[1, "tema_negocio"] == "Pix e transferências"


def test_build_business_insights_surfaces_top_negative_category_and_model_risk():
    from dashboard.app import build_business_insights

    unified_dataframe = pd.DataFrame(
        [
            {"fonte": "google_play", "data_publicacao": "2026-03-31", "texto_original": "bom"},
            {"fonte": "youtube", "data_publicacao": "2026-03-31", "texto_original": "ruim"},
        ]
    )
    primary_model_dataframe = pd.DataFrame(
        [
            {
                "fonte": "youtube",
                "categoria_problema": "Nubank emprestimo",
                "sentimento_previsto": "Negativo",
                "data_publicacao": "2026-03-31",
                "predicao_incerta": True,
            },
            {
                "fonte": "youtube",
                "categoria_problema": "Nubank emprestimo",
                "sentimento_previsto": "Negativo",
                "data_publicacao": "2026-03-31",
                "predicao_incerta": False,
            },
            {
                "fonte": "google_play",
                "categoria_problema": "Nubank cartao",
                "sentimento_previsto": "Positivo",
                "data_publicacao": "2026-03-31",
                "predicao_incerta": False,
            },
        ]
    )
    modelos_dataframe = pd.DataFrame(
        [
            {
                "modelo": "LogisticRegression",
                "modelo_principal": True,
                "proveniencia": "classic_runtime",
                "f1_neutro": 0.22,
                "accuracy": 0.84,
            }
        ]
    )

    insights = build_business_insights(
        unified_dataframe,
        primary_model_dataframe,
        modelos_dataframe,
    )

    assert len(insights) >= 3
    joined = " ".join(insights).lower()
    assert "emprestimo" in joined
    assert "negativ" in joined
    assert "neutro" in joined
    assert "confianca" in joined or "ambigu" in joined


def test_build_google_play_insights_surfaces_channel_specific_reading():
    from dashboard.app import build_google_play_insights

    unified_dataframe = pd.DataFrame(
        [
            {"fonte": "google_play", "data_publicacao": "2026-01-31", "texto_original": "bom"},
            {"fonte": "google_play", "data_publicacao": "2026-02-28", "texto_original": "ruim"},
            {"fonte": "youtube", "data_publicacao": "2026-02-28", "texto_original": "ok"},
        ]
    )
    primary_model_dataframe = pd.DataFrame(
        [
            {
                "fonte": "google_play",
                "sentimento_previsto": "Positivo",
                "cluster_label": "App e estabilidade",
                "cluster_signal": "Positivo",
                "cluster_keywords": "app, rapido, estavel",
                "cluster_action": "Priorizar estabilidade, velocidade e reducao de erros no app.",
                "predicao_incerta": False,
            },
            {
                "fonte": "google_play",
                "sentimento_previsto": "Negativo",
                "cluster_label": "Conta bloqueada / acesso",
                "cluster_signal": "Negativo",
                "cluster_keywords": "conta, acesso, bloqueio",
                "cluster_action": "Reduzir friccao de acesso e simplificar desbloqueio.",
                "predicao_incerta": True,
            },
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "cluster_label": "Conta bloqueada / acesso",
                "cluster_signal": "Negativo",
                "cluster_keywords": "conta, acesso, bloqueio",
                "cluster_action": "Reduzir friccao de acesso e simplificar desbloqueio.",
                "predicao_incerta": True,
            },
        ]
    )

    insights = build_google_play_insights(unified_dataframe, primary_model_dataframe)

    assert insights["total_registros"] == 2
    assert insights["participacao_base"] > 0
    assert insights["top_theme"] in {"App e estabilidade", "Conta bloqueada / acesso"}
    joined = " ".join(insights["insights"]).lower()
    assert "google play" in joined
    assert "fric" in joined or "satisf" in joined


def test_build_trend_insights_reports_direction_and_recent_pressure():
    from dashboard.app import build_trend_insights

    primary_model_dataframe = pd.DataFrame(
        [
            {"fonte": "google_play", "data_publicacao": "2026-01-31", "sentimento_previsto": "Positivo"},
            {"fonte": "google_play", "data_publicacao": "2026-01-31", "sentimento_previsto": "Negativo"},
            {"fonte": "google_play", "data_publicacao": "2026-02-28", "sentimento_previsto": "Negativo"},
            {"fonte": "google_play", "data_publicacao": "2026-03-31", "sentimento_previsto": "Negativo"},
            {"fonte": "youtube", "data_publicacao": "2026-03-31", "sentimento_previsto": "Positivo"},
            {"fonte": "youtube", "data_publicacao": "2026-04-30", "sentimento_previsto": "Negativo"},
        ]
    )

    insights = build_trend_insights(primary_model_dataframe)

    assert len(insights) >= 2
    joined = " ".join(insights).lower()
    assert "participação negativa" in joined or "média móvel" in joined
    assert "google play" in joined


def test_build_business_storyline_prioritizes_business_language():
    from dashboard.app import build_business_storyline

    primary_model_dataframe = pd.DataFrame(
        [
            {
                "fonte": "google_play",
                "categoria_problema": None,
                "sentimento_previsto": "Positivo",
                "predicao_incerta": False,
            },
            {
                "fonte": "youtube",
                "categoria_problema": "Nubank emprestimo",
                "sentimento_previsto": "Negativo",
                "predicao_incerta": True,
            },
            {
                "fonte": "youtube",
                "categoria_problema": "Nubank emprestimo",
                "sentimento_previsto": "Negativo",
                "predicao_incerta": True,
            },
        ]
    )

    story = build_business_storyline(primary_model_dataframe)

    assert "satisf" in story["headline"].lower() or "fric" in story["headline"].lower()
    assert "modelo principal" not in story["headline"].lower()
    assert len(story["business_risks"]) >= 2


def test_build_theme_priority_table_prefers_business_theme_over_raw_category():
    from dashboard.app import build_theme_priority_table

    dataframe = pd.DataFrame(
        [
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Conta bloqueada e acesso",
                "categoria_problema": "Sem informacao",
                "texto_original": "conta bloqueada e acesso ao app",
            },
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Conta bloqueada e acesso",
                "categoria_problema": "Sem informacao",
                "texto_original": "nao consigo acessar a conta",
            },
            {
                "fonte": "google_play",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Pix e transferências",
                "categoria_problema": "Sem informacao",
                "texto_original": "pix nao caiu",
            },
        ]
    )

    theme_table = build_theme_priority_table(dataframe, top_n=5)

    assert theme_table.iloc[0]["tema"] == "Conta bloqueada e acesso"
    assert "acao_sugerida" in theme_table.columns
    assert theme_table.iloc[0]["acao_sugerida"]


def test_extract_topics_clusters_full_base_with_embeddings(monkeypatch):
    from src.topics.topic_modeling import extract_topics

    dataframe = pd.DataFrame(
        [
            {"texto_limpo": "conta bloqueada acesso ao app", "texto_original": "conta bloqueada acesso ao app", "fonte": "youtube", "sentimento_previsto": "Negativo"},
            {"texto_limpo": "nao consigo entrar conta bloqueada", "texto_original": "nao consigo entrar conta bloqueada", "fonte": "youtube", "sentimento_previsto": "Negativo"},
            {"texto_limpo": "app rapido e estavel", "texto_original": "app rapido e estavel", "fonte": "google_play", "sentimento_previsto": "Positivo"},
            {"texto_limpo": "funciona bem e sem erro", "texto_original": "funciona bem e sem erro", "fonte": "google_play", "sentimento_previsto": "Positivo"},
        ]
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    monkeypatch.setattr(
        "src.topics.topic_modeling.build_text_embeddings",
        lambda texts, **kwargs: embeddings[: len(texts)],
    )

    clustered_df, topic_summary = extract_topics(dataframe, n_clusters=2)

    assert {"cluster_id", "cluster_label", "cluster_signal", "topico"}.issubset(clustered_df.columns)
    assert len(topic_summary) == 2
    assert set(topic_summary["cluster_signal"]) == {"Positivo", "Negativo"}
    assert topic_summary["palavras_chave"].astype(str).str.len().min() > 0


def test_build_theme_priority_table_uses_cluster_signals_when_available():
    from dashboard.app import build_theme_priority_table

    dataframe = pd.DataFrame(
        [
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "cluster_label": "Conta bloqueada / acesso",
                "cluster_signal": "Negativo",
                "cluster_keywords": "conta bloqueada, acesso, login",
                "texto_original": "conta bloqueada e acesso ao app",
            },
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "cluster_label": "Conta bloqueada / acesso",
                "cluster_signal": "Negativo",
                "cluster_keywords": "conta bloqueada, acesso, login",
                "texto_original": "nao consigo acessar a conta",
            },
            {
                "fonte": "google_play",
                "sentimento_previsto": "Positivo",
                "cluster_label": "App e estabilidade",
                "cluster_signal": "Positivo",
                "cluster_keywords": "app, estavel, rapido",
                "texto_original": "app rapido e estavel",
            },
        ]
    )

    positive_table = build_theme_priority_table(
        dataframe,
        top_n=5,
        sentiment_value="Positivo",
    )
    negative_table = build_theme_priority_table(
        dataframe,
        top_n=5,
        sentiment_value="Negativo",
    )

    assert positive_table.iloc[0]["tema"] == "App e estabilidade"
    assert negative_table.iloc[0]["tema"] == "Conta bloqueada / acesso"
    assert positive_table.iloc[0]["sentimento_dominante"] == "Positivo"
    assert negative_table.iloc[0]["sentimento_dominante"] == "Negativo"


def test_build_theme_priority_table_excludes_mixed_clusters_from_positive_and_negative_sections():
    from dashboard.app import build_theme_priority_table

    dataframe = pd.DataFrame(
        [
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "cluster_label": "Emprestimo e credito",
                "cluster_signal": "Misto",
                "cluster_keywords": "emprestimo, credito, limite",
                "texto_original": "emprestimo com limite baixo",
            },
            {
                "fonte": "google_play",
                "sentimento_previsto": "Positivo",
                "cluster_label": "Emprestimo e credito",
                "cluster_signal": "Misto",
                "cluster_keywords": "emprestimo, credito, limite",
                "texto_original": "gostei do emprestimo",
            },
        ]
    )

    positive_table = build_theme_priority_table(dataframe, top_n=5, sentiment_value="Positivo")
    negative_table = build_theme_priority_table(dataframe, top_n=5, sentiment_value="Negativo")
    mixed_table = build_theme_priority_table(dataframe, top_n=5, sentiment_value="Misto")

    assert positive_table.empty
    assert negative_table.empty
    assert not mixed_table.empty
    assert mixed_table.iloc[0]["tema"] == "Emprestimo e credito"


def test_build_audience_insights_emphasizes_business_audiences():
    from dashboard.app import build_audience_insights

    unified_dataframe = pd.DataFrame(
        [
            {"fonte": "youtube", "data_publicacao": "2026-03-31", "texto_original": "ok"},
            {"fonte": "google_play", "data_publicacao": "2026-03-31", "texto_original": "ok"},
        ]
    )
    primary_model_dataframe = pd.DataFrame(
        [
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Conta bloqueada e acesso",
                "predicao_incerta": True,
            },
            {
                "fonte": "google_play",
                "sentimento_previsto": "Positivo",
                "tema_negocio": "App e estabilidade",
                "predicao_incerta": False,
            },
        ]
    )

    insights = build_audience_insights(unified_dataframe, primary_model_dataframe)

    assert len(insights["investidor"]) >= 1
    assert len(insights["usuario"]) >= 1
    assert len(insights["interno"]) >= 1
    joined = " ".join(insights["interno"]).lower()
    assert "acesso" in joined or "conta bloqueada" in joined


def test_build_audience_insights_emphasizes_source_risk_over_positive_share():
    from dashboard.app import build_audience_insights

    unified_dataframe = pd.DataFrame(
        [
            {"fonte": "google_play", "data_publicacao": "2026-03-31", "texto_original": "ok"},
            {"fonte": "youtube", "data_publicacao": "2026-03-31", "texto_original": "ok"},
            {"fonte": "consumidor_gov", "data_publicacao": "2026-03-31", "texto_original": "ok"},
        ]
    )
    primary_model_dataframe = pd.DataFrame(
        [
            {
                "fonte": "google_play",
                "sentimento_previsto": "Positivo",
                "tema_negocio": "App e estabilidade",
                "predicao_incerta": False,
            },
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Conta bloqueada e acesso",
                "predicao_incerta": True,
            },
            {
                "fonte": "consumidor_gov",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Cartão e fatura",
                "predicao_incerta": False,
            },
        ]
    )

    insights = build_audience_insights(unified_dataframe, primary_model_dataframe)

    assert any("risco" in item.lower() or "negativo" in item.lower() for item in insights["investidor"])
    assert any("usuario" in item.lower() or "cliente" in item.lower() for item in insights["usuario"])
    assert any("acao" in item.lower() or "prioridade" in item.lower() for item in insights["interno"])


def test_build_business_storyline_prioritizes_channels_and_themes():
    from dashboard.app import build_business_storyline

    primary_model_dataframe = pd.DataFrame(
        [
            {
                "fonte": "google_play",
                "sentimento_previsto": "Positivo",
                "tema_negocio": "App e estabilidade",
                "predicao_incerta": False,
            },
            {
                "fonte": "youtube",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Conta bloqueada e acesso",
                "predicao_incerta": True,
            },
            {
                "fonte": "consumidor_gov",
                "sentimento_previsto": "Negativo",
                "tema_negocio": "Cartão e fatura",
                "predicao_incerta": False,
            },
        ]
    )

    story = build_business_storyline(primary_model_dataframe)

    assert "canal" in story["headline"].lower()
    assert "tema" in story["headline"].lower() or "friccao" in story["headline"].lower()
    assert any("prioridade" in item.lower() for item in story["business_risks"])


def test_build_audience_insights_supports_bert_full_dataset_columns():
    from dashboard.app import build_audience_insights

    unified_dataframe = pd.DataFrame(
        [
            {"fonte": "youtube", "data_publicacao": "2026-03-31", "texto_original": "ok"},
            {"fonte": "google_play", "data_publicacao": "2026-03-31", "texto_original": "ok"},
        ]
    )
    bert_dataframe = pd.DataFrame(
        [
            {
                "fonte": "youtube",
                "sentimento_previsto_bert": "Negativo",
                "tema_negocio": "Conta bloqueada e acesso",
                "predicao_incerta_bert": True,
            },
            {
                "fonte": "google_play",
                "sentimento_previsto_bert": "Positivo",
                "tema_negocio": "App e estabilidade",
                "predicao_incerta_bert": False,
            },
        ]
    )

    insights = build_audience_insights(
        unified_dataframe,
        bert_dataframe,
        sentiment_column="sentimento_previsto_bert",
    )

    assert len(insights["investidor"]) >= 1
    assert len(insights["usuario"]) >= 1
    assert len(insights["interno"]) >= 1
    assert any("baixa confianca" in item.lower() for item in insights["investidor"])


def test_build_comment_examples_handles_missing_youtube_columns():
    from dashboard.app import build_comment_examples

    youtube_dataframe = pd.DataFrame(
        [
            {
                "sentimento_previsto_bert": "Positivo",
                "texto_original": "gostei muito do app",
                "data_publicacao": pd.Timestamp("2026-03-31"),
                "texto_limpo": "gostei muito app",
            },
            {
                "sentimento_previsto_bert": "Negativo",
                "texto_original": "app travando",
                "data_publicacao": pd.Timestamp("2026-03-31"),
                "texto_limpo": "app travando",
            },
        ]
    )

    examples = build_comment_examples(youtube_dataframe, examples_per_sentiment=1)

    assert not examples.empty
    assert "usuario" in examples.columns
    assert "titulo" in examples.columns
    assert set(examples["sentimento"]) <= {"Positivo", "Negativo", "Neutro"}


def test_process_consumidor_gov_generates_silver_contract_and_clean_text(
    tmp_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    input_dir = raw_dir / "consumidor_gov"
    input_dir.mkdir(parents=True)
    source_file = input_dir / "consumidor_2026_03.csv"
    pd.DataFrame(
        [
            {
                "Nome Fantasia": "Nubank",
                "Data Abertura": "2026-03-04",
                "Nota do Consumidor": "2",
                "Grupo Problema": "Cobrança / Contestação",
                "Situação": "Finalizada avaliada",
                "UF": "SP",
                "Assunto": "Cartão de crédito",
                "Problema": "Cobrança indevida no aplicativo",
                "Cidade": "São Paulo",
            },
            {
                "Nome Fantasia": "Outra Empresa",
                "Data Abertura": "2026-03-05",
                "Nota do Consumidor": "5",
                "Grupo Problema": "Atendimento / SAC",
                "Situação": "Resolvida",
                "UF": "RJ",
                "Assunto": "Conta",
                "Problema": "Sem acesso",
                "Cidade": "Rio de Janeiro",
            },
        ]
    ).to_csv(source_file, sep=";", index=False, encoding="utf-8")

    monkeypatch.setattr("src.preprocessing.process_consumidor_gov.RAW_DIR", raw_dir)

    process_consumidor_gov()

    output_path = input_dir / "consumidor_gov_processed.csv"
    result = pd.read_csv(output_path)

    assert list(result.columns) == [
        "id_registro",
        "fonte",
        "data_publicacao",
        "texto_original",
        "texto_limpo",
        "nota",
        "status_reclamacao",
        "categoria_problema",
        "uf",
        "versao_app",
        "sentimento_real",
        "data_processamento",
        "assunto",
        "problema_detalhado",
        "cidade",
        "empresa",
        "arquivo_origem",
    ]
    assert len(result) == 1
    assert result.loc[0, "fonte"] == "consumidor_gov"
    assert result.loc[0, "status_reclamacao"] == "Finalizada avaliada"
    assert result.loc[0, "categoria_problema"] == "Cobrança / Contestação"
    assert result.loc[0, "uf"] == "SP"
    assert result.loc[0, "versao_app"] == "Não aplicável"
    assert "Cartão de crédito" in result.loc[0, "texto_original"]
    assert "Cobrança indevida no aplicativo" in result.loc[0, "texto_original"]
    assert "cobrança indevida no aplicativo" in result.loc[0, "texto_limpo"]


def test_build_unified_bronze_keeps_consumidor_gov_textual_records(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    gov_dir = raw_dir / "consumidor_gov"
    gov_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id_registro": "cg_1",
                "fonte": "consumidor_gov",
                "data_publicacao": "2026-03-04",
                "texto_original": "Assunto: Cartão. Problema: cobrança indevida.",
                "texto_limpo": "assunto cartao problema cobranca indevida",
                "nota": 1,
                "status_reclamacao": "Finalizada avaliada",
                "categoria_problema": "Cobrança / Contestação",
                "uf": "SP",
                "versao_app": "Não aplicável",
                "sentimento_real": None,
                "data_processamento": "2026-05-26 10:00:00",
            }
        ]
    ).to_csv(gov_dir / "consumidor_gov_processed.csv", index=False)

    monkeypatch.setattr("src.pipelines.run_preprocessing.RAW_DIR", raw_dir)
    monkeypatch.setattr("src.pipelines.run_preprocessing.PROJECT_ROOT", tmp_path)

    result = build_unified_bronze()

    assert len(result) == 1
    assert result.loc[0, "fonte"] == "consumidor_gov"
    assert "cobrança indevida" in result.loc[0, "texto_original"].lower()


def test_build_unified_bronze_backfills_missing_sources_from_legacy_dataset(
    tmp_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    gov_dir = raw_dir / "consumidor_gov"
    gov_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id_registro": "cg_1",
                "fonte": "consumidor_gov",
                "data_publicacao": "2026-03-04",
                "texto_original": "Assunto: Cartão. Problema: cobrança indevida.",
                "nota": 1,
                "status_reclamacao": "Finalizada avaliada",
                "categoria_problema": "Cobrança / Contestação",
                "uf": "SP",
                "versao_app": "Não aplicável",
                "sentimento_real": None,
            }
        ]
    ).to_csv(gov_dir / "consumidor_gov_processed.csv", index=False)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "id_registro": "gp_1",
                "fonte": "google_play",
                "data_publicacao": "2026-03-04",
                "texto_original": "App muito bom",
                "nota": 5,
                "categoria": None,
                "status": None,
                "UF": None,
                "sentimento_real": "Positivo",
            },
            {
                "id_registro": "yt_1",
                "fonte": "youtube",
                "data_publicacao": "2026-03-04",
                "texto_original": "Video esclarecedor",
                "nota": None,
                "categoria": "Nubank atendimento",
                "status": None,
                "UF": None,
                "sentimento_real": "Positivo",
            },
        ]
    ).to_csv(processed_dir / "unified_dataset.csv", index=False)

    monkeypatch.setattr("src.pipelines.run_preprocessing.RAW_DIR", raw_dir)
    monkeypatch.setattr("src.pipelines.run_preprocessing.PROJECT_ROOT", tmp_path)

    result = build_unified_bronze()

    assert set(result["fonte"]) == {"consumidor_gov", "google_play", "youtube"}


def test_process_consumidor_gov_falls_back_to_sql_when_raw_files_are_missing(
    tmp_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    input_dir = raw_dir / "consumidor_gov"
    input_dir.mkdir(parents=True)

    sql_rows = pd.DataFrame(
        [
            {
                "id_registro": "cg_100",
                "fonte": "consumidor_gov",
                "data_publicacao": "2026-03-04",
                "texto_original": "Assunto: Cartão | Problema: cobrança indevida | Categoria: Cobrança / Contestação",
                "texto_limpo": "assunto cartão problema cobrança indevida categoria cobrança contestação",
                "nota": 1,
                "status_reclamacao": "Não Avaliada",
                "categoria_problema": "Cobrança / Contestação",
                "uf": "SP",
                "versao_app": "Não aplicável",
                "sentimento_real": None,
                "data_processamento": "2026-05-26 10:00:00",
            }
        ]
    )

    monkeypatch.setattr("src.preprocessing.process_consumidor_gov.RAW_DIR", raw_dir)
    monkeypatch.setattr(
        "src.preprocessing.process_consumidor_gov._load_consumidor_gov_from_sql",
        lambda: sql_rows,
    )

    process_consumidor_gov()

    output_path = input_dir / "consumidor_gov_processed.csv"
    result = pd.read_csv(output_path)

    assert len(result) == 1
    assert result.loc[0, "id_registro"] == "cg_100"
    assert result.loc[0, "fonte"] == "consumidor_gov"
