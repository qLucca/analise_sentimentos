# Model Validation And Streamlit Insights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validar os modelos classicos e o BERTimbau em um fluxo unico, gerar artefatos `gold/dashboard` consistentes e expor insights analiticos no Streamlit.

**Architecture:** A implementacao centraliza a avaliacao de modelos em `src/models/`, separa o resumo comparativo da previsao final do modelo principal e faz o dashboard consumir esses contratos explicitamente. O fluxo passa a gerar um `model_comparison_summary.csv` rico, um dataset previsto do modelo principal e insights textuais e visuais derivados de metricas e topicos reais.

**Tech Stack:** Python 3.11, pandas, scikit-learn, transformers, Streamlit, pytest, SQL Server

---

### Task 1: Definir o contrato do resumo de modelos

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`
- Create: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\models\model_summary.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_model_summary_row_contains_required_fields():
    from src.models.model_summary import build_model_summary_row

    row = build_model_summary_row(
        model_name="LogisticRegression",
        metrics={
            "accuracy": 0.8,
            "precision_macro": 0.7,
            "recall_macro": 0.6,
            "f1_macro": 0.65,
            "roc_auc_macro": 0.81,
            "roc_auc_weighted": 0.9,
            "f1_negativo": 0.62,
            "f1_neutro": 0.31,
            "f1_positivo": 0.88,
        },
        is_primary=False,
        insight_summary="baseline estavel",
    )

    assert row["modelo"] == "LogisticRegression"
    assert row["modelo_principal"] is False
    assert row["insight_resumo"] == "baseline estavel"
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
        }
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_build_model_summary_row_contains_required_fields -v`
Expected: `FAIL` with `ModuleNotFoundError` or missing function error for `src.models.model_summary`

- [ ] **Step 3: Write minimal implementation**

```python
def build_model_summary_row(
    model_name: str,
    metrics: dict,
    is_primary: bool,
    insight_summary: str,
) -> dict:
    return {
        "modelo": model_name,
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "roc_auc_macro": metrics.get("roc_auc_macro"),
        "roc_auc_weighted": metrics.get("roc_auc_weighted"),
        "f1_negativo": metrics.get("f1_negativo"),
        "f1_neutro": metrics.get("f1_neutro"),
        "f1_positivo": metrics.get("f1_positivo"),
        "modelo_principal": is_primary,
        "insight_resumo": insight_summary,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_build_model_summary_row_contains_required_fields -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py src/models/model_summary.py
git commit -m "feat: define model summary contract"
```

### Task 2: Escolher o modelo principal com criterio unico

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\models\model_summary.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_choose_primary_model_prefers_roc_auc_then_f1_then_accuracy():
    from src.models.model_summary import choose_primary_model

    rows = [
        {"modelo": "A", "roc_auc_macro": 0.91, "f1_macro": 0.60, "accuracy": 0.88},
        {"modelo": "B", "roc_auc_macro": 0.91, "f1_macro": 0.62, "accuracy": 0.87},
        {"modelo": "C", "roc_auc_macro": 0.90, "f1_macro": 0.80, "accuracy": 0.95},
    ]

    winner = choose_primary_model(rows)

    assert winner["modelo"] == "B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_choose_primary_model_prefers_roc_auc_then_f1_then_accuracy -v`
Expected: `FAIL` with missing `choose_primary_model`

- [ ] **Step 3: Write minimal implementation**

```python
def choose_primary_model(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            row.get("roc_auc_macro", float("-inf")),
            row.get("f1_macro", float("-inf")),
            row.get("accuracy", float("-inf")),
        ),
        reverse=True,
    )[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_choose_primary_model_prefers_roc_auc_then_f1_then_accuracy -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py src/models/model_summary.py
git commit -m "feat: centralize primary model selection"
```

### Task 3: Gerar comparacao classica reproduzivel

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\models\train_sentiment_model.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\pipelines\run_training.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_training_writes_model_comparison_summary_with_classic_metrics(tmp_path, monkeypatch):
    from src.pipelines.run_training import run

    silver_path = tmp_path / "silver.csv"
    dashboard_summary = tmp_path / "model_comparison_summary.csv"
    dashboard_predicted = tmp_path / "predicted.csv"
    pd.DataFrame(
        [
            {"texto_limpo": "muito bom", "sentimento_real": "Positivo"},
            {"texto_limpo": "muito ruim", "sentimento_real": "Negativo"},
            {"texto_limpo": "mais ou menos", "sentimento_real": "Neutro"},
            {"texto_limpo": "otimo atendimento", "sentimento_real": "Positivo"},
            {"texto_limpo": "app travando", "sentimento_real": "Negativo"},
            {"texto_limpo": "servico ok", "sentimento_real": "Neutro"},
        ]
    ).to_csv(silver_path, index=False)

    monkeypatch.setattr("src.pipelines.run_training.SILVER_UNIFIED_DATASET_PATH", silver_path)
    monkeypatch.setattr("src.pipelines.run_training.GOLD_DASHBOARD_MODEL_COMPARISON_PATH", dashboard_summary)
    monkeypatch.setattr("src.pipelines.run_training.GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH", dashboard_predicted)

    run()

    summary_df = pd.read_csv(dashboard_summary)
    assert "modelo" in summary_df.columns
    assert "f1_macro" in summary_df.columns
    assert len(summary_df) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_run_training_writes_model_comparison_summary_with_classic_metrics -v`
Expected: `FAIL` because the summary does not yet contain the comparative contract

- [ ] **Step 3: Write minimal implementation**

```python
def run() -> tuple[pd.DataFrame, dict]:
    setup_logging()
    df = pd.read_csv(SILVER_UNIFIED_DATASET_PATH)
    model, metrics, vectorizer = train_sentiment_model(df)
    metrics.setdefault("id_execucao", str(uuid4()))
    predicted = predict_sentiment(df, model, vectorizer)
    predicted.to_csv(GOLD_DASHBOARD_CLASSIC_SENTIMENT_PATH, index=False)

    row = build_model_summary_row(
        model_name="LogisticRegression",
        metrics=metrics,
        is_primary=True,
        insight_summary="Modelo classico baseline com pipeline reproduzivel.",
    )
    pd.DataFrame([row]).to_csv(GOLD_DASHBOARD_MODEL_COMPARISON_PATH, index=False)
    return predicted, metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_run_training_writes_model_comparison_summary_with_classic_metrics -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py src/models/train_sentiment_model.py src/pipelines/run_training.py
git commit -m "feat: write reproducible classic model summary"
```

### Task 4: Reativar o BERTimbau como comparador oficial

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`
- Create: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\models\bertimbau_runner.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\pipelines\run_training.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_merge_model_summaries_marks_only_the_selected_primary_model():
    from src.models.model_summary import merge_model_summaries

    classic_rows = [
        {"modelo": "LogisticRegression", "roc_auc_macro": 0.88, "f1_macro": 0.63, "accuracy": 0.85}
    ]
    bert_row = {"modelo": "BERTimbau", "roc_auc_macro": 0.91, "f1_macro": 0.61, "accuracy": 0.90}

    merged = merge_model_summaries(classic_rows, bert_row)

    winners = [row["modelo"] for row in merged if row["modelo_principal"]]
    assert winners == ["BERTimbau"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_merge_model_summaries_marks_only_the_selected_primary_model -v`
Expected: `FAIL` because merge logic does not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
def merge_model_summaries(classic_rows: list[dict], bert_row: dict | None) -> list[dict]:
    rows = [*classic_rows]
    if bert_row:
        rows.append(bert_row)
    winner = choose_primary_model(rows)
    for row in rows:
        row["modelo_principal"] = row["modelo"] == winner["modelo"]
    return rows
```

Also create a `run_bertimbau_evaluation(...)` function that returns either a summary row plus prediction path or `None` with a clear explanation when dependencies or weights are unavailable.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_merge_model_summaries_marks_only_the_selected_primary_model -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py src/models/model_summary.py src/models/bertimbau_runner.py src/pipelines/run_training.py
git commit -m "feat: add bertimbau comparison flow"
```

### Task 5: Publicar o dataset previsto do modelo principal

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\utils\paths.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\pipelines\run_training.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_get_dashboard_dataset_paths_exposes_primary_model_dataset():
    from src.utils.paths import get_dashboard_dataset_paths

    paths = get_dashboard_dataset_paths()

    assert "Modelo principal" in paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_get_dashboard_dataset_paths_exposes_primary_model_dataset -v`
Expected: `FAIL` because the path key does not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH = GOLD_DASHBOARD_DIR / "primary_model_predictions.csv"

def get_dashboard_dataset_paths() -> dict[str, Path]:
    return {
        "Base unificada": GOLD_DASHBOARD_UNIFIED_DATASET_PATH,
        "Resumo de modelos": GOLD_DASHBOARD_MODEL_COMPARISON_PATH,
        "YouTube + BERTimbau": GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH,
        "Consumidor.gov": RAW_CONSUMIDOR_GOV_PROCESSED_PATH,
        "Modelo principal": GOLD_DASHBOARD_PRIMARY_MODEL_DATASET_PATH,
    }
```

Update training to write the chosen prediction dataset to `primary_model_predictions.csv`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_get_dashboard_dataset_paths_exposes_primary_model_dataset -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py src/utils/paths.py src/pipelines/run_training.py
git commit -m "feat: publish primary model dataset for dashboard"
```

### Task 6: Exibir insights de modelos e topicos no Streamlit

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\dashboard\app.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_model_interpretation_prefers_primary_model_flag():
    from dashboard.app import build_model_interpretation

    df = pd.DataFrame(
        [
            {
                "modelo": "LogisticRegression",
                "accuracy": 0.85,
                "roc_auc_macro": 0.88,
                "f1_macro": 0.63,
                "f1_neutro": 0.20,
                "modelo_principal": False,
                "insight_resumo": "baseline",
            },
            {
                "modelo": "BERTimbau",
                "accuracy": 0.90,
                "roc_auc_macro": 0.91,
                "f1_macro": 0.61,
                "f1_neutro": 0.08,
                "modelo_principal": True,
                "insight_resumo": "melhor discriminacao global",
            },
        ]
    )

    interpretation = build_model_interpretation(df)

    assert "BERTimbau" in interpretation
    assert "melhor discriminacao global" in interpretation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_build_model_interpretation_prefers_primary_model_flag -v`
Expected: `FAIL` because the interpretation does not yet use `modelo_principal` and `insight_resumo`

- [ ] **Step 3: Write minimal implementation**

```python
def build_model_interpretation(modelos_dataframe: pd.DataFrame) -> str:
    if modelos_dataframe.empty:
        return "Ainda nao ha comparacao de modelos disponivel para interpretacao."

    primary_rows = modelos_dataframe.loc[modelos_dataframe["modelo_principal"] == True]
    primary_row = primary_rows.iloc[0] if not primary_rows.empty else modelos_dataframe.sort_values(
        ["accuracy", "roc_auc_macro", "f1_macro"], ascending=False
    ).iloc[0]

    summary = str(primary_row.get("insight_resumo", "")).strip()
    if summary:
        return f"Modelo principal: {primary_row['modelo']}. {summary}"
    return f"Modelo principal: {primary_row['modelo']}."
```

Also update the Streamlit "Modelos" section to:

- display the primary model as a headline card
- show the comparative table
- show top negative topics and keywords using the `gold` outputs
- preview the primary model dataset instead of a hard-coded BERT-only view

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_build_model_interpretation_prefers_primary_model_flag -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py dashboard/app.py
git commit -m "feat: add model insights to streamlit"
```

### Task 7: Validar o pipeline end-to-end e o dashboard

**Files:**
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\pipelines\run_topics.py`
- Modify: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\src\pipelines\run_load_sqlserver.py`
- Test: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_load_sqlserver_reads_model_summary_and_primary_dataset(tmp_path, monkeypatch):
    from src.pipelines.run_load_sqlserver import run

    summary_path = tmp_path / "model_comparison_summary.csv"
    primary_path = tmp_path / "primary_model_predictions.csv"
    silver_path = tmp_path / "silver.csv"
    sentiment_path = tmp_path / "gold_sentiment_analysis.csv"
    topics_path = tmp_path / "gold_topic_analysis.csv"

    pd.DataFrame([{"modelo": "LogisticRegression", "modelo_principal": True}]).to_csv(summary_path, index=False)
    pd.DataFrame([{"id_registro": "1"}]).to_csv(primary_path, index=False)
    pd.DataFrame([{"id_registro": "1"}]).to_csv(silver_path, index=False)
    pd.DataFrame([{"id_registro": "1"}]).to_csv(sentiment_path, index=False)
    pd.DataFrame([{"id_topico": 1}]).to_csv(topics_path, index=False)

    calls = []
    monkeypatch.setattr("src.pipelines.run_load_sqlserver.SILVER_UNIFIED_DATASET_PATH", silver_path)
    monkeypatch.setattr("src.pipelines.run_load_sqlserver.GOLD_ANALYTICS_SENTIMENT_PATH", sentiment_path)
    monkeypatch.setattr("src.pipelines.run_load_sqlserver.GOLD_TOPICS_ANALYSIS_PATH", topics_path)
    monkeypatch.setattr("src.pipelines.run_load_sqlserver.GOLD_DASHBOARD_MODEL_COMPARISON_PATH", summary_path)
    monkeypatch.setattr("src.database.load_gold_to_sqlserver.load_gold_dataframe", lambda df, table_name, schema='gold', if_exists='append': calls.append(table_name))
    monkeypatch.setattr("src.database.load_silver_to_sqlserver.load_silver_reviews", lambda df, table_name='reviews_cleaned', schema='silver': calls.append(table_name))

    run()

    assert "reviews_cleaned" in calls
    assert "sentiment_analysis" in calls
    assert "topic_analysis" in calls
    assert "model_metrics" in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_run_load_sqlserver_reads_model_summary_and_primary_dataset -v`
Expected: `FAIL` due to incomplete integration expectations

- [ ] **Step 3: Write minimal implementation**

```python
def run() -> None:
    setup_logging()
    silver = pd.read_csv(SILVER_UNIFIED_DATASET_PATH)
    load_silver_reviews(silver)

    sentiment = pd.read_csv(GOLD_ANALYTICS_SENTIMENT_PATH)
    topics = pd.read_csv(GOLD_TOPICS_ANALYSIS_PATH)
    load_gold_dataframe(sentiment, "sentiment_analysis")
    load_gold_dataframe(topics, "topic_analysis")

    if GOLD_DASHBOARD_MODEL_COMPARISON_PATH.exists():
        metrics = pd.read_csv(GOLD_DASHBOARD_MODEL_COMPARISON_PATH)
        load_gold_dataframe(metrics, "model_metrics")
```

Adjust only if the failing test exposes a real integration gap, such as missing primary dataset synchronization or contract mismatches.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_run_load_sqlserver_reads_model_summary_and_primary_dataset -v`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_regressions.py src/pipelines/run_topics.py src/pipelines/run_load_sqlserver.py
git commit -m "feat: align end-to-end dashboard artifacts and sql load"
```

### Task 8: Verificacao final

**Files:**
- Verify only: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\tests\`
- Verify only: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\dashboard\app.py`
- Verify only: `C:\Users\Lucca\Documents\analise_sentimentov2\analise_sentimentos\data\gold\dashboard\`

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests -v`
Expected: all tests pass with `0 failed`

- [ ] **Step 2: Run the training and topics pipeline**

Run: `python -m src.pipelines.run_training`
Expected: files `data/gold/dashboard/model_comparison_summary.csv` and `data/gold/dashboard/primary_model_predictions.csv` are written

Run: `python -m src.pipelines.run_topics`
Expected: files `data/gold/analytics/gold_sentiment_analysis.csv` and `data/gold/topics/gold_topic_analysis.csv` are written

- [ ] **Step 3: Launch the dashboard for manual validation**

Run: `streamlit run dashboard/app.py`
Expected: dashboard opens with updated "Modelos" insights, comparative metrics and topic-based interpretation

- [ ] **Step 4: Load SQL Server and verify counts**

Run: `python -m src.pipelines.run_load_sqlserver`
Expected: `silver.reviews_cleaned`, `gold.sentiment_analysis`, `gold.topic_analysis` and `gold.model_metrics` receive rows without schema errors

- [ ] **Step 5: Commit**

```bash
git add src/models src/pipelines src/utils dashboard/app.py tests
git commit -m "feat: validate models and publish streamlit insights"
```
