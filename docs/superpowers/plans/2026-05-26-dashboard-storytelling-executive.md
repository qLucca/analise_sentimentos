# Dashboard Storytelling Executivo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganizar a home do Streamlit para contar uma história executiva baseada nos dados, destacando risco, impacto por canal e recomendações acionáveis para investidor, usuário e time interno.

**Architecture:** Vamos manter a base de dados e os artefatos atuais, mas mudar a camada de apresentação no `dashboard/app.py` para priorizar leitura de negócio em vez de leitura técnica. A home vai seguir uma sequência fixa: panorama do negócio, risco por canal, temas críticos e recomendações por público. Os helpers de insight vão ser ajustados para privilegiar contraste entre fontes, e a seção técnica de modelos vai ficar como apoio secundário.

**Tech Stack:** Python, Streamlit, pandas, matplotlib, seaborn, pytest.

---

### Task 1: Consolidar a narrativa executiva da home

**Files:**
- Modify: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/dashboard/app.py:1880-2185`
- Test: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/tests/test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_modelos_section_prioritizes_business_story_over_model_table():
    from dashboard.app import build_business_storyline

    primary_model_dataframe = pd.DataFrame(
        [
            {"fonte": "google_play", "sentimento_previsto": "Positivo", "tema_negocio": "App e estabilidade"},
            {"fonte": "youtube", "sentimento_previsto": "Negativo", "tema_negocio": "Conta bloqueada e acesso"},
            {"fonte": "consumidor_gov", "sentimento_previsto": "Negativo", "tema_negocio": "Cartão e fatura"},
        ]
    )

    story = build_business_storyline(primary_model_dataframe)

    assert "canal" in story["headline"].lower()
    assert "tema" in story["headline"].lower() or "friccao" in story["headline"].lower()
    assert any("prioridade" in item.lower() for item in story["business_risks"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_render_modelos_section_prioritizes_business_story_over_model_table -v`
Expected: FAIL if the current storytelling still reads like a model summary instead of a business narrative.

- [ ] **Step 3: Write minimal implementation**

```python
def build_business_storyline(primary_model_dataframe: pd.DataFrame, sentiment_column: str = "sentimento_previsto") -> dict[str, object]:
    ...
    return {
        "headline": (
            f"A conversa mostra sinal favoravel no agregado, mas o risco se concentra em {top_theme} "
            f"e no canal {top_negative_source}."
        ),
        "subheadline": (
            "A leitura abaixo prioriza risco, impacto no cliente e prioridade operacional por canal."
        ),
        "business_risks": [
            f"O canal com maior concentracao de sinais negativos e {top_negative_source}, pedindo acao imediata.",
            f"O tema mais sensivel continua sendo {top_theme}, com volume relevante de friccao.",
            f"{uncertain_share:.1%} das previsoes ficaram em baixa confianca, o que pede leitura cautelosa.",
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_render_modelos_section_prioritizes_business_story_over_model_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py tests/test_pipeline_regressions.py
git commit -m "feat: tighten executive storytelling on dashboard home"
```

### Task 2: Rebalance audience insights for investidor, usuário e interno

**Files:**
- Modify: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/dashboard/app.py:942-1245`
- Test: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/tests/test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
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
            {"fonte": "google_play", "sentimento_previsto": "Positivo", "tema_negocio": "App e estabilidade", "predicao_incerta": False},
            {"fonte": "youtube", "sentimento_previsto": "Negativo", "tema_negocio": "Conta bloqueada e acesso", "predicao_incerta": True},
            {"fonte": "consumidor_gov", "sentimento_previsto": "Negativo", "tema_negocio": "Cartão e fatura", "predicao_incerta": False},
        ]
    )

    insights = build_audience_insights(unified_dataframe, primary_model_dataframe)

    assert any("risco" in item.lower() or "negativo" in item.lower() for item in insights["investidor"])
    assert any("usuario" in item.lower() or "cliente" in item.lower() for item in insights["usuario"])
    assert any("acao" in item.lower() or "prioridade" in item.lower() for item in insights["interno"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_build_audience_insights_emphasizes_source_risk_over_positive_share -v`
Expected: FAIL until the insight text is tuned to business language.

- [ ] **Step 3: Write minimal implementation**

```python
def build_audience_insights(...):
    ...
    investor = [
        f"O sinal positivo no agregado existe, mas o risco real se concentra em {top_theme['tema']} e no canal {dominant_source}.",
        f"{negative_share:.1%} da base ainda aparece como negativa, o que merece leitura de reputacao e churn.",
        f"{uncertain_share:.1%} das previsoes estao em baixa confianca, entao o agregado nao deve ser lido de forma absoluta.",
    ]
    user = [
        f"Para o cliente, o problema mais visivel e {top_theme['tema']}, que tende a afetar uso diario e percepcao de confianca.",
        f"O canal mais recorrente no volume e {dominant_source}, o que ajuda a entender onde a experiencia aparece com mais força.",
    ]
    internal = [
        f"Prioridade operacional: {top_theme['acao_sugerida']}",
        f"Focar primeiro no canal {dominant_source} porque ele concentra a maior parte do sinal negativo.",
        f"Revisar manualmente amostras em baixa confianca para validar se ha ruído de classificação ou problema real.",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_build_audience_insights_emphasizes_source_risk_over_positive_share -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py tests/test_pipeline_regressions.py
git commit -m "feat: sharpen audience insights in dashboard"
```

### Task 3: Remove technical emphasis from the home and surface business actions

**Files:**
- Modify: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/dashboard/app.py:1475-2185`
- Test: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/tests/test_pipeline_regressions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_modelos_section_surfaces_business_actions_before_technical_table():
    from dashboard.app import build_business_insights

    primary_model_dataframe = pd.DataFrame(
        [
            {"fonte": "google_play", "sentimento_previsto": "Positivo", "tema_negocio": "App e estabilidade", "predicao_incerta": False},
            {"fonte": "youtube", "sentimento_previsto": "Negativo", "tema_negocio": "Conta bloqueada e acesso", "predicao_incerta": True},
            {"fonte": "consumidor_gov", "sentimento_previsto": "Negativo", "tema_negocio": "Cartão e fatura", "predicao_incerta": False},
        ]
    )
    insights = build_business_insights(primary_model_dataframe, primary_model_dataframe, pd.DataFrame())

    assert any("friccao" in item.lower() or "atrito" in item.lower() for item in insights)
    assert any("canal" in item.lower() for item in insights)
    assert any("tema" in item.lower() for item in insights)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_regressions.py::test_render_modelos_section_surfaces_business_actions_before_technical_table -v`
Expected: FAIL if the current home still leans too much into technical model comparison.

- [ ] **Step 3: Write minimal implementation**

```python
def build_business_insights(...):
    ...
    insights = [
        f"A conversa geral ainda favorece o Nubank, mas {negative_share:.1%} da base já pede resposta em experiência e suporte.",
        f"O maior foco de fricção aparece em {top_negative_category['categoria']}, com {int(top_negative_category['quantidade'])} ocorrências negativas.",
        f"O tema mais sensível e {top_theme['tema']}, com prioridade clara no canal {top_source}.",
        f"{uncertain_share:.1%} das previsões estão em zona de baixa confiança, então parte do painel precisa ser lida como tendência, não como verdade absoluta.",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_regressions.py::test_render_modelos_section_surfaces_business_actions_before_technical_table -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py tests/test_pipeline_regressions.py
git commit -m "feat: prioritize business actions in dashboard home"
```

### Task 4: Regenerate artifacts and validate dashboard smoke test

**Files:**
- Modify: none
- Test: `C:/Users/Lucca/Documents/analise_sentimentov2/analise_sentimentos/tests/test_pipeline_regressions.py`

- [ ] **Step 1: Run the preprocessing pipeline**

Run: `python -m src.pipelines.run_preprocessing`
Expected: unified silver and dashboard base refreshed.

- [ ] **Step 2: Run model training and inference**

Run: `$env:BERTIMBAU_ENABLE_TRAINING='1'; python -m src.pipelines.run_training`
Expected: refreshed classic predictions, full-base BERT export, and updated comparison summary.

- [ ] **Step 3: Run topic and SQL loaders**

Run: `python -m src.pipelines.run_topics`
Run: `python -m src.pipelines.run_load_sqlserver`
Expected: topic and SQL artifacts aligned with the refreshed outputs.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -q`
Expected: all regression tests pass.

- [ ] **Step 5: Smoke test the dashboard**

Run: `(Invoke-WebRequest -Uri 'http://localhost:8501/' -UseBasicParsing).StatusCode`
Expected: `200`

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py src/preprocessing/text_preprocessing.py src/topics/topic_modeling.py tests/test_preprocessing.py tests/test_pipeline_regressions.py
git commit -m "feat: refine executive storytelling and text cleaning"
```
