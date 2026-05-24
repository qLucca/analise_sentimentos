# Model Comparison With ROC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** comparar `LogisticRegression`, `MultinomialNB` e `LinearSVC` no mesmo conjunto de treino/teste e gerar métricas de classificação e curva ROC multiclasse antes de partir para `BERTimbau`.

**Architecture:** reutilizar a base textual pré-processada e o mesmo `TF-IDF` para os três modelos clássicos. Centralizar o split estratificado, o treino, a avaliação textual, o cálculo de ROC one-vs-rest e o salvamento dos artefatos em módulos de `src/models`.

**Tech Stack:** `pandas`, `scikit-learn`, `matplotlib`, `joblib`

---

### Task 1: Expandir o treino para múltiplos modelos

**Files:**
- Modify: `C:\Users\Lucca\Documents\New project\nubank-sentiment-analysis\src\models\train_sentiment_model.py`
- Test: validação manual via execução local do pipeline de treino

- [ ] Adicionar importações de `MultinomialNB` e `LinearSVC`.
- [ ] Criar uma estrutura de fábrica simples para instanciar os três modelos clássicos.
- [ ] Garantir que os três modelos usem o mesmo `train_test_split` estratificado e o mesmo `TF-IDF`.
- [ ] Retornar predições e scores necessários para avaliação posterior, além do modelo e do vetorizador.

### Task 2: Expandir a avaliação com ROC multiclasse

**Files:**
- Modify: `C:\Users\Lucca\Documents\New project\nubank-sentiment-analysis\src\models\evaluate_model.py`
- Test: validação manual via execução local da avaliação

- [ ] Adicionar cálculo de `classification_report`, `confusion_matrix`, `accuracy`, `precision_macro`, `recall_macro` e `f1_macro`.
- [ ] Adicionar cálculo de `roc_auc` multiclasse no formato one-vs-rest.
- [ ] Tratar a diferença entre modelos com `predict_proba` e modelos com `decision_function`.
- [ ] Retornar estrutura serializável com métricas por modelo.

### Task 3: Gerar e salvar gráfico ROC

**Files:**
- Create: `C:\Users\Lucca\Documents\New project\nubank-sentiment-analysis\src\models\plot_roc_curve.py`
- Test: validação manual abrindo o PNG gerado

- [ ] Criar função para binarizar as classes `Negativo`, `Neutro` e `Positivo`.
- [ ] Gerar curvas ROC one-vs-rest por classe.
- [ ] Salvar PNG por modelo em diretório de artefatos do projeto.
- [ ] Retornar caminho do gráfico salvo para facilitar o resumo final.

### Task 4: Integrar tudo em um fluxo de comparação

**Files:**
- Modify: `C:\Users\Lucca\Documents\New project\nubank-sentiment-analysis\src\models\predict_sentiment.py`
- Create or Modify: `C:\Users\Lucca\Documents\New project\nubank-sentiment-analysis\src\models\run_model_comparison.py`
- Test: execução local completa do fluxo

- [ ] Ler a base `textual_dataset_preprocessed.csv`.
- [ ] Separar dados rotulados e não rotulados.
- [ ] Rodar os três modelos no mesmo split.
- [ ] Salvar tabela consolidada de métricas e curva ROC de cada modelo.
- [ ] Selecionar o melhor modelo clássico para posterior predição no YouTube e comparação com `BERTimbau`.

### Task 5: Verificação final

**Files:**
- No direct file edits required

- [ ] Executar o script de comparação ponta a ponta.
- [ ] Confirmar que as métricas foram salvas e que os PNGs de ROC existem.
- [ ] Registrar no resumo final qual modelo clássico ficou melhor e quais limitações permaneceram, especialmente na classe `Neutro`.
