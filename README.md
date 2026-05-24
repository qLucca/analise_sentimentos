# Nubank Sentiment Analysis

Projeto técnico para operacionalizar o trabalho acadêmico "Análise de Sentimentos sobre o Nubank: Atendimento ao Cliente e Segurança em Plataformas Digitais".

## Problema de pesquisa

Como a percepção de consumidores brasileiros sobre o atendimento ao cliente e a segurança do Nubank se distribui entre sentimentos positivos, negativos e neutros nas plataformas Google Play Store, Reclame Aqui e Consumidor.gov.br, no período de outubro de 2025 a março de 2026, e quais tópicos recorrentes estão associados a cada categoria de sentimento?

## Objetivo

Desenvolver um pipeline completo de Ciência de Dados para coletar, padronizar, tratar, classificar e visualizar avaliações e reclamações públicas sobre atendimento ao cliente e segurança do Nubank.

## Fontes de dados

- Google Play Store
- Reclame Aqui
- Consumidor.gov.br

## Método de coleta por fonte

- `Google Play Store`: biblioteca `google-play-scraper`. Justificativa: método mais simples e estável para avaliações públicas do app, sem assumir acesso à API oficial do desenvolvedor.
- `Reclame Aqui`: scraping com `requests` + `BeautifulSoup` como primeira tentativa e `Selenium` apenas se houver dependência de JavaScript, paginação dinâmica ou bloqueio estrutural.
- `Consumidor.gov.br`: ingestão de dados abertos ou arquivo público baixado manualmente/automaticamente. `Selenium` fica como alternativa para automação de navegação/download, se necessário.

## Arquitetura

```text
Fontes públicas
  -> scripts Python de coleta
  -> data/raw/
  -> padronização local
  -> data/bronze/
  -> NLP e limpeza
  -> SQL Server / silver
  -> modelagem de sentimentos
  -> clustering de tópicos
  -> SQL Server / gold
  -> Streamlit
```

## Por que manter dados brutos fora do SQL Server

- Preserva a rastreabilidade da coleta original por fonte.
- Evita levar ruído operacional e campos desnecessários para o banco analítico.
- Mantém `raw` e `bronze` como camadas locais de ingestão e padronização inicial.
- Usa SQL Server apenas para dados tratados, resultados analíticos e consumo do dashboard.

## Estrutura do projeto

O projeto foi organizado em camadas de coleta, pré-processamento, modelagem, tópicos, carga em banco, dashboard e documentação técnica. A pasta `data/raw/` armazena apenas dados brutos, `data/bronze/` guarda arquivos padronizados locais, e os schemas `silver` e `gold` concentram a camada analítica no SQL Server.

## Configuração do ambiente

Versão recomendada do interpretador:

- `Python 3.11`

Justificativa:

- boa compatibilidade com `pandas`, `scikit-learn`, `nltk`, `pyodbc`, `sqlalchemy` e `streamlit`
- reduz risco de incompatibilidade de bibliotecas de NLP e ML em ambiente local Windows

1. Crie e ative um ambiente virtual com `Python 3.11`.
2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Copie `.env.example` para `.env` e preencha as credenciais do SQL Server.
4. Execute os scripts SQL da pasta `sql/` para criar banco, schemas, tabelas e views.

Exemplo de criação do ambiente no Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Execução dos scripts

Executar por etapa:

```bash
python -m src.pipelines.run_ingestion
python -m src.pipelines.run_preprocessing
python -m src.pipelines.run_training
python -m src.pipelines.run_topics
python -m src.pipelines.run_load_sqlserver
python -m src.pipelines.run_full_pipeline
```

## Dashboard

Para iniciar o dashboard:

```bash
streamlit run dashboard/app.py
```

O dashboard foi desenhado para consultar preferencialmente views e tabelas do SQL Server no schema `gold`.

## Limitações conhecidas

- A estrutura do Reclame Aqui pode mudar e exigir ajuste no scraping.
- Algumas fontes podem impor bloqueios, paginação limitada ou dependência de JavaScript.
- O período de outubro de 2025 a março de 2026 dependerá da disponibilidade pública dos registros coletados.
- Quando a automação total não for viável, o projeto documenta caminhos de ingestão manual sem descaracterizar a proposta acadêmica.
- O ambiente local precisa usar um interpretador compatível com NLP/ML; a base do projeto foi preparada para `Python 3.11`.
