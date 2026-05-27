# Arquitetura Medalhao

## Objetivo

Definir com clareza onde cada tipo de dado deve ficar para evitar mistura entre dados brutos, dados tratados, saidas analiticas e artefatos exploratorios.

## Regras

- `src/` guarda apenas codigo reutilizavel e entrypoints do pipeline.
- `data/` guarda dados operacionais do pipeline.
- `artifacts/` guarda figuras, relatorios e modelos gerados em experimentos ou estudos.
- `dashboard/` guarda a aplicacao Streamlit.
- `notebooks/` guarda exploracao e comparativos.

## Camadas de dados

### Raw

- caminho base: `data/raw/`
- conteudo: arquivos coletados por fonte sem tratamento analitico
- exemplos: comentarios do YouTube, reviews da Google Play, base do Consumidor.gov

### Bronze

- caminho base: `data/bronze/`
- conteudo: dados padronizados e unificados apos ingestao
- objetivo: consolidar schema e rastreabilidade entre fontes

### Silver

- caminho base: `data/silver/`
- conteudo: dados tratados, limpos e prontos para modelagem
- objetivo: remover ruido e preparar a base para NLP, metricas e carga analitica

### Gold

- caminho base: `data/gold/`
- conteudo: datasets finais de consumo analitico
- objetivo: alimentar dashboard, SQL Server e saidas finais do projeto

## Pastas recomendadas

- `data/bronze/unified/`
- `data/silver/unified/`
- `data/silver/preprocessing/`
- `data/gold/dashboard/`
- `data/gold/analytics/`
- `data/gold/topics/`

## Notebook e exploracao

- `data/sandbox/notebooks/`: dados temporarios gerados em notebooks
- `artifacts/figures/notebooks/`: graficos exportados
- `artifacts/reports/notebooks/`: CSVs e tabelas gerados em estudos

## Regra pratica

- se o arquivo alimenta o app, o SQL Server ou a etapa oficial do pipeline, ele deve ficar em `gold`
- se o arquivo foi gerado em exploracao e ainda nao virou entrega oficial, ele deve ficar em `sandbox` ou `artifacts`
