# Dashboard

O dashboard em Streamlit consome preferencialmente dados finais da camada `gold`, seja por arquivos locais em `data/gold/dashboard/` ou por views do schema `gold` no SQL Server.

A primeira versao inclui:

- metricas gerais
- sentimentos por fonte
- evolucao temporal
- resumo por fonte
- comparacao de modelos
- recortes do dataset do YouTube previsto pelo BERTimbau

Regra de organizacao:

- arquivos necessarios para a aplicacao devem ficar em `data/gold/dashboard/`
- saidas experimentais de notebooks nao devem ser usadas diretamente pelo app sem antes virarem artefatos oficiais da gold
