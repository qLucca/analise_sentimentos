# Pipeline

1. Coleta ou ingestao dos dados brutos em `data/raw/`
2. Padronizacao local e unificacao em `data/bronze/`
3. Limpeza textual, normalizacao e geracao da camada `data/silver/`
4. Treinamento e inferencia de sentimentos a partir da silver
5. Geracao de saidas analiticas e datasets de consumo em `data/gold/`
6. Extracao de topicos
7. Carga da silver e da gold no SQL Server
8. Consumo analitico no dashboard Streamlit

## Convencao de pastas

- `data/bronze/unified/`: bases unificadas apos coleta e padronizacao inicial
- `data/silver/unified/`: bases limpas e prontas para modelagem
- `data/silver/preprocessing/`: saidas intermediarias de limpeza textual
- `data/gold/dashboard/`: arquivos finais consumidos pelo dashboard
- `data/gold/analytics/`: saidas analiticas finais
- `data/gold/topics/`: resultados de modelagem de topicos
- `data/sandbox/notebooks/`: dados gerados em exploracao e experimentacao
