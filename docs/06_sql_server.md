# SQL Server

O SQL Server é usado apenas para camadas tratadas e analíticas:

- `silver.reviews_cleaned`
- `gold.sentiment_analysis`
- `gold.topic_analysis`
- `gold.model_metrics`
- views `gold.*` para o dashboard

Os dados brutos e bronze permanecem fora do banco para manter rastreabilidade e evitar carga desnecessária de dados operacionais.
