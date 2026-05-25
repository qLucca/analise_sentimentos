USE NubankSentimentAnalysis;
GO

SELECT 'silver.reviews_cleaned' AS tabela, COUNT(*) AS total_registros FROM silver.reviews_cleaned
UNION ALL
SELECT 'gold.sentiment_analysis', COUNT(*) FROM gold.sentiment_analysis
UNION ALL
SELECT 'gold.topic_analysis', COUNT(*) FROM gold.topic_analysis
UNION ALL
SELECT 'gold.model_metrics', COUNT(*) FROM gold.model_metrics;
GO

SELECT fonte, COUNT(*) AS quantidade
FROM silver.reviews_cleaned
GROUP BY fonte;
GO

SELECT sentimento_previsto, COUNT(*) AS quantidade
FROM gold.sentiment_analysis
GROUP BY sentimento_previsto;
GO

SELECT
    SUM(CASE WHEN texto_limpo IS NULL OR LTRIM(RTRIM(texto_limpo)) = '' THEN 1 ELSE 0 END) AS registros_sem_texto_limpo,
    SUM(CASE WHEN categoria_problema IS NULL THEN 1 ELSE 0 END) AS registros_sem_categoria_problema,
    SUM(CASE WHEN status_reclamacao IS NULL THEN 1 ELSE 0 END) AS registros_sem_status_reclamacao,
    SUM(CASE WHEN uf IS NULL THEN 1 ELSE 0 END) AS registros_sem_uf
FROM silver.reviews_cleaned;
GO

SELECT TOP 20
    id_registro,
    fonte,
    categoria_problema,
    status_reclamacao,
    uf,
    versao_app
FROM silver.reviews_cleaned
ORDER BY data_processamento DESC;
GO
