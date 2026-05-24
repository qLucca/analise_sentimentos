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
