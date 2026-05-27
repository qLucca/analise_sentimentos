USE NubankSentimentAnalysis;
GO

CREATE OR ALTER VIEW gold.vw_general_metrics AS
SELECT
    COUNT(*) AS total_registros,
    MIN(data_publicacao) AS periodo_inicial,
    MAX(data_publicacao) AS periodo_final,
    COUNT(DISTINCT fonte) AS total_fontes
FROM gold.sentiment_analysis;
GO

CREATE OR ALTER VIEW gold.vw_sentiment_by_source AS
SELECT
    fonte,
    sentimento_previsto,
    COUNT(*) AS quantidade
FROM gold.sentiment_analysis
GROUP BY fonte, sentimento_previsto;
GO

CREATE OR ALTER VIEW gold.vw_sentiment_by_month AS
SELECT
    CAST(DATEFROMPARTS(YEAR(data_publicacao), MONTH(data_publicacao), 1) AS DATE) AS mes_referencia,
    sentimento_previsto,
    COUNT(*) AS quantidade
FROM gold.sentiment_analysis
WHERE data_publicacao IS NOT NULL
GROUP BY DATEFROMPARTS(YEAR(data_publicacao), MONTH(data_publicacao), 1), sentimento_previsto;
GO

CREATE OR ALTER VIEW gold.vw_topics_by_sentiment AS
SELECT
    sentimento_previsto,
    topico,
    COUNT(*) AS quantidade
FROM gold.sentiment_analysis
WHERE topico IS NOT NULL
GROUP BY sentimento_previsto, topico;
GO

CREATE OR ALTER VIEW gold.vw_negative_topics AS
SELECT
    topico,
    COUNT(*) AS quantidade
FROM gold.sentiment_analysis
WHERE sentimento_previsto = 'Negativo'
GROUP BY topico;
GO

CREATE OR ALTER VIEW gold.vw_source_summary AS
SELECT
    fonte,
    COUNT(*) AS total_registros,
    SUM(CASE WHEN sentimento_previsto = 'Positivo' THEN 1 ELSE 0 END) AS positivos,
    SUM(CASE WHEN sentimento_previsto = 'Neutro' THEN 1 ELSE 0 END) AS neutros,
    SUM(CASE WHEN sentimento_previsto = 'Negativo' THEN 1 ELSE 0 END) AS negativos
FROM gold.sentiment_analysis
GROUP BY fonte;
GO
