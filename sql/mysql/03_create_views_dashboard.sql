USE gold;

CREATE OR REPLACE VIEW vw_source_summary AS
SELECT
    fonte,
    COUNT(*) AS total_registros,
    SUM(CASE WHEN sentimento_previsto_bert = 'Positivo' THEN 1 ELSE 0 END) AS positivos,
    SUM(CASE WHEN sentimento_previsto_bert = 'Neutro' THEN 1 ELSE 0 END) AS neutros,
    SUM(CASE WHEN sentimento_previsto_bert = 'Negativo' THEN 1 ELSE 0 END) AS negativos
FROM sentiment_analysis
GROUP BY fonte;
