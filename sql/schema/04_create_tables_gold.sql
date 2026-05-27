USE NubankSentimentAnalysis;
GO

IF OBJECT_ID('gold.sentiment_analysis', 'U') IS NOT NULL
BEGIN
    DROP TABLE gold.sentiment_analysis;
END;
GO

CREATE TABLE gold.sentiment_analysis (
    id_registro NVARCHAR(100) NOT NULL PRIMARY KEY,
    fonte NVARCHAR(50) NOT NULL,
    data_publicacao DATE NULL,
    texto_original NVARCHAR(MAX) NULL,
    texto_limpo NVARCHAR(MAX) NULL,
    nota FLOAT NULL,
    status_reclamacao NVARCHAR(100) NULL,
    categoria_problema NVARCHAR(255) NULL,
    uf NVARCHAR(10) NULL,
    versao_app NVARCHAR(50) NULL,
    sentimento_real NVARCHAR(20) NULL,
    sentimento_previsto NVARCHAR(20) NULL,
    topico NVARCHAR(255) NULL,
    data_processamento DATETIME2 NOT NULL DEFAULT SYSDATETIME()
);
GO

IF OBJECT_ID('gold.topic_analysis', 'U') IS NOT NULL
BEGIN
    DROP TABLE gold.topic_analysis;
END;
GO

CREATE TABLE gold.topic_analysis (
    id_topico INT NOT NULL PRIMARY KEY,
    nome_topico NVARCHAR(255) NOT NULL,
    palavras_chave NVARCHAR(MAX) NULL,
    quantidade_registros INT NOT NULL,
    sentimento_predominante NVARCHAR(20) NULL,
    fonte_predominante NVARCHAR(50) NULL,
    data_processamento DATETIME2 NOT NULL DEFAULT SYSDATETIME()
);
GO

IF OBJECT_ID('gold.model_metrics', 'U') IS NOT NULL
BEGIN
    DROP TABLE gold.model_metrics;
END;
GO

CREATE TABLE gold.model_metrics (
    id_execucao NVARCHAR(100) NOT NULL PRIMARY KEY,
    modelo NVARCHAR(100) NOT NULL,
    vetorizador NVARCHAR(100) NOT NULL,
    accuracy FLOAT NULL,
    precision_macro FLOAT NULL,
    recall_macro FLOAT NULL,
    f1_macro FLOAT NULL,
    data_treinamento DATETIME2 NOT NULL,
    observacoes NVARCHAR(MAX) NULL
);
GO
