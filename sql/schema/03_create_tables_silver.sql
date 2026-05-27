USE NubankSentimentAnalysis;
GO

IF OBJECT_ID('silver.reviews_cleaned', 'U') IS NOT NULL
BEGIN
    DROP TABLE silver.reviews_cleaned;
END;
GO

CREATE TABLE silver.reviews_cleaned (
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
    data_processamento DATETIME2 NOT NULL DEFAULT SYSDATETIME()
);
GO
