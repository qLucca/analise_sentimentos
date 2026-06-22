USE silver;

CREATE TABLE IF NOT EXISTS reviews_cleaned (
    id_registro VARCHAR(255) PRIMARY KEY,
    fonte VARCHAR(100),
    data_publicacao DATETIME NULL,
    titulo TEXT NULL,
    texto_original LONGTEXT NULL,
    nota DECIMAL(10,2) NULL,
    usuario VARCHAR(255) NULL,
    categoria VARCHAR(255) NULL,
    status VARCHAR(255) NULL,
    sentimento_real VARCHAR(50) NULL,
    texto_limpo LONGTEXT NULL,
    data_processamento DATETIME NULL
);

USE gold;

CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id_registro VARCHAR(255) PRIMARY KEY,
    fonte VARCHAR(100),
    data_publicacao DATETIME NULL,
    titulo TEXT NULL,
    texto_original LONGTEXT NULL,
    nota DECIMAL(10,2) NULL,
    usuario VARCHAR(255) NULL,
    categoria VARCHAR(255) NULL,
    status VARCHAR(255) NULL,
    sentimento_real VARCHAR(50) NULL,
    texto_limpo LONGTEXT NULL,
    data_processamento DATETIME NULL,
    sentimento_previsto_bert VARCHAR(50) NULL
);
