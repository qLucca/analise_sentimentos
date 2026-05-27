# Deploy no Azure com Databricks

## Objetivo
Este documento descreve uma arquitetura Azure recomendada para o projeto `analise_sentimentos`, com foco em:

- arquitetura medalhao no storage
- execucao dos pipelines em Azure Databricks
- persistencia opcional em Azure SQL Database
- publicacao do dashboard Streamlit fora do Databricks

A proposta parte do estado atual do repositorio e tenta preservar a estrutura ja existente em `src/`, `data/`, `sql/` e `dashboard/`.

## Visao Geral da Arquitetura

### Componentes principais
- `Azure Data Lake Storage Gen2`
  - armazena as camadas `raw`, `bronze`, `silver` e `gold`
- `Azure Databricks`
  - executa ingestao, preprocessamento, treino, analise de topicos e geracao de datasets finais
- `Azure SQL Database`
  - opcional para persistir `silver` e `gold` em formato relacional
- `Azure App Service`
  - hospeda o `dashboard/app.py`

### Arquitetura alvo
1. Os dados entram no Data Lake na camada `raw`.
2. Os jobs do Databricks processam a camada `raw` para `bronze`.
3. O preprocessamento gera a camada `silver`.
4. Os jobs de modelagem e topicos geram a camada `gold`.
5. Opcionalmente, os datasets `silver` e `gold` sao carregados no Azure SQL Database.
6. O dashboard Streamlit consome os dados finais do SQL ou da camada `gold`.

## Recomendacao para este projeto

### O que vai para o Databricks
- `src/collectors/`
- `src/preprocessing/`
- `src/features/`
- `src/topics/`
- `src/pipelines/`
- notebooks analiticos que vao virar apoio de engenharia ou modelagem

### O que fica fora do Databricks inicialmente
- `dashboard/app.py`
- hospedagem do dashboard
- scripts locais do IDE e configuracoes da maquina

### O que precisa de atencao antes da migracao
- `src/pipelines/run_training.py` ainda depende de `src.models.*`
- esse ponto deve ser resolvido antes de fechar a parte de treino no Azure

## Mapeamento do repositorio para Azure

### Estrutura local atual
O projeto ja usa a organizacao abaixo em `src/utils/paths.py`:

- `data/raw/`
- `data/bronze/`
- `data/silver/`
- `data/gold/`
- `data/sandbox/`
- `artifacts/`

### Estrutura recomendada no Data Lake
No Azure Data Lake Storage Gen2, use uma estrutura equivalente:

```text
abfss://nubank-sentiment@<storage-account>.dfs.core.windows.net/
├─ raw/
│  ├─ google_play/
│  ├─ youtube/
│  └─ consumidor_gov/
├─ bronze/
│  └─ unified/
├─ silver/
│  ├─ preprocessing/
│  ├─ unified/
│  └─ sentiment/
├─ gold/
│  ├─ analytics/
│  ├─ topics/
│  ├─ dashboard/
│  └─ sqlserver/
├─ sandbox/
│  └─ notebooks/
└─ artifacts/
   ├─ models/
   ├─ figures/
   └─ reports/
```

### Mapeamento de caminhos importantes
Mapeamento sugerido entre o projeto local e o lake:

| Local | Azure Data Lake |
|---|---|
| `data/raw/` | `raw/` |
| `data/bronze/unified/dados_unificados_bronze.csv` | `bronze/unified/dados_unificados_bronze.csv` |
| `data/silver/unified/dados_unificados_silver.csv` | `silver/unified/dados_unificados_silver.csv` |
| `data/silver/preprocessing/textual_dataset_preprocessed.csv` | `silver/preprocessing/textual_dataset_preprocessed.csv` |
| `data/gold/analytics/gold_sentiment_analysis.csv` | `gold/analytics/gold_sentiment_analysis.csv` |
| `data/gold/topics/gold_topic_analysis.csv` | `gold/topics/gold_topic_analysis.csv` |
| `data/gold/dashboard/unified_dataset.csv` | `gold/dashboard/unified_dataset.csv` |
| `data/gold/dashboard/model_comparison_summary.csv` | `gold/dashboard/model_comparison_summary.csv` |
| `data/gold/dashboard/youtube_with_predicted_sentiment_bertimbau.csv` | `gold/dashboard/youtube_with_predicted_sentiment_bertimbau.csv` |
| `artifacts/models/` | `artifacts/models/` |
| `artifacts/figures/notebooks/` | `artifacts/figures/notebooks/` |
| `artifacts/reports/notebooks/` | `artifacts/reports/notebooks/` |

## Mapeamento dos pipelines para Jobs do Databricks

### Job 1: ingestao
Arquivo atual:
- `src/pipelines/run_ingestion.py`

Responsabilidade:
- coletar ou consolidar os dados de entrada
- escrever arquivos na camada `raw`

Saida esperada:
- `raw/google_play/...`
- `raw/youtube/...`
- `raw/consumidor_gov/...`

### Job 2: preprocessamento
Arquivo atual:
- `src/pipelines/run_preprocessing.py`

Responsabilidade:
- ler os arquivos tratados por fonte
- harmonizar colunas
- construir `bronze`
- aplicar limpeza textual
- gerar `silver`
- publicar dataset unificado para dashboard

Saidas esperadas:
- `bronze/unified/dados_unificados_bronze.csv`
- `silver/unified/dados_unificados_silver.csv`
- `gold/dashboard/unified_dataset.csv`

### Job 3: treinamento
Arquivo atual:
- `src/pipelines/run_training.py`

Responsabilidade:
- ler `silver`
- treinar o modelo
- gerar previsoes
- salvar dataset com sentimento previsto
- gerar metricas do modelo

Saidas esperadas:
- `gold/dashboard/dados_com_sentimento_previsto.csv`
- `gold/dashboard/model_comparison_summary.csv`

Observacao:
- esse job depende da resolucao de `src.models.*`

### Job 4: topicos
Arquivo atual:
- `src/pipelines/run_topics.py`

Responsabilidade:
- ler dataset com sentimento previsto
- executar modelagem de topicos
- gerar bases analiticas finais

Saidas esperadas:
- `gold/analytics/gold_sentiment_analysis.csv`
- `gold/topics/gold_topic_analysis.csv`

### Job 5: carga SQL
Arquivo atual:
- `src/pipelines/run_load_sqlserver.py`

Responsabilidade:
- carregar `silver` e `gold` para o Azure SQL Database

Tabelas de destino esperadas:
- `silver.reviews_cleaned`
- `gold.sentiment_analysis`
- `gold.topic_analysis`
- `gold.model_metrics`

### Job 6: pipeline completo
Arquivo atual:
- `src/pipelines/run_full_pipeline.py`

Responsabilidade:
- orquestrar tudo ponta a ponta
- consolidar metricas e cargas finais

Uso recomendado:
- manter como job de orquestracao final ou job manual de validacao

## Mapeamento da camada SQL

### Scripts SQL do projeto
Os scripts atuais ficam organizados em:

- `sql/schema/`
- `sql/views/`
- `sql/validation/`

### Ordem de execucao sugerida
1. `sql/schema/01_create_database.sql`
2. `sql/schema/02_create_schemas.sql`
3. `sql/schema/03_create_tables_silver.sql`
4. `sql/schema/04_create_tables_gold.sql`
5. `sql/views/05_create_views_dashboard.sql`
6. `sql/validation/06_validation_queries.sql`

### Quando manter Azure SQL Database
Use Azure SQL Database se voce quer:
- dashboard lendo dados por SQL
- consumo relacional por outras ferramentas
- compatibilidade com o fluxo atual de `src/database/`

### Quando simplificar sem SQL no inicio
Voce pode comecar sem Azure SQL se:
- o dashboard puder ler os arquivos finais da camada `gold`
- o foco inicial for somente pipeline e ML

## Estrutura recomendada dentro do Databricks

### Catalog, schema e volumes
Uma estrutura simples e clara seria:

```text
Catalog: nubank_sentiment
Schemas:
- raw
- bronze
- silver
- gold
- artifacts
Volumes:
- raw_files
- bronze_files
- silver_files
- gold_files
- notebook_artifacts
```

### Exemplo de caminhos em volumes
Exemplos de paths no Databricks:

```text
/Volumes/nubank_sentiment/raw/raw_files/google_play/google_play_reviews.csv
/Volumes/nubank_sentiment/bronze/bronze_files/unified/dados_unificados_bronze.csv
/Volumes/nubank_sentiment/silver/silver_files/unified/dados_unificados_silver.csv
/Volumes/nubank_sentiment/gold/gold_files/dashboard/unified_dataset.csv
/Volumes/nubank_sentiment/artifacts/notebook_artifacts/figures/notebooks/
```

## Como adaptar o codigo

### Opcao 1: modo local e modo Azure
O melhor caminho para este repositorio e manter dois modos de execucao:

- `modo local`
  - continua usando `Path` e `data/...`
- `modo azure`
  - usa paths em `/Volumes/...`

### Sugestao de chave de ambiente
Criar uma variavel como:

```text
APP_ENV=local
APP_ENV=azure
```

### Comportamento esperado
- `APP_ENV=local`
  - usa `src/utils/paths.py` com caminhos locais
- `APP_ENV=azure`
  - usa caminhos do Data Lake montados em volumes

### Arquivos que mais precisarao de ajuste
- `src/utils/paths.py`
- `src/pipelines/run_ingestion.py`
- `src/pipelines/run_preprocessing.py`
- `src/pipelines/run_training.py`
- `src/pipelines/run_topics.py`
- `src/pipelines/run_load_sqlserver.py`
- `dashboard/app.py`

## Passo a passo de implantacao

### Fase 1: preparar a fundacao Azure
1. Criar o Resource Group.
2. Criar o workspace do Azure Databricks.
3. Criar o Azure Data Lake Storage Gen2.
4. Definir a convention de pastas `raw`, `bronze`, `silver`, `gold`, `artifacts`.
5. Configurar acesso do Databricks ao storage via Unity Catalog.

Resultado esperado:
- Databricks acessando o storage com governanca centralizada.

### Fase 2: preparar o projeto para rodar no Databricks
1. Resolver a dependencia ausente de `src.models.*`.
2. Ajustar `src/utils/paths.py` para suportar execucao local e Azure.
3. Garantir que os pipelines escrevam em caminhos cloud quando `APP_ENV=azure`.
4. Validar o preprocessamento e a geracao da camada `silver`.
5. Validar a geracao de `gold/dashboard`, `gold/analytics` e `gold/topics`.

Resultado esperado:
- o pipeline roda no Databricks e grava corretamente no lake.

### Fase 3: subir os jobs
1. Criar um job de ingestao.
2. Criar um job de preprocessamento.
3. Criar um job de treinamento.
4. Criar um job de topicos.
5. Criar um job de carga SQL.
6. Opcionalmente, criar um job orquestrador com a ordem completa.

Resultado esperado:
- execucao automatizada e repetivel.

### Fase 4: estruturar a camada SQL
1. Criar o Azure SQL Database.
2. Executar os scripts de `sql/schema/`.
3. Executar as views de `sql/views/`.
4. Rodar as queries de `sql/validation/`.
5. Validar a carga vinda do Databricks.

Resultado esperado:
- banco pronto para consumo analitico e dashboard.

### Fase 5: publicar o dashboard
1. Containerizar o Streamlit.
2. Subir o app no Azure App Service.
3. Definir variaveis de ambiente e strings de conexao.
4. Configurar se o app vai ler do Azure SQL ou da camada `gold`.
5. Validar filtros, datas e datasets finais.

Resultado esperado:
- dashboard publico com acesso controlado aos dados finais.

## Ordem de implementacao recomendada

### Semana 1
- fechar a arquitetura
- corrigir `run_training.py`
- preparar `paths.py` para local e Azure

### Semana 2
- criar storage
- configurar Databricks
- validar preprocessamento no lake

### Semana 3
- subir jobs de treino e topicos
- gerar a camada `gold`
- testar carga no Azure SQL

### Semana 4
- publicar o dashboard
- ajustar monitoramento e rotina operacional

## Decisoes recomendadas

### Decisao 1: manter o dashboard fora do Databricks
Recomendacao:
- sim

Motivo:
- reduz custo e acoplamento
- simplifica a operacao do app
- deixa o Databricks focado em dados e ML

### Decisao 2: usar Azure SQL no inicio
Recomendacao:
- opcional

Motivo:
- se o objetivo inicial for acelerar a migracao, o dashboard pode ler direto da camada `gold`
- se houver necessidade relacional, mantenha o Azure SQL

### Decisao 3: usar notebooks como producao
Recomendacao:
- nao

Motivo:
- notebooks devem apoiar exploracao
- pipeline oficial deve viver em `src/pipelines/`

## Riscos e cuidados

### Risco 1: caminhos locais no codigo
Se algum script continuar preso a caminhos locais, a execucao em Databricks quebra.

### Risco 2: dependencia ausente no treino
Sem resolver `src.models.*`, a etapa de treinamento fica incompleta.

### Risco 3: misturar dado oficial com artefato de notebook
Mantenha:
- `gold` para saida oficial
- `artifacts` e `sandbox` para experimentos

### Risco 4: subir tudo de uma vez
Migrar tudo em um unico movimento aumenta o risco de quebra. O melhor caminho e por fases.

## Checklist final

### Antes do deploy
- [ ] corrigir `run_training.py`
- [ ] revisar `paths.py`
- [ ] decidir se o dashboard vai ler SQL ou `gold`
- [ ] confirmar naming final dos datasets do dashboard

### Durante o deploy
- [ ] criar workspace Databricks
- [ ] criar Data Lake
- [ ] configurar Unity Catalog
- [ ] criar jobs
- [ ] validar escrita de `raw`, `bronze`, `silver` e `gold`

### Depois do deploy
- [ ] validar SQL
- [ ] publicar dashboard
- [ ] revisar logs e monitoramento
- [ ] documentar operacao recorrente

## Referencias oficiais
- Azure Databricks workspace: https://learn.microsoft.com/en-us/azure/databricks/admin/workspace/
- Unity Catalog e acesso ao storage: https://learn.microsoft.com/en-us/azure/databricks/connect/unity-catalog/
- Unity Catalog volumes: https://learn.microsoft.com/en-us/azure/databricks/volumes/
- Python wheel jobs: https://learn.microsoft.com/en-us/azure/databricks/jobs/how-to/use-python-wheels-in-workflows
- Azure SQL Database quickstart: https://learn.microsoft.com/en-us/azure/azure-sql/database/quickstart-content-reference-guide?view=azuresql
- Azure App Service custom container: https://learn.microsoft.com/en-us/azure/app-service/quickstart-custom-container
