# Design: Validacao de Modelos e Insights no Streamlit

## Objetivo

Evoluir o projeto para:

- validar de forma consistente os modelos classicos e o BERTimbau
- gerar artefatos `gold/dashboard` consumiveis pelo Streamlit
- expor no dashboard comparacao de modelos e insights analiticos prontos para leitura academica

## Escopo

Este trabalho cobre:

- consolidacao da etapa de avaliacao de modelos
- reativacao do fluxo comparativo do BERTimbau como benchmark oficial
- geracao de um resumo estruturado de metricas para o dashboard
- exibicao de insights explicativos dentro do Streamlit

Este trabalho nao cobre:

- redesenho completo da coleta
- refatoracao ampla do dashboard fora da secao de modelos e insights
- tuning pesado de hiperparametros

## Estado Atual

Hoje o projeto possui:

- pipeline local funcional para preprocessamento, treino classico, topicos e carga SQL Server
- dashboard Streamlit lendo arquivos em `data/gold/dashboard/`
- comparacao visual de modelos no dashboard, mas dependente de um CSV incompleto e de logica antiga
- scripts experimentais em `src/training/` para modelos classicos e BERTimbau, sem um fluxo unico e reproduzivel

Os principais gaps atuais sao:

- falta uma etapa unica que compare modelos com o mesmo protocolo de validacao
- o BERTimbau ainda nao participa do pipeline oficial consumido pelo dashboard
- os insights do dashboard estao mais descritivos do que analiticamente ancorados nas metricas reais e nos topicos gerados

## Abordagens Consideradas

### 1. Validar apenas modelos classicos

Vantagens:

- menor complexidade
- execucao mais barata e mais rapida

Desvantagens:

- enfraquece a comparacao academica
- deixa o dashboard sem a referencia semantica mais forte

### 2. Validar classicos e BERTimbau em um fluxo comparativo unico

Vantagens:

- melhor equilibrio entre rigor analitico e utilidade pratica
- fortalece a narrativa de benchmark
- permite escolher e justificar o melhor modelo no dashboard

Desvantagens:

- exige reativar e padronizar a avaliacao do BERTimbau
- aumenta o tempo de execucao

### 3. Destacar apenas o BERTimbau

Vantagens:

- painel mais enxuto
- narrativa mais direta

Desvantagens:

- perde baseline reproduzivel e comparabilidade
- reduz confianca na escolha do modelo

## Abordagem Recomendada

Adotar a abordagem 2.

O sistema passara a tratar os modelos classicos e o BERTimbau como candidatos oficiais avaliados sob um protocolo comum. O pipeline gerara um resumo unificado de metricas e o dashboard usara esse resumo para mostrar ranking, justificativa do melhor modelo e insights interpretativos.

## Design Tecnico

## 1. Camada de avaliacao de modelos

Criar uma camada de avaliacao oficial em `src/models/` ou modulo adjacente que:

- leia a base `silver` textual rotulada
- execute split estratificado consistente
- avalie os modelos classicos atuais
- reative a avaliacao do BERTimbau sem depender de notebook/script solto
- produza metricas padronizadas por modelo

Saidas esperadas:

- `data/gold/dashboard/model_comparison_summary.csv`
- opcionalmente artefatos auxiliares de apoio, como curvas ROC e matrizes de confusao serializadas para exibicao futura

Colunas minimas do resumo:

- `modelo`
- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `roc_auc_macro`
- `roc_auc_weighted`
- `f1_negativo`
- `f1_neutro`
- `f1_positivo`
- `modelo_principal`
- `insight_resumo`

## 2. Escolha do modelo principal

O pipeline deve escolher explicitamente o modelo principal para consumo analitico.

Criterio recomendado:

- priorizar `roc_auc_macro`
- desempatar por `f1_macro`
- desempatar por `accuracy`

Esse criterio precisa ser centralizado em codigo para que:

- o dashboard nao adivinhe o vencedor
- a previsao final do YouTube e os insights usem a mesma decisao do pipeline

## 3. Dataset final para dashboard

O dashboard deve continuar lendo artefatos em `data/gold/dashboard/`, mas com contratos mais claros:

- `unified_dataset.csv` para panorama consolidado
- `model_comparison_summary.csv` para comparacao e narrativa dos modelos
- dataset previsto do modelo principal para distribuicoes, exemplos e exploracao por sentimento

Se o modelo principal for o BERTimbau, o dashboard deve apontar para o arquivo previsto dele como referencia principal.

## 4. Insights no Streamlit

O Streamlit deve ganhar uma leitura mais analitica dos resultados, com foco em interpretacao e nao apenas tabela.

Elementos recomendados:

- card com modelo vencedor e justificativa curta
- comparativo de metricas com destaque visual do melhor valor por coluna
- insight textual automatico sobre desempenho geral e fragilidade por classe
- secao com principais topicos negativos e sua distribuicao
- exemplos recentes de comentarios por sentimento
- leitura executiva de volumes, proporcoes e principal foco de friccao

## 5. Integracao com topicos

Os insights do dashboard devem usar a saida de `gold.topic_analysis` e/ou `gold_sentiment_analysis.csv` para mostrar:

- topicos negativos dominantes
- palavras-chave associadas
- volume por topico

Isso liga a validacao do modelo com interpretabilidade de negocio.

## Fluxo de Dados

```text
silver rotulada
  -> avaliacao classicos
  -> avaliacao bertimbau
  -> consolidacao de metricas
  -> escolha do modelo principal
  -> previsao final do dataset analitico
  -> topicos sobre saida prevista
  -> artefatos gold/dashboard
  -> Streamlit
```

## Tratamento de Erros

- se o BERTimbau nao puder rodar por dependencia ausente, o pipeline deve falhar com mensagem clara ou registrar fallback explicito, sem fingir comparacao completa
- se o CSV de comparacao estiver sem colunas obrigatorias, o dashboard deve mostrar erro orientativo
- se nao houver dados suficientes para alguma classe, a avaliacao deve registrar isso em `insight_resumo`

## Testes

Adicionar testes para:

- schema do `model_comparison_summary.csv`
- criterio de escolha do modelo principal
- compatibilidade do dashboard com o novo resumo
- geracao de insights textuais com dataset minimo

## Riscos

- custo e tempo de execucao do BERTimbau
- inconsistencias entre dataset previsto do modelo principal e dataset usado nos insights
- dependencia do dashboard em colunas antigas

Mitigacoes:

- manter o contrato de colunas explicitamente testado
- centralizar a escolha do modelo principal
- tratar o BERTimbau como etapa oficial, nao como script paralelo

## Sucesso Esperado

Ao final, o projeto deve permitir:

- comparar modelos com metricas consistentes
- justificar qual modelo sera usado na analise final
- mostrar no Streamlit insights claros sobre sentimento e topicos
- manter `gold` e SQL Server alinhados com o consumo analitico
